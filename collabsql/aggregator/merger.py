from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from collabsql.aggregator.join_executor import execute_join_via_id_exchange, is_join_query
from collabsql.protocol.messages import AgentSqlResult, DqcpMessageType
from collabsql.utils.sql_exec import execute_sql, result_sets_equal


DQCP_MERGE_MARKER = "-- DQCP-MERGE --"


@dataclass
class MergeResult:
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    predicted_sql: str = ""
    participating_agents: list[str] = field(default_factory=list)
    error: str = ""
    composition_mode: str = ""
    composition_latency_ms: float = 0.0

    def to_message(self) -> dict[str, Any]:
        return {
            "msg_type": DqcpMessageType.TERMINATE.value,
            "payload": {
                "success": self.success,
                "columns": self.columns,
                "row_count": len(self.rows),
                "predicted_sql": self.predicted_sql,
                "participating_agents": self.participating_agents,
                "error": self.error,
                "composition_mode": self.composition_mode,
                "composition_latency_ms": self.composition_latency_ms,
            },
        }


class ResultMerger:
    """
    Lightweight aggregator: merges partial agent results without touching
    CollabSQL policy SQL generation or validation internals.
    """

    def merge(
        self,
        *,
        agent_results: list[AgentSqlResult],
        split_type: str,
        source_db_path: str | None = None,
        ground_truth_sql: str | None = None,
        agent_db_paths: dict[str, str] | None = None,
    ) -> MergeResult:
        ok_results = [r for r in agent_results if r.success and r.sql]
        if not ok_results:
            return MergeResult(success=False, error="No successful agent SQL to merge")

        if len(ok_results) == 1:
            r = ok_results[0]
            return MergeResult(
                success=True,
                columns=r.columns,
                rows=r.rows,
                predicted_sql=r.sql,
                participating_agents=[r.agent_id],
            )

        base_sql = ok_results[0].sql
        predicted_sql = f" {DQCP_MERGE_MARKER} ".join(r.sql for r in ok_results)
        participating = [r.agent_id for r in ok_results]

        if split_type in ("horizontal", "vertical") and is_join_query(base_sql) and agent_db_paths:
            cols, rows, _ = execute_join_via_id_exchange(base_sql, agent_db_paths)
            if rows:
                return MergeResult(
                    success=True,
                    columns=cols,
                    rows=rows,
                    predicted_sql=f"{base_sql} -- DQCP-JOIN-ID-EXCHANGE --",
                    participating_agents=participating,
                )

        if split_type == "horizontal":
            columns, rows = self._merge_horizontal(ok_results)
        elif split_type == "vertical":
            columns, rows = self._merge_vertical(ok_results, base_sql)
        else:
            columns, rows = self._merge_union(ok_results)

        return MergeResult(
            success=bool(rows or columns),
            columns=columns,
            rows=rows,
            predicted_sql=predicted_sql,
            participating_agents=participating,
        )

    def evaluate(
        self,
        merge: MergeResult,
        ground_truth_sql: str,
        eval_db_path: str,
    ) -> bool:
        if not merge.success:
            return False
        _, gt_rows, gt_err = execute_sql(eval_db_path, ground_truth_sql)
        if gt_err:
            return False
        return result_sets_equal(merge.rows, gt_rows)

    def _merge_horizontal(self, results: list[AgentSqlResult]) -> tuple[list[str], list[list[Any]]]:
        sqls = [r.sql for r in results]
        if all(_is_scalar_aggregate(s) for s in sqls):
            return self._merge_scalar_aggregates(results)
        if all(_is_count_query(s) for s in sqls):
            return self._merge_counts(results)
        if _has_order_limit(sqls[0]):
            return self._merge_order_limit(results, sqls[0])
        return self._merge_union(results)

    def _merge_vertical(
        self, results: list[AgentSqlResult], sql: str
    ) -> tuple[list[str], list[list[Any]]]:
        """Prefer the shard that produced rows; otherwise union partials."""
        with_rows = [r for r in results if r.rows]
        if len(with_rows) == 1:
            r = with_rows[0]
            return r.columns, r.rows
        if with_rows and all(_is_scalar_aggregate(r.sql) for r in with_rows):
            return self._merge_scalar_aggregates(with_rows)
        if with_rows and all(_is_count_query(r.sql) for r in with_rows):
            return self._merge_counts(with_rows)
        if with_rows and _has_order_limit(sql):
            return self._merge_order_limit(with_rows, sql)
        if with_rows:
            return self._merge_union(with_rows)
        return self._merge_union(results)

    def _merge_order_limit(self, results: list[AgentSqlResult], sql: str) -> tuple[list[str], list[list[Any]]]:
        columns, rows = self._merge_union(results)
        if not rows:
            return columns, rows
        limit = _parse_limit(sql)
        descending = _is_desc_order(sql)
        order_expr = _parse_order_expression(sql)
        if order_expr and len(columns) == 1:
            sorted_rows = sorted(rows, key=lambda r: _sort_key(r[0]), reverse=descending)
        elif order_expr and len(columns) > 1:
            sorted_rows = sorted(rows, key=lambda r: _sort_key(r[0]), reverse=descending)
        elif len(columns) == 1:
            sorted_rows = sorted(rows, key=lambda r: _sort_key(r[0]), reverse=descending)
        else:
            sorted_rows = rows
        if limit is not None:
            sorted_rows = sorted_rows[:limit]
        return columns, sorted_rows

    def _merge_scalar_aggregates(self, results: list[AgentSqlResult]) -> tuple[list[str], list[list[Any]]]:
        cols = results[0].columns or ["value"]
        fn = _dominant_agg_fn(results[0].sql)
        values: list[Any] = []
        for r in results:
            if not r.rows:
                continue
            values.append(r.rows[0][0])

        if not values:
            return cols, []

        if fn == "MAX":
            merged = max(values, key=_sort_key)
        elif fn == "MIN":
            merged = min(values, key=_sort_key)
        elif fn == "SUM":
            merged = sum(_to_number(v) for v in values)
        elif fn == "COUNT":
            merged = sum(int(_to_number(v)) for v in values)
        elif fn == "AVG":
            merged = sum(_to_number(v) for v in values) / len(values)
        else:
            return self._merge_union(results)
        return cols, [[merged]]

    def _merge_counts(self, results: list[AgentSqlResult]) -> tuple[list[str], list[list[Any]]]:
        cols = results[0].columns or ["count"]
        total = 0
        for r in results:
            if r.rows:
                total += int(_to_number(r.rows[0][0]))
        return cols, [[total]]

    def _merge_union(self, results: list[AgentSqlResult]) -> tuple[list[str], list[list[Any]]]:
        all_cols: list[str] = []
        all_rows: list[list[Any]] = []
        for r in results:
            for c in r.columns:
                if c not in all_cols:
                    all_cols.append(c)
            for row in r.rows:
                aligned = []
                for c in all_cols:
                    if c in r.columns:
                        aligned.append(row[r.columns.index(c)])
                    else:
                        aligned.append(None)
                all_rows.append(aligned)
        deduped = []
        seen = set()
        for row in all_rows:
            key = tuple(row)
            if key not in seen:
                seen.add(key)
                deduped.append(row)
        return all_cols, deduped


def _is_scalar_aggregate(sql: str) -> bool:
    upper = sql.upper()
    return bool(re.search(r"\bSELECT\s+(MAX|MIN|SUM|AVG|COUNT)\s*\(", upper))


def _is_count_query(sql: str) -> bool:
    return bool(re.search(r"\bCOUNT\s*\(", sql, re.IGNORECASE))


def _dominant_agg_fn(sql: str) -> str:
    upper = sql.upper()
    for fn in ("MAX", "MIN", "SUM", "AVG", "COUNT"):
        if re.search(rf"\b{fn}\s*\(", upper):
            return fn
    return "UNION"


def _has_order_limit(sql: str) -> bool:
    upper = sql.upper()
    return "ORDER BY" in upper and "LIMIT" in upper


def _parse_limit(sql: str) -> int | None:
    m = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _is_desc_order(sql: str) -> bool:
    m = re.search(r"\bORDER\s+BY\b(.*?)(?:\bLIMIT\b|$)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return False
    segment = m.group(1)
    return bool(re.search(r"\bDESC\b", segment, re.IGNORECASE))


def _parse_order_expression(sql: str) -> str | None:
    m = re.search(r"\bORDER\s+BY\s+(.*?)(?:\bLIMIT\b|$)", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    expr = m.group(1).strip()
    expr = re.sub(r"\b(DESC|ASC)\b", "", expr, flags=re.IGNORECASE).strip()
    return expr or None


def _to_number(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _sort_key(v: Any) -> tuple[int, Any]:
    if isinstance(v, (int, float)):
        return (0, v)
    if v is None:
        return (2, 0)
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))
