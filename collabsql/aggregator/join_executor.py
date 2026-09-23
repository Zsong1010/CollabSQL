from __future__ import annotations

import re
from typing import Any

from collabsql.protocol.messages import AgentSqlResult
from collabsql.utils.sql_exec import execute_sql


def is_join_query(sql: str) -> bool:
    return bool(re.search(r"\bJOIN\b", sql, re.IGNORECASE))


def execute_join_via_id_exchange(
    sql: str,
    agent_db_paths: dict[str, str],
) -> tuple[list[str], list[list[Any]], list[AgentSqlResult]]:
    """
    Decompose JOIN into:
      Phase 1 — driver-table ID (+ optional sort key) collection per shard
      Phase 2 — projection lookup on join table for selected IDs only

    Cross-shard payload is ID lists (and compact sort keys), never full rows.
    """
    parsed = _parse_join_sql(sql)
    if parsed is None:
        return [], [], []

    driver_alias = parsed["driver_alias"]
    driver_table = parsed["driver_table"]
    join_alias = parsed["join_alias"]
    join_table = parsed["join_table"]
    key = parsed["join_key"]
    driver_where, join_where = _split_where(sql, driver_alias, join_alias)
    order_expr, descending, limit = _parse_order_limit(sql, driver_alias)

    if order_expr:
        phase1_select = f'"{key}", ({order_expr}) AS _collab_sort'
    else:
        phase1_select = f'"{key}"'

    phase1_sql = (
        f'SELECT {phase1_select} FROM "{driver_table}" AS {driver_alias} {driver_where}'
    ).strip()

    ranked: list[tuple[Any, Any]] = []
    phase1_results: list[AgentSqlResult] = []
    for agent_id, db_path in agent_db_paths.items():
        cols, rows, err = execute_sql(db_path, phase1_sql)
        phase1_results.append(
            AgentSqlResult(
                agent_id=agent_id,
                need_id="need_ids",
                success=err is None,
                sql=phase1_sql,
                columns=cols,
                rows=rows,
                error=err or "",
            )
        )
        if not rows:
            continue
        if order_expr and len(rows[0]) >= 2:
            for row in rows:
                if row[1] is None:
                    continue
                ranked.append((row[1], row[0]))
        else:
            for row in rows:
                ranked.append((None, row[0]))

    if not ranked:
        return [], [], phase1_results

    selected_ids = _select_top_ids(ranked, descending=descending, limit=limit)
    id_literals = ", ".join(_sql_literal(v) for v in selected_ids)
    select_clause = parsed["select_clause"]
    join_pred = f" AND {join_where}" if join_where else ""
    phase2_sql = (
        f'SELECT {select_clause} FROM "{join_table}" AS {join_alias} '
        f'WHERE "{key}" IN ({id_literals}){join_pred}'
    )

    merged_rows: list[list[Any]] = []
    out_cols: list[str] = []
    phase2_results: list[AgentSqlResult] = []
    for agent_id, db_path in agent_db_paths.items():
        cols, rows, err = execute_sql(db_path, phase2_sql)
        phase2_results.append(
            AgentSqlResult(
                agent_id=agent_id,
                need_id="need_proj",
                success=err is None,
                sql=phase2_sql,
                columns=cols,
                rows=rows,
                error=err or "",
            )
        )
        if cols and not out_cols:
            out_cols = cols
        merged_rows.extend(rows)

    deduped = _dedupe_rows(merged_rows)
    if order_expr and limit and len(out_cols) == 1:
        # Preserve global order when projection is 1:1 with selected ids
        pass
    return out_cols, deduped, phase1_results + phase2_results


def _parse_join_sql(sql: str) -> dict[str, str] | None:
    patterns = [
        (
            r"SELECT\s+(?P<select>.+?)\s+FROM\s+"
            r'"?(?P<t1>\w+)"?\s+AS\s+(?P<a1>\w+)\s+'
            r"(?:INNER\s+)?JOIN\s+"
            r'"?(?P<t2>\w+)"?\s+AS\s+(?P<a2>\w+)\s+'
            r"ON\s+(?P<on>.+?)(?:\s+WHERE|\s+ORDER|\s+GROUP|\s+LIMIT|$)"
        ),
    ]
    for pat in patterns:
        m = re.search(pat, sql, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        on = m.group("on")
        km = re.search(
            r'(?P<a1>\w+)\.(?:"(?P<k1>[^"]+)"|(\w+))\s*=\s*(?P<a2>\w+)\.(?:"(?P<k2>[^"]+)"|(\w+))',
            on,
            re.IGNORECASE,
        )
        if not km:
            continue
        key = km.group("k1") or km.group(2) or km.group("k2") or km.group(3)
        return {
            "select_clause": m.group("select").strip(),
            "driver_table": m.group("t1"),
            "driver_alias": m.group("a1"),
            "join_table": m.group("t2"),
            "join_alias": m.group("a2"),
            "join_key": key,
        }
    return None


def _split_where(sql: str, driver_alias: str, join_alias: str) -> tuple[str, str]:
    wm = re.search(r"\bWHERE\b(.*?)(?:\bORDER\b|\bGROUP\b|\bLIMIT\b|$)", sql, re.IGNORECASE | re.DOTALL)
    if not wm:
        return "", ""
    clause = wm.group(1).strip()
    parts = _split_predicates(clause)
    driver_parts: list[str] = []
    join_parts: list[str] = []
    for part in parts:
        refs_driver = bool(re.search(rf"\b{re.escape(driver_alias)}\.", part))
        refs_join = bool(re.search(rf"\b{re.escape(join_alias)}\.", part))
        cleaned = part
        if refs_driver:
            cleaned = re.sub(rf"\b{re.escape(driver_alias)}\.", "", cleaned)
            driver_parts.append(cleaned)
        elif refs_join:
            cleaned = re.sub(rf"\b{re.escape(join_alias)}\.", "", cleaned)
            join_parts.append(cleaned)
        else:
            driver_parts.append(part)
    driver_where = f"WHERE {' AND '.join(driver_parts)}" if driver_parts else ""
    join_where = " AND ".join(join_parts) if join_parts else ""
    return driver_where, join_where


def _split_predicates(clause: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    tokens = re.split(r"(\s+AND\s+)", clause, flags=re.IGNORECASE)
    for tok in tokens:
        if re.fullmatch(r"\s+AND\s+", tok, flags=re.IGNORECASE):
            if depth == 0 and buf:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(tok)
            continue
        depth += tok.count("(") - tok.count(")")
        buf.append(tok)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _parse_order_limit(sql: str, driver_alias: str) -> tuple[str | None, bool, int | None]:
    om = re.search(
        r"\bORDER\s+BY\s+(.*?)\s+LIMIT\s+(\d+)",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not om:
        ol = re.search(r"\bORDER\s+BY\s+(.*?)(?:\bLIMIT\b|$)", sql, re.IGNORECASE | re.DOTALL)
        if not ol:
            return None, False, None
        expr = ol.group(1).strip()
        expr = re.sub(r"\bDESC\b.*", "", expr, flags=re.IGNORECASE).strip()
        descending = bool(re.search(r"\bDESC\b", ol.group(1), re.IGNORECASE))
        expr = re.sub(rf"\b{re.escape(driver_alias)}\.", "", expr)
        return expr, descending, None

    expr = om.group(1).strip()
    descending = bool(re.search(r"\bDESC\b", expr, re.IGNORECASE))
    expr = re.sub(r"\bDESC\b", "", expr, flags=re.IGNORECASE).strip()
    expr = re.sub(r"\bASC\b", "", expr, flags=re.IGNORECASE).strip()
    expr = re.sub(rf"\b{re.escape(driver_alias)}\.", "", expr)
    return expr, descending, int(om.group(2))


def _select_top_ids(
    ranked: list[tuple[Any, Any]],
    *,
    descending: bool,
    limit: int | None,
) -> list[Any]:
    if ranked and ranked[0][0] is not None:
        ordered = sorted(ranked, key=lambda x: _sort_key(x[0]), reverse=descending)
        if limit is not None:
            ordered = ordered[:limit]
        seen: set[Any] = set()
        ids: list[Any] = []
        for _, id_val in ordered:
            if id_val in seen:
                continue
            seen.add(id_val)
            ids.append(id_val)
        return ids

    ids: list[Any] = []
    seen: set[Any] = set()
    for _, id_val in ranked:
        if id_val in seen:
            continue
        seen.add(id_val)
        ids.append(id_val)
    return ids


def _dedupe_rows(rows: list[list[Any]]) -> list[list[Any]]:
    deduped: list[list[Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _sort_key(v: Any) -> tuple[int, Any]:
    if v is None:
        return (-1, 0)
    if isinstance(v, (int, float)):
        return (0, v)
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"
