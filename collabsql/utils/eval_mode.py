from __future__ import annotations

import time
from typing import Any

from collabsql.aggregator.merger import DQCP_MERGE_MARKER, MergeResult
from collabsql.protocol.messages import AgentSqlResult
from collabsql.utils.sql_exec import execute_sql, result_sets_equal


def resolve_eval_mode(
    *,
    split_type: str | None,
    backend: str,
    requested: str = "source_db",
) -> str:
    """
    Vertical distributed runs must compare shard-merged rows, not replay SQL on source_db.
    """
    if backend == "vertical_cached" or split_type == "vertical":
        return "shard_rows"
    return requested


def extract_primary_sql(merge: MergeResult, agent_results: list[AgentSqlResult]) -> str:
    """Pick the synthesized SQL for integrated-DB evaluation (DQCP Result Synthesis)."""
    if merge.predicted_sql and DQCP_MERGE_MARKER not in merge.predicted_sql:
        sql = (
            merge.predicted_sql.replace("-- DQCP-JOIN-ID-EXCHANGE --", "")
            .replace("-- QCP-JOIN-ID-EXCHANGE --", "")
            .strip()
        )
        if sql.upper().startswith("SELECT"):
            return sql

    for r in agent_results:
        if r.success and r.sql and r.sql.strip().upper().startswith("SELECT"):
            return r.sql.strip()

    if merge.predicted_sql:
        parts = merge.predicted_sql
        for m in (DQCP_MERGE_MARKER, "-- QCP-MERGE --"):
            parts = parts.replace(m, "\x1e")
        for p in (x.strip() for x in parts.split("\x1e")):
            if p.upper().startswith("SELECT"):
                return p
    return ""


def evaluate_ex(
    *,
    merge: MergeResult,
    agent_results: list[AgentSqlResult],
    ground_truth_sql: str,
    eval_db_path: str,
    eval_mode: str = "source_db",
    pred_db_path: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    Evaluate execution accuracy.

    source_db (default, paper-aligned):
      Execute synthesized SQL on integrated source database — standard BIRD EX.

    shard_rows (ablation):
      Compare merged partial rows from shard execution.

    shard_union:
      Execute synthesized SQL on a DB that unions active agents' shard rows;
      compare to gold on source DB. Reflects data coverage without brittle
      result-set merging (JOIN/aggregates work if the union has the rows).
    """
    meta: dict[str, Any] = {"eval_mode": eval_mode, "sql_latency_ms": 0.0}
    if not ground_truth_sql:
        return 0, meta

    t_sql = time.perf_counter()
    _, gt_rows, gt_err = execute_sql(eval_db_path, ground_truth_sql)
    if gt_err:
        meta["gt_error"] = gt_err
        meta["sql_latency_ms"] = round((time.perf_counter() - t_sql) * 1000.0, 3)
        return 0, meta

    if eval_mode == "shard_rows":
        if not merge.success:
            meta["sql_latency_ms"] = round((time.perf_counter() - t_sql) * 1000.0, 3)
            return 0, meta
        ok = result_sets_equal(merge.rows, gt_rows)
        meta["predicted_sql"] = merge.predicted_sql
        meta["sql_latency_ms"] = round((time.perf_counter() - t_sql) * 1000.0, 3)
        return int(ok), meta

    primary_sql = extract_primary_sql(merge, agent_results)
    meta["predicted_sql"] = primary_sql
    if not primary_sql:
        meta["sql_latency_ms"] = round((time.perf_counter() - t_sql) * 1000.0, 3)
        return 0, meta

    run_db = pred_db_path if (eval_mode == "shard_union" and pred_db_path) else eval_db_path
    if eval_mode == "shard_union":
        meta["pred_db_path"] = run_db
        if not pred_db_path:
            meta["sql_error"] = "shard_union requires pred_db_path"
            meta["sql_latency_ms"] = round((time.perf_counter() - t_sql) * 1000.0, 3)
            return 0, meta

    _, pred_rows, pred_err = execute_sql(run_db, primary_sql)
    meta["sql_latency_ms"] = round((time.perf_counter() - t_sql) * 1000.0, 3)
    if pred_err:
        meta["sql_error"] = pred_err
        return 0, meta

    ok = result_sets_equal(pred_rows, gt_rows)
    meta["pred_row_count"] = len(pred_rows)
    return int(ok), meta
