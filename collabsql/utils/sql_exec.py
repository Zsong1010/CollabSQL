from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


def _first_sql_statement(sql: str) -> str:
    """Keep a single executable statement (models sometimes emit trailing notes)."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        return ""
    # Split on semicolons outside quotes — take the first non-empty chunk.
    parts = re.split(r";\s*(?=(?:[^'\"]|'[^']*'|\"[^\"]*\")*$)", text)
    for part in parts:
        stmt = part.strip()
        if stmt:
            return stmt
    return text


def execute_sql(db_path: Path | str, sql: str) -> tuple[list[str], list[list[Any]], str | None]:
    if not sql:
        return [], [], "empty sql"
    sql = _first_sql_statement(sql)
    if not sql:
        return [], [], "empty sql"
    try:
        conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
        cur = conn.cursor()
        conn.execute("BEGIN TRANSACTION;")
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchall()]
        conn.rollback()
        conn.close()
        return cols, rows, None
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return [], [], str(exc)
    except Exception as exc:  # noqa: BLE001 — keep distributed eval resilient
        return [], [], str(exc)


def result_sets_equal(predicted: list[list[Any]], ground_truth: list[list[Any]]) -> bool:
    return {_normalize_row(r) for r in predicted} == {_normalize_row(r) for r in ground_truth}


def _normalize_row(row: list[Any]) -> tuple[Any, ...]:
    return tuple(_normalize_value(v) for v in row)


def _normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 6)
    return v


def extract_ground_truth_sql(row: Any) -> str | None:
    reward = row.get("reward_model") if hasattr(row, "get") else None
    if reward is None:
        return None
    if isinstance(reward, dict):
        return reward.get("ground_truth")
    return None
