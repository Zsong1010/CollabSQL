"""Column coverage helpers for vertical (column-sharded) fragments."""
from __future__ import annotations

import re
from typing import Mapping


def extract_sql_columns(sql: str) -> set[str]:
    if not sql:
        return set()
    return set(re.findall(r"`([^`]+)`", sql))


def agent_column_set(columns_by_table: Mapping[str, list[str]]) -> set[str]:
    cols: set[str] = set()
    for table_cols in columns_by_table.values():
        cols.update(table_cols)
    return cols


def agent_covers_sql(columns_by_table: Mapping[str, list[str]], sql: str) -> bool:
    needed = extract_sql_columns(sql)
    if not needed:
        return True
    have = agent_column_set(columns_by_table)
    return needed.issubset(have)


def select_capable_agents(
    agents: Mapping[str, Mapping[str, list[str]]],
    sql: str,
) -> list[str]:
    """Return agent_ids whose vertical shard schema covers all columns in sql."""
    if not sql:
        return sorted(agents.keys())
    capable = [aid for aid, cols in agents.items() if agent_covers_sql(cols, sql)]
    return capable or sorted(agents.keys())
