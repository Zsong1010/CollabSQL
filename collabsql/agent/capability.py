from __future__ import annotations

import sqlite3
from pathlib import Path

from collabsql.protocol.messages import CapabilityBroadcast


def introspect_sqlite(db_path: Path, agent_id: str, description: str, split_type: str) -> CapabilityBroadcast:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    column_summary: dict[str, list[str]] = {}
    for table in tables:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
        column_summary[table] = cols
    conn.close()
    return CapabilityBroadcast(
        agent_id=agent_id,
        db_path=str(db_path),
        tables=tables,
        column_summary=column_summary,
        split_type=split_type,
        description=description,
    )
