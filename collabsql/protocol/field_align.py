from __future__ import annotations

from collabsql.protocol.messages import CapabilityBroadcast, FieldAlignment


def align_fields(capabilities: list[CapabilityBroadcast]) -> list[FieldAlignment]:
    """Consensus alignment on shared table/column names — no global schema required."""
    by_table: dict[str, dict[str, set[str]]] = {}
    for cap in capabilities:
        for table, columns in cap.column_summary.items():
            by_table.setdefault(table, {}).setdefault(cap.agent_id, set()).update(columns)

    alignments: list[FieldAlignment] = []
    for table, agent_cols in by_table.items():
        if len(agent_cols) < 2:
            continue
        shared = None
        participants = sorted(agent_cols.keys())
        for cols in agent_cols.values():
            shared = cols if shared is None else shared & cols
        if not shared:
            continue
        aligned = sorted(shared)
        join_keys = _pick_join_keys(aligned)
        alignments.append(
            FieldAlignment(
                table=table,
                aligned_columns=aligned,
                participating_agents=participants,
                join_keys=join_keys,
            )
        )
    return alignments


def _pick_join_keys(columns: list[str]) -> list[str]:
    preferred = ("id", "code", "key", "uuid", "pk")
    keys: list[str] = []
    for col in columns:
        lower = col.lower()
        if lower.endswith("id") or lower.endswith("code") or any(p in lower for p in preferred):
            keys.append(col)
    if not keys and columns:
        keys.append(columns[0])
    return keys[:3]
