"""Runtime follow-up instantiation Γ(q_t, r_t, B_t) for DQCP (paper Algorithm 1)."""
from __future__ import annotations

from typing import Any

from collabsql.protocol.partial_result import PartialResult


def _is_id_column(name: str) -> bool:
    low = name.lower()
    return low.endswith("id") or low.endswith("code") or low in {"pk", "key"}


def is_identifier_heavy(partial: PartialResult) -> bool:
    """True when evidence is mainly join keys (short-term scratchpad material)."""
    if not partial.columns:
        return False
    id_cols = [c for c in partial.columns if _is_id_column(c)]
    if not id_cols:
        return False
    return len(id_cols) >= max(1, len(partial.columns) - 1)


def instantiate_followups(
    *,
    user_query: str,
    partial: PartialResult,
    other_agent_ids: list[str],
    seen: set[str],
    max_id_preview: int = 8,
) -> list[dict[str, Any]]:
    """
    Paper Γ: if local evidence exposes identifiers but not full answer attributes,
    instantiate enrichment sub-queries for other capable agents.
    """
    if not other_agent_ids or not partial.rows or not is_identifier_heavy(partial):
        return []

    id_cols = [c for c in partial.columns if _is_id_column(c)]
    preview_vals: list[str] = []
    for row in partial.rows[:max_id_preview]:
        preview_vals.append(",".join(str(v) for v in row[: len(id_cols)]))
    preview = "; ".join(preview_vals)

    question = (
        f"Using identifier evidence on {', '.join(id_cols)} "
        f"(examples: {preview}), retrieve the attributes required to answer: {user_query}"
    )
    key = question.strip().lower()
    if key in seen:
        return []
    seen.add(key)
    return [
        {
            "question": question,
            "target_agents": list(other_agent_ids),
            "originator": partial.agent_id,
            "parent_need_id": partial.need_id,
            "scratchpad_keys": [f"{partial.need_id}_{partial.agent_id}"],
        }
    ]
