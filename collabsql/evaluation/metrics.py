"""Evaluation metrics: EX and communication cost."""
from __future__ import annotations

from typing import Any


def execution_accuracy(results: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(results)
    correct = sum(int(r.get("execution_accuracy", 0)) for r in results)
    pct = round(correct / total * 100, 2) if total else 0.0
    return {"correct": correct, "total": total, "execution_accuracy_pct": pct}


CROSS_NODE_MSG_TYPES = frozenset(
    {
        "need_request",
        "need_response",
        "scratchpad_update",
        "field_alignment",
        "terminate",
        "need_dispatch",
        "id_exchange",
        "field_align",
        "agent_response",
        "partial_result",
        "merge_result",
        "merge_complete",
        "task_decomposition",
        "message_passing",
    }
)


def _protocol_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("dqcp_messages") or row.get("qcp_messages") or [])


def count_messages(dqcp_messages: list[dict[str, Any]]) -> int:
    """Count cross-node DQCP protocol messages (excludes capability broadcast)."""
    count = 0
    for msg in dqcp_messages:
        msg_type = str(msg.get("msg_type", ""))
        if msg_type == "capability_broadcast":
            continue
        if msg_type in CROSS_NODE_MSG_TYPES or msg_type:
            count += 1
    return count


def communication_cost(results: list[dict[str, Any]]) -> dict[str, float]:
    totals = [count_messages(_protocol_messages(r)) for r in results]
    n = len(totals)
    avg = round(sum(totals) / n, 2) if n else 0.0
    return {
        "messages_total": sum(totals),
        "messages_per_query": avg,
        "queries": n,
    }


def summarize_run(
    results: list[dict[str, Any]],
    *,
    method: str,
    topology: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "topology": topology,
        **execution_accuracy(results),
        **communication_cost(results),
    }
