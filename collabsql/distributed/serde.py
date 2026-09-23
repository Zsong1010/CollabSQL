"""JSON serialization for DQCP agent payloads over HTTP."""
from __future__ import annotations

from typing import Any

from collabsql.protocol.messages import CapabilityBroadcast, IdExchange
from collabsql.protocol.partial_result import AgentResponse, PartialResult, ResultClassification


def id_exchange_to_dict(obj: IdExchange) -> dict[str, Any]:
    return {
        "agent_id": obj.agent_id,
        "table": obj.table,
        "key_column": obj.key_column,
        "ids": list(obj.ids),
        "need_id": obj.need_id,
    }


def id_exchange_from_dict(data: dict[str, Any]) -> IdExchange:
    return IdExchange(
        agent_id=str(data.get("agent_id", "")),
        table=str(data.get("table", "")),
        key_column=str(data.get("key_column", "")),
        ids=list(data.get("ids") or []),
        need_id=str(data.get("need_id", "")),
    )


def scratchpad_to_dict(scratchpad: dict[str, IdExchange]) -> dict[str, Any]:
    return {k: id_exchange_to_dict(v) for k, v in scratchpad.items()}


def scratchpad_from_dict(data: dict[str, Any] | None) -> dict[str, IdExchange]:
    if not data:
        return {}
    return {k: id_exchange_from_dict(v) for k, v in data.items()}


def partial_result_to_dict(pr: PartialResult) -> dict[str, Any]:
    return pr.to_dict()


def partial_result_from_dict(data: dict[str, Any] | None) -> PartialResult | None:
    if not data:
        return None
    cls = data.get("classification", ResultClassification.LONG_TERM.value)
    try:
        classification = ResultClassification(cls)
    except ValueError:
        classification = ResultClassification.LONG_TERM
    return PartialResult(
        need_id=str(data.get("need_id", "")),
        agent_id=str(data.get("agent_id", "")),
        question=str(data.get("question", "")),
        columns=list(data.get("columns") or []),
        rows=list(data.get("rows") or []),
        classification=classification,
        sql_used=str(data.get("sql_used", "")),
        metadata=dict(data.get("metadata") or {}),
    )


def agent_response_to_dict(resp: AgentResponse) -> dict[str, Any]:
    return {
        "success": bool(resp.success),
        "partial_result": partial_result_to_dict(resp.partial_result) if resp.partial_result else None,
        "new_need": resp.new_need,
        "feedback": resp.feedback,
        "sql_used": resp.sql_used,
    }


def agent_response_from_dict(data: dict[str, Any]) -> AgentResponse:
    return AgentResponse(
        success=bool(data.get("success")),
        partial_result=partial_result_from_dict(data.get("partial_result")),
        new_need=data.get("new_need"),
        feedback=str(data.get("feedback", "")),
        sql_used=str(data.get("sql_used", "")),
    )


def capability_to_dict(cap: CapabilityBroadcast) -> dict[str, Any]:
    return {
        "agent_id": cap.agent_id,
        "db_path": cap.db_path,
        "tables": list(cap.tables),
        "column_summary": dict(cap.column_summary),
        "split_type": cap.split_type,
        "description": cap.description,
    }


def capability_from_dict(data: dict[str, Any]) -> CapabilityBroadcast:
    return CapabilityBroadcast(
        agent_id=str(data.get("agent_id", "")),
        db_path=str(data.get("db_path", "")),
        tables=list(data.get("tables") or []),
        column_summary=dict(data.get("column_summary") or {}),
        split_type=str(data.get("split_type", "random")),
        description=str(data.get("description", "")),
    )
