from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DqcpMessageType(str, Enum):
    """DQCP standardized message types (paper Section 3.3 / Algorithm 1)."""

    CAPABILITY_BROADCAST = "capability_broadcast"
    NEED_REQUEST = "need_request"
    NEED_RESPONSE = "need_response"
    SCRATCHPAD_UPDATE = "scratchpad_update"
    FIELD_ALIGNMENT = "field_alignment"
    TERMINATE = "terminate"

    # Backward-compatible aliases for older serialized traces
    NEED_DISPATCH = "need_request"
    PARTIAL_RESULT = "need_response"
    ID_EXCHANGE = "scratchpad_update"
    MERGE_COMPLETE = "terminate"


# Deprecated alias
QcpMessageType = DqcpMessageType


@dataclass
class CapabilityBroadcast:
    agent_id: str
    db_path: str
    tables: list[str]
    column_summary: dict[str, list[str]]
    split_type: str
    description: str

    def to_message(self) -> dict[str, Any]:
        return {
            "msg_type": DqcpMessageType.CAPABILITY_BROADCAST.value,
            "sender_id": self.agent_id,
            "payload": asdict(self),
        }


@dataclass
class FieldAlignment:
    """Align joinable fields across agents without a global schema."""

    table: str
    aligned_columns: list[str]
    participating_agents: list[str]
    join_keys: list[str] = field(default_factory=list)

    def to_message(self) -> dict[str, Any]:
        return {
            "msg_type": DqcpMessageType.FIELD_ALIGNMENT.value,
            "payload": asdict(self),
        }


@dataclass
class IdExchange:
    """Minimal cross-agent payload: only identifier lists (ID-only communication)."""

    agent_id: str
    table: str
    key_column: str
    ids: list[Any]
    need_id: str

    def size_bytes(self) -> int:
        return sum(len(str(v).encode("utf-8")) for v in self.ids)

    def to_message(self) -> dict[str, Any]:
        return {
            "msg_type": DqcpMessageType.SCRATCHPAD_UPDATE.value,
            "sender_id": self.agent_id,
            "payload": asdict(self),
        }


@dataclass
class AgentSqlResult:
    agent_id: str
    need_id: str
    success: bool
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    error: str = ""
    id_exchange: IdExchange | None = None

    def to_partial_message(self) -> dict[str, Any]:
        payload = {
            "agent_id": self.agent_id,
            "need_id": self.need_id,
            "success": self.success,
            "sql": self.sql,
            "columns": self.columns,
            "row_count": len(self.rows),
            "error": self.error,
        }
        if self.id_exchange is not None:
            payload["id_exchange"] = asdict(self.id_exchange)
        return {
            "msg_type": DqcpMessageType.NEED_RESPONSE.value,
            "sender_id": self.agent_id,
            "payload": payload,
        }
