from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResultClassification(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class PartialResult:
    need_id: str
    agent_id: str
    question: str
    columns: list[str]
    rows: list[list[Any]]
    classification: ResultClassification = ResultClassification.LONG_TERM
    sql_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_id": self.need_id,
            "agent_id": self.agent_id,
            "question": self.question,
            "columns": self.columns,
            "rows": self.rows,
            "classification": self.classification.value,
            "sql_used": self.sql_used,
            "metadata": self.metadata,
        }

    def to_table_str(self, max_rows: int = 20) -> str:
        header = " | ".join(self.columns)
        lines = [header, "-" * max(len(header), 1)]
        for row in self.rows[:max_rows]:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)


@dataclass
class AgentResponse:
    success: bool
    partial_result: PartialResult | None = None
    new_need: dict[str, Any] | None = None
    feedback: str = ""
    sql_used: str = ""
