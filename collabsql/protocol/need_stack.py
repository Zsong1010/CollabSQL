from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeedNode:
    need_id: str
    question: str
    target_agents: list[str]
    originator: str
    parent_need_id: str | None = None
    status: str = "pending"
    scratchpad_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_id": self.need_id,
            "question": self.question,
            "target_agents": self.target_agents,
            "originator": self.originator,
            "parent_need_id": self.parent_need_id,
            "status": self.status,
            "scratchpad_keys": self.scratchpad_keys,
        }


class NeedStack:
    """LIFO Need Stack Q in DQCP protocol state Ω = (Q, Σ) (paper Section 3.3)."""

    def __init__(self) -> None:
        self._stack: list[NeedNode] = []
        self._counter = 0

    def push(
        self,
        question: str,
        target_agents: list[str],
        originator: str,
        *,
        need_id: str | None = None,
        parent_need_id: str | None = None,
        scratchpad_keys: list[str] | None = None,
    ) -> NeedNode:
        node = NeedNode(
            need_id=need_id or f"need_{self._counter}",
            question=question,
            target_agents=target_agents,
            originator=originator,
            parent_need_id=parent_need_id,
            scratchpad_keys=scratchpad_keys or [],
        )
        self._stack.append(node)
        if need_id is None:
            self._counter += 1
        return node

    def push_user_need(self, question: str, target_agents: list[str]) -> NeedNode:
        node = NeedNode(
            need_id="need_0",
            question=question,
            target_agents=target_agents,
            originator="coordinator",
        )
        self._stack.insert(0, node)
        if self._counter == 0:
            self._counter = 1
        return node

    def pop(self) -> NeedNode | None:
        if not self._stack:
            return None
        return self._stack.pop()

    def is_empty(self) -> bool:
        return not self._stack
