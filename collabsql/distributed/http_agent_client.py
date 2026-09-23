"""HTTP client that mirrors CollabDataAgent for the Coordinator process."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from collabsql.distributed.http_stats import HttpCommStats, SharedHttpStats
from collabsql.distributed.serde import (
    agent_response_from_dict,
    capability_from_dict,
    scratchpad_to_dict,
)
from collabsql.protocol.messages import CapabilityBroadcast, IdExchange
from collabsql.protocol.partial_result import AgentResponse
from collabsql.utils.plan_loader import AgentPlanEntry


@dataclass
class _RemoteEntry:
    """Stand-in for AgentPlanEntry so pipeline can read agent.entry.db_path."""

    agent_id: str
    db_path: Path = field(default_factory=lambda: Path("/remote"))
    tables: list[str] = field(default_factory=list)
    description: str = "remote HTTP data agent"
    split_type: str = "random"
    columns: dict[str, list[str]] | None = None


@dataclass
class HttpCollabDataAgent:
    """
    Drop-in remote proxy for CollabDataAgent.
    All DQCP agent calls go over HTTP; byte counts are recorded in shared stats.
    """

    base_url: str
    agent_id: str
    shared_stats: SharedHttpStats
    timeout_s: float = 300.0
    _db_id: str = ""
    _capability: CapabilityBroadcast | None = field(default=None, repr=False)
    entry: AgentPlanEntry | _RemoteEntry = field(init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.entry = _RemoteEntry(agent_id=self.agent_id, split_type="random")

    @property
    def capability(self) -> CapabilityBroadcast:
        if self._capability is None:
            self._capability = capability_from_dict(self._get_json("/capability"))
        return self._capability

    @property
    def stats(self) -> HttpCommStats:
        return self.shared_stats.stats

    def bind_db(self, db_id: str) -> None:
        """Tell the remote agent which shard DB to use for the next query."""
        self._db_id = db_id
        payload = self._post_json("/bind", {"db_id": db_id})
        self._capability = capability_from_dict(payload["capability"])
        self.entry = _RemoteEntry(
            agent_id=self.agent_id,
            db_path=Path(str(payload.get("db_path") or "/remote")),
            tables=list(self._capability.tables),
            description=self._capability.description,
            split_type=self._capability.split_type,
            columns=dict(self._capability.column_summary),
        )

    def to_mschema_text(self, max_examples: int = 3) -> str:
        out = self._post_json("/mschema", {"max_examples": max_examples, "db_id": self._db_id})
        return str(out.get("text", ""))

    def _column_map(self) -> dict[str, list[str]]:
        out = self._post_json("/column_map", {"db_id": self._db_id})
        return dict(out.get("columns") or {})

    def self_assess(self, user_query: str, *, sql_hint: str | None = None) -> tuple[bool, str]:
        out = self._post_json(
            "/self_assess",
            {"user_query": user_query, "sql_hint": sql_hint, "db_id": self._db_id},
        )
        return bool(out.get("can", True)), str(out.get("reason", ""))

    def answer_need(
        self,
        need_id: str,
        question: str,
        scratchpad: dict[str, IdExchange],
        scratchpad_keys: list[str] | None,
        *,
        prompt: list[dict[str, str]],
        db_id: str,
        data_source: str,
        question_index: int | None,
    ) -> AgentResponse:
        out = self._post_json(
            "/answer_need",
            {
                "need_id": need_id,
                "question": question,
                "scratchpad": scratchpad_to_dict(scratchpad),
                "scratchpad_keys": scratchpad_keys,
                "prompt": prompt,
                "db_id": db_id or self._db_id,
                "data_source": data_source,
                "question_index": question_index,
            },
        )
        return agent_response_from_dict(out)

    def _get_json(self, path: str) -> dict[str, Any]:
        return self._request("GET", path, None)

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        req_bytes = 0
        if body is not None:
            raw = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
            data = raw
            req_bytes = len(raw)
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                resp_raw = resp.read()
                self.stats.record(request_bytes=req_bytes, response_bytes=len(resp_raw))
                return json.loads(resp_raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read() if e.fp else b""
            self.stats.record(request_bytes=req_bytes, response_bytes=len(err_body))
            raise RuntimeError(f"HTTP {e.code} from {url}: {err_body[:500]!r}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"HTTP connection failed to {url}: {e}") from e
