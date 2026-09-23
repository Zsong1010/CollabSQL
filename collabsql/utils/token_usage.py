"""Process-wide LLM token accounting for CollabSQL coordinator / composition calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    by_role: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        role: str = "llm",
    ) -> None:
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        t = int(total_tokens) if total_tokens is not None else p + c
        self.prompt_tokens += p
        self.completion_tokens += c
        self.total_tokens += t
        self.llm_calls += 1
        slot = self.by_role.setdefault(
            role, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        )
        slot["prompt_tokens"] += p
        slot["completion_tokens"] += c
        slot["total_tokens"] += t
        slot["llm_calls"] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "by_role": {k: dict(v) for k, v in self.by_role.items()},
        }

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0
        self.by_role.clear()


QUERY_TOKEN_METER = TokenUsage()


def record_openai_usage(usage: Any, *, role: str) -> None:
    if usage is None:
        return
    if isinstance(usage, dict):
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        t = int(usage.get("total_tokens") or (p + c))
    else:
        p = int(getattr(usage, "prompt_tokens", 0) or 0)
        c = int(getattr(usage, "completion_tokens", 0) or 0)
        t = int(getattr(usage, "total_tokens", 0) or (p + c))
    QUERY_TOKEN_METER.add(prompt_tokens=p, completion_tokens=c, total_tokens=t, role=role)


def wrap_llm_client(llm: Any, *, role: str = "composition") -> Any:
    """Monkey-patch LLMClient.chat to record token usage from API JSON."""
    if getattr(llm, "_collab_token_wrapped", False):
        return llm
    import requests

    def chat(system: str, user: str, *, response_format=None):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict[str, Any] = {"model": llm.model, "messages": messages}
        if not str(llm.model).startswith("o3"):
            payload["temperature"] = llm.temperature
        if response_format:
            payload["response_format"] = response_format
        headers = {
            "Authorization": f"Bearer {llm.api_key}",
            "Content-Type": "application/json",
        }
        session = requests.Session()
        session.trust_env = False
        resp = session.post(llm.api_url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        record_openai_usage(data.get("usage"), role=role)
        return data["choices"][0]["message"]["content"] or ""

    llm.chat = chat  # type: ignore[method-assign]
    llm._collab_token_wrapped = True
    return llm
