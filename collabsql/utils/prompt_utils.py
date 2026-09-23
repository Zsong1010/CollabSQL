from __future__ import annotations

import copy
import re
from typing import Any

from collabsql.protocol.messages import IdExchange


def _normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return copy.deepcopy(list(prompt))


def extract_question(prompt: Any) -> str:
    messages = _normalize_prompt(prompt)
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        match = re.search(r"Question:\s*(.+?)(?:\n|$)", content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if lines:
            return lines[-1]
    return ""


def inject_scratchpad(prompt: Any, scratchpad: dict[str, IdExchange]) -> list[dict[str, str]]:
    messages = _normalize_prompt(prompt)
    if not scratchpad:
        return messages

    lines = ["[DQCP Scratchpad — ID lists only, no row payloads]"]
    for key, bundle in scratchpad.items():
        preview = ", ".join(repr(v) for v in bundle.ids[:50])
        suffix = " ..." if len(bundle.ids) > 50 else ""
        lines.append(
            f"- {key}: {bundle.table}.{bundle.key_column} IN ({preview}{suffix}) "
            f"[{len(bundle.ids)} ids from agent {bundle.agent_id}]"
        )

    block = "\n".join(lines)
    for msg in messages:
        if msg.get("role") == "user":
            msg["content"] = f"{msg['content']}\n\n{block}"
            break
    return messages
