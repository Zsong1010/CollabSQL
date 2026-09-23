"""Coordinator-side LLM helpers (paper Fig 2/3 Step 1.2 Decompose)."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

QCP_DB_ROOT = Path(
    os.environ.get("COLLABSQL_QCP_ROOT")
    or os.environ.get("QCP_DB_ROOT")
    or ""
)

QUERY_PLAN_PROMPT = """You are the Query Coordinator in CollabSQL (DQCP).
Decompose the user query into sequential natural language sub-questions (information needs).
Each sub-question should be answerable by one or more specific agents given their schemas.
Sub-questions are pushed onto a stack and processed LIFO; the original user question will be
kept as need_0 and processed last after dependency sub-needs.

Return JSON only:
{
  "needs": [
    {
      "need_id": "need_1",
      "question": "natural language sub-question",
      "target_agents": ["A1"],
      "depends_on_scratchpad": []
    }
  ],
  "reasoning": "plan explanation"
}

Order needs so dependencies are resolved first (higher need_id processed before lower).
Only use agent ids from the participating list. Prefer 1-4 focused sub-needs."""


def prefer_llm_coordinator() -> bool:
    raw = os.environ.get("COLLABSQL_LLM_COORDINATOR", "auto").strip().lower()
    if raw in {"0", "false", "no", "rule"}:
        return False
    if raw in {"1", "true", "yes", "llm"}:
        return True
    # auto: enable when an OpenAI-compatible endpoint is configured
    return bool(
        os.environ.get("COLLABSQL_LLM_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("MODEL_API_KEY")
    )


def _extract_json_obj(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _chat_json(*, system: str, user: str) -> dict[str, Any]:
    """Call local/OpenAI-compatible chat and parse JSON object."""
    api_base = (
        os.environ.get("COLLABSQL_LLM_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).rstrip("/")
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("COLLABSQL_LLM_API_KEY")
        or os.environ.get("MODEL_API_KEY")
        or "EMPTY"
    )
    if not api_base and not (api_key and api_key != "EMPTY"):
        # optional external LLM client fallback
        if not QCP_DB_ROOT.exists():
            return {}
        if str(QCP_DB_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(QCP_DB_ROOT / "src"))
        from qcp_db.config import Config
        from qcp_db.llm.client import LLMClient

        config = Config.load(QCP_DB_ROOT / "config.yaml")
        setup = os.environ.get("QCP_SETUP", "dashscope")
        models = (
            config.setup_dashscope
            if setup == "dashscope" and config.setup_dashscope
            else config.setup_a
        )
        llm = LLMClient(config, models.coordinator)
        return llm.chat_json(system, user) or {}

    from openai import OpenAI

    from collabsql.agent.collab_policy_wrapper import _resolve_vllm_model_id
    from collabsql.config import CollabSqlConfig

    model = _resolve_vllm_model_id(
        config=CollabSqlConfig(),
        api_base=api_base or "http://127.0.0.1:8001/v1",
        api_key=api_key,
    )
    # Prefer explicit served name for local CollabSQL Qwen-SQL-7B
    model = os.environ.get("COLLABSQL_LLM_MODEL_NAME") or model
    from collabsql.utils.token_usage import record_openai_usage

    client = OpenAI(base_url=api_base, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    record_openai_usage(getattr(resp, "usage", None), role="decompose")
    content = resp.choices[0].message.content or ""
    return _extract_json_obj(content)


def llm_decompose_needs(
    *,
    user_query: str,
    participating_agents: list[str],
    agent_schemas: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Paper Step 1.2: Coordinator (via LLM) decomposes into dependency-aware sub-queries.
    Returns list of need dicts; empty list means caller should fall back to rule plan.
    """
    # Prefer compact schema blurbs to keep prompt short
    schema_payload = {
        aid: (agent_schemas.get(aid) or "")[:2500] for aid in participating_agents
    }
    user = (
        f"{QUERY_PLAN_PROMPT}\n\n"
        f"User query: {user_query}\n\n"
        f"Participating agents: {participating_agents}\n\n"
        f"Agent schemas:\n{json.dumps(schema_payload, indent=2, ensure_ascii=False)}"
    )
    try:
        result = _chat_json(
            system="You are the CollabSQL Query Coordinator. Reply with JSON only.",
            user=user,
        )
    except Exception:
        return []

    needs_raw = result.get("needs") if isinstance(result, dict) else None
    if not isinstance(needs_raw, list) or not needs_raw:
        return []

    valid_agents = set(participating_agents)
    out: list[dict[str, Any]] = []
    for i, n in enumerate(needs_raw, start=1):
        if not isinstance(n, dict):
            continue
        q = str(n.get("question") or "").strip()
        if not q:
            continue
        targets = [a for a in (n.get("target_agents") or participating_agents) if a in valid_agents]
        if not targets:
            targets = list(participating_agents)
        out.append(
            {
                "need_id": str(n.get("need_id") or f"need_{i}"),
                "question": q,
                "target_agents": targets,
                "depends_on_scratchpad": list(n.get("depends_on_scratchpad") or []),
            }
        )
    return out
