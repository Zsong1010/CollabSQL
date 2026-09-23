"""
Data Agent HTTP service.

Each process owns ONE agent_id (A1..AK) and only opens that agent's sqlite shards.
Coordinator never mounts these files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from collabsql.agent.collab_data_agent import CollabDataAgent
from collabsql.agent.collab_policy_wrapper import ApiCollabBackend, CachedCollabBackend
from collabsql.config import CollabSqlConfig
from collabsql.distributed.serde import agent_response_to_dict, capability_to_dict
from collabsql.utils.plan_loader import find_plan


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


AGENT_ID = _env("COLLABSQL_AGENT_ID", "A1")
FRAGMENTS_ROOT = Path(_env("COLLABSQL_FRAGMENTS_ROOT", "/data/fragments/random_n4"))
CACHED_PARQUET = Path(_env("COLLABSQL_CACHED_PARQUET", "/data/cache.parquet"))
BACKEND = _env("COLLABSQL_BACKEND", "cached")
HOST = _env("COLLABSQL_AGENT_HOST", "0.0.0.0")
PORT = int(_env("COLLABSQL_AGENT_PORT", "8100"))

app = FastAPI(title=f"CollabSQL Data Agent {AGENT_ID}", version="1.0")

_backend = None
_agent: CollabDataAgent | None = None
_bound_db_id: str = ""


def get_backend():
    global _backend
    if _backend is None:
        if BACKEND == "api":
            _backend = ApiCollabBackend(CollabSqlConfig())
        else:
            if not CACHED_PARQUET.exists():
                raise RuntimeError(f"Cached parquet not found: {CACHED_PARQUET}")
            _backend = CachedCollabBackend(CACHED_PARQUET)
    return _backend


def _load_agent(db_id: str) -> CollabDataAgent:
    global _agent, _bound_db_id
    if _agent is not None and _bound_db_id == db_id:
        return _agent
    plan = find_plan(FRAGMENTS_ROOT, db_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"No plan for db_id={db_id} under {FRAGMENTS_ROOT}")
    entry = next((a for a in plan.agents if a.agent_id == AGENT_ID), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Agent {AGENT_ID} not in plan for {db_id}")
    if not entry.db_path.exists():
        # Prefer local shard next to plan.json
        local = FRAGMENTS_ROOT / db_id / f"{db_id}_{AGENT_ID}.sqlite"
        if local.exists():
            from collabsql.utils.plan_loader import AgentPlanEntry

            entry = AgentPlanEntry(
                agent_id=entry.agent_id,
                db_path=local,
                tables=entry.tables,
                description=entry.description,
                split_type=entry.split_type,
                columns=entry.columns,
            )
        else:
            raise HTTPException(status_code=404, detail=f"Shard missing: {entry.db_path}")
    _agent = CollabDataAgent(entry=entry, backend=get_backend())
    _bound_db_id = db_id
    return _agent


class BindBody(BaseModel):
    db_id: str


class MschemaBody(BaseModel):
    db_id: str = ""
    max_examples: int = 3


class ColumnMapBody(BaseModel):
    db_id: str = ""


class SelfAssessBody(BaseModel):
    user_query: str
    sql_hint: str | None = None
    db_id: str = ""


class AnswerNeedBody(BaseModel):
    need_id: str
    question: str
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    scratchpad_keys: list[str] | None = None
    prompt: list[dict[str, str]]
    db_id: str
    data_source: str = "bird"
    question_index: int | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "agent_id": AGENT_ID,
        "fragments_root": str(FRAGMENTS_ROOT),
        "bound_db_id": _bound_db_id,
        "backend": BACKEND,
    }


@app.get("/capability")
def capability() -> dict[str, Any]:
    if _agent is None:
        raise HTTPException(status_code=400, detail="Call /bind first")
    return capability_to_dict(_agent.capability)


@app.post("/bind")
def bind(body: BindBody) -> dict[str, Any]:
    agent = _load_agent(body.db_id)
    return {
        "agent_id": AGENT_ID,
        "db_id": body.db_id,
        "db_path": str(agent.entry.db_path),
        "capability": capability_to_dict(agent.capability),
    }


@app.post("/mschema")
def mschema(body: MschemaBody) -> dict[str, Any]:
    db_id = body.db_id or _bound_db_id
    if not db_id:
        raise HTTPException(status_code=400, detail="db_id required")
    agent = _load_agent(db_id)
    return {"text": agent.to_mschema_text(max_examples=body.max_examples)}


@app.post("/column_map")
def column_map(body: ColumnMapBody) -> dict[str, Any]:
    db_id = body.db_id or _bound_db_id
    if not db_id:
        raise HTTPException(status_code=400, detail="db_id required")
    agent = _load_agent(db_id)
    return {"columns": agent._column_map()}


@app.post("/self_assess")
def self_assess(body: SelfAssessBody) -> dict[str, Any]:
    db_id = body.db_id or _bound_db_id
    if not db_id:
        raise HTTPException(status_code=400, detail="db_id required")
    agent = _load_agent(db_id)
    can, reason = agent.self_assess(body.user_query, sql_hint=body.sql_hint)
    return {"can": can, "reason": reason}


@app.post("/answer_need")
def answer_need(body: AnswerNeedBody) -> dict[str, Any]:
    from collabsql.distributed.serde import scratchpad_from_dict

    agent = _load_agent(body.db_id)
    scratchpad = scratchpad_from_dict(body.scratchpad)
    resp = agent.answer_need(
        body.need_id,
        body.question,
        scratchpad,
        body.scratchpad_keys,
        prompt=body.prompt,
        db_id=body.db_id,
        data_source=body.data_source,
        question_index=body.question_index,
    )
    return agent_response_to_dict(resp)


def main() -> None:
    global AGENT_ID
    parser = argparse.ArgumentParser(description="CollabSQL Data Agent HTTP server")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--agent-id", default=AGENT_ID)
    args = parser.parse_args()
    AGENT_ID = args.agent_id
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
