"""
Coordinator HTTP service.

Runs existing DQCP workflow; talks to Data Agents only via HTTP.
Does not mount or open any shard databases.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from collabsql.distributed.http_agent_client import HttpCollabDataAgent
from collabsql.distributed.http_stats import SharedHttpStats
from collabsql.evaluation.metrics import count_messages
from collabsql.pipeline import run_dqcp_pipeline
from collabsql.utils.plan_loader import AgentPlanEntry, FragmentationPlan


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


AGENT_URLS = [
    u.strip()
    for u in _env(
        "COLLABSQL_AGENT_URLS",
        "http://agent-a1:8101,http://agent-a2:8102,http://agent-a3:8103,http://agent-a4:8104",
    ).split(",")
    if u.strip()
]
AGENT_IDS = [u.strip() for u in _env("COLLABSQL_AGENT_IDS", "A1,A2,A3,A4").split(",") if u.strip()]
HOST = _env("COLLABSQL_COORD_HOST", "0.0.0.0")
PORT = int(_env("COLLABSQL_COORD_PORT", "8200"))
SPLIT_TYPE = _env("COLLABSQL_SPLIT_TYPE", "random")

app = FastAPI(title="CollabSQL Coordinator", version="1.0")


class _NoOpBackend:
    """Coordinator never runs local EFPL; agents own the frozen policy."""


class QueryBody(BaseModel):
    db_id: str
    question: str
    prompt: list[dict[str, str]]
    question_index: int | None = None
    data_source: str = "bird"
    force_all_agents: bool = False
    prefer_llm_composition: bool | None = None


def _stub_plan(db_id: str, agent_ids: list[str]) -> FragmentationPlan:
    agents = [
        AgentPlanEntry(
            agent_id=aid,
            db_path=Path(f"/remote/{aid}"),
            tables=[],
            description=f"Remote HTTP agent {aid}",
            split_type=SPLIT_TYPE,
        )
        for aid in agent_ids
    ]
    return FragmentationPlan(
        db_id=db_id,
        source_db=Path("/remote/source_unavailable"),
        split_type=SPLIT_TYPE,
        num_agents=len(agents),
        agents=agents,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "agent_urls": AGENT_URLS, "agent_ids": AGENT_IDS, "split_type": SPLIT_TYPE}


@app.post("/query")
def query(body: QueryBody) -> dict[str, Any]:
    if len(AGENT_URLS) != len(AGENT_IDS):
        raise HTTPException(status_code=500, detail="COLLABSQL_AGENT_URLS / COLLABSQL_AGENT_IDS length mismatch")

    shared = SharedHttpStats()
    agents: dict[str, HttpCollabDataAgent] = {
        aid: HttpCollabDataAgent(base_url=url, agent_id=aid, shared_stats=shared)
        for aid, url in zip(AGENT_IDS, AGENT_URLS)
    }

    plan = _stub_plan(body.db_id, AGENT_IDS)
    t0 = time.perf_counter()
    try:
        result = run_dqcp_pipeline(
            plan=plan,
            backend=_NoOpBackend(),  # type: ignore[arg-type]
            prompt=body.prompt,
            db_id=body.db_id,
            data_source=body.data_source,
            question=body.question,
            question_index=body.question_index,
            force_all_agents=body.force_all_agents,
            prefer_llm_composition=body.prefer_llm_composition,
            agents=agents,  # type: ignore[arg-type]
            agent_db_paths={},
            source_db_path=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    wall_ms = (time.perf_counter() - t0) * 1000.0

    merge = result.merge
    http = shared.stats.as_dict()
    return {
        "db_id": body.db_id,
        "question_index": body.question_index,
        "success": bool(merge.success),
        "predicted_sql": merge.predicted_sql or "",
        "columns": list(merge.columns or []),
        "rows": list(merge.rows or []),
        "participating_agents": list(result.participating_agents),
        "composition_mode": getattr(merge, "composition_mode", ""),
        "decompose_mode": result.decompose_mode,
        "error": getattr(merge, "error", "") or "",
        "metrics": {
            "e2e_latency_ms": result.e2e_latency_ms,
            "coord_latency_ms": result.coord_latency_ms,
            "composition_latency_ms": result.composition_latency_ms,
            "wall_latency_ms": round(wall_ms, 3),
            "messages_per_query": count_messages(result.dqcp_messages),
            "communication_bytes": http["http_total_bytes"],
            "http_request_bytes": http["http_request_bytes"],
            "http_response_bytes": http["http_response_bytes"],
            "http_requests": http["http_requests"],
            "token_usage": result.token_usage,
        },
        "dqcp_messages": result.dqcp_messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CollabSQL Coordinator HTTP server")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
