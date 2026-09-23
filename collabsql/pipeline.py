from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from collabsql.agent.collab_data_agent import CollabDataAgent
from collabsql.agent.collab_policy_wrapper import CollabPolicyBackend
from collabsql.aggregator.merger import MergeResult
from collabsql.aggregator.result_synthesis import CollabResultSynthesizer
from collabsql.coordinator.dqcp_coordinator import CollabCoordinator
from collabsql.protocol.field_align import align_fields
from collabsql.utils.plan_loader import FragmentationPlan, find_plan
from collabsql.utils.token_usage import QUERY_TOKEN_METER

_UNSET = object()


@dataclass
class PipelineResult:
    merge: MergeResult
    coord_trace: list[dict[str, Any]]
    dqcp_messages: list[dict[str, Any]]
    participating_agents: list[str]
    agent_partials: list[Any]
    coord_latency_ms: float = 0.0
    composition_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    decompose_mode: str = "rule"
    token_usage: dict[str, Any] | None = None


def build_collab_agents(plan: FragmentationPlan, backend: CollabPolicyBackend) -> dict[str, CollabDataAgent]:
    agents: dict[str, CollabDataAgent] = {}
    for entry in plan.agents:
        if not entry.db_path.exists():
            continue
        agents[entry.agent_id] = CollabDataAgent(entry=entry, backend=backend)
    return agents


def run_dqcp_pipeline(
    *,
    plan: FragmentationPlan,
    backend: CollabPolicyBackend,
    prompt: list[dict[str, str]],
    db_id: str,
    data_source: str,
    question: str,
    question_index: int | None,
    force_all_agents: bool = False,
    prefer_llm_composition: bool | None = None,
    agents: dict[str, Any] | None = None,
    agent_db_paths: dict[str, str] | None = None,
    source_db_path: Any = _UNSET,
    skip_bind: bool = False,
) -> PipelineResult:
    """
    Online DQCP (paper Steps 1–2):
      Step 1 collaborative execution (Need Stack + Scratchpad + Γ follow-ups)
      Step 2 provenance-constrained result composition (align + synthesize)

    Optional ``agents``: pre-built local or HTTP Data Agents (distributed deployment).
    Pass ``source_db_path=None`` / ``agent_db_paths={}`` so Coordinator never opens DBs.
    """
    t_e2e = time.perf_counter()
    QUERY_TOKEN_METER.reset()
    agents = agents if agents is not None else build_collab_agents(plan, backend)
    if not skip_bind:
        for agent in agents.values():
            if hasattr(agent, "bind_db"):
                agent.bind_db(db_id)
    coordinator = CollabCoordinator.from_plan(plan, agents)
    sql_hint = None
    if hasattr(backend, "peek_sql"):
        sql_hint = backend.peek_sql(db_id=db_id, question_index=question_index) or None

    t_coord = time.perf_counter()
    coord = coordinator.process_query(
        user_query=question,
        prompt=prompt,
        db_id=db_id,
        data_source=data_source,
        question_index=question_index,
        sql_hint=sql_hint,
        force_all_agents=force_all_agents,
    )
    coord_latency_ms = (time.perf_counter() - t_coord) * 1000.0

    capabilities = [a.capability for a in agents.values()]
    messages = list(coord.dqcp_messages)
    for fa in align_fields(capabilities):
        messages.append(fa.to_message())

    resolved_agent_db_paths = (
        agent_db_paths
        if agent_db_paths is not None
        else {aid: str(a.entry.db_path) for aid, a in agents.items()}
    )
    resolved_source_db = (
        str(plan.source_db) if source_db_path is _UNSET else source_db_path
    )

    synthesizer = CollabResultSynthesizer(prefer_llm=prefer_llm_composition)
    merge = synthesizer.synthesize(
        user_query=question,
        partials=coord.final_partials,
        split_type=plan.split_type,
        agent_db_paths=resolved_agent_db_paths,
        source_db_path=resolved_source_db,
    )
    messages.append(merge.to_message())
    composition_latency_ms = float(merge.composition_latency_ms or 0.0)
    e2e_latency_ms = (time.perf_counter() - t_e2e) * 1000.0

    return PipelineResult(
        merge=merge,
        coord_trace=coord.stack_trace,
        dqcp_messages=messages,
        participating_agents=coord.participating_agents,
        agent_partials=coord.final_partials,
        coord_latency_ms=round(coord_latency_ms, 3),
        composition_latency_ms=round(composition_latency_ms, 3),
        e2e_latency_ms=round(e2e_latency_ms, 3),
        decompose_mode=getattr(coord, "decompose_mode", "rule"),
        token_usage=QUERY_TOKEN_METER.as_dict(),
    )


def load_plan_for_db(fragments_dir: Path, db_id: str) -> FragmentationPlan | None:
    return find_plan(fragments_dir, db_id)
