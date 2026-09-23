"""Shared helpers for CollabSQL distributed evaluation."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from collabsql.agent.collab_data_agent import CollabDataAgent
from collabsql.agent.collab_policy_wrapper import (
    ApiCollabBackend,
    CollabPolicyBackend,
    SubprocessCollabBackend,
    build_collab_cached_backend,
)
from collabsql.config import CollabSqlConfig
from collabsql.utils.plan_loader import FragmentationPlan
from collabsql.utils.sql_exec import extract_ground_truth_sql


def build_backend(
    *,
    config: CollabSqlConfig,
    backend: str,
    cached_parquet: Path | None = None,
    topology: str | None = None,
) -> CollabPolicyBackend:
    if backend in ("cached", "vertical_cached"):
        if cached_parquet is None:
            raise ValueError("cached_parquet is required for cached backends")
        return build_collab_cached_backend(
            Path(cached_parquet),
            backend=backend,
            topology=topology or "horizontal_n4",
        )
    if backend == "api":
        return ApiCollabBackend(config)
    return SubprocessCollabBackend(config)


def build_agents(plan: FragmentationPlan, backend: CollabPolicyBackend) -> dict[str, CollabDataAgent]:
    agents: dict[str, CollabDataAgent] = {}
    for entry in plan.agents:
        if not entry.db_path.exists():
            continue
        agents[entry.agent_id] = CollabDataAgent(entry=entry, backend=backend)
    return agents


# Backward-compatible aliases used by older call sites
_build_agents = build_agents


def _extract_gt(row: Any) -> str:
    gt = extract_ground_truth_sql(row)
    if gt:
        return gt
    reward = row.get("reward_model")
    if isinstance(reward, str):
        try:
            reward = ast.literal_eval(reward)
        except (ValueError, SyntaxError):
            return ""
    if isinstance(reward, dict):
        return str(reward.get("ground_truth") or "")
    return ""
