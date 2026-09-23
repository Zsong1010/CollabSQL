from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class AgentPlanEntry:
    agent_id: str
    db_path: Path
    tables: list[str]
    description: str
    split_type: str
    columns: dict[str, list[str]] | None = None


@dataclass
class FragmentationPlan:
    db_id: str
    source_db: Path
    split_type: str
    num_agents: int
    agents: list[AgentPlanEntry]
    seed: int = 42


def _infer_repo_root(plan_file: Path) -> Path:
    """Locate CollabSQL_Open root from .../data/spider/fragments/<topo>/<db>/plan.json."""
    resolved = plan_file.resolve()
    parts = resolved.parts
    if "data" in parts:
        idx = parts.index("data")
        if idx > 0:
            return Path(*parts[:idx])
    # Fallback: fragments/<topo>/<db>/plan.json → up 4
    return resolved.parents[4]


@lru_cache(maxsize=32)
def load_plan(plan_path: str) -> FragmentationPlan:
    raw = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    plan_file = Path(plan_path)
    repo_root = _infer_repo_root(plan_file)

    def _resolve_db_path(p: str) -> Path:
        path = Path(p)
        if path.is_absolute() and path.exists():
            return path
        if not path.is_absolute():
            candidate = repo_root / path
            if candidate.exists():
                return candidate
        # Prefer shard next to plan.json (portable OSS layout)
        local = plan_file.parent / path.name
        if local.exists():
            return local
        return path

    agents = [
        AgentPlanEntry(
            agent_id=a["agent_id"],
            db_path=_resolve_db_path(a["db_path"]),
            tables=a.get("tables", []),
            description=a.get("description", ""),
            split_type=a.get("split_type", raw.get("split_type", "horizontal")),
            columns=a.get("metadata", {}).get("columns"),
        )
        for a in raw["agents"]
    ]
    source = _resolve_db_path(raw["source_db"])
    return FragmentationPlan(
        db_id=raw["db_id"],
        source_db=source,
        split_type=raw.get("split_type", "horizontal"),
        num_agents=raw.get("num_agents", len(agents)),
        agents=agents,
        seed=raw.get("seed", 42),
    )


def find_plan(fragments_dir: Path, db_id: str) -> FragmentationPlan | None:
    plan_path = fragments_dir / db_id / "plan.json"
    if not plan_path.exists():
        return None
    return load_plan(str(plan_path.resolve()))


def agent_index(agent_id: str) -> int:
    """Parse A{k} → k. Returns 0 on failure."""
    if not agent_id:
        return 0
    suffix = agent_id[1:] if agent_id[0] in "Aa" else agent_id
    try:
        return int(suffix)
    except ValueError:
        return 0


def prefix_plan(plan: FragmentationPlan, num_active: int) -> FragmentationPlan:
    """
    Keep agents A1..A{num_active} from a fixed N-way plan (same shard files).
    Used for dynamic data-agent join evaluation without re-fragmenting.
    """
    if num_active < 1:
        raise ValueError("num_active must be >= 1")
    kept = [a for a in plan.agents if 1 <= agent_index(a.agent_id) <= num_active]
    kept.sort(key=lambda a: agent_index(a.agent_id))
    return FragmentationPlan(
        db_id=plan.db_id,
        source_db=plan.source_db,
        split_type=plan.split_type,
        num_agents=num_active,
        agents=kept,
        seed=plan.seed,
    )


def should_use_distributed(plan: FragmentationPlan | None) -> bool:
    if plan is None:
        return False
    existing = [a for a in plan.agents if a.db_path.exists()]
    return len(existing) >= 2
