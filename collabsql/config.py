from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env(*names: str, default: str = "") -> str:
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


@dataclass
class CollabSqlConfig:
    """Runtime config via COLLABSQL_* environment variables."""

    repo_root: Path = field(default_factory=_repo_root)
    # Subprocess backend: set COLLABSQL_INFER_DIR to a generation checkout if needed.
    # Prefer --backend api or cached for online eval.
    policy_infer_dir: Path = field(
        default_factory=lambda: Path(
            _env("COLLABSQL_INFER_DIR", default=str(_repo_root() / "EFPL"))
        )
    )
    fragments_dir: Path = field(
        default_factory=lambda: _repo_root() / "data" / "spider_test" / "fragments"
    )
    default_db_root: Path = field(
        default_factory=lambda: Path(
            _env(
                "COLLABSQL_DB_ROOT",
                default=str(_repo_root() / "data" / "spider_test" / "databases"),
            )
        )
    )
    default_model: Path = field(
        default_factory=lambda: Path(
            _env(
                "COLLABSQL_MODEL",
                default=str(Path.home() / "models" / "Qwen-SQL-7B" / "consolidated_model"),
            )
        )
    )
    python_bin: Path = field(
        default_factory=lambda: Path(
            _env("COLLABSQL_PYTHON", default=os.environ.get("PYTHON", "python3"))
        )
    )
    gpu: str = field(default_factory=lambda: _env("GPU", default="0"))
    n_trajectories: int = 1
    batch_size: int = 1
    rollout_backend: str = "vllm"
    max_iterations: int = 5
    gpu_mem_util: float = 0.35
    work_dir: Path = field(default_factory=lambda: _repo_root() / "outputs" / "collabsql")

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
