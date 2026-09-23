from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from collabsql.config import CollabSqlConfig
from collabsql.protocol.messages import AgentSqlResult, IdExchange
from collabsql.utils.prompt_utils import inject_scratchpad
from collabsql.utils.sql_exec import execute_sql


@dataclass
class FragmentAgentSpec:
    agent_id: str
    db_path: Path
    tables: list[str]
    description: str
    split_type: str


class CollabPolicyBackend(ABC):
    @abstractmethod
    def infer_sql(
        self,
        *,
        prompt: list[dict[str, str]],
        db_id: str,
        data_source: str,
        db_path: Path,
        question: str,
        scratchpad: dict[str, IdExchange],
        work_tag: str,
        question_index: int | None = None,
    ) -> AgentSqlResult:
        raise NotImplementedError


class SubprocessCollabBackend(CollabPolicyBackend):
    """Wrap CollabSQL policy inference via subprocess — SQLAct core untouched."""

    def __init__(self, config: CollabSqlConfig) -> None:
        self.config = config
        self.config.ensure_dirs()

    def infer_sql(
        self,
        *,
        prompt: list[dict[str, str]],
        db_id: str,
        data_source: str,
        db_path: Path,
        question: str,
        scratchpad: dict[str, IdExchange],
        work_tag: str,
        question_index: int | None = None,
    ) -> AgentSqlResult:
        need_id = work_tag.split("_")[0] if "_" in work_tag else "need_0"
        agent_id = work_tag.split("_")[-1] if "_" in work_tag else "A1"

        enriched_prompt = inject_scratchpad(prompt, scratchpad)
        row = {
            "prompt": enriched_prompt,
            "db_id": db_id,
            "data": data_source,
            "data_source": data_source,
            "reward_model": {"ground_truth": ""},
        }

        run_dir = self.config.work_dir / f"efpl_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        input_parquet = run_dir / "input.parquet"
        output_parquet = run_dir / "output.parquet"
        pd.DataFrame([row]).to_parquet(input_parquet, index=False)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.gpu
        env.setdefault("HF_HUB_OFFLINE", "0")
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_IB_DISABLE", "1")

        db_root = db_path.parent if db_path.suffix == ".sqlite" else db_path
        cmd = [
            str(self.config.python_bin),
            "-m",
            "verl.trainer.main_generation",
            "trainer.nnodes=1",
            "trainer.n_gpus_per_node=1",
            f"data.path={input_parquet}",
            "data.prompt_key=prompt",
            "data.n_samples=1",
            f"data.batch_size={self.config.batch_size}",
            f"data.output_path={output_parquet}",
            f"+data.base_db_path={self.config.default_db_root}",
            f"model.path={self.config.default_model}",
            "+model.trust_remote_code=True",
            f"rollout.name={self.config.rollout_backend}",
            "rollout.temperature=0.8",
            "rollout.top_k=50",
            "rollout.top_p=0.7",
            "rollout.prompt_length=3096",
            "rollout.response_length=5096",
            "rollout.tensor_model_parallel_size=1",
            f"rollout.gpu_memory_utilization={self.config.gpu_mem_util}",
            "rollout.enforce_eager=True",
            "+rollout.task_type=sql",
            "+rollout.port=30000",
            f"+rollout.max_iterations={self.config.max_iterations}",
            "+rollout.sql.max_start_length=3048",
            "+rollout.sql.max_prompt_length=3096",
            "+rollout.sql.max_response_length=5096",
            "+rollout.sql.max_obs_length=1024",
            f"+rollout.sql.db_path={db_root}",
            f"+rollout.n_trajectories={self.config.n_trajectories}",
            "+rollout.sampling_params.max_new_tokens=1024",
            f"hydra.run.dir={run_dir / 'hydra'}",
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.config.policy_infer_dir),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return AgentSqlResult(
                    agent_id=agent_id,
                    need_id=need_id,
                    success=False,
                    error=(proc.stderr or proc.stdout or "CollabSQL inference subprocess failed")[-2000:],
                )
            if not output_parquet.exists():
                return AgentSqlResult(
                    agent_id=agent_id,
                    need_id=need_id,
                    success=False,
                    error="CollabSQL policy produced no output parquet",
                )

            out_df = pd.read_parquet(output_parquet)
            sql = _extract_sql_from_output(out_df.iloc[0])
            if not sql:
                return AgentSqlResult(
                    agent_id=agent_id,
                    need_id=need_id,
                    success=False,
                    error="No SQL extracted from CollabSQL policy output",
                )

            cols, rows, err = execute_sql(db_path, sql)
            if err:
                return AgentSqlResult(
                    agent_id=agent_id,
                    need_id=need_id,
                    success=False,
                    sql=sql,
                    error=err,
                )
            return AgentSqlResult(
                agent_id=agent_id,
                need_id=need_id,
                success=True,
                sql=sql,
                columns=cols,
                rows=rows,
            )
        finally:
            if os.environ.get("COLLABSQL_KEEP_WORKDIR", "0") != "1":
                shutil.rmtree(run_dir, ignore_errors=True)


class ApiCollabBackend(CollabPolicyBackend):
    """Live local inference via a persistent OpenAI-compatible vLLM server (no cached parquet)."""

    def __init__(self, config: CollabSqlConfig) -> None:
        self.config = config
        self.api_base = (
            os.environ.get("COLLABSQL_LLM_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or "http://127.0.0.1:8001/v1"
        ).rstrip("/")
        self.api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("COLLABSQL_LLM_API_KEY")
            or "EMPTY"
        )
        self.model_name = _resolve_vllm_model_id(
            config=config,
            api_base=self.api_base,
            api_key=self.api_key,
        )

    def infer_sql(
        self,
        *,
        prompt: list[dict[str, str]],
        db_id: str,
        data_source: str,
        db_path: Path,
        question: str,
        scratchpad: dict[str, IdExchange],
        work_tag: str,
        question_index: int | None = None,
    ) -> AgentSqlResult:
        need_id = work_tag.split("_")[0] if "_" in work_tag else "need_0"
        agent_id = work_tag.split("_")[-1] if "_" in work_tag else "A1"

        enriched_prompt = inject_scratchpad(prompt, scratchpad)
        messages = []
        for msg in enriched_prompt:
            role = str(msg.get("role", "user"))
            if role not in ("system", "user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": str(msg.get("content", ""))})

        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            return AgentSqlResult(
                agent_id=agent_id,
                need_id=need_id,
                success=False,
                error=f"CollabSQL API inference failed: {exc}",
            )

        sql = _extract_sql_from_text(text)
        if not sql:
            return AgentSqlResult(
                agent_id=agent_id,
                need_id=need_id,
                success=False,
                error="No SQL extracted from CollabSQL API response",
            )

        # Shard execution is validated again in CollabDataAgent; keep SQL for synthesis.
        cols, rows, err = execute_sql(db_path, sql)
        return AgentSqlResult(
            agent_id=agent_id,
            need_id=need_id,
            success=True,
            sql=sql,
            columns=cols or [],
            rows=rows or [],
            error=err or "",
        )


class CachedCollabBackend(CollabPolicyBackend):
    """Fast path: reuse precomputed CollabSQL policy parquet rows when available."""

    def __init__(self, cached_parquet: Path) -> None:
        self.df = pd.read_parquet(cached_parquet)

    def _lookup_row(self, *, db_id: str, question_index: int | None) -> pd.Series | None:
        if question_index is not None and 0 <= question_index < len(self.df):
            return self.df.iloc[question_index]
        subset = self.df[self.df["db_id"] == db_id]
        if subset.empty:
            return None
        return subset.iloc[0]

    def _pick_sql(self, row: pd.Series) -> str:
        return _extract_sql_from_output(row)

    def infer_sql(
        self,
        *,
        prompt: list[dict[str, str]],
        db_id: str,
        data_source: str,
        db_path: Path,
        question: str,
        scratchpad: dict[str, IdExchange],
        work_tag: str,
        question_index: int | None = None,
    ) -> AgentSqlResult:
        need_id = work_tag.split("_")[0] if "_" in work_tag else "need_0"
        agent_id = work_tag.split("_")[-1] if "_" in work_tag else "A1"
        row = self._lookup_row(db_id=db_id, question_index=question_index)
        if row is None:
            return AgentSqlResult(
                agent_id=agent_id,
                need_id=need_id,
                success=False,
                error=f"No cached CollabSQL row for db_id={db_id}",
            )
        sql = self._pick_sql(row)
        if not sql:
            return AgentSqlResult(
                agent_id=agent_id,
                need_id=need_id,
                success=False,
                error="Cached row has no SQL",
            )
        cols, rows, err = execute_sql(db_path, sql)
        if err:
            return AgentSqlResult(agent_id=agent_id, need_id=need_id, success=False, sql=sql, error=err)
        return AgentSqlResult(
            agent_id=agent_id,
            need_id=need_id,
            success=True,
            sql=sql,
            columns=cols,
            rows=rows,
        )

    def peek_sql(
        self,
        *,
        db_id: str,
        question_index: int | None = None,
    ) -> str:
        row = self._lookup_row(db_id=db_id, question_index=question_index)
        if row is None:
            return ""
        return self._pick_sql(row)


class RandomTrajCachedBackend(CachedCollabBackend):
    """NoRL fallback: sample a random trajectory SQL (no exec-feedback best-of selection)."""

    def __init__(self, cached_parquet: Path, seed: int = 0) -> None:
        super().__init__(cached_parquet)
        self.seed = seed

    def _pick_sql(self, row: pd.Series) -> str:
        import random as _random

        results = row.get("result", [])
        if hasattr(results, "tolist"):
            results = results.tolist()
        cands: list[str] = []
        if isinstance(results, (list, tuple)):
            for item in results:
                if isinstance(item, str) and item.strip().upper().startswith("SELECT"):
                    cands.append(item.strip().strip('"'))
        if not cands:
            fulls = row.get("full_responses", [])
            if hasattr(fulls, "tolist"):
                fulls = fulls.tolist()
            if isinstance(fulls, (list, tuple)):
                for item in fulls:
                    if not isinstance(item, str):
                        continue
                    sql = _extract_sql_from_text(item)
                    if sql:
                        cands.append(sql)
        if not cands:
            return ""
        rng = _random.Random(self.seed)
        return rng.choice(cands)


class VerticalCachedCollabBackend(CachedCollabBackend):
    """
    Marker backend: same centralized EFPL parquet (read-only), but evaluation must use
    shard-local execution + shard_rows merge — never source_db SQL replay.
    """

    requires_shard_eval = True


def is_vertical_shard_eval_backend(backend: CollabPolicyBackend) -> bool:
    return isinstance(backend, VerticalCachedCollabBackend) or getattr(
        backend, "requires_shard_eval", False
    )


def build_collab_cached_backend(
    cached_parquet: Path,
    *,
    backend: str,
    topology: str | None = None,
) -> CollabPolicyBackend:
    """Pick cached backend. Use vertical_cached for honest shard-local vertical eval."""
    if backend == "vertical_cached":
        return VerticalCachedCollabBackend(cached_parquet)
    if backend == "cached":
        return CachedCollabBackend(cached_parquet)
    raise ValueError(f"build_collab_cached_backend does not support backend={backend!r}")


def _resolve_vllm_model_id(*, config: CollabSqlConfig, api_base: str, api_key: str) -> str:
    """Pick a model id that vLLM actually serves (never a bare basename like consolidated_model)."""
    preferred = [
        os.environ.get("COLLABSQL_LLM_MODEL_NAME"),
        str(config.default_model),
    ]
    preferred = [p for p in preferred if p]

    try:
        from openai import OpenAI

        client = OpenAI(base_url=api_base.rstrip("/"), api_key=api_key)
        available = [m.id for m in client.models.list().data]
        if not available:
            return str(config.default_model)

        available_set = set(available)
        for candidate in preferred:
            if candidate in available_set:
                return candidate
            cand_base = os.path.basename(candidate.rstrip("/"))
            for model_id in available:
                if model_id.rstrip("/") == candidate.rstrip("/"):
                    return model_id
                if os.path.basename(model_id.rstrip("/")) == cand_base and "/" in model_id:
                    return model_id
        return available[0]
    except Exception:
        return str(config.default_model)


def _extract_sql_from_text(text: str) -> str:
    if not text:
        return ""
    blocks = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        sql = blocks[-1].strip()
        if sql.upper().startswith("SELECT"):
            return sql
    blocks = re.findall(r"<sql>(.*?)</sql>", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        sql = blocks[-1].strip()
        if sql.upper().startswith("SELECT"):
            return sql
    fenced = re.findall(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        sql = fenced[-1].strip()
        if sql.upper().startswith("SELECT"):
            return sql
    match = re.search(r"\b(SELECT\b.+?)(?:;|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_sql_from_output(row: pd.Series) -> str:
    results = row.get("result", [])
    if hasattr(results, "tolist"):
        results = results.tolist()
    if isinstance(results, (list, tuple)) and results:
        first = results[0]
        if isinstance(first, str) and first.strip().upper().startswith("SELECT"):
            return first.strip().strip('"')

    fulls = row.get("full_responses", [])
    if hasattr(fulls, "tolist"):
        fulls = fulls.tolist()
    if isinstance(fulls, (list, tuple)):
        for item in fulls:
            if not isinstance(item, str):
                continue
            sql = _extract_sql_from_text(item)
            if sql:
                return sql
    return ""


class LocalCollabAgent:
    """Local agent shell: capability broadcast + CollabSQL policy delegation."""

    def __init__(self, spec: FragmentAgentSpec, backend: CollabPolicyBackend) -> None:
        self.spec = spec
        self.backend = backend
        from collabsql.agent.capability import introspect_sqlite

        self.capability = introspect_sqlite(
            spec.db_path,
            spec.agent_id,
            spec.description,
            spec.split_type,
        )

    def answer_need(
        self,
        *,
        need_id: str,
        question: str,
        prompt: list[dict[str, str]],
        db_id: str,
        data_source: str,
        scratchpad: dict[str, IdExchange],
        question_index: int | None = None,
    ) -> AgentSqlResult:
        result = self.backend.infer_sql(
            prompt=prompt,
            db_id=db_id,
            data_source=data_source,
            db_path=self.spec.db_path,
            question=question,
            scratchpad=scratchpad,
            work_tag=f"{need_id}_{self.spec.agent_id}",
            question_index=question_index,
        )
        result.agent_id = self.spec.agent_id
        result.need_id = need_id
        return result

    def capability_message(self) -> dict[str, Any]:
        return self.capability.to_message()
