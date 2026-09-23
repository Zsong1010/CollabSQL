from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from collabsql.aggregator.merger import MergeResult, ResultMerger
from collabsql.protocol.messages import AgentSqlResult
from collabsql.protocol.partial_result import PartialResult
from collabsql.utils.eval_mode import extract_primary_sql


QCP_DB_ROOT = Path(
    __import__("os").environ.get("COLLABSQL_QCP_ROOT")
    or __import__("os").environ.get("QCP_DB_ROOT")
    or ""
)


def _env_prefer_llm() -> bool:
    """Paper default: try LLM composition when a result agent is available."""
    raw = os.environ.get("COLLABSQL_LLM_COMPOSITION", "auto").strip().lower()
    if raw in {"0", "false", "no", "rule"}:
        return False
    if raw in {"1", "true", "yes", "llm"}:
        return True
    return True  # auto → prefer LLM if loadable


class CollabResultSynthesizer:
    """
    CollabSQL Result Composition (paper Step 2):
      2.1 Identifier-level alignment (field align / ID exchange)
      2.2 Provenance-grounded composition (LLM when available; rule fallback)
    """

    def __init__(self, *, prefer_llm: bool | None = None) -> None:
        self.rule_merger = ResultMerger()
        self._llm_agent = None
        self.prefer_llm = _env_prefer_llm() if prefer_llm is None else prefer_llm

    def _maybe_load_llm_agent(self):
        if self._llm_agent is not None:
            return self._llm_agent
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MODEL_API_KEY")
        api_base = os.environ.get("COLLABSQL_LLM_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        if (not api_key and not api_base) or not QCP_DB_ROOT.exists():
            return None
        try:
            if str(QCP_DB_ROOT / "src") not in sys.path:
                sys.path.insert(0, str(QCP_DB_ROOT / "src"))
            from qcp_db.agents.result_agent import ResultAgent
            from qcp_db.config import Config
            from qcp_db.llm.client import LLMClient

            config = Config.load(QCP_DB_ROOT / "config.yaml")
            if api_base:
                from collabsql.agent.collab_policy_wrapper import _resolve_vllm_model_id
                from collabsql.config import CollabSqlConfig

                cfg = CollabSqlConfig()
                model_name = _resolve_vllm_model_id(
                    config=cfg,
                    api_base=api_base.rstrip("/"),
                    api_key=api_key or "EMPTY",
                )
                config.apply_llm_provider(
                    provider="local",
                    model_api_url=api_base,
                    model_api_key=api_key or "EMPTY",
                    model_name=model_name,
                )
                from qcp_db.config import ModelSetup

                models = ModelSetup(
                    coordinator=model_name,
                    data_agent=model_name,
                    result_agent=model_name,
                    baseline=model_name,
                )
            else:
                setup = os.environ.get("QCP_SETUP", "dashscope")
                models = (
                    config.setup_dashscope
                    if setup == "dashscope" and config.setup_dashscope
                    else config.setup_a
                )
            llm = LLMClient(config, models.result_agent)
            from collabsql.utils.token_usage import wrap_llm_client

            wrap_llm_client(llm, role="composition")
            self._llm_agent = ResultAgent.create(
                llm, QCP_DB_ROOT / "prompts" / "result_agent.txt"
            )
            return self._llm_agent
        except Exception:
            return None

    def _rule_merge(
        self,
        *,
        partials: list[PartialResult],
        split_type: str,
        agent_db_paths: dict[str, str],
        source_db_path: str | None,
    ) -> MergeResult:
        agent_results = [
            AgentSqlResult(
                agent_id=p.agent_id,
                need_id=p.need_id,
                success=True,
                sql=p.sql_used,
                columns=p.columns,
                rows=p.rows,
            )
            for p in partials
            if p.sql_used
        ]
        t0 = time.perf_counter()
        merge = self.rule_merger.merge(
            agent_results=agent_results,
            split_type=split_type,
            source_db_path=source_db_path,
            agent_db_paths=agent_db_paths,
        )
        merge.composition_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        merge.composition_mode = "rule"
        sqls = [p.sql_used.strip() for p in partials if p.sql_used and p.sql_used.strip()]
        if split_type != "vertical" and sqls and len(set(sqls)) == 1:
            merge.predicted_sql = sqls[0]
            merge.composition_mode = "rule_identical_sql"
        if not merge.predicted_sql:
            merge.predicted_sql = extract_primary_sql(merge, agent_results)
        if not merge.participating_agents:
            merge.participating_agents = [p.agent_id for p in partials]
        return merge

    def synthesize(
        self,
        *,
        user_query: str,
        partials: list[PartialResult],
        split_type: str,
        agent_db_paths: dict[str, str],
        source_db_path: str | None = None,
    ) -> MergeResult:
        if not partials:
            return MergeResult(success=False, error="no partial results", composition_mode="empty")

        if split_type == "vertical":
            with_rows = [p for p in partials if p.rows]
            if len(with_rows) == 1:
                p0 = with_rows[0]
                return MergeResult(
                    success=True,
                    columns=p0.columns,
                    rows=p0.rows,
                    predicted_sql=p0.sql_used,
                    participating_agents=[p0.agent_id],
                    composition_mode="vertical_single",
                )

        # Paper Step 2.2: prefer LLM-guided provenance-grounded composition when available.
        llm_agent = None
        if self.prefer_llm and split_type != "vertical" and len(partials) > 1:
            llm_agent = self._maybe_load_llm_agent()
        if llm_agent is not None:
            t0 = time.perf_counter()
            try:
                final = llm_agent.synthesize(user_query, partials, {})
            except Exception:
                final = None
            latency = round((time.perf_counter() - t0) * 1000.0, 3)
            sqls = [p.sql_used.strip() for p in partials if p.sql_used and p.sql_used.strip()]
            predicted = sqls[0] if sqls else (partials[0].sql_used or "")
            if final and getattr(final, "rows", None):
                return MergeResult(
                    success=True,
                    columns=list(getattr(final, "columns", []) or []),
                    rows=list(final.rows),
                    predicted_sql=predicted,
                    participating_agents=[p.agent_id for p in partials],
                    composition_mode="llm",
                    composition_latency_ms=latency,
                )
            # LLM attempted but failed → fall through to rule; keep latency signal
            rule = self._rule_merge(
                partials=partials,
                split_type=split_type,
                agent_db_paths=agent_db_paths,
                source_db_path=source_db_path,
            )
            rule.composition_mode = f"{rule.composition_mode}+llm_failed"
            rule.composition_latency_ms = round(latency + rule.composition_latency_ms, 3)
            return rule

        return self._rule_merge(
            partials=partials,
            split_type=split_type,
            agent_db_paths=agent_db_paths,
            source_db_path=source_db_path,
        )
