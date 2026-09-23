from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from collabsql.agent.capability import introspect_sqlite
from collabsql.agent.collab_policy_wrapper import CollabPolicyBackend
from collabsql.coordinator.followup import is_identifier_heavy
from collabsql.protocol.messages import IdExchange
from collabsql.protocol.partial_result import AgentResponse, PartialResult
from collabsql.utils.plan_loader import AgentPlanEntry
from collabsql.utils.vertical_columns import agent_covers_sql
from collabsql.utils.sql_exec import execute_sql


@dataclass
class CollabDataAgent:
    """
    CollabSQL Data Agent with shared EFPL policy π_θ on the local shard (paper Section 3.2).
    Performs local SQL generation, execution-feedback analysis, and optional Need raising.
    """

    entry: AgentPlanEntry
    backend: CollabPolicyBackend
    max_feedback_iterations: int = 3
    capability: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.capability is None:
            self.capability = introspect_sqlite(
                self.entry.db_path,
                self.entry.agent_id,
                self.entry.description,
                self.entry.split_type,
            )

    @property
    def agent_id(self) -> str:
        return self.entry.agent_id

    @property
    def registration_description(self) -> str:
        return (
            f"{self.entry.description} Tables: {', '.join(self.entry.tables)}. "
            f"Query capability: SQL via CollabSQL policy on local shard."
        )

    def to_mschema_text(self, max_examples: int = 3) -> str:
        conn = sqlite3.connect(f"file:{self.entry.db_path}?mode=ro", uri=True)
        lines = [f"【DB_ID】 {self.entry.agent_id}", ""]
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for table in tables:
            lines.append(f"# Table: {table}")
            lines.append("[")
            for col in conn.execute(f'PRAGMA table_info("{table}")').fetchall():
                name, dtype = col[1], col[2] or "TEXT"
                examples = []
                try:
                    cur = conn.execute(
                        f'SELECT DISTINCT "{name}" FROM "{table}" '
                        f'WHERE "{name}" IS NOT NULL LIMIT {max_examples}'
                    )
                    examples = [str(r[0]) for r in cur.fetchall()]
                except sqlite3.Error:
                    pass
                ex = ", Examples: [" + ", ".join(examples) + "]" if examples else ""
                lines.append(f"  ({name}, {dtype}{ex})")
            lines.append("]")
            lines.append("")
        conn.close()
        return "\n".join(lines)

    def _column_map(self) -> dict[str, list[str]]:
        return self.entry.columns or {}

    def _shard_covers_sql(self, sql: str) -> bool:
        if self.entry.split_type != "vertical":
            return True
        return agent_covers_sql(self._column_map(), sql)

    def self_assess(self, user_query: str, *, sql_hint: str | None = None) -> tuple[bool, str]:
        text = user_query.lower()
        if self.entry.split_type == "vertical":
            if sql_hint and not self._shard_covers_sql(sql_hint):
                return False, "Vertical shard missing columns for cached SQL"
            for table in self._column_map() or self.capability.tables:
                if table.lower() in text:
                    return True, f"Local shard contains table {table}"
            return bool(self._column_map()), "Vertical shard participates when schema overlaps"
        for table in self.capability.tables:
            if table.lower() in text:
                return True, f"Local shard contains table {table}"
        return True, "Default participate for horizontal shard collaboration"

    def answer_need(
        self,
        need_id: str,
        question: str,
        scratchpad: dict[str, IdExchange],
        scratchpad_keys: list[str] | None,
        *,
        prompt: list[dict[str, str]],
        db_id: str,
        data_source: str,
        question_index: int | None,
    ) -> AgentResponse:
        last_sql = ""
        last_error = ""
        for iteration in range(self.max_feedback_iterations):
            result = self.backend.infer_sql(
                prompt=prompt,
                db_id=db_id,
                data_source=data_source,
                db_path=self.entry.db_path,
                question=question or question,
                scratchpad=scratchpad,
                work_tag=f"{need_id}_{self.agent_id}",
                question_index=question_index,
            )
            last_sql = result.sql
            # Cached / centralized SQL often fails on random (and some horizontal) shards
            # due to missing columns/rows. For non-vertical topologies under source_db
            # evaluation we still keep the SQL so synthesis can replay on the integrated DB.
            if result.sql and result.sql.strip().upper().startswith("SELECT"):
                if self.entry.split_type == "vertical" and not self._shard_covers_sql(result.sql):
                    last_error = "Vertical shard schema does not cover cached SQL columns"
                    continue
                if self.entry.split_type != "vertical" and (
                    not result.success or result.error
                ):
                    return AgentResponse(
                        success=True,
                        partial_result=PartialResult(
                            need_id=need_id,
                            agent_id=self.agent_id,
                            question=question,
                            columns=result.columns or [],
                            rows=result.rows or [],
                            sql_used=result.sql,
                            metadata={"shard_exec_error": result.error or "backend_failed"},
                        ),
                        sql_used=result.sql,
                        feedback="accept_sql_despite_shard_error",
                    )
            if not result.success or not result.sql:
                last_error = result.error or "CollabSQL policy produced no SQL"
                continue
            if self.entry.split_type == "vertical" and not self._shard_covers_sql(result.sql):
                last_error = "Vertical shard schema does not cover cached SQL columns"
                continue

            cols, rows, err = execute_sql(self.entry.db_path, result.sql)
            if err:
                if self.entry.split_type != "vertical" and result.sql.strip().upper().startswith("SELECT"):
                    # Horizontal/random cached SQL may fail on a local shard; keep for source_db.
                    return AgentResponse(
                        success=True,
                        partial_result=PartialResult(
                            need_id=need_id,
                            agent_id=self.agent_id,
                            question=question,
                            columns=cols or [],
                            rows=rows or [],
                            sql_used=result.sql,
                            metadata={"shard_exec_error": err},
                        ),
                        sql_used=result.sql,
                        feedback="accept_sql_despite_shard_error",
                    )
                last_error = err
                continue

            if not rows and iteration < self.max_feedback_iterations - 1:
                if self.entry.split_type != "vertical" and result.sql.strip().upper().startswith("SELECT"):
                    return AgentResponse(
                        success=True,
                        partial_result=PartialResult(
                            need_id=need_id,
                            agent_id=self.agent_id,
                            question=question,
                            columns=cols or [],
                            rows=[],
                            sql_used=result.sql,
                            metadata={"shard_empty": True},
                        ),
                        sql_used=result.sql,
                        feedback="accept_sql_despite_empty_shard",
                    )
                last_error = "Empty result on local shard"
                continue

            partial = PartialResult(
                need_id=need_id,
                agent_id=self.agent_id,
                question=question,
                columns=cols,
                rows=rows,
                sql_used=result.sql,
            )
            # Paper: local evidence may raise a follow-up need for missing attributes.
            new_need = None
            if is_identifier_heavy(partial) and self.entry.split_type in {
                "vertical",
                "random",
            }:
                new_need = {
                    "question": (
                        f"Enrich identifier evidence from {self.agent_id} to fully "
                        f"answer: {question}"
                    ),
                }
            return AgentResponse(
                success=True,
                partial_result=partial,
                sql_used=result.sql,
                feedback="accept",
                new_need=new_need,
            )

        return AgentResponse(
            success=False,
            feedback=last_error or "feedback loop exhausted",
            sql_used=last_sql,
        )
