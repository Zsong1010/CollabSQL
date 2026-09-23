from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from collabsql.agent.collab_data_agent import CollabDataAgent
from collabsql.coordinator.coord_llm import llm_decompose_needs, prefer_llm_coordinator
from collabsql.coordinator.followup import instantiate_followups, is_identifier_heavy
from collabsql.protocol.messages import DqcpMessageType, IdExchange
from collabsql.protocol.need_stack import NeedNode, NeedStack
from collabsql.protocol.partial_result import (
    AgentResponse,
    PartialResult,
    ResultClassification,
)
from collabsql.utils.plan_loader import FragmentationPlan
from collabsql.utils.vertical_columns import select_capable_agents


@dataclass
class CoordinatorResult:
    final_partials: list[PartialResult]
    scratchpad: dict[str, IdExchange]
    stack_trace: list[dict[str, Any]]
    dqcp_messages: list[dict[str, Any]]
    participating_agents: list[str]
    decompose_mode: str = "rule"


class CollabCoordinator:
    """
    CollabSQL Query Coordinator (DQCP, paper Algorithm 1 / Fig 2–3):
    capability registration → LLM Decompose (fallback rule) → Pop/Route/LocalExec →
    Γ follow-ups → Scratchpad → Identifier alignment / synthesis (caller Step 2).
    """

    def __init__(
        self,
        plan: FragmentationPlan,
        agents: dict[str, Any],
        *,
        max_needs: int = 8,
        prefer_llm_decompose: bool | None = None,
    ) -> None:
        self.plan = plan
        self.agents = agents  # local CollabDataAgent or HttpCollabDataAgent
        self.max_needs = max_needs
        self.prefer_llm_decompose = (
            prefer_llm_coordinator() if prefer_llm_decompose is None else prefer_llm_decompose
        )
        self.decompose_mode = "rule"

    @classmethod
    def from_plan(
        cls, plan: FragmentationPlan, agents: dict[str, Any]
    ) -> "CollabCoordinator":
        return cls(plan=plan, agents=agents)

    def register_capabilities(self) -> list[dict[str, Any]]:
        return [agent.capability.to_message() for agent in self.agents.values()]

    def select_agents(
        self,
        user_query: str,
        *,
        sql_hint: str | None = None,
        force_all: bool = False,
    ) -> list[str]:
        if force_all:
            return sorted(self.agents.keys())

        if self.plan.split_type == "vertical" and sql_hint:
            colmap = {
                agent_id: (agent._column_map() or {})
                for agent_id, agent in self.agents.items()
            }
            capable = select_capable_agents(colmap, sql_hint)
            if capable:
                return capable

        selected: list[str] = []
        for agent_id, agent in self.agents.items():
            can, _ = agent.self_assess(user_query, sql_hint=sql_hint)
            if can:
                selected.append(agent_id)
        return selected or sorted(self.agents.keys())

    def _rule_query_plan(self, user_query: str, participating: list[str]) -> list[NeedNode]:
        """Rule fallback: broadcast original query as a single root need."""
        return [
            NeedNode(
                need_id="need_0",
                question=user_query,
                target_agents=participating,
                originator="coordinator",
            )
        ]

    def create_query_plan(self, user_query: str, participating: list[str]) -> list[NeedNode]:
        """
        Paper Step 1.2 Decompose & Dispatch:
          Coordinator (via LLM) → dependency-aware sub-queries; else rule single need_0.
        """
        if self.prefer_llm_decompose and participating:
            schemas = {
                aid: self.agents[aid].to_mschema_text()
                for aid in participating
                if aid in self.agents
            }
            planned = llm_decompose_needs(
                user_query=user_query,
                participating_agents=participating,
                agent_schemas=schemas,
            )
            if planned:
                self.decompose_mode = "llm"
                return [
                    NeedNode(
                        need_id=n["need_id"],
                        question=n["question"],
                        target_agents=n["target_agents"],
                        originator="coordinator",
                        scratchpad_keys=list(n.get("depends_on_scratchpad") or []),
                    )
                    for n in planned
                ]
            self.decompose_mode = "rule+llm_failed"
        else:
            self.decompose_mode = "rule"
        return self._rule_query_plan(user_query, participating)

    def classify_result(self, partial: PartialResult, need_id: str) -> ResultClassification:
        """Short-Term → Scratchpad Σ; Long-Term → final composition."""
        if is_identifier_heavy(partial):
            partial.metadata["scratchpad_key"] = f"{partial.need_id}_{partial.agent_id}"
            return ResultClassification.SHORT_TERM
        if need_id in ("need_main", "need_0"):
            return ResultClassification.LONG_TERM
        if len(partial.columns) == 1:
            col = partial.columns[0].lower()
            if col.endswith("id") or col.endswith("code"):
                partial.metadata["scratchpad_key"] = f"{partial.need_id}_{partial.agent_id}"
                return ResultClassification.SHORT_TERM
        return ResultClassification.LONG_TERM

    def process_query(
        self,
        *,
        user_query: str,
        prompt: list[dict[str, str]],
        db_id: str,
        data_source: str,
        question_index: int | None,
        sql_hint: str | None = None,
        force_all_agents: bool = False,
    ) -> CoordinatorResult:
        dqcp_messages: list[dict[str, Any]] = []
        dqcp_messages.extend(self.register_capabilities())

        participating = self.select_agents(
            user_query, sql_hint=sql_hint, force_all=force_all_agents
        )
        dqcp_messages.append(
            {
                "msg_type": "agent_selection",
                "payload": {"participating_agents": participating},
            }
        )

        # Algorithm 1 / Fig 3: Q0 ← Decompose(q, C) via LLM when available
        sub_needs = self.create_query_plan(user_query, participating)
        stack = NeedStack()
        for need in reversed(sub_needs):
            stack.push(
                need.question,
                need.target_agents,
                need.originator,
                need_id=need.need_id,
                scratchpad_keys=need.scratchpad_keys,
            )
        # Paper: keep original user query as need_0 (processed last) when LLM returned deps
        has_need0 = any(n.need_id in ("need_0", "need_main") for n in sub_needs)
        if self.decompose_mode.startswith("llm") and not has_need0:
            stack.push_user_need(user_query, participating)

        dqcp_messages.append(
            {
                "msg_type": "decompose",
                "payload": {
                    "mode": self.decompose_mode,
                    "needs": [n.to_dict() for n in sub_needs],
                },
            }
        )

        scratchpad: dict[str, IdExchange] = {}
        long_term: list[PartialResult] = []
        trace: list[dict[str, Any]] = [{"action": "decompose", "mode": self.decompose_mode}]
        seen_followups: set[str] = set()
        needs_processed = 0

        while not stack.is_empty() and needs_processed < self.max_needs:
            need = stack.pop()
            assert need is not None
            needs_processed += 1
            need.status = "in_progress"
            trace.append({"action": "pop", "need": need.to_dict()})
            dqcp_messages.append(
                {
                    "msg_type": DqcpMessageType.NEED_REQUEST.value,
                    "payload": {
                        "need_id": need.need_id,
                        "question": need.question,
                        "target_agents": need.target_agents,
                    },
                }
            )

            scratch_keys = need.scratchpad_keys or []
            answered = False
            for agent_id in need.target_agents:
                agent = self.agents.get(agent_id)
                if agent is None:
                    continue
                response: AgentResponse = agent.answer_need(
                    need.need_id,
                    need.question,
                    scratchpad,
                    scratch_keys,
                    prompt=prompt,
                    db_id=db_id,
                    data_source=data_source,
                    question_index=question_index,
                )
                trace.append(
                    {
                        "action": "agent_response",
                        "agent_id": agent_id,
                        "need_id": need.need_id,
                        "success": response.success,
                        "sql": response.sql_used,
                        "feedback": response.feedback,
                        "raised_new_need": bool(response.new_need),
                    }
                )
                dqcp_messages.append(
                    {
                        "msg_type": DqcpMessageType.NEED_RESPONSE.value,
                        "sender_id": agent_id,
                        "payload": {
                            "success": response.success,
                            "sql": response.sql_used,
                            "feedback": response.feedback,
                        },
                    }
                )

                # Agent-raised follow-up (optional path)
                if response.new_need:
                    fq = str(response.new_need.get("question", "")).strip()
                    if fq and fq.lower() not in seen_followups:
                        seen_followups.add(fq.lower())
                        stack.push(
                            fq,
                            list(response.new_need.get("target_agents") or self.agents.keys()),
                            agent_id,
                            parent_need_id=need.need_id,
                        )
                        trace.append({"action": "followup_from_agent", "question": fq})
                    answered = True
                    break

                if response.success and response.partial_result:
                    pr = response.partial_result
                    pr.classification = self.classify_result(pr, need.need_id)
                    if pr.classification == ResultClassification.SHORT_TERM:
                        key = pr.metadata.get(
                            "scratchpad_key", f"{pr.need_id}_{pr.agent_id}"
                        )
                        scratchpad[key] = IdExchange(
                            agent_id=pr.agent_id,
                            table=pr.columns[0] if pr.columns else "unknown",
                            key_column=pr.columns[0] if pr.columns else "id",
                            ids=[row[0] for row in pr.rows[:500]],
                            need_id=pr.need_id,
                        )
                        dqcp_messages.append(scratchpad[key].to_message())
                        # Paper Γ(q_t, r_t, B_t): instantiate enrichment sub-queries
                        others = [a for a in participating if a != pr.agent_id]
                        for fu in instantiate_followups(
                            user_query=user_query,
                            partial=pr,
                            other_agent_ids=others,
                            seen=seen_followups,
                        ):
                            stack.push(
                                fu["question"],
                                fu["target_agents"],
                                fu["originator"],
                                parent_need_id=fu.get("parent_need_id"),
                                scratchpad_keys=fu.get("scratchpad_keys"),
                            )
                            trace.append(
                                {
                                    "action": "followup_gamma",
                                    "question": fu["question"],
                                    "target_agents": fu["target_agents"],
                                }
                            )
                            dqcp_messages.append(
                                {
                                    "msg_type": "followup_instantiate",
                                    "payload": fu,
                                }
                            )
                    else:
                        long_term.append(pr)
                    need.status = "completed"
                    answered = True
                    if (
                        self.plan.split_type == "vertical"
                        and sql_hint
                        and pr.rows
                        and pr.classification == ResultClassification.LONG_TERM
                    ):
                        break

            if not answered:
                need.status = "failed"
                trace.append({"action": "need_failed", "need_id": need.need_id})

        dqcp_messages.append({"msg_type": DqcpMessageType.TERMINATE.value, "payload": {}})

        # If only short-term evidence exists, still expose it for composition
        if not long_term and scratchpad:
            for key, bundle in scratchpad.items():
                long_term.append(
                    PartialResult(
                        need_id=bundle.need_id,
                        agent_id=bundle.agent_id,
                        question=user_query,
                        columns=[bundle.key_column],
                        rows=[[i] for i in bundle.ids[:50]],
                        classification=ResultClassification.LONG_TERM,
                        metadata={"from_scratchpad": key},
                    )
                )

        return CoordinatorResult(
            final_partials=long_term,
            scratchpad=scratchpad,
            stack_trace=trace,
            dqcp_messages=dqcp_messages,
            participating_agents=participating,
            decompose_mode=self.decompose_mode,
        )
