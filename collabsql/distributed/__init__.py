"""Real HTTP-based multi-process deployment for CollabSQL (Coordinator + Data Agents)."""

from collabsql.distributed.http_agent_client import HttpCollabDataAgent, HttpCommStats

__all__ = ["HttpCollabDataAgent", "HttpCommStats"]
