"""Accumulate real HTTP request/response byte counts."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HttpCommStats:
    requests: int = 0
    request_bytes: int = 0
    response_bytes: int = 0

    def record(self, *, request_bytes: int, response_bytes: int) -> None:
        self.requests += 1
        self.request_bytes += max(0, int(request_bytes))
        self.response_bytes += max(0, int(response_bytes))

    @property
    def total_bytes(self) -> int:
        return self.request_bytes + self.response_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "http_requests": self.requests,
            "http_request_bytes": self.request_bytes,
            "http_response_bytes": self.response_bytes,
            "http_total_bytes": self.total_bytes,
        }

    def reset(self) -> None:
        self.requests = 0
        self.request_bytes = 0
        self.response_bytes = 0


@dataclass
class SharedHttpStats:
    """Shared across all HttpCollabDataAgent clients for one query."""

    stats: HttpCommStats = field(default_factory=HttpCommStats)

    def reset(self) -> None:
        self.stats.reset()
