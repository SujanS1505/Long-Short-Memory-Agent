"""Short-term memory manager.

Maintains a FIFO buffer of recent interactions constrained by interaction
count and approximate token budget.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List

from utils.token_counter import TokenCounter


@dataclass(slots=True)
class Interaction:
    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, str] = field(default_factory=dict)

    def as_text(self) -> str:
        return f"{self.role.upper()}: {self.content}"


class ShortTermMemory:
    """FIFO short-term memory with count and token limits."""

    def __init__(
        self,
        max_interactions: int = 10,
        max_tokens: int = 2_000,
        overflow_archive_size: int = 4,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if max_interactions <= 0:
            raise ValueError("max_interactions must be > 0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if overflow_archive_size <= 0:
            raise ValueError("overflow_archive_size must be > 0")

        self.max_interactions = max_interactions
        self.max_tokens = max_tokens
        self.overflow_archive_size = overflow_archive_size
        self._buffer: Deque[Interaction] = deque()
        self._token_counter = token_counter or TokenCounter()

    def add(self, role: str, content: str, metadata: Dict[str, str] | None = None) -> None:
        self._buffer.append(
            Interaction(role=role, content=content, metadata=metadata or {})
        )

    def size(self) -> int:
        return len(self._buffer)

    def interactions(self) -> List[Interaction]:
        return list(self._buffer)

    def formatted_context(self) -> str:
        if not self._buffer:
            return "(empty)"
        return "\n".join(item.as_text() for item in self._buffer)

    def token_count(self) -> int:
        return self._token_counter.estimate_messages_tokens(
            item.as_text() for item in self._buffer
        )

    def should_overflow(self) -> bool:
        return self.size() > self.max_interactions or self.token_count() > self.max_tokens

    def pop_oldest_for_archive(self, count: int | None = None) -> List[Interaction]:
        if not self._buffer:
            return []

        items_to_pop = min(count or self.overflow_archive_size, len(self._buffer))
        popped: List[Interaction] = []
        for _ in range(items_to_pop):
            popped.append(self._buffer.popleft())
        return popped
