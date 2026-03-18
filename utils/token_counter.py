"""Token counting utilities.

Provides a lightweight token approximation to keep STM bounded without
requiring provider-specific tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class TokenCounter:
    """Approximate token counting based on characters and words.

    This approximation is intentionally model-agnostic and fast.
    For most English text, char_count / 4 is a practical estimate.
    """

    chars_per_token: float = 4.0

    def estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        approx = int(len(text) / self.chars_per_token)
        return max(1, approx)

    def estimate_messages_tokens(self, messages: Iterable[str]) -> int:
        return sum(self.estimate_text_tokens(message) for message in messages)
