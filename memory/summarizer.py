"""Summarization logic for archiving STM chunks to LTM."""

from __future__ import annotations

from typing import Iterable

from llm.model import LLMStrategy
from memory.short_term import Interaction


class ConversationSummarizer:
    def __init__(self, llm: LLMStrategy) -> None:
        self._llm = llm

    def summarize(self, interactions: Iterable[Interaction]) -> str:
        items = list(interactions)
        if not items:
            return ""

        transcript = "\n".join(f"{it.role.upper()}: {it.content}" for it in items)
        prompt = (
            "You are summarizing a conversation archive for long-term memory.\n"
            "Keep all critical facts, decisions, named entities, constraints, and numbers.\n"
            "Avoid speculation. If facts are missing, write 'unknown' for that detail.\n"
            "Return concise bullet points and a final one-line summary.\n\n"
            f"TRANSCRIPT:\n{transcript}\n"
        )
        summary = self._llm.generate(prompt=prompt, max_tokens=280, temperature=0.0).strip()

        if not summary:
            # Conservative fallback retains the actual content if model returns empty output.
            return "Archived transcript:\n" + transcript

        return summary
