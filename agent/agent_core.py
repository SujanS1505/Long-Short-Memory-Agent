"""Main hierarchical memory agent orchestration (pipeline pattern)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from agent.prompt_template import PromptBuilder
from embeddings.embedding_model import EmbeddingStrategy
from llm.model import LLMStrategy
from memory.long_term import LongTermRepository
from memory.short_term import Interaction, ShortTermMemory
from memory.summarizer import ConversationSummarizer
from retrieval.retriever import MemoryRetriever, RetrievedMemory

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentConfig:
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.25
    archive_importance_floor: float = 0.5


class HierarchicalMemoryAgent:
    """Pipeline:
    1) store user in STM
    2) retrieve LTM context
    3) build prompt
    4) generate response
    5) store assistant in STM
    6) overflow -> summarize + archive to LTM
    """

    def __init__(
        self,
        stm: ShortTermMemory,
        ltm_repository: LongTermRepository,
        embedder: EmbeddingStrategy,
        llm: LLMStrategy,
        summarizer: ConversationSummarizer,
        retriever: MemoryRetriever,
        prompt_builder: PromptBuilder,
        config: AgentConfig | None = None,
    ) -> None:
        self.stm = stm
        self.ltm_repository = ltm_repository
        self.embedder = embedder
        self.llm = llm
        self.summarizer = summarizer
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.config = config or AgentConfig()

    def add_to_stm(self, role: str, content: str) -> None:
        self.stm.add(role=role, content=content)

    def retrieve_from_ltm(self, query: str) -> List[RetrievedMemory]:
        return self.retriever.retrieve(
            query,
            top_k=self.config.retrieval_top_k,
            min_score=self.config.retrieval_min_score,
        )

    def summarize_and_archive(self) -> int:
        archived_count = 0

        while self.stm.should_overflow() and self.stm.size() > 0:
            oldest = self.stm.pop_oldest_for_archive()
            if self._archive_chunk(oldest):
                archived_count += 1

        return archived_count

    def flush_stm_to_ltm(self) -> int:
        """Force archive all remaining STM interactions.

        This is useful at session end so recent context is retained for future runs.
        """
        archived_count = 0
        while self.stm.size() > 0:
            oldest = self.stm.pop_oldest_for_archive()
            if self._archive_chunk(oldest):
                archived_count += 1
        return archived_count

    def build_prompt(self, user_input: str, retrieved: List[RetrievedMemory]) -> str:
        return self.prompt_builder.build_prompt(
            user_input=user_input,
            short_term_memory=self.stm.formatted_context(),
            retrieved_memories=retrieved,
        )

    def generate_response(self, prompt: str) -> str:
        try:
            return self.llm.generate(prompt=prompt, max_tokens=320, temperature=0.2)
        except Exception as exc:
            logger.exception("Primary LLM generation failed: %s", exc)
            return (
                "I am temporarily unable to reach the primary model backend. "
                "Please verify Bedrock credentials/model access and try again."
            )

    def memory_loop(self, user_input: str) -> str:
        self.add_to_stm("user", user_input)

        retrieval_query = self._build_retrieval_query(user_input)
        retrieved = self.retrieve_from_ltm(retrieval_query)
        response = self._history_query_answer(user_input, retrieved)
        if response is None:
            prompt = self.build_prompt(user_input, retrieved)
            response = self.generate_response(prompt)

        self.add_to_stm("assistant", response)
        self.summarize_and_archive()
        return response

    def _archive_chunk(self, interactions: List[Interaction]) -> bool:
        if not interactions:
            return False

        summary = self.summarizer.summarize(interactions)
        if not summary.strip():
            return False

        # Preserve semantic richness and exact terms for robust topic retrieval.
        transcript_lines = [f"{item.role.upper()}: {item.content}" for item in interactions]
        transcript = "\n".join(transcript_lines)
        archive_text = (
            "ARCHIVE SUMMARY:\n"
            f"{summary.strip()}\n\n"
            "ARCHIVE TRANSCRIPT:\n"
            f"{transcript[:1800]}"
        )

        importance = self._score_importance(interactions, archive_text)
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "stm_archive",
            "importance": f"{importance:.2f}",
            "interaction_count": str(len(interactions)),
        }
        embedding = self.embedder.embed(archive_text)
        entry_id = self.ltm_repository.add_entry(archive_text, embedding, metadata)
        if entry_id:
            logger.info("Archived STM chunk to LTM entry_id=%s", entry_id)
            return True
        return False

    def _history_query_answer(self, user_input: str, retrieved: List[RetrievedMemory]) -> str | None:
        query = user_input.lower().strip()
        asks_history = bool(
            re.search(r"\b(did we discuss|have we discussed|did we talk about|previously discuss)\b", query)
        )
        if not asks_history:
            return None

        topic = self._extract_history_topic(user_input)
        topic_tokens = self._tokenize(topic or user_input)
        found_in_stm = self._topic_exists_in_stm(topic_tokens)
        found_in_ltm = len(retrieved) > 0

        if not found_in_stm and not found_in_ltm:
            return "I do not see matching prior memory on that topic in short-term or long-term storage."

        snippets = []
        for item in retrieved[:2]:
            compact = " ".join(item.text.split())
            snippets.append(compact[:180])

        if found_in_stm:
            snippets.insert(0, "Recent chat contains this topic.")

        joined = " | ".join(snippets)
        return f"Yes. We previously discussed this topic. Relevant memory: {joined}"

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}

    @staticmethod
    def _extract_history_topic(user_input: str) -> str:
        patterns = [
            r"did we discuss anything on (.+)",
            r"did we discuss (.+)",
            r"have we discussed (.+)",
            r"did we talk about (.+)",
            r"previously discuss(?:ed)? (.+)",
        ]
        text = user_input.strip().rstrip("?.!")
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return text

    def _topic_exists_in_stm(self, topic_tokens: set[str]) -> bool:
        if not topic_tokens:
            return False

        for interaction in self.stm.interactions():
            if interaction.role != "user":
                continue
            text_tokens = self._tokenize(interaction.content)
            if topic_tokens.intersection(text_tokens):
                return True
        return False

    def _build_retrieval_query(self, user_input: str) -> str:
        query = user_input.strip()
        lower = query.lower()
        if re.search(r"\b(did we discuss|have we discussed|did we talk about|previously discuss)\b", lower):
            topic = self._extract_history_topic(query)
            if topic:
                return topic
        return query

    @staticmethod
    def _score_importance(interactions: List[Interaction], summary: str) -> float:
        text = " ".join(item.content for item in interactions).lower() + " " + summary.lower()
        importance = 0.4

        if any(token in text for token in ["remember", "important", "deadline", "password", "preference"]):
            importance += 0.3
        if any(char.isdigit() for char in text):
            importance += 0.2
        if len(summary) > 240:
            importance += 0.1

        return min(1.0, importance)
