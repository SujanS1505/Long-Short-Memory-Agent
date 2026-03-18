"""Prompt builder for grounded memory-aware responses."""

from __future__ import annotations

from typing import Iterable

from retrieval.retriever import RetrievedMemory


class PromptBuilder:
    TEMPLATE = (
        "SYSTEM:\n"
        "You are an AI assistant with memory. Your goal is to provide seamless, conversational answers.\n\n"
        "RETRIEVED CONTEXT:\n"
        "{retrieved_context}\n\n"
        "RECENT CONVERSATION:\n"
        "{short_term_memory}\n\n"
        "USER QUERY:\n"
        "{user_input}\n\n"
        "INSTRUCTIONS:\n"
        "- If the retrieved context or recent conversation contains information relevant to the user query, combine that information to answer the query naturally.\n"
        "- If the topic hasn't been discussed previously (no relevant context), provide a fresh, comprehensive response on the topic from your own knowledge.\n"
        "- NEVER output the literal words 'RETRIEVED CONTEXT:' or display the raw context scores/summaries in your response.\n"
        "- Provide a natural, direct answer.\n\n"
        "ASSISTANT:\n"
    )

    def build_prompt(
        self,
        user_input: str,
        short_term_memory: str,
        retrieved_memories: Iterable[RetrievedMemory],
    ) -> str:
        context_parts = []
        for idx, item in enumerate(retrieved_memories, start=1):
            context_parts.append(f"[{idx}] score={item.score:.3f} | {item.text}")

        retrieved_context = "\n".join(context_parts) if context_parts else "(no relevant long-term memory found)"

        return self.TEMPLATE.format(
            retrieved_context=retrieved_context,
            short_term_memory=short_term_memory,
            user_input=user_input,
        )
