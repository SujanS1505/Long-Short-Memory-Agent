"""Example entrypoint for the hierarchical memory agent."""

from __future__ import annotations

import logging
import os
import re

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from agent.agent_core import AgentConfig, HierarchicalMemoryAgent
from agent.prompt_template import PromptBuilder
from embeddings.embedding_model import EmbeddingFactory
from llm.model import LLMFactory
from memory.long_term import FaissLongTermMemoryRepository
from memory.short_term import ShortTermMemory
from memory.summarizer import ConversationSummarizer
from retrieval.retriever import MemoryRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

if load_dotenv is not None:
    load_dotenv()


def _select_llm_provider() -> str:
    configured = os.getenv("LLM_PROVIDER", "").strip().lower()
    if configured and configured not in {"auto", "default"}:
        return configured

    has_aws = bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
    return "bedrock" if has_aws else "rule_based"


def _select_embedding_provider() -> str:
    configured = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if configured and configured not in {"auto", "default"}:
        return configured

    has_aws = bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
    return "bedrock" if has_aws else "hashing"


def build_agent() -> HierarchicalMemoryAgent:
    embedding_provider = _select_embedding_provider()
    embedding_model = os.getenv("EMBEDDING_MODEL", "")

    llm_provider = _select_llm_provider()
    llm_model = os.getenv(
        "LLM_MODEL",
        os.getenv("BEDROCK_LLM_MODEL_ID", "mistral.mistral-7b-instruct-v0:2"),
    )

    logger.info("Selected embedding provider=%s model=%s", embedding_provider, embedding_model or "default")
    logger.info("Selected llm provider=%s model=%s", llm_provider, llm_model or "default")

    embedder = EmbeddingFactory.create(provider=embedding_provider, model_name=embedding_model or None)
    llm = LLMFactory.create(provider=llm_provider, model_name=llm_model or None)

    default_model_slug = embedding_model or "default"
    default_model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", default_model_slug)
    default_persist_dir = f"./data/ltm/{embedding_provider}_{default_model_slug}_{embedder.dimension}"
    ltm_persist_dir = os.getenv("LTM_PERSIST_DIR", default_persist_dir)

    stm = ShortTermMemory(max_interactions=10, max_tokens=900, overflow_archive_size=6)
    ltm = FaissLongTermMemoryRepository(dimension=embedder.dimension, persist_dir=ltm_persist_dir)

    summarizer = ConversationSummarizer(llm=llm)
    retriever = MemoryRetriever(repository=ltm, embedder=embedder)
    prompt_builder = PromptBuilder()

    config = AgentConfig(retrieval_top_k=5, retrieval_min_score=0.02)
    return HierarchicalMemoryAgent(
        stm=stm,
        ltm_repository=ltm,
        embedder=embedder,
        llm=llm,
        summarizer=summarizer,
        retriever=retriever,
        prompt_builder=prompt_builder,
        config=config,
    )


def demo_archival_and_recall(agent: HierarchicalMemoryAgent) -> None:
    demo_interactions = int(os.getenv("DEMO_INTERACTIONS", "0"))
    if demo_interactions <= 0:
        print("Skipping synthetic demo. Set DEMO_INTERACTIONS>0 to enable it.")
        return

    print(f"Running archival demo with {demo_interactions} synthetic interactions...")
    for i in range(demo_interactions):
        fact = f"Fact #{i}: project milestone code is M-{1000 + i}."
        user_text = f"Please remember this detail: {fact}"
        _ = agent.memory_loop(user_text)

    print(f"STM size after overflow management: {agent.stm.size()}")
    print(f"LTM size after archival: {agent.ltm_repository.size()}")

    query = "What was the project milestone code around Fact #10?"
    retrieved = agent.retrieve_from_ltm(query)
    print(f"Retrieved memories for query: {len(retrieved)}")
    for idx, item in enumerate(retrieved[:3], start=1):
        print(f"{idx}. score={item.score:.3f} | {item.text[:160]}")


def interactive_chat(agent: HierarchicalMemoryAgent) -> None:
    print("\nInteractive mode started. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        response = agent.memory_loop(user_input)
        print(f"Assistant: {response}\n")

    archived = agent.flush_stm_to_ltm()
    print(f"Session archived chunks on exit: {archived}")


if __name__ == "__main__":
    assistant = build_agent()
    demo_archival_and_recall(assistant)
    interactive_chat(assistant)
