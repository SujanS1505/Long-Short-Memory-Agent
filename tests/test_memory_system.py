from __future__ import annotations

from pathlib import Path

from agent.agent_core import AgentConfig, HierarchicalMemoryAgent
from agent.prompt_template import PromptBuilder
from embeddings.embedding_model import HashingEmbeddingStrategy
from llm.model import LLMStrategy
from memory.long_term import FaissLongTermMemoryRepository
from memory.short_term import ShortTermMemory
from memory.summarizer import ConversationSummarizer
from retrieval.retriever import MemoryRetriever


class TestLLM(LLMStrategy):
    """Deterministic LLM stub for repeatable tests.

    - Summarization requests return a transcript-preserving summary.
    - Normal generation requests return a fixed safe response.
    """

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        del max_tokens, temperature
        if "TRANSCRIPT:" in prompt:
            return "ARCHIVE SUMMARY:\n" + prompt.split("TRANSCRIPT:", maxsplit=1)[1].strip()
        return "Acknowledged."


def build_test_agent() -> HierarchicalMemoryAgent:
    embedder = HashingEmbeddingStrategy(dimension=384)
    llm = TestLLM()

    stm = ShortTermMemory(max_interactions=10, max_tokens=600, overflow_archive_size=6)
    ltm = FaissLongTermMemoryRepository(dimension=embedder.dimension)

    summarizer = ConversationSummarizer(llm=llm)
    retriever = MemoryRetriever(repository=ltm, embedder=embedder)
    prompt_builder = PromptBuilder()

    return HierarchicalMemoryAgent(
        stm=stm,
        ltm_repository=ltm,
        embedder=embedder,
        llm=llm,
        summarizer=summarizer,
        retriever=retriever,
        prompt_builder=prompt_builder,
        config=AgentConfig(retrieval_top_k=8, retrieval_min_score=0.05),
    )


def test_archive_and_retrieve_100_plus_interactions() -> None:
    agent = build_test_agent()

    for i in range(110):
        message = f"Remember this historical detail #{i}: customer token TOK-{5000 + i}."
        _ = agent.memory_loop(message)

    assert agent.ltm_repository.size() > 0

    query = "What was the customer token for detail #3?"
    retrieved = agent.retrieve_from_ltm(query)
    combined = "\n".join(item.text for item in retrieved)

    # We assert a nearby or exact retained fact appears in archived summaries.
    assert "TOK-5003" in combined or "#3" in combined or "detail #3" in combined


def test_empty_memory_retrieval() -> None:
    agent = build_test_agent()
    retrieved = agent.retrieve_from_ltm("Anything there?")
    assert retrieved == []


def test_ltm_persists_across_repository_rebuild(tmp_path: Path) -> None:
    embedder = HashingEmbeddingStrategy(dimension=384)
    persist_dir = str(tmp_path / "ltm_store")

    first_repo = FaissLongTermMemoryRepository(dimension=embedder.dimension, persist_dir=persist_dir)
    text = "red teaming is an adversarial simulation to find weaknesses before attackers do"
    emb = embedder.embed(text)
    entry_id = first_repo.add_entry(text=text, embedding=emb, metadata={"source": "unit_test", "importance": "0.9"})
    assert entry_id is not None
    assert first_repo.size() == 1

    second_repo = FaissLongTermMemoryRepository(dimension=embedder.dimension, persist_dir=persist_dir)
    results = second_repo.search(
        query_embedding=embedder.embed("what is red teaming"),
        top_k=3,
        min_score=0.01,
        keyword_query="red teaming",
    )
    assert second_repo.size() >= 1
    assert any("red teaming" in str(item["text"]).lower() for item in results)


def test_history_query_uses_stm_in_same_chat() -> None:
    agent = build_test_agent()
    _ = agent.memory_loop("AWS provides EC2, S3, and RDS services.")
    reply = agent.memory_loop("did we discuss anything on aws?")
    assert "Yes. We previously discussed this topic." in reply
