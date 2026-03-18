"""Retriever for long-term memory similarity search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from embeddings.embedding_model import EmbeddingStrategy
from memory.long_term import LongTermRepository


@dataclass(slots=True)
class RetrievedMemory:
    text: str
    metadata: dict
    score: float


class MemoryRetriever:
    def __init__(self, repository: LongTermRepository, embedder: EmbeddingStrategy) -> None:
        self._repository = repository
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.25) -> List[RetrievedMemory]:
        if not query.strip() or self._repository.size() == 0:
            return []

        query_embedding = self._embedder.embed(query)
        raw_results = self._repository.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            keyword_query=query,
        )

        return [
            RetrievedMemory(
                text=item["text"],
                metadata=item["metadata"],
                score=float(item["hybrid_score"]),
            )
            for item in raw_results
        ]
