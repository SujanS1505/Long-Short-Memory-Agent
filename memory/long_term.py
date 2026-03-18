"""Long-term memory repository with FAISS-first vector retrieval.

Implements:
- Repository pattern via abstract base class
- Singleton index manager for vector DB instance reuse
- Semantic and hybrid retrieval with dedup safeguards
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
PERSIST_DIR_NOT_CONFIGURED = "Persistence directory is not configured"


@dataclass(slots=True)
class MemoryEntry:
    text: str
    embedding: List[float]
    metadata: Dict[str, str] = field(default_factory=dict)
    entry_id: str = ""


class LongTermRepository(ABC):
    @abstractmethod
    def add_entry(self, text: str, embedding: List[float], metadata: Dict[str, str]) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.2,
        keyword_query: str | None = None,
    ) -> List[Dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError


class _VectorIndexSingleton:
    """Singleton manager for vector index instances."""

    _instances: Dict[Tuple[int, str], "_VectorIndexSingleton"] = {}

    def __new__(cls, dimension: int, persist_key: str = "memory") -> "_VectorIndexSingleton":
        key = (dimension, persist_key)
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._init_store(dimension)
            cls._instances[key] = instance
        return cls._instances[key]

    def _init_store(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors = np.zeros((0, dimension), dtype=np.float32)
        self.has_faiss = False
        self.faiss_index = None

        try:
            import faiss

            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.has_faiss = True
            logger.info("Using FAISS IndexFlatIP for long-term memory")
        except Exception:
            logger.warning(
                "FAISS is unavailable; using numpy fallback for vector retrieval"
            )

    def add(self, vector: np.ndarray) -> None:
        normalized = self._normalize(vector)

        if normalized.shape[0] != self.dimension:
            if self.vectors.shape[0] == 0:
                logger.warning(
                    "Vector dimension mismatch on empty index (expected=%d got=%d). Reinitializing index.",
                    self.dimension,
                    normalized.shape[0],
                )
                self._init_store(int(normalized.shape[0]))
                normalized = self._normalize(vector)
            else:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {normalized.shape[0]}"
                )

        if self.has_faiss and self.faiss_index is not None:
            self.faiss_index.add(normalized.reshape(1, -1))
        self.vectors = np.vstack([self.vectors, normalized])

    def load_vectors(self, vectors: np.ndarray) -> None:
        normalized_vectors = vectors.astype(np.float32)
        self.vectors = normalized_vectors

        if self.has_faiss and self.faiss_index is not None:
            try:
                import faiss

                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                if normalized_vectors.shape[0] > 0:
                    self.faiss_index.add(normalized_vectors)
            except Exception:
                logger.exception("Failed to rebuild FAISS index from persisted vectors")

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query = self._normalize(query_vector)

        if self.vectors.shape[0] == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        if self.has_faiss and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query.reshape(1, -1), top_k)
            return scores[0], indices[0]

        similarities = np.dot(self.vectors, query.T).flatten()
        ranked = np.argsort(similarities)[::-1][:top_k]
        return similarities[ranked], ranked

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)


class FaissLongTermMemoryRepository(LongTermRepository):
    def __init__(self, dimension: int, persist_dir: str | None = None, auto_persist: bool = True) -> None:
        self._persist_dir = Path(persist_dir).resolve() if persist_dir else None
        persist_key = str(self._persist_dir) if self._persist_dir else "memory"
        self._index = _VectorIndexSingleton(dimension, persist_key=persist_key)
        self._entries: List[MemoryEntry] = []
        self._dedup_hashes: set[str] = set()
        self._auto_persist = auto_persist

        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def add_entry(self, text: str, embedding: List[float], metadata: Dict[str, str]) -> str | None:
        if not text.strip():
            return None

        text_hash = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
        if text_hash in self._dedup_hashes:
            return None

        entry_id = hashlib.sha256(
            f"{datetime.now(timezone.utc).isoformat()}::{text_hash}".encode("utf-8")
        ).hexdigest()[:16]

        payload = MemoryEntry(
            text=text,
            embedding=embedding,
            metadata={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": metadata.get("source", "unknown"),
                "importance": str(metadata.get("importance", "0.5")),
                **metadata,
            },
            entry_id=entry_id,
        )

        vector = np.array(embedding, dtype=np.float32)
        self._index.add(vector)
        self._entries.append(payload)
        self._dedup_hashes.add(text_hash)
        if self._persist_dir is not None and self._auto_persist:
            self._save_to_disk()
        return entry_id

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.2,
        keyword_query: str | None = None,
    ) -> List[Dict[str, object]]:
        if not self._entries:
            return []

        scores, indices = self._index.search(np.array(query_embedding, dtype=np.float32), top_k)
        results: List[Dict[str, object]] = []

        keywords = self._tokenize(keyword_query or "")
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._entries):
                continue

            entry = self._entries[idx]
            semantic_score = float(score)
            keyword_score = 0.0
            if keywords:
                text_words = self._tokenize(entry.text)
                overlap = len(keywords.intersection(text_words))
                keyword_score = overlap / max(1, len(keywords))

            hybrid_score = 0.8 * semantic_score + 0.2 * keyword_score
            if hybrid_score < min_score:
                continue

            results.append(
                {
                    "entry_id": entry.entry_id,
                    "text": entry.text,
                    "metadata": entry.metadata,
                    "semantic_score": round(semantic_score, 4),
                    "keyword_score": round(keyword_score, 4),
                    "hybrid_score": round(hybrid_score, 4),
                }
            )

        return sorted(results, key=lambda x: x["hybrid_score"], reverse=True)

    def size(self) -> int:
        return len(self._entries)

    def get_entry_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def _metadata_path(self) -> Path:
        if self._persist_dir is None:
            raise RuntimeError(PERSIST_DIR_NOT_CONFIGURED)
        return self._persist_dir / "entries.jsonl"

    def _vectors_path(self) -> Path:
        if self._persist_dir is None:
            raise RuntimeError(PERSIST_DIR_NOT_CONFIGURED)
        return self._persist_dir / "vectors.npy"

    def _faiss_path(self) -> Path:
        if self._persist_dir is None:
            raise RuntimeError(PERSIST_DIR_NOT_CONFIGURED)
        return self._persist_dir / "index.faiss"

    def _save_to_disk(self) -> None:
        if self._persist_dir is None:
            return

        meta_path = self._metadata_path()
        vectors_path = self._vectors_path()

        with meta_path.open("w", encoding="utf-8") as f:
            for entry in self._entries:
                row = {
                    "entry_id": entry.entry_id,
                    "text": entry.text,
                    "metadata": entry.metadata,
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        np.save(vectors_path, self._index.vectors)

        if self._index.has_faiss and self._index.faiss_index is not None:
            try:
                import faiss

                faiss.write_index(self._index.faiss_index, str(self._faiss_path()))
            except Exception:
                logger.exception("Failed to persist FAISS index to disk")

    def _load_from_disk(self) -> None:
        if self._persist_dir is None:
            return

        meta_path = self._metadata_path()
        vectors_path = self._vectors_path()

        if not meta_path.exists() or not vectors_path.exists():
            return

        loaded_entries: List[MemoryEntry] = []
        dedup_hashes: set[str] = set()
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("text", ""))
                metadata = row.get("metadata", {}) or {}
                entry_id = str(row.get("entry_id", ""))
                loaded_entries.append(
                    MemoryEntry(text=text, embedding=[], metadata=metadata, entry_id=entry_id)
                )
                dedup_hashes.add(hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest())

        vectors = np.load(vectors_path)
        if vectors.ndim != 2 or vectors.shape[1] != self._index.dimension:
            logger.warning("Persisted vectors shape mismatch; ignoring persisted state")
            return

        if vectors.shape[0] != len(loaded_entries):
            logger.warning("Persisted entries/vectors count mismatch; ignoring persisted state")
            return

        self._index.load_vectors(vectors)
        for i, entry in enumerate(loaded_entries):
            entry.embedding = [float(x) for x in vectors[i].tolist()]

        self._entries = loaded_entries
        self._dedup_hashes = dedup_hashes
        logger.info("Loaded %d long-term memory entries from disk", len(self._entries))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}
