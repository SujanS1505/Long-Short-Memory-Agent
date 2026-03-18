"""Embedding model strategies and factory."""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import List, Sequence


class EmbeddingStrategy(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class HashingEmbeddingStrategy(EmbeddingStrategy):
    """Deterministic local embedding fallback with no external dependencies."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        if not text:
            return vector

        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector


class HuggingFaceEmbeddingStrategy(EmbeddingStrategy):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for HuggingFace embeddings"
            ) from exc

        self._model = SentenceTransformer(model_name)
        sample = self._model.encode("dimension_probe", normalize_embeddings=True)
        self._dimension = int(len(sample))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in embedding]


class BedrockEmbeddingStrategy(EmbeddingStrategy):
    """AWS Bedrock Titan embedding strategy."""

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v1",
        region_name: str | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for Bedrock embeddings") from exc

        self._model_id = model_id
        self._client = boto3.client(
            "bedrock-runtime", region_name=region_name or os.getenv("AWS_REGION", "us-east-1")
        )
        self._dimension = 1536
        # Probe once so downstream vector stores are initialized with the actual model dimension.
        self._dimension = len(self.embed("dimension_probe"))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        import json

        payload = {"inputText": text}
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        data = json.loads(response["body"].read())
        embedding = data["embedding"]
        self._dimension = len(embedding)
        return [float(x) for x in embedding]


class EmbeddingFactory:
    """Factory for creating embedding strategies."""

    @staticmethod
    def create(provider: str = "hashing", model_name: str | None = None) -> EmbeddingStrategy:
        normalized = provider.lower().strip()
        if normalized in {"hashing", "local", "fallback"}:
            return HashingEmbeddingStrategy()
        if normalized in {"hf", "huggingface"}:
            return HuggingFaceEmbeddingStrategy(
                model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )
        if normalized in {"bedrock", "aws"}:
            return BedrockEmbeddingStrategy(model_id=model_name or "amazon.titan-embed-text-v1")
        raise ValueError(f"Unsupported embedding provider: {provider}")
