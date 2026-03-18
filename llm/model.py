"""LLM model strategies and factory."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Optional


class LLMStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        raise NotImplementedError


class RuleBasedLLMStrategy(LLMStrategy):
    """Safe default fallback for local testing when no external LLM is configured."""

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        del max_tokens, temperature

        lower_prompt = prompt.lower()
        user_query = self._extract_section(prompt, "USER QUERY:", "INSTRUCTIONS:").strip()
        retrieved_context = self._extract_section(
            prompt,
            "RETRIEVED CONTEXT:",
            "RECENT CONVERSATION:",
        )
        recent_conversation = self._extract_section(
            prompt,
            "RECENT CONVERSATION:",
            "USER QUERY:",
        )
        memory_text = f"{retrieved_context}\n{recent_conversation}"

        if "TRANSCRIPT:" in prompt:
            transcript = prompt.split("TRANSCRIPT:", maxsplit=1)[1].strip()
            return (
                "ARCHIVE SUMMARY:\n"
                + "- Key conversation details were archived for long-term retrieval.\n"
                + f"- Transcript excerpt: {transcript[:1200]}"
            )

        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
        if user_query.lower() in greetings:
            return "Hello. I am ready to help."

        fact_answer = self._answer_fact_lookup(user_query, memory_text)
        if fact_answer:
            return fact_answer

        definition_answer = self._answer_definition_lookup(user_query, memory_text)
        if definition_answer:
            return definition_answer

        if retrieved_context.strip() and "(no relevant long-term memory found)" not in retrieved_context.lower():
            evidence_lines = [
                line.strip()
                for line in retrieved_context.splitlines()
                if line.strip() and not line.strip().startswith("[") and "score=" not in line
            ]
            if evidence_lines:
                return f"Based on memory: {evidence_lines[0][:320]}"

        if "retrieved context:\n(no relevant long-term memory found)" in lower_prompt:
            if user_query:
                return (
                    "I do not have grounded memory for that yet. "
                    f"Please provide more details and I will store them. Question received: {user_query}"
                )
            return "I do not have grounded memory for that yet."

        if user_query:
            return (
                "I have memory context but cannot extract a high-confidence exact answer. "
                f"Please provide one more specific detail so I can answer precisely: {user_query}"
            )
        return "I can help with that."

    @staticmethod
    def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
        if start_marker not in text:
            return ""
        section = text.split(start_marker, maxsplit=1)[1]
        if end_marker in section:
            return section.split(end_marker, maxsplit=1)[0]
        return section

    @staticmethod
    def _answer_fact_lookup(user_query: str, memory_text: str) -> str | None:
        query_match = re.search(r"fact\s*#\s*(\d+)", user_query, re.IGNORECASE)
        if not query_match:
            return None

        fact_number = query_match.group(1)
        # Matches lines such as: Fact #10: project milestone code is M-1010.
        pattern = rf"Fact\s*#\s*{fact_number}\s*:\s*([^\n\.]+(?:\.[^\n]*)?)"
        memory_match = re.search(pattern, memory_text, re.IGNORECASE)
        if memory_match:
            detail = memory_match.group(1).strip()
            return f"From memory: Fact #{fact_number}: {detail}"
        return None

    @staticmethod
    def _answer_definition_lookup(user_query: str, memory_text: str) -> str | None:
        match = re.search(r"what\s+is\s+([\w\s-]+)\??", user_query, re.IGNORECASE)
        if not match:
            return None

        term = match.group(1).strip().lower()
        if not term:
            return None

        patterns = [
            rf"{re.escape(term)}\s+is\s+([^\n\.]+(?:\.[^\n]*)?)",
            rf"{re.escape(term)}\s*:\s*([^\n\.]+(?:\.[^\n]*)?)",
        ]
        for pattern in patterns:
            memory_match = re.search(pattern, memory_text, re.IGNORECASE)
            if memory_match:
                definition = memory_match.group(1).strip()
                return f"From memory: {term} is {definition}"

        return None


class HuggingFaceLLMStrategy(LLMStrategy):
    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers is required for HuggingFace LLM") from exc

        self._pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        result = self._pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
        return str(result[0]["generated_text"]).strip()


class BedrockMistralLLMStrategy(LLMStrategy):
    """Bedrock wrapper for Mistral family models."""

    def __init__(
        self,
        model_id: str = "mistral.mistral-7b-instruct-v0:2",
        region_name: Optional[str] = None,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for Bedrock LLM") from exc

        self._model_id = model_id
        self._client = boto3.client(
            "bedrock-runtime", region_name=region_name or os.getenv("AWS_REGION", "us-east-1")
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        data = json.loads(response["body"].read())

        for key in ("outputs", "generation", "generated_text", "output"):
            if key in data:
                value = data[key]
                if isinstance(value, list) and value:
                    first = value[0]
                    if isinstance(first, dict):
                        return str(first.get("text", "")).strip()
                    return str(first).strip()
                if isinstance(value, str):
                    return value.strip()

        return str(data).strip()


class LLMFactory:
    """Factory for creating LLM strategies."""

    @staticmethod
    def create(provider: str = "rule_based", model_name: str | None = None) -> LLMStrategy:
        normalized = provider.lower().strip()
        if normalized in {"rule_based", "fallback", "local"}:
            return RuleBasedLLMStrategy()
        if normalized in {"hf", "huggingface"}:
            return HuggingFaceLLMStrategy(model_name=model_name or "google/flan-t5-base")
        if normalized in {"bedrock", "aws"}:
            return BedrockMistralLLMStrategy(
                model_id=model_name or "mistral.mistral-7b-instruct-v0:2"
            )
        raise ValueError(f"Unsupported LLM provider: {provider}")
