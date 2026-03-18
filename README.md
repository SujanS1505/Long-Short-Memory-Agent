# Hierarchical Memory Management System

Production-ready dual-layer memory system for an autonomous AI agent.

Detailed flow documentation: see `README_PIPELINE.md`.

## Features

- Short-Term Memory (STM): FIFO queue with interaction and token limits
- Long-Term Memory (LTM): FAISS-first semantic repository with metadata
- Summarization-based archiving from STM to LTM
- Dynamic context injection into prompts
- Hybrid retrieval: semantic + keyword overlap
- Strategy/Factory patterns for LLM and embeddings
- Pipeline-style agent loop

## Structure

```text
project/
├── memory/
│   ├── short_term.py
│   ├── long_term.py
│   ├── summarizer.py
├── retrieval/
│   ├── retriever.py
├── agent/
│   ├── prompt_template.py
│   ├── agent_core.py
├── embeddings/
│   ├── embedding_model.py
├── llm/
│   ├── model.py
├── utils/
│   ├── token_counter.py
├── tests/
│   ├── test_memory_system.py
└── main.py
```

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Optional environment variables:

- `EMBEDDING_PROVIDER` = `hashing` | `huggingface` | `bedrock`
- `EMBEDDING_MODEL` = provider-specific model id
- `LLM_PROVIDER` = `rule_based` | `huggingface` | `bedrock`
- `LLM_MODEL` = provider-specific model id
- AWS credentials should be loaded from your environment or `.env`, never hardcoded.

3. Run demo + chat:

```bash
python main.py
```

4. Run tests:

```bash
pytest -q
```
