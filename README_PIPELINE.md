# Hierarchical Memory Agent Pipeline Guide

This document explains how the full memory pipeline works during runtime.

## 1) What this system is solving

Large-context chat agents fail in two common ways:

- They forget old details once context window is full.
- They hallucinate when they cannot find grounded facts.

This project fixes that with two memory layers:

- STM (Short-Term Memory): fast recent buffer for the latest turns.
- LTM (Long-Term Memory): archived summaries in a vector index for semantic retrieval.

## 2) High-level architecture

- STM manager: memory/short_term.py
- LTM repository: memory/long_term.py
- Summarizer: memory/summarizer.py
- Retriever: retrieval/retriever.py
- Prompt builder: agent/prompt_template.py
- Agent orchestrator: agent/agent_core.py
- LLM/Embedding strategies: llm/model.py, embeddings/embedding_model.py
- Entrypoint: main.py

## 3) End-to-end request lifecycle

For each user message, the runtime executes this pipeline:

1. Accept user input.
2. Save user input into STM.
3. Create query embedding and retrieve top-k related LTM memories.
4. Build a grounded prompt containing:
   - Retrieved LTM context
   - Recent STM conversation
   - Current user query
   - Anti-hallucination instructions
5. Send prompt to selected LLM.
6. Save assistant response to STM.
7. If STM exceeds limits:
   - Pop oldest K interactions
   - Summarize them
   - Embed summary
   - Store summary + metadata in LTM
   - Remove popped items from STM

## 4) How overflow and archiving work

STM has two limits:

- max_interactions (count-based)
- max_tokens (approximate token budget)

When either limit is exceeded, the overflow loop archives old chunks into LTM.
This prevents context overflow while preserving history in searchable form.

## 5) Retrieval quality and scoring

LTM retrieval uses hybrid scoring:

- Semantic similarity (vector score)
- Keyword overlap (lexical match)

Final score is a weighted combination. Results under min_score are filtered out.

## 6) Why your previous responses looked generic

Your .env had LLM_PROVIDER=rule_based, which forced fallback behavior.
That fallback is useful for offline testing but not for high-quality answers.

Now the default is:

- LLM_PROVIDER=auto
- If AWS credentials exist, Bedrock is selected.
- If not, rule-based fallback is used.

## 7) Run modes

### Bedrock primary mode (recommended)

- Set AWS credentials and region.
- Keep LLM_PROVIDER=auto (or set bedrock explicitly).
- Run: python main.py

By default, synthetic demo is disabled so your real chats are not polluted.
Set DEMO_INTERACTIONS to a positive number only when you want test data.

### Local fallback mode

- Set LLM_PROVIDER=rule_based.
- Useful for debugging pipeline mechanics without cloud calls.

## 8) How memory grounding prevents hallucination

The prompt always includes retrieved memory context and explicit instruction to avoid invention.
If no relevant context is found, the model should answer with uncertainty rather than guessing.

## 9) Observing internals while running

Look for logs like:

- Selected providers at startup (embedding + llm)
- Archived STM chunk to LTM entry_id=...

These confirm retrieval and archival are active.

## 10) Common troubleshooting

### Problem: responses are generic

- Check .env value for LLM_PROVIDER.
- Ensure startup log shows llm provider=bedrock.

### Problem: no retrieval hits

- Lower retrieval_min_score in main.py.
- Ensure archived summaries contain key facts.

### Problem: Bedrock errors

- Verify credentials and region.
- Confirm model access is enabled in your Bedrock account.

## 11) Suggested production hardening

- LTM persistence is implemented via LTM_PERSIST_DIR (FAISS vectors + metadata on disk).
- Add retry/backoff around Bedrock calls.
- Add metrics for retrieval hit rate and answer grounding rate.
- Add evaluation tests for fact recall at turn 100+.

## 12) Cross-session memory behavior

To preserve memory across fresh conversations and app restarts:

- Keep LTM_PERSIST_DIR set (default ./data/ltm).
- On exit, interactive mode flushes remaining STM into LTM.
- On next startup, repository reloads persisted entries automatically.

This enables follow-up prompts like "did we discuss red teaming?" to recover prior sessions.
