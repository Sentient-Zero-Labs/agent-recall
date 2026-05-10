# Recall

Persistent memory layer for AI agents. Local-first. Inspectable. Framework-agnostic.

Built as the implementation anchor for the [Building Effective Tools for AI](https://read.sentientzerolabs.com) series by Sentient Zero Labs.

---

## What it does

Recall gives AI agents persistent, structured memory across sessions.

- **Store** conversation transcripts → extracted as typed facts, preferences, decisions, procedures
- **Search** memories using hybrid retrieval (BM25 + vector in v0.2)
- **Inspect** stored memories with pagination
- **Delete** specific memories
- **Observe** every operation via structured `ToolCallRecord` logging

## Install

```bash
pip install recall-memory
```

With optional vector search support:

```bash
pip install "recall-memory[embeddings]"
```

With Postgres backend:

```bash
pip install "recall-memory[postgres]"
```

## Quick start

```bash
# 1. Set your Anthropic API key (used for memory extraction)
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Start the server once to initialize the database
recall serve --port 8000 &

# 3. Create an API token for your user
recall create-token my-user-id

#    Output:
#      Token created for user: my-user-id
#
#        Bearer <token>
#
#      Store this token securely — it will not be shown again.

# 4. Stop the background server and restart normally
kill %1
recall serve --port 8000
```

Add to your MCP client (Claude Desktop example):

```json
{
  "mcpServers": {
    "recall": {
      "url": "http://localhost:8000/mcp",
      "transport": "streamable-http",
      "headers": {
        "Authorization": "Bearer <your-token>"
      }
    }
  }
}
```

## Five tools

| Tool | Purpose | Pattern |
|------|---------|---------|
| `store_memory` | Store conversation for extraction | Async-acknowledge, idempotent |
| `search_memories` | Hybrid retrieval by query | Synchronous, paginated |
| `inspect_memories` | List all memories | Synchronous, paginated |
| `delete_memory` | Remove one memory by ID | Synchronous |
| `get_memory_stats` | Counts + pending queue | Synchronous, fast health check |

## Architecture

```
Agent
  │
  │  MCP (Streamable HTTP)
  │
  ▼
┌──────────────────────┐
│  BearerAuthMiddleware │  ← validates token, injects user_id via ContextVar
│  TimeoutMiddleware    │  ← 30s hard cap via asyncio.wait_for
│  LoggingMiddleware    │  ← ToolCallRecord per call
├──────────────────────┤
│  store_memory         │  ← enqueue + return fast (<10ms)
│  search_memories      │  ← BM25 search (v0.1), +vectors (v0.2)
│  inspect_memories     │  ← paginated list
│  delete_memory        │  ← ownership-checked delete
│  get_memory_stats     │  ← health check
└──────────────────────┘
  │                   │
  │ SQLite (WAL mode)  │  asyncio.Queue
  │                   ▼
  │         ┌──────────────────┐
  │         │ ExtractionWorker │  ← processes store_memory jobs async
  │         └──────────────────┘
  ▼
recall.db
```

## Development

```bash
git clone https://github.com/ppritish51/recall
cd recall

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Used by the extraction worker to classify memories via Claude |
| `RECALL_DB_PATH` | No | `recall.db` | Path to the SQLite database file |

Copy `.env.example` to `.env` and fill in your values, or export them directly.

## Versions

- **v0.1** (current): SQLite + BM25 search + 5 MCP tools. LLM extraction via Claude Haiku — classifies conversation text into typed memories (fact, preference, decision, procedure).
- **v0.2** (Memory series): Vector embeddings, hybrid BM25+vector RRF search, Postgres support, memory decay + contradiction resolution.

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Sentient Zero Labs](https://sentientzerolabs.com).
Newsletter: [read.sentientzerolabs.com](https://read.sentientzerolabs.com).
