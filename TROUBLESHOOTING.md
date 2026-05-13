# Recall — Troubleshooting Guide

Common issues when running, testing, and integrating Recall.

---

## 1. `--db` flag ignored — server uses wrong database

**Symptom:** `recall serve --db /path/to/custom.db` starts, but memories and tokens land in `recall.db` in the current directory instead of the path you specified.

**Root cause (pre-v0.3.1):** `recall.db.connection` read `RECALL_DB_PATH` at module import time. The CLI called `os.environ.setdefault("RECALL_DB_PATH", args.db)` *after* the module was already imported, so `_DB_PATH` was already resolved and the flag had no effect.

**Fix (v0.3.1+):** The CLI now calls `set_db_path(db_path)` before any server import, which updates the already-imported module-level variable. Upgrade to v0.3.1 or later.

**Workaround on older versions:**
```bash
# Set RECALL_DB_PATH *before* starting the server
export RECALL_DB_PATH=/path/to/custom.db
recall serve --host 127.0.0.1 --port 8000
```

**Critical rule:** The `recall create-token` call must always point to the same DB file the server is using:
```bash
# Server using /data/recall.db
export RECALL_DB_PATH=/data/recall.db
recall serve &

# Token must go into the SAME file
recall create-token alice --db /data/recall.db
```

If server and `create-token` point at different files, every request returns **401 Unauthorized** with no error detail (the token hash simply isn't in the DB the server is reading from).

---

## 2. `grep "Bearer "` extracts broken token

**Symptom:** Authorization header is corrupted; server returns `Invalid HTTP request received.` or `400 Bad Request`.

**Root cause:** `recall create-token` outputs two lines containing "Bearer":
```
  Bearer xJ8kAp...qT4           ← the actual token (indented with 2 spaces)
Add it to your MCP client as: Authorization: Bearer <token>   ← help text
```

`grep "Bearer "` matches both. Piping to `awk '{print $2}'` on the second match gives the literal string `<token>`, which gets newline-joined to the real token, embedding a `\n` in the Authorization header.

**Fix:** Match only the indented token output line:
```bash
TOKEN=$(recall create-token alice --db recall.db | grep "^  Bearer " | awk '{print $2}')
# Verify — should be 43 chars, no whitespace
echo "len=${#TOKEN}"
```

---

## 3. FastMCP requires `Accept` header and `initialize` handshake

**Symptom:** Direct `POST /mcp` with a `tools/call` payload returns HTTP 406 or an empty response.

**Root cause:** FastMCP's streamable-http transport enforces two requirements:
1. `Accept: application/json, text/event-stream` header on every request
2. An `initialize` RPC call before any `tools/call` — this establishes the MCP session

**Fix:** Always initialize first and carry the session ID:
```python
import httpx, json

BASE = "http://localhost:8000/mcp"
HDR = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",   # required
}

# Step 1: initialize
r = httpx.post(BASE, json={
    "jsonrpc": "2.0", "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "my-client", "version": "0"}}
}, headers=HDR)
session_id = r.headers.get("mcp-session-id")   # may be None — some transports omit it

# Step 2: tool calls carry the session header
tool_hdrs = {**HDR, **({"mcp-session-id": session_id} if session_id else {})}
r2 = httpx.post(BASE, json={
    "jsonrpc": "2.0", "id": 2,
    "method": "tools/call",
    "params": {"name": "search_memories", "arguments": {"query": "Python"}}
}, headers=tool_hdrs)

# Responses are SSE-framed: strip the "data: " prefix
text = r2.text.strip()
if text.startswith("data:"):
    text = text[len("data:"):].strip()
data = json.loads(text)
```

---

## 4. ANTHROPIC_API_KEY not found — server fails to start

**Symptom:** `RuntimeError: ANTHROPIC_API_KEY environment variable is not set.`

**Root cause:** The key lives in `recall/.env` (or wherever you placed it). Shell tricks like `source .env | cut -d= -f2-` often include surrounding quotes or whitespace that invalidate the key.

**Reliable extraction:**
```bash
# From .env file
export ANTHROPIC_API_KEY=$(python3 -c "
from pathlib import Path
for line in Path('recall/.env').read_text().splitlines():
    if line.startswith('ANTHROPIC_API_KEY='):
        print(line.split('=', 1)[1].strip())
        break
")

# Verify
echo "Key starts with: ${ANTHROPIC_API_KEY:0:15}"
```

Or install `python-dotenv` and use:
```bash
export ANTHROPIC_API_KEY=$(python3 -c "from dotenv import dotenv_values; print(dotenv_values('recall/.env')['ANTHROPIC_API_KEY'])")
```

---

## 5. 401 after `delete_namespace_data` — expected behaviour

**Symptom:** Calling `delete_namespace_data` with `confirm="DELETE MY DATA"` succeeds, but all subsequent requests return 401.

**This is correct behaviour.** `delete_namespace_data` revokes all API tokens for the user as part of the GDPR erasure — the auth credential is invalidated alongside all stored data.

To continue using Recall after erasure, create a new token:
```bash
recall create-token alice --db recall.db
```

---

## 6. Extraction returns 0 memories — BM25-only mode

**Symptom:** `store_memory` returns `status=ok, queued=True`, but `search_memories` returns empty results even after waiting several seconds.

**Check 1 — wait for async extraction:**
`store_memory` queues a job and returns immediately. The `ExtractionWorker` runs async and calls Claude Haiku to extract memories. Check pending operations:
```bash
recall status --db recall.db
# Shows: "Pending extractions: N"
```

Wait until pending extractions reach 0, then search.

**Check 2 — verify ANTHROPIC_API_KEY is set in the server process:**
The extraction worker uses the Anthropic API. If the key is missing, extraction falls back to a stub (raw text stored as a single fact). Check server logs for `extraction_failed` entries.

**Check 3 — embeddings in BM25-only mode:**
If `sentence-transformers` is not installed, Recall runs in BM25-only mode. BM25 still works — keyword matching returns results. If you're getting truly empty results, the issue is likely the DB path mismatch (Issue 1 above), not the search algorithm.

---

## 7. MMR scores all show 0.000

**Symptom:** `search_memories` with `mmr_lambda=0.3` returns results but all `score` fields are `0.0`.

**Root cause:** MMR scores are only non-zero when sentence embeddings are available. Without `sentence-transformers`, the system uses BM25 ranking and MMR falls back to top-k order — scores from the BM25 RRF fusion are not surfaced in the response.

**Fix:** Install the embeddings optional extra:
```bash
pip install "szl-recall[embeddings]"
# Downloads BAAI/bge-small-en-v1.5 (~130MB on first run)
```

---

## 8. Token hash mismatch — how to verify

If you suspect a token was created against the wrong DB, verify directly:
```python
from recall.security import hash_token
import sqlite3

raw_token = "your-raw-token-here"
db_path = "recall.db"

h = hash_token(raw_token)
db = sqlite3.connect(db_path)
row = db.execute("SELECT namespace, revoked FROM api_tokens WHERE token_hash = ?", (h,)).fetchone()
print(row)  # None = wrong DB or wrong token
```

---

## Identity Conventions (namespace as principal_id)

`namespace` in Recall is a **string isolation key** — it can be a human user, an agent, a project, or any principal. All memories are scoped to it.

Recommended naming conventions:
```bash
# Human user
recall create-token user:alice --db recall.db

# AI agent (agent has its own memory namespace)
recall create-token agent:code-reviewer-v2 --db recall.db

# Project-scoped shared memory
recall create-token project:stripe-integration --db recall.db

# Team/org scope
recall create-token team:backend-engineers --db recall.db
```

This gives you logical separation in a single DB. An agent's domain knowledge (`agent:code-reviewer-v2`) is completely isolated from a user's personal preferences (`user:alice`). If you want an agent to have access to both, create two tokens and query both namespaces, then merge the results at the application level.

A proper two-key principal model (`owner_id` + `agent_id` composite isolation) is planned for v0.4.
