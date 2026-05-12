"""Recall MCP server — all 6 tools, auth middleware, 30s timeout."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import aiosqlite
import anthropic
from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from recall.db.connection import get_db_path, init_db
from recall.decay import DecayWorker
from recall.logging import LoggingMiddleware
from recall.security import hash_token, validate_tool_descriptions
from recall.worker import ExtractionWorker

logger = logging.getLogger(__name__)

user_id_ctx: ContextVar[str] = ContextVar("user_id", default="")
extraction_worker: ExtractionWorker | None = None
decay_worker: DecayWorker | None = None

_VALID_MEMORY_TYPES = frozenset({"preference", "fact", "decision", "procedure"})

_MERGE_PROMPT = """\
Given these related memories about the same topic, produce ONE canonical memory that \
captures all distinct information. Be specific. Preserve concrete details. Do not generalize.

Memories:
{memories}

Return ONLY a JSON object: {{"text": "...", "type": "preference|fact|decision|procedure", "importance": 0.0-1.0}}"""


# ── Middleware ────────────────────────────────────────────────────────────────

class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # Agent Card is publicly discoverable — no auth required
        if request.url.path.startswith("/.well-known"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        user_id = await _validate_token(token)
        if not user_id:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        user_id_ctx.set(user_id)
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Any:
        try:
            return await asyncio.wait_for(call_next(request), timeout=30.0)
        except asyncio.TimeoutError:
            return JSONResponse(
                {
                    "status": "error",
                    "error": "Tool call exceeded 30s limit.",
                    "code": "TOOL_TIMEOUT",
                },
                status_code=504,
            )


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[None]:
    global extraction_worker, decay_worker
    await init_db()
    _validate_server_descriptions()
    extraction_worker = ExtractionWorker()
    await extraction_worker.start()
    decay_worker = DecayWorker()
    await decay_worker.start()
    await _recover_orphaned_operations()
    yield
    if extraction_worker:
        await extraction_worker.stop()
    if decay_worker:
        await decay_worker.stop()


async def _recover_orphaned_operations() -> None:
    """Mark any queued/processing operations left over from a prior crash as failed."""
    async with aiosqlite.connect(get_db_path()) as db:
        cursor = await db.execute(
            "UPDATE operations SET status = 'failed', updated_at = datetime('now') "
            "WHERE status IN ('queued', 'processing')",
        )
        await db.commit()
    if cursor.rowcount:
        logger.warning(
            "startup_orphan_recovery",
            extra={"count": cursor.rowcount},
        )


def _validate_server_descriptions() -> None:
    """Validate all tool descriptions against poisoning patterns. Fail fast."""
    tool_functions = [
        store_memory, search_memories, inspect_memories,
        delete_memory, get_memory_stats, consolidate_memories,
    ]
    descriptions = {
        f.__name__: (f.__doc__ or "").split("\n")[0].strip()
        for f in tool_functions
    }
    validate_tool_descriptions(descriptions)


# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("recall", lifespan=_lifespan)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def store_memory(
    text: str,
    topic: str,
    idempotency_key: str,
    session_id: str = "",
    agent_id: str = "",
) -> dict:
    """Store conversation messages for background memory extraction.
    Returns immediately — extraction runs async. Safe to retry with the same key."""
    user_id = user_id_ctx.get()

    job_id = str(uuid.uuid4())
    async with aiosqlite.connect(get_db_path()) as db:
        cursor = await db.execute(
            "INSERT OR IGNORE INTO operations (id, idempotency_key, user_id, status) "
            "VALUES (?,?,?,'queued')",
            (job_id, idempotency_key, user_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            return {"status": "ok", "data": {"queued": False, "cached": True}, "error": None}

    if extraction_worker:
        await extraction_worker.enqueue({
            "job_id": job_id,
            "user_id": user_id,
            "text": text,
            "topic": topic,
            "session_id": session_id,
            "agent_id": agent_id,
        })

    return {"status": "ok", "data": {"queued": True, "job_id": job_id}, "error": None}


@mcp.tool()
async def search_memories(
    query: str,
    limit: int = 20,
    recency_weight: float = 0.3,
    mmr_lambda: float = 0.5,
    score_threshold: float = 0.0,
    max_tokens: int | None = None,
) -> dict:
    """Search memories using hybrid retrieval (~200-500ms).

    recency_weight 0-1 upweights recent results.
    mmr_lambda 0-1: 0=max diversity, 1=pure relevance (default 0.5).
    score_threshold: drop candidates below this hybrid score (default 0.0 = keep all).
    max_tokens: trim results to fit within this token budget (None = no limit).
    """
    if not 0.0 <= recency_weight <= 1.0:
        return {"status": "error", "error": "recency_weight must be between 0.0 and 1.0.", "code": "INVALID_PARAM"}
    if not 0.0 <= mmr_lambda <= 1.0:
        return {"status": "error", "error": "mmr_lambda must be between 0.0 and 1.0.", "code": "INVALID_PARAM"}
    user_id = user_id_ctx.get()
    results = await _hybrid_search(
        user_id, query, min(limit, 50), recency_weight, mmr_lambda, score_threshold
    )
    if max_tokens is not None:
        results = _apply_budget(results, max_tokens)
    return {"status": "ok", "data": {"results": results, "total": len(results)}, "error": None}


@mcp.tool()
async def inspect_memories(limit: int = 20, offset: int = 0) -> dict:
    """List stored memories with pagination. Default 20 per page, max 50."""
    limit = min(limit, 50)
    user_id = user_id_ctx.get()

    async with aiosqlite.connect(get_db_path()) as db:
        rows = await db.execute_fetchall(
            "SELECT id, text, topic, importance, type, created_at "
            "FROM memories WHERE user_id = ? AND valid_until IS NULL "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        )
        count = await db.execute_fetchall(
            "SELECT COUNT(*) FROM memories WHERE user_id = ? AND valid_until IS NULL",
            (user_id,),
        )
        total = count[0][0]

    memories = [
        {
            "id": r[0],
            "text": r[1],
            "topic": r[2],
            "importance": r[3],
            "type": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]
    return {
        "status": "ok",
        "data": {
            "memories": memories,
            "total": total,
            "has_more": offset + limit < total,
            "next_offset": offset + limit if offset + limit < total else None,
        },
        "error": None,
    }


@mcp.tool()
async def delete_memory(memory_id: str) -> dict:
    """Permanently delete a memory by ID. This action cannot be undone."""
    user_id = user_id_ctx.get()

    async with aiosqlite.connect(get_db_path()) as db:
        cursor = await db.execute(
            "DELETE FROM memories WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            return {
                "status": "error",
                "error": (
                    f"Memory '{memory_id}' not found. "
                    "Use inspect_memories to list valid IDs."
                ),
                "code": "MEMORY_NOT_FOUND",
            }

    return {"status": "ok", "data": {"deleted": memory_id}, "error": None}


@mcp.tool()
async def get_memory_stats() -> dict:
    """Return memory counts and storage stats for the current user. Fast health check."""
    user_id = user_id_ctx.get()

    async with aiosqlite.connect(get_db_path()) as db:
        by_type = await db.execute_fetchall(
            "SELECT type, COUNT(*) FROM memories "
            "WHERE user_id = ? AND valid_until IS NULL GROUP BY type",
            (user_id,),
        )
        pending = await db.execute_fetchall(
            "SELECT COUNT(*) FROM operations WHERE user_id = ? AND status = 'queued'",
            (user_id,),
        )

    return {
        "status": "ok",
        "data": {
            "by_type": dict(by_type),
            "total": sum(r[1] for r in by_type),
            "pending_extractions": pending[0][0],
        },
        "error": None,
    }


@mcp.tool()
async def consolidate_memories(
    topic: str,
    similarity_threshold: float = 0.85,
    dry_run: bool = False,
) -> dict:
    """Find semantically similar memories in a topic and merge them into canonical facts.
    Requires embeddings extra: pip install 'szl-recall[embeddings]'. Returns a diff."""
    user_id = user_id_ctx.get()

    # Check embeddings available before any DB work
    from recall.embeddings import embed
    probe = embed(["test"])
    if probe is None:
        return {
            "status": "error",
            "error": "consolidate_memories requires embeddings: pip install 'szl-recall[embeddings]'",
            "code": "EMBEDDINGS_REQUIRED",
        }

    # Fetch all active memories for this topic
    async with aiosqlite.connect(get_db_path()) as db:
        rows = await db.execute_fetchall(
            "SELECT id, text, type, importance "
            "FROM memories WHERE user_id = ? AND topic = ? AND valid_until IS NULL "
            "ORDER BY created_at DESC",
            (user_id, topic),
        )

    if not rows:
        return {
            "status": "ok",
            "data": {
                "groups_found": 0, "memories_consolidated": 0,
                "memories_created": 0, "deleted_ids": [], "created": [], "dry_run": dry_run,
            },
            "error": None,
        }

    memories = [{"id": r[0], "text": r[1], "type": r[2], "importance": r[3]} for r in rows]

    # Compute embeddings for all fetched memories
    texts = [m["text"] for m in memories]
    vecs = embed(texts)
    if vecs is None:
        return {
            "status": "error",
            "error": "consolidate_memories requires embeddings: pip install 'szl-recall[embeddings]'",
            "code": "EMBEDDINGS_REQUIRED",
        }
    for i, m in enumerate(memories):
        m["_vec"] = vecs[i]

    # Group by cosine similarity — only merge groups with 2+ members
    groups = _greedy_cluster(memories, similarity_threshold)
    merge_groups = [g for g in groups if len(g) >= 2]

    if not merge_groups:
        return {
            "status": "ok",
            "data": {
                "groups_found": 0, "memories_consolidated": 0,
                "memories_created": 0, "deleted_ids": [], "created": [], "dry_run": dry_run,
            },
            "error": None,
        }

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error": "ANTHROPIC_API_KEY not set.",
            "code": "NO_API_KEY",
        }

    llm_client = anthropic.AsyncAnthropic(api_key=api_key)
    deleted_ids: list[str] = []
    created_memories: list[dict] = []

    try:
        for group in merge_groups:
            merged = await _llm_merge(llm_client, group, topic)
            if merged:
                deleted_ids.extend(m["id"] for m in group)
                created_memories.append(merged)
    finally:
        await llm_client.close()

    if dry_run:
        return {
            "status": "ok",
            "data": {
                "groups_found": len(merge_groups),
                "memories_consolidated": len(deleted_ids),
                "memories_created": len(created_memories),
                "deleted_ids": deleted_ids,
                "created": [{"text": m["text"], "type": m["type"], "importance": m["importance"]} for m in created_memories],
                "dry_run": True,
            },
            "error": None,
        }

    # Persist: insert canonical memories, supersede originals
    now = datetime.now(timezone.utc).isoformat()
    persisted: list[dict] = []
    async with aiosqlite.connect(get_db_path()) as db:
        for merged in created_memories:
            new_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO memories (id, user_id, text, type, topic, importance, created_at, valid_from) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (new_id, user_id, merged["text"], merged["type"], topic, merged["importance"], now, now),
            )
            persisted.append({"id": new_id, "text": merged["text"], "type": merged["type"], "topic": topic})
        for del_id in deleted_ids:
            await db.execute(
                "UPDATE memories SET valid_until = ? WHERE id = ? AND user_id = ?",
                (now, del_id, user_id),
            )
        await db.commit()

    return {
        "status": "ok",
        "data": {
            "groups_found": len(merge_groups),
            "memories_consolidated": len(deleted_ids),
            "memories_created": len(persisted),
            "deleted_ids": deleted_ids,
            "created": persisted,
            "dry_run": False,
        },
        "error": None,
    }


@mcp.tool()
async def delete_user_data(confirm: str) -> dict:
    """Permanently delete ALL data for the current user. Irreversible.

    Pass confirm='DELETE MY DATA' exactly to proceed.
    Deletes: all memories, A2A tasks, pending operations, and tool-call records.
    Revokes (does not delete) API tokens so the user knows their auth is gone.
    """
    if confirm != "DELETE MY DATA":
        return {
            "status": "error",
            "error": "Pass confirm='DELETE MY DATA' exactly to proceed.",
            "code": "CONFIRM_REQUIRED",
        }
    user_id = user_id_ctx.get()
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        await db.execute("DELETE FROM a2a_tasks WHERE user_id = ?", (user_id,))
        await db.execute("DELETE FROM operations WHERE user_id = ?", (user_id,))
        await db.execute(
            "DELETE FROM tool_call_records WHERE user_id = ? AND tool_name != 'delete_user_data'",
            (user_id,),
        )
        r = await db.execute(
            "UPDATE api_tokens SET revoked = 1 WHERE user_id = ?", (user_id,)
        )
        tokens_revoked = r.rowcount
        await db.commit()
    return {
        "status": "ok",
        "data": {"tokens_revoked": tokens_revoked},
        "error": None,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _validate_token(token: str) -> str | None:
    if not token:
        return None
    token_hash = hash_token(token)
    db_path = get_db_path()
    async with aiosqlite.connect(db_path) as db:
        rows = await db.execute_fetchall(
            "SELECT user_id FROM api_tokens WHERE token_hash = ? AND revoked = 0",
            (token_hash,),
        )
    return rows[0][0] if rows else None


def _parse_ts(ts: str) -> datetime:
    """Parse ISO8601 timestamp from SQLite, ensuring UTC timezone."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _bm25_ranks(query: str, texts: list[str]) -> list[int] | None:
    """Return per-document BM25 rank (1-based). None if rank_bm25 not installed.

    Uses BM25Plus over BM25Okapi: BM25Okapi IDF collapses to 0 with N=2 when a term
    appears in exactly one document (log(1.5/1.5)=0), causing all scores to be 0.
    BM25Plus adds a lower-bound delta that keeps scores positive for small corpora.
    """
    try:
        from rank_bm25 import BM25Plus
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Plus(tokenized)
        scores = list(bm25.get_scores(query.lower().split()))
        # rank = position in descending score order (0-based → 1-based)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(order):
            ranks[idx] = rank + 1  # 1-based
        # Docs with zero BM25Plus score have no query term overlap — push to worst rank
        for i, s in enumerate(scores):
            if s == 0:
                ranks[i] = len(scores) + 1
        return ranks
    except ImportError:
        return None


def _dense_ranks(query: str, texts: list[str]) -> list[int] | None:
    """Return per-document cosine similarity rank (1-based). None if model unavailable."""
    from recall.embeddings import cosine_scores, embed, embed_query
    q_vec = embed_query(query)
    if q_vec is None:
        return None
    doc_vecs = embed(texts)
    if doc_vecs is None:
        return None
    sims = cosine_scores(q_vec, doc_vecs)
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    ranks = [0] * len(sims)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks


def _mmr_rerank(
    candidates: list[dict],
    query_vec: "np.ndarray | None",
    mmr_lambda: float,
    limit: int,
) -> list[dict]:
    """Max-Marginal Relevance reranking for diversity.

    Iteratively selects the candidate that maximises:
      mmr_lambda * sim(query, doc) - (1 - mmr_lambda) * max(sim(doc, selected))

    Falls back to candidates[:limit] when embeddings are unavailable.
    """
    if query_vec is None or len(candidates) <= limit:
        return candidates[:limit]

    from recall.embeddings import embed
    import numpy as np

    doc_vecs = embed([m["text"] for m in candidates])
    if doc_vecs is None:
        return candidates[:limit]

    q_sims = (doc_vecs @ query_vec).tolist()

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected) < limit and remaining:
        best_i, best_score = None, float("-inf")
        for i in remaining:
            redundancy = float(np.max(doc_vecs[selected] @ doc_vecs[i])) if selected else 0.0
            mmr_score = mmr_lambda * q_sims[i] - (1.0 - mmr_lambda) * redundancy
            if mmr_score > best_score:
                best_score, best_i = mmr_score, i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected]


def _rrf_fuse(
    bm25_ranks: list[int] | None,
    dense_ranks: list[int] | None,
    k: int = 60,
) -> list[float]:
    """Reciprocal Rank Fusion. Returns a score per document (higher = better match)."""
    n = len(bm25_ranks or dense_ranks or [])
    if n == 0:
        return []
    scores = [0.0] * n
    for ranks in (bm25_ranks, dense_ranks):
        if ranks is None:
            continue
        for i, r in enumerate(ranks):
            if r <= n:  # r == n+1 means zero BM25 signal — skip
                scores[i] += 1.0 / (k + r)
    return scores


def _greedy_cluster(memories: list[dict], threshold: float) -> list[list[dict]]:
    """Group memories by cosine similarity using greedy assignment.

    Each unassigned memory starts a new cluster. Subsequent memories join the
    cluster of the first memory they exceed `threshold` similarity with.
    Vectors must be pre-computed and stored as m['_vec'].
    """
    import numpy as np

    if len(memories) <= 1:
        return [[m] for m in memories]

    vecs = np.stack([m["_vec"] for m in memories])
    sim_matrix = vecs @ vecs.T  # cosine similarity (vectors already L2-normalized)

    assigned = [False] * len(memories)
    groups: list[list[dict]] = []

    for i in range(len(memories)):
        if assigned[i]:
            continue
        group = [memories[i]]
        assigned[i] = True
        for j in range(i + 1, len(memories)):
            if not assigned[j] and float(sim_matrix[i, j]) >= threshold:
                group.append(memories[j])
                assigned[j] = True
        groups.append(group)

    return groups


async def _llm_merge(
    client: anthropic.AsyncAnthropic, group: list[dict], topic: str
) -> dict | None:
    """Call Claude Haiku to merge a group of similar memories into one canonical memory."""
    memories_str = "\n".join(f"{i + 1}. {m['text']}" for i, m in enumerate(group))
    prompt = _MERGE_PROMPT.format(memories=memories_str)

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        merged = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("consolidation_merge_parse_failed", extra={"topic": topic, "raw": raw[:100]})
        return None

    mem_type = merged.get("type", "fact")
    if mem_type not in _VALID_MEMORY_TYPES:
        mem_type = "fact"
    text = str(merged.get("text", "")).strip()
    if not text:
        return None

    return {
        "text": text,
        "type": mem_type,
        "importance": max(0.0, min(1.0, float(merged.get("importance", 0.5)))),
    }


def _estimate_tokens(text: str) -> int:
    """Estimate token count using the ~4 chars/token GPT-family heuristic."""
    return max(1, len(text) // 4)


def _apply_budget(memories: list[dict], max_tokens: int) -> list[dict]:
    """Greedily trim the ranked result list to fit within a token budget.

    Always returns at least one memory even if it exceeds the budget.
    """
    kept: list[dict] = []
    used = 0
    for m in memories:
        cost = _estimate_tokens(m.get("text", ""))
        if kept and used + cost > max_tokens:
            break
        kept.append(m)
        used += cost
    return kept


async def _hybrid_search(
    user_id: str,
    query: str,
    limit: int,
    recency_weight: float = 0.3,
    mmr_lambda: float = 0.5,
    score_threshold: float = 0.0,
) -> list[dict]:
    """BM25 + dense RRF(k=60) hybrid retrieval with MMR diversification.

    Components (Park et al. 2023 + MemoryBank):
      w_rrf      · RRF(BM25, cosine)
      w_recency  · exp(-Δt / (1 + access_count))   — recency with strength modifier
      w_import   · importance * decay_score          — decay-adjusted importance
      w_strength · log(1+ac) / log(1+max_ac)        — access frequency (strength)

    Weights interpolate linearly with recency_weight (always sum to 1.0):
      recency_weight=0 → (w_rrf=0.70, w_recency=0.00, w_import=0.20, w_strength=0.10)
      recency_weight=1 → (w_rrf=0.40, w_recency=0.40, w_import=0.10, w_strength=0.10)

    After scoring, applies score_threshold filtering then MMR reranking for diversity.
    Updates access_count and last_accessed for returned memories.
    """
    async with aiosqlite.connect(get_db_path()) as db:
        rows = await db.execute_fetchall(
            "SELECT id, text, topic, importance, type, created_at, access_count, decay_score "
            "FROM memories WHERE user_id = ? AND valid_until IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, limit * 5),
        )

    if not rows:
        return []

    memories = [
        {
            "id": r[0], "text": r[1], "topic": r[2], "importance": r[3],
            "type": r[4], "created_at": r[5], "access_count": r[6] or 0,
            "decay_score": r[7],
        }
        for r in rows
    ]
    texts = [m["text"] for m in memories]

    bm25_r = _bm25_ranks(query, texts)
    dense_r = _dense_ranks(query, texts)
    rrf = _rrf_fuse(bm25_r, dense_r, k=60)

    now = datetime.now(timezone.utc)
    max_ac = max((m["access_count"] for m in memories), default=1) or 1

    w_rrf      = 0.70 - 0.30 * recency_weight
    w_recency  = 0.40 * recency_weight
    w_import   = 0.20 - 0.10 * recency_weight
    w_strength = 0.10

    scored: list[tuple[float, dict]] = []
    for i, m in enumerate(memories):
        if rrf[i] == 0:
            continue  # no BM25 or dense signal — suppress from results
        age_days = (now - _parse_ts(m["created_at"])).total_seconds() / 86400
        ac = m["access_count"]
        recency = math.exp(-age_days / (1 + ac))
        strength = math.log(1 + ac) / math.log(1 + max_ac)
        decay = m["decay_score"]
        effective_importance = (m.get("importance") or 0.5) * (decay if decay is not None else 1.0)
        score = (
            w_rrf      * rrf[i]
            + w_recency  * recency
            + w_import   * effective_importance
            + w_strength * strength
        )
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Pre-MMR pool: top limit*3 candidates, filtered by score_threshold
    pre_mmr = [m for s, m in scored[: limit * 3] if s >= score_threshold]

    # MMR diversification — embed_query is fast (model already warm from _dense_ranks)
    from recall.embeddings import embed_query as _eq
    results = _mmr_rerank(pre_mmr, _eq(query), mmr_lambda, limit)

    # Update access tracking for returned memories
    if results:
        returned_ids = [m["id"] for m in results]
        try:
            async with aiosqlite.connect(get_db_path()) as db:
                for mid in returned_ids:
                    await db.execute(
                        "UPDATE memories SET access_count = access_count + 1, "
                        "last_accessed = datetime('now') WHERE id = ?",
                        (mid,),
                    )
                await db.commit()
        except Exception as exc:
            logger.warning("access_count_update_failed", extra={"error": str(exc)})

    return results


# ── ASGI app factory ──────────────────────────────────────────────────────────

def create_app():
    """Return the Starlette app with all middleware and A2A routes wired. Used by uvicorn."""
    from recall.a2a import create_a2a_router, create_well_known_router

    app = mcp.http_app(transport="streamable-http")
    app.mount("/a2a", create_a2a_router(user_id_ctx))
    app.mount("/.well-known", create_well_known_router())
    # Starlette applies middleware in reverse order — LoggingMiddleware runs outermost (sees full duration)
    app.add_middleware(LoggingMiddleware, user_id_ctx=user_id_ctx)
    app.add_middleware(TimeoutMiddleware)
    app.add_middleware(BearerAuthMiddleware)
    return app
