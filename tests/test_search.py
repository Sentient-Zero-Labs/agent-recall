"""Search ranking and correctness tests — no API calls.

Inserts memories directly via MemoryClient (and raw SQL for v2 fields),
backdates created_at via raw SQL, calls _hybrid_search to verify ranking
and filtering behavior.
"""

from __future__ import annotations

import uuid

import aiosqlite
import pytest

from recall.db.connection import get_db_path
from recall.server import _hybrid_search


class TestSearchCorrectness:
    async def test_superseded_memory_excluded_from_search(self, client, user_id):
        """Memories with valid_until set must not appear in search results."""
        old = await client.store(user_id, "User preferred language is Go for backend work", "tech")
        # Manually supersede the old memory — simulates what _handle_contradiction does
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE memories SET valid_until = datetime('now') WHERE id = ?", (old.id,)
            )
            await db.commit()

        # Active replacement
        new = await client.store(user_id, "User preferred language is Python for backend work", "tech")

        results = await _hybrid_search(user_id, "preferred language backend", limit=10, recency_weight=0.3)
        ids = [r["id"] for r in results]
        assert old.id not in ids, "superseded memory must not appear in search results"
        assert new.id in ids, "active memory must appear in search results"

    async def test_superseded_memory_excluded_from_inspect(self, client, user_id):
        """inspect_memories must only return active (valid_until IS NULL) memories."""
        from recall.server import inspect_memories
        from recall.server import user_id_ctx

        active = await client.store(user_id, "Active memory about Python preferences", "tech")
        old = await client.store(user_id, "Old superseded memory about Go preferences", "tech")
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE memories SET valid_until = datetime('now') WHERE id = ?", (old.id,)
            )
            await db.commit()

        token = user_id_ctx.set(user_id)
        try:
            result = await inspect_memories(limit=50)
        finally:
            user_id_ctx.reset(token)

        ids = [m["id"] for m in result["data"]["memories"]]
        assert active.id in ids
        assert old.id not in ids
        assert result["data"]["total"] == 1

    async def test_stats_counts_only_active_memories(self, client, user_id):
        """get_memory_stats must not count superseded memories in total."""
        from recall.server import get_memory_stats, user_id_ctx

        await client.store(user_id, "Active memory one", "tech")
        await client.store(user_id, "Active memory two", "tech")
        old = await client.store(user_id, "Superseded memory", "tech")
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE memories SET valid_until = datetime('now') WHERE id = ?", (old.id,)
            )
            await db.commit()

        token = user_id_ctx.set(user_id)
        try:
            result = await get_memory_stats()
        finally:
            user_id_ctx.reset(token)

        assert result["data"]["total"] == 2, "stats total must exclude superseded memories"

    async def test_idempotency_is_atomic(self, client, user_id):
        """Two concurrent store_memory calls with the same idempotency_key produce exactly one job."""
        import asyncio
        from recall.server import store_memory, user_id_ctx

        token = user_id_ctx.set(user_id)
        key = f"idem-{uuid.uuid4()}"
        try:
            results = await asyncio.gather(
                store_memory("text A", "topic", key),
                store_memory("text B", "topic", key),
            )
        finally:
            user_id_ctx.reset(token)

        # Exactly one must succeed (queued=True), the other must be cached
        queued = [r for r in results if r["data"].get("queued")]
        cached = [r for r in results if r["data"].get("cached")]
        assert len(queued) == 1, "exactly one request should queue the job"
        assert len(cached) == 1, "the duplicate must be returned as cached"

        # Verify only one row exists in DB
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM operations WHERE idempotency_key = ?", (key,)
            )
        assert len(rows) == 1, "only one operation row must exist for a given idempotency_key"


class TestSearchRanking:
    async def test_recency_weight_1_prefers_newer_memory(self, client, user_id):
        """recency_weight=1.0 ranks the newest memory first when BM25 relevance is equal."""
        old = await client.store(user_id, "Python is great for backend development", "tech")
        new = await client.store(user_id, "Python development is good for backend", "tech")

        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE memories SET created_at = datetime('now', '-30 days') WHERE id = ?",
                (old.id,),
            )
            await db.commit()

        results = await _hybrid_search(user_id, "Python backend", limit=10, recency_weight=1.0)
        assert len(results) >= 1
        assert results[0]["id"] == new.id, "recency_weight=1.0 should rank newest first"

    async def test_recency_weight_0_prefers_relevant_text(self, client, user_id):
        """recency_weight=0.0 ranks the more BM25-relevant memory first, ignoring age."""
        relevant = await client.store(
            user_id, "Python FastAPI backend API development framework", "tech"
        )
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE memories SET created_at = datetime('now', '-30 days') WHERE id = ?",
                (relevant.id,),
            )
            await db.commit()

        await client.store(user_id, "I went hiking in the mountains last weekend", "personal")

        results = await _hybrid_search(user_id, "Python API backend", limit=10, recency_weight=0.0)
        assert len(results) >= 1
        assert results[0]["id"] == relevant.id, "recency_weight=0.0 should rank BM25-relevant first"

    async def test_no_match_returns_empty(self, client, user_id):
        """A query with zero relevance signal returns empty list — not recency-surfaced junk."""
        await client.store(user_id, "I prefer dark mode in my IDE", "prefs")
        await client.store(user_id, "My favorite coffee is Ethiopian Yirgacheffe", "food")

        results = await _hybrid_search(user_id, "quantumfluxcapacitor", limit=10, recency_weight=0.3)
        assert results == [], "Zero relevance should return empty list"
