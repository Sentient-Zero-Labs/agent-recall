"""Search ranking tests — no API calls.

Inserts memories directly via MemoryClient, backdates via raw SQL,
and calls _hybrid_search to verify ranking behavior.
"""

from __future__ import annotations

import aiosqlite
import pytest

from recall.db.connection import get_db_path
from recall.server import _hybrid_search


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
