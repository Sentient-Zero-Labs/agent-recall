"""Decay scoring tests — no API calls.

Tests the DecayWorker and its integration with _hybrid_search.
All tests manipulate created_at / last_accessed / access_count via raw SQL
to control decay inputs deterministically.
"""

from __future__ import annotations

import aiosqlite
import pytest

from recall.db.connection import get_db_path
from recall.decay import DecayWorker


@pytest.fixture
async def decay_worker(tmp_db):
    """DecayWorker backed by the test database."""
    w = DecayWorker()
    return w


class TestDecayScoring:
    async def test_decay_score_null_before_first_run(self, client, user_id):
        """A freshly stored memory has no decay_score yet — DecayWorker has not run."""
        m = await client.store(user_id, "User prefers Python for backend work", "tech")
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT decay_score FROM memories WHERE id = ?", (m.id,)
            )
        assert rows[0][0] is None, "decay_score must be NULL before any decay run"

    async def test_decay_score_set_after_run(self, client, user_id, decay_worker):
        """run_once() writes a decay_score to all active memories."""
        await client.store(user_id, "User prefers dark mode in IDE", "prefs")
        count = await decay_worker.run_once()
        assert count >= 1
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT decay_score FROM memories WHERE user_id = ? AND valid_until IS NULL",
                (user_id,),
            )
        for (score,) in rows:
            assert score is not None, "run_once must set decay_score on every active memory"
            assert 0.0 < score <= 1.0, f"decay_score must be in (0, 1], got {score}"

    async def test_old_memory_has_lower_decay_than_new(self, client, user_id, decay_worker):
        """A 90-day-old memory should have a lower decay_score than a fresh one."""
        old = await client.store(user_id, "User liked Java for enterprise work", "tech")
        new = await client.store(user_id, "User prefers TypeScript for frontend", "tech")

        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE memories SET created_at = datetime('now', '-90 days'), "
                "last_accessed = datetime('now', '-90 days') WHERE id = ?",
                (old.id,),
            )
            await db.commit()

        await decay_worker.run_once()

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id, decay_score FROM memories WHERE id IN (?, ?)",
                (old.id, new.id),
            )
        scores = {r[0]: r[1] for r in rows}
        assert scores[old.id] < scores[new.id], (
            f"90-day-old memory ({scores[old.id]:.3f}) should decay more than "
            f"fresh memory ({scores[new.id]:.3f})"
        )

    async def test_frequently_accessed_memory_decays_slower(self, client, user_id, decay_worker):
        """A memory accessed 10+ times should retain higher decay_score than one never accessed."""
        low_ac = await client.store(user_id, "User prefers light theme in VSCode", "prefs")
        high_ac = await client.store(user_id, "User prefers dark theme in VSCode", "prefs")

        async with aiosqlite.connect(get_db_path()) as db:
            # Both backdated 60 days, but high_ac has been accessed many times
            for mid in (low_ac.id, high_ac.id):
                await db.execute(
                    "UPDATE memories SET created_at = datetime('now', '-60 days'), "
                    "last_accessed = datetime('now', '-60 days') WHERE id = ?",
                    (mid,),
                )
            await db.execute(
                "UPDATE memories SET access_count = 10 WHERE id = ?", (high_ac.id,)
            )
            await db.commit()

        await decay_worker.run_once()

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id, decay_score FROM memories WHERE id IN (?, ?)",
                (low_ac.id, high_ac.id),
            )
        scores = {r[0]: r[1] for r in rows}
        assert scores[high_ac.id] > scores[low_ac.id], (
            "Frequently accessed memory should have higher decay_score (slower decay)"
        )

    async def test_decay_does_not_modify_importance(self, client, user_id, decay_worker):
        """DecayWorker updates decay_score only — importance is never modified."""
        m = await client.store(user_id, "User is a senior software engineer at Acme Corp", "work")
        # Manually set importance to a known value
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE memories SET importance = 0.9, "
                "created_at = datetime('now', '-180 days'), "
                "last_accessed = datetime('now', '-180 days') WHERE id = ?",
                (m.id,),
            )
            await db.commit()

        await decay_worker.run_once()

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT importance, decay_score FROM memories WHERE id = ?", (m.id,)
            )
        importance, decay_score = rows[0]
        assert importance == pytest.approx(0.9), "run_once must not modify importance"
        assert decay_score is not None and decay_score < 0.5, (
            "180-day-old memory should have decay_score < 0.5"
        )

    async def test_decay_only_affects_active_memories(self, client, user_id, decay_worker):
        """Superseded memories (valid_until set) must not get a decay_score."""
        active = await client.store(user_id, "User uses FastAPI for APIs", "tech")
        superseded = await client.store(user_id, "User used Flask for APIs", "tech")

        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "UPDATE memories SET valid_until = datetime('now') WHERE id = ?",
                (superseded.id,),
            )
            await db.commit()

        await decay_worker.run_once()

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id, decay_score FROM memories WHERE id IN (?, ?)",
                (active.id, superseded.id),
            )
        scores = {r[0]: r[1] for r in rows}
        assert scores[active.id] is not None, "active memory must have decay_score after run"
        assert scores[superseded.id] is None, "superseded memory must not have decay_score set"

    async def test_run_once_returns_count(self, client, user_id, decay_worker):
        """run_once() must return the exact count of memories it updated."""
        await client.store(user_id, "memory one", "general")
        await client.store(user_id, "memory two", "general")
        await client.store(user_id, "memory three", "general")

        count = await decay_worker.run_once()
        assert count == 3

    async def test_decay_worker_start_stop(self, tmp_db):
        """start() then immediate stop() must not raise or leave dangling tasks."""
        w = DecayWorker()
        await w.start()
        await w.stop()
        # If we reach here without exception, the test passes

    async def test_decay_score_affects_search_ranking(self, client, user_id, decay_worker):
        """A memory with low decay_score should rank below one with high decay_score
        when BM25 relevance and base importance are equal and recency_weight=0."""
        from recall.server import _hybrid_search

        fresh = await client.store(user_id, "Python is the best language for data science", "tech")
        stale = await client.store(user_id, "Python is the best language for data science work", "tech")

        async with aiosqlite.connect(get_db_path()) as db:
            # Stale: 180 days old, never accessed
            await db.execute(
                "UPDATE memories SET created_at = datetime('now', '-180 days'), "
                "last_accessed = datetime('now', '-180 days'), importance = 0.5 WHERE id = ?",
                (stale.id,),
            )
            # Fresh: just created, access_count = 5 (simulates active use)
            await db.execute(
                "UPDATE memories SET access_count = 5, importance = 0.5 WHERE id = ?",
                (fresh.id,),
            )
            await db.commit()

        await decay_worker.run_once()

        # recency_weight=0 → scoring driven by relevance + importance * decay_score
        results = await _hybrid_search(user_id, "Python data science language", limit=10, recency_weight=0.0)
        ids = [r["id"] for r in results]
        assert len(ids) >= 2
        assert ids.index(fresh.id) < ids.index(stale.id), (
            "Fresh memory with higher decay_score should rank above stale memory"
        )
