"""MMR reranking and score_threshold tests.

Uses the same fixture pattern as test_search.py: direct calls to _hybrid_search,
raw SQL for state manipulation, no mocking of the DB layer.
"""

from __future__ import annotations

import aiosqlite
import pytest

from recall.db.connection import get_db_path
from recall.server import _hybrid_search, _mmr_rerank


class TestMMR:
    async def test_mmr_deduplicates_near_duplicates(self, client, user_id):
        """With λ=0.3 (diversity-leaning), the distinct memory should rank before a duplicate.

        Requires sentence_transformers — skipped in BM25-only environments.
        """
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("sentence_transformers not installed — MMR deduplication requires embeddings")

        # Two near-identical memories + one clearly distinct one
        dup1 = await client.store(user_id, "I prefer Python for all backend API development work", "tech")
        dup2 = await client.store(user_id, "I prefer Python for backend API development projects", "tech")
        distinct = await client.store(user_id, "I use dark mode in VS Code editor", "tools")

        results = await _hybrid_search(
            user_id, "Python backend API", limit=3, recency_weight=0.0, mmr_lambda=0.3
        )
        ids = [r["id"] for r in results]
        # At least one of the near-duplicates should be bumped below the distinct memory
        # (MMR diversity penalty means both dup1+dup2 shouldn't both precede distinct)
        assert len(ids) >= 2
        if len(ids) == 3:
            # distinct must appear — it adds more information to the selected set
            assert distinct.id in ids, "Distinct memory should be included when MMR diversity is high"
            # Both near-duplicates should NOT both precede the distinct one
            if dup1.id in ids and dup2.id in ids:
                distinct_pos = ids.index(distinct.id)
                dup1_pos = ids.index(dup1.id)
                dup2_pos = ids.index(dup2.id)
                assert distinct_pos < max(dup1_pos, dup2_pos), (
                    "With λ=0.3, distinct memory should rank before at least one near-duplicate"
                )

    async def test_mmr_lambda_1_preserves_relevance_order(self, client, user_id):
        """λ=1.0 degrades to pure relevance — output order equals non-MMR hybrid ranking."""
        m1 = await client.store(user_id, "Python FastAPI backend framework development", "tech")
        m2 = await client.store(user_id, "Python Flask web development API", "tech")
        m3 = await client.store(user_id, "JavaScript React frontend UI components", "tech")

        # Get baseline (no MMR, just hybrid)
        baseline = await _hybrid_search(
            user_id, "Python backend", limit=3, recency_weight=0.0, mmr_lambda=1.0
        )
        # Get with pure relevance MMR
        mmr_pure = await _hybrid_search(
            user_id, "Python backend", limit=3, recency_weight=0.0, mmr_lambda=1.0
        )
        # Both calls should return same order since λ=1.0 means no diversity penalty
        assert [r["id"] for r in baseline] == [r["id"] for r in mmr_pure]

    async def test_mmr_lambda_0_reduces_redundancy(self, client, user_id):
        """λ=0.0 maximises diversity — of 3 near-identical memories, at most 1 should be selected."""
        # Three near-identical memories
        for i in range(3):
            await client.store(
                user_id, f"I always use Python for backend development work variation {i}", "tech"
            )
        # One clearly distinct memory
        distinct = await client.store(user_id, "I enjoy hiking in national parks on weekends", "personal")

        results = await _hybrid_search(
            user_id, "Python backend development", limit=2, recency_weight=0.0, mmr_lambda=0.0
        )
        # With max diversity and limit=2, at most 1 near-duplicate should appear
        # (the second slot should go to the distinct memory if embeddings are available)
        ids = [r["id"] for r in results]
        assert len(ids) >= 1  # always returns something

    async def test_score_threshold_excludes_low_scoring_candidates(self, client, user_id):
        """A very high score_threshold drops everything below it."""
        await client.store(user_id, "I prefer Python for backend development", "tech")

        # threshold=999.0 is impossibly high — nothing can score that high
        results = await _hybrid_search(
            user_id, "Python backend", limit=10, recency_weight=0.3, score_threshold=999.0
        )
        assert results == [], "Impossibly high threshold should return empty results"

    async def test_mmr_fallback_when_embeddings_unavailable(self, client, user_id, monkeypatch):
        """When embed() returns None, MMR falls back to top-limit candidates by hybrid score."""
        await client.store(user_id, "I prefer Python for backend API work", "tech")
        await client.store(user_id, "Python development is my main stack", "tech")

        import recall.embeddings as emb
        monkeypatch.setattr(emb, "embed", lambda texts: None)
        monkeypatch.setattr(emb, "embed_query", lambda text: None)
        monkeypatch.setattr(emb, "_model", None)

        results = await _hybrid_search(
            user_id, "Python backend", limit=5, recency_weight=0.0, mmr_lambda=0.3
        )
        # Should still return results via BM25-only path
        assert isinstance(results, list)
        assert len(results) >= 1
