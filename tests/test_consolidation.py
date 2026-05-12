"""Consolidation MCP tool tests.

Most tests run without an API key (edge cases, error conditions, dry_run).
The actual LLM merge test is skipped when ANTHROPIC_API_KEY is not set.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest

from recall.db.connection import get_db_path
from recall.server import consolidate_memories, user_id_ctx

_needs_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skip live consolidation tests",
)


@pytest.fixture
def set_user(user_id):
    """Set user_id_ctx for the duration of a test."""
    token = user_id_ctx.set(user_id)
    yield
    user_id_ctx.reset(token)


class TestConsolidationEdgeCases:
    async def test_empty_topic_returns_ok(self, client, user_id, set_user, monkeypatch):
        """No memories in topic → ok with zeros, no LLM call needed."""
        _patch_embed(monkeypatch)
        result = await consolidate_memories(topic="nonexistent-topic")
        assert result["status"] == "ok"
        assert result["data"]["groups_found"] == 0
        assert result["data"]["memories_consolidated"] == 0
        assert result["data"]["deleted_ids"] == []

    async def test_single_memory_topic_noop(self, client, user_id, set_user, monkeypatch):
        """One memory in topic → no merge possible, return ok with zeros."""
        _patch_embed(monkeypatch)
        await client.store(user_id, "User prefers Python for backend work", "solo-topic")
        result = await consolidate_memories(topic="solo-topic")
        assert result["status"] == "ok"
        assert result["data"]["groups_found"] == 0
        assert result["data"]["memories_consolidated"] == 0

    async def test_embeddings_required_error(self, client, user_id, set_user, monkeypatch):
        """When embed() returns None, must return EMBEDDINGS_REQUIRED error."""
        monkeypatch.setattr("recall.server.embed", lambda texts: None, raising=False)
        from recall import server as s
        monkeypatch.setattr(s, "embed", lambda texts: None)

        await client.store(user_id, "User likes Python", "tech")
        await client.store(user_id, "User prefers Python", "tech")

        # Patch at the import point inside the tool
        with patch("recall.embeddings.embed", return_value=None):
            result = await consolidate_memories(topic="tech")

        assert result["status"] == "error"
        assert result["code"] == "EMBEDDINGS_REQUIRED"

    async def test_dry_run_no_mutations(self, client, user_id, set_user, monkeypatch):
        """dry_run=True must return a plan without modifying the database."""
        _patch_embed(monkeypatch, similar=True)
        _patch_llm_merge(monkeypatch)

        await client.store(user_id, "User uses Python for backend development", "tech")
        await client.store(user_id, "User prefers Python for backend services", "tech")

        result = await consolidate_memories(topic="tech", dry_run=True)

        assert result["status"] == "ok"
        assert result["data"]["dry_run"] is True

        # Database must be unchanged
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM memories WHERE user_id = ? AND topic = ? AND valid_until IS NULL",
                (user_id, "tech"),
            )
        assert len(rows) == 2, "dry_run must not supersede any memories"

    async def test_topic_isolation(self, client, user_id, set_user, monkeypatch):
        """Consolidating topic A must not touch memories in topic B."""
        _patch_embed(monkeypatch, similar=True)
        _patch_llm_merge(monkeypatch)

        await client.store(user_id, "User uses Python for backend development", "tech")
        await client.store(user_id, "User prefers Python for backend services", "tech")
        b1 = await client.store(user_id, "User goes hiking every weekend", "personal")
        b2 = await client.store(user_id, "User enjoys outdoor activities on weekends", "personal")

        await consolidate_memories(topic="tech")

        # Topic B memories must still be active
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM memories WHERE id IN (?, ?) AND valid_until IS NULL",
                (b1.id, b2.id),
            )
        assert len(rows) == 2, "consolidation must not affect memories in other topics"

    async def test_consolidation_result_shape(self, client, user_id, set_user, monkeypatch):
        """Response shape must include all required fields."""
        _patch_embed(monkeypatch, similar=True)
        _patch_llm_merge(monkeypatch)

        await client.store(user_id, "User uses Python for backend development", "tech")
        await client.store(user_id, "User prefers Python for backend services", "tech")

        result = await consolidate_memories(topic="tech")

        assert result["status"] == "ok"
        data = result["data"]
        assert "groups_found" in data
        assert "memories_consolidated" in data
        assert "memories_created" in data
        assert "deleted_ids" in data
        assert "created" in data
        assert "dry_run" in data
        assert data["dry_run"] is False

    @_needs_api
    async def test_live_merge_supersedes_originals(self, client, user_id, set_user):
        """Live test: merged memories must be superseded, canonical memory must be created."""
        from recall.embeddings import embed, vec_to_blob

        texts = [
            "User strongly prefers Python for backend API development",
            "User prefers using Python when building backend services",
            "User likes Python for server-side development work",
        ]

        ids = []
        vecs = embed(texts)
        if vecs is None:
            pytest.skip("embeddings not available")

        for i, text in enumerate(texts):
            m = await client.store(user_id, text, "tech-live")
            ids.append(m.id)

        result = await consolidate_memories(topic="tech-live", similarity_threshold=0.7)

        assert result["status"] == "ok"
        data = result["data"]
        assert data["memories_consolidated"] >= 2, "should have merged at least 2 memories"
        assert data["memories_created"] >= 1, "should have created at least 1 canonical memory"

        # Original memories must now be superseded
        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id, valid_until FROM memories WHERE id IN (?, ?, ?)",
                tuple(ids),
            )
        superseded = [r for r in rows if r[1] is not None]
        assert len(superseded) >= 2, "merged originals must have valid_until set"


# ── Test helpers ──────────────────────────────────────────────────────────────

def _patch_embed(monkeypatch, similar: bool = False):
    """Patch embed() to return fake vectors. similar=True makes all vectors identical
    (cosine similarity = 1.0, will cluster together above any reasonable threshold)."""
    import numpy as np

    def fake_embed(texts):
        if similar:
            # All identical unit vectors → cosine sim = 1.0 → cluster together
            vec = np.ones(4, dtype=np.float32)
            vec /= np.linalg.norm(vec)
            return np.stack([vec] * len(texts))
        # Each text gets a unique random unit vector → will not cluster
        rng = np.random.default_rng(seed=42)
        vecs = rng.random((len(texts), 4)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    import recall.server as s
    monkeypatch.setattr(s, "embed", fake_embed, raising=False)

    import recall.embeddings as e
    monkeypatch.setattr(e, "embed", fake_embed)


def _patch_llm_merge(monkeypatch):
    """Patch _llm_merge to return a deterministic merged memory without API calls."""
    import recall.server as s

    async def fake_merge(client, group, topic):
        texts = " / ".join(m["text"][:30] for m in group)
        return {
            "text": f"Canonical: {texts}",
            "type": "preference",
            "importance": 0.7,
        }

    monkeypatch.setattr(s, "_llm_merge", fake_merge)
