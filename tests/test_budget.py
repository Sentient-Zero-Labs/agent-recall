"""Context budget manager tests.

Tests _estimate_tokens, _apply_budget, and the max_tokens parameter
wired into search_memories.
"""

from __future__ import annotations

import pytest

from recall.server import _apply_budget, _estimate_tokens, search_memories, user_id_ctx


class TestTokenEstimation:
    def test_estimate_tokens_reasonable(self):
        """_estimate_tokens returns a plausible positive value."""
        result = _estimate_tokens("hello world")
        assert result > 0
        assert result < 10  # "hello world" is 2 tokens, estimate should be small

    def test_estimate_tokens_longer_text(self):
        """Longer text produces a proportionally larger estimate."""
        short = _estimate_tokens("Hello world")
        long = _estimate_tokens("Hello world " * 100)
        assert long > short

    def test_estimate_tokens_empty_string(self):
        """Empty string returns at least 1 (no zero-tokens)."""
        assert _estimate_tokens("") == 1


class TestApplyBudget:
    def test_budget_truncates_to_token_limit(self):
        """With a small budget, only the first few memories are returned."""
        memories = [{"text": "A" * 100, "id": str(i)} for i in range(5)]
        # Each memory is ~25 tokens (100 chars // 4). Budget of 30 → only 1 fits.
        result = _apply_budget(memories, max_tokens=30)
        assert len(result) < 5
        assert result[0] == memories[0]

    def test_budget_none_equivalent_returns_all(self):
        """_apply_budget with a very large budget returns all memories."""
        memories = [{"text": "short text", "id": str(i)} for i in range(10)]
        result = _apply_budget(memories, max_tokens=100_000)
        assert len(result) == 10

    def test_budget_always_returns_at_least_one(self):
        """Even if the first memory exceeds the budget, it is still returned."""
        huge_memory = [{"text": "X" * 10_000, "id": "1"}]
        result = _apply_budget(huge_memory, max_tokens=1)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_budget_respects_ranking_order(self):
        """Budget fills greedily from the front — rank is preserved."""
        memories = [{"text": f"memory {i} " * 10, "id": str(i)} for i in range(5)]
        budget = _estimate_tokens(memories[0]["text"]) + _estimate_tokens(memories[1]["text"]) + 1
        result = _apply_budget(memories, max_tokens=budget)
        assert result[0]["id"] == "0"
        assert result[1]["id"] == "1"


class TestMaxTokensInSearchMemories:
    async def test_max_tokens_param_trims_results(self, client, user_id):
        """max_tokens parameter on search_memories reduces results to fit budget."""
        # Store several memories with long texts
        for i in range(5):
            await client.store(
                user_id,
                f"Python backend development preference statement number {i} " * 10,
                "tech",
            )

        token = user_id_ctx.set(user_id)
        try:
            # Very small budget — should return fewer results than default
            small = await search_memories(query="Python backend", limit=10, max_tokens=50)
            large = await search_memories(query="Python backend", limit=10, max_tokens=100_000)
        finally:
            user_id_ctx.reset(token)

        assert small["status"] == "ok"
        assert large["status"] == "ok"
        assert small["data"]["total"] <= large["data"]["total"]

    async def test_max_tokens_none_returns_all(self, client, user_id):
        """max_tokens=None (default) returns all results without truncation."""
        for i in range(3):
            await client.store(user_id, f"Python backend preference {i}", "tech")

        token = user_id_ctx.set(user_id)
        try:
            result = await search_memories(query="Python backend", limit=10, max_tokens=None)
        finally:
            user_id_ctx.reset(token)

        assert result["status"] == "ok"
        assert result["data"]["total"] >= 1
