"""Integration tests for MemoryClient — requires a real SQLite database."""

from __future__ import annotations

import pytest

from recall.client import MemoryClient
from recall.models import MemoryType, MemoryUnit


class TestMemoryClientStore:
    async def test_store_returns_memory_unit(self, client: MemoryClient, user_id: str):
        m = await client.store(user_id=user_id, text="Prefers Python over JS", topic="tech")
        assert isinstance(m, MemoryUnit)
        assert m.user_id == user_id
        assert m.text == "Prefers Python over JS"
        assert m.topic == "tech"

    async def test_store_persists(self, client: MemoryClient, user_id: str):
        m = await client.store(user_id=user_id, text="Uses vim", topic="tools")
        fetched = await client.get(m.id, user_id)
        assert fetched is not None
        assert fetched.text == "Uses vim"

    async def test_store_with_type(self, client: MemoryClient, user_id: str):
        m = await client.store(
            user_id=user_id,
            text="Always deploys on Fridays",
            topic="ops",
            memory_type=MemoryType.DECISION,
        )
        assert m.type == MemoryType.DECISION


class TestMemoryClientGet:
    async def test_get_nonexistent(self, client: MemoryClient, user_id: str):
        result = await client.get("nonexistent-id", user_id)
        assert result is None

    async def test_get_wrong_user(self, client: MemoryClient, user_id: str):
        m = await client.store(user_id=user_id, text="Private", topic="test")
        result = await client.get(m.id, "different-user")
        assert result is None


class TestMemoryClientSearch:
    async def test_search_finds_match(self, client: MemoryClient, user_id: str):
        await client.store(user_id=user_id, text="Loves hiking on weekends", topic="hobbies")
        await client.store(user_id=user_id, text="Prefers coffee over tea", topic="food")

        results = await client.search(user_id, "hiking")
        assert any("hiking" in m.text for m in results)

    async def test_search_no_match(self, client: MemoryClient, user_id: str):
        await client.store(user_id=user_id, text="Likes jazz music", topic="music")
        results = await client.search(user_id, "skydiving")
        assert results == []


class TestMemoryClientList:
    async def test_list_returns_all(self, client: MemoryClient, user_id: str):
        for i in range(5):
            await client.store(user_id=user_id, text=f"Memory {i}", topic="test")
        memories, total = await client.list_all(user_id)
        assert total == 5
        assert len(memories) == 5

    async def test_list_pagination(self, client: MemoryClient, user_id: str):
        for i in range(5):
            await client.store(user_id=user_id, text=f"Memory {i}", topic="test")
        page1, total = await client.list_all(user_id, limit=3, offset=0)
        page2, _ = await client.list_all(user_id, limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2
        assert total == 5

    async def test_list_isolates_by_user(self, client: MemoryClient, user_id: str):
        await client.store(user_id=user_id, text="User A memory", topic="test")
        await client.store(user_id="other-user", text="User B memory", topic="test")
        memories, total = await client.list_all(user_id)
        assert total == 1
        assert memories[0].user_id == user_id


class TestMemoryClientDelete:
    async def test_delete_returns_true(self, client: MemoryClient, user_id: str):
        m = await client.store(user_id=user_id, text="To be deleted", topic="test")
        deleted = await client.delete(m.id, user_id)
        assert deleted is True

    async def test_delete_removes_from_db(self, client: MemoryClient, user_id: str):
        m = await client.store(user_id=user_id, text="To be deleted", topic="test")
        await client.delete(m.id, user_id)
        fetched = await client.get(m.id, user_id)
        assert fetched is None

    async def test_delete_nonexistent_returns_false(self, client: MemoryClient, user_id: str):
        deleted = await client.delete("nonexistent-id", user_id)
        assert deleted is False

    async def test_delete_all(self, client: MemoryClient, user_id: str):
        for i in range(3):
            await client.store(user_id=user_id, text=f"Memory {i}", topic="test")
        count = await client.delete_all(user_id)
        assert count == 3
        _, total = await client.list_all(user_id)
        assert total == 0
