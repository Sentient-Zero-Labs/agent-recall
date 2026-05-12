"""GDPR erasure tests — delete_user_data MCP tool.

Uses the client/user_id fixtures from conftest.py. Tests call delete_user_data
directly (not via HTTP) to verify row-level deletion in the DB.
"""

from __future__ import annotations

import json
import uuid

import aiosqlite
import pytest

from recall.db.connection import get_db_path
from recall.security import hash_token
from recall.server import _validate_token, delete_user_data, user_id_ctx


class TestGDPRErasure:
    async def test_wrong_confirmation_returns_error(self, client, user_id):
        """Any string other than 'DELETE MY DATA' must be rejected."""
        token = user_id_ctx.set(user_id)
        try:
            result = await delete_user_data("yes")
        finally:
            user_id_ctx.reset(token)

        assert result["status"] == "error"
        assert result["code"] == "CONFIRM_REQUIRED"

    async def test_deletes_all_memories(self, client, user_id):
        """After erasure, no memories remain for the user."""
        await client.store(user_id, "I prefer Python for backend work", "tech")
        await client.store(user_id, "I use VS Code as my main editor", "tools")
        await client.store(user_id, "I enjoy hiking on weekends", "personal")

        token = user_id_ctx.set(user_id)
        try:
            result = await delete_user_data("DELETE MY DATA")
        finally:
            user_id_ctx.reset(token)

        assert result["status"] == "ok"

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM memories WHERE user_id = ?", (user_id,)
            )
        assert rows == [], "All memories must be deleted for the user"

    async def test_revokes_tokens(self, client, user_id):
        """After erasure, previously valid tokens are revoked."""
        raw_token = f"test-gdpr-token-{uuid.uuid4()}"
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "INSERT INTO api_tokens (id, token_hash, user_id, revoked) VALUES (?,?,?,0)",
                (str(uuid.uuid4()), hash_token(raw_token), user_id),
            )
            await db.commit()

        # Token works before erasure
        assert await _validate_token(raw_token) == user_id

        token = user_id_ctx.set(user_id)
        try:
            result = await delete_user_data("DELETE MY DATA")
        finally:
            user_id_ctx.reset(token)

        assert result["status"] == "ok"
        assert result["data"]["tokens_revoked"] >= 1

        # Token must no longer validate after erasure
        assert await _validate_token(raw_token) is None

    async def test_deletes_operations(self, client, user_id):
        """After erasure, pending/complete operations are removed for the user."""
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "INSERT INTO operations (id, idempotency_key, user_id, status) VALUES (?,?,?,?)",
                (str(uuid.uuid4()), f"key-{uuid.uuid4()}", user_id, "queued"),
            )
            await db.commit()

        token = user_id_ctx.set(user_id)
        try:
            await delete_user_data("DELETE MY DATA")
        finally:
            user_id_ctx.reset(token)

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM operations WHERE user_id = ?", (user_id,)
            )
        assert rows == [], "All operations must be deleted for the user"

    async def test_deletes_a2a_tasks(self, client, user_id):
        """After erasure, A2A tasks stored in the DB are removed for the user."""
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "INSERT INTO a2a_tasks (id, user_id, status, input) VALUES (?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    user_id,
                    "completed",
                    json.dumps({"text": "test", "topic": "general"}),
                ),
            )
            await db.commit()

        token = user_id_ctx.set(user_id)
        try:
            await delete_user_data("DELETE MY DATA")
        finally:
            user_id_ctx.reset(token)

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM a2a_tasks WHERE user_id = ?", (user_id,)
            )
        assert rows == [], "All A2A tasks must be deleted for the user"
