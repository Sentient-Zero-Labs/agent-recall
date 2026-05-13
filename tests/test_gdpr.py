"""GDPR erasure tests — delete_namespace_data MCP tool.

Uses the client/namespace fixtures from conftest.py. Tests call delete_namespace_data
directly (not via HTTP) to verify row-level deletion in the DB.
"""

from __future__ import annotations

import json
import uuid

import aiosqlite
import pytest

from recall.db.connection import get_db_path
from recall.security import hash_token
from recall.server import _validate_token, delete_namespace_data, namespace_ctx


class TestGDPRErasure:
    async def test_wrong_confirmation_returns_error(self, client, namespace):
        """Any string other than 'DELETE MY DATA' must be rejected."""
        token = namespace_ctx.set(namespace)
        try:
            result = await delete_namespace_data("yes")
        finally:
            namespace_ctx.reset(token)

        assert result["status"] == "error"
        assert result["code"] == "CONFIRM_REQUIRED"

    async def test_deletes_all_memories(self, client, namespace):
        """After erasure, no memories remain for the user."""
        await client.store(namespace, "I prefer Python for backend work", "tech")
        await client.store(namespace, "I use VS Code as my main editor", "tools")
        await client.store(namespace, "I enjoy hiking on weekends", "personal")

        token = namespace_ctx.set(namespace)
        try:
            result = await delete_namespace_data("DELETE MY DATA")
        finally:
            namespace_ctx.reset(token)

        assert result["status"] == "ok"

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM memories WHERE namespace = ?", (namespace,)
            )
        assert rows == [], "All memories must be deleted for the user"

    async def test_revokes_tokens(self, client, namespace):
        """After erasure, previously valid tokens are revoked."""
        raw_token = f"test-gdpr-token-{uuid.uuid4()}"
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "INSERT INTO api_tokens (id, token_hash, namespace, revoked) VALUES (?,?,?,0)",
                (str(uuid.uuid4()), hash_token(raw_token), namespace),
            )
            await db.commit()

        # Token works before erasure
        assert await _validate_token(raw_token) == namespace

        token = namespace_ctx.set(namespace)
        try:
            result = await delete_namespace_data("DELETE MY DATA")
        finally:
            namespace_ctx.reset(token)

        assert result["status"] == "ok"
        assert result["data"]["tokens_revoked"] >= 1

        # Token must no longer validate after erasure
        assert await _validate_token(raw_token) is None

    async def test_deletes_operations(self, client, namespace):
        """After erasure, pending/complete operations are removed for the user."""
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "INSERT INTO operations (id, idempotency_key, namespace, status) VALUES (?,?,?,?)",
                (str(uuid.uuid4()), f"key-{uuid.uuid4()}", namespace, "queued"),
            )
            await db.commit()

        token = namespace_ctx.set(namespace)
        try:
            await delete_namespace_data("DELETE MY DATA")
        finally:
            namespace_ctx.reset(token)

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM operations WHERE namespace = ?", (namespace,)
            )
        assert rows == [], "All operations must be deleted for the user"

    async def test_deletes_a2a_tasks(self, client, namespace):
        """After erasure, A2A tasks stored in the DB are removed for the user."""
        async with aiosqlite.connect(get_db_path()) as db:
            await db.execute(
                "INSERT INTO a2a_tasks (id, namespace, status, input) VALUES (?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    namespace,
                    "completed",
                    json.dumps({"text": "test", "topic": "general"}),
                ),
            )
            await db.commit()

        token = namespace_ctx.set(namespace)
        try:
            await delete_namespace_data("DELETE MY DATA")
        finally:
            namespace_ctx.reset(token)

        async with aiosqlite.connect(get_db_path()) as db:
            rows = await db.execute_fetchall(
                "SELECT id FROM a2a_tasks WHERE namespace = ?", (namespace,)
            )
        assert rows == [], "All A2A tasks must be deleted for the user"
