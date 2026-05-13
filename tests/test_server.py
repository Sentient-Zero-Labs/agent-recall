"""HTTP smoke tests for the Recall MCP server — auth, all 5 tools, error paths."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
import pytest
import pytest_asyncio

from recall.db.connection import init_db, set_db_path
from recall.security import hash_token
from recall.server import create_app

TEST_TOKEN = "smoke-test-token-abc123xyz"
TEST_USER = "smoke-test-user-001"
_MCP_PROTO = "2024-11-05"


@pytest_asyncio.fixture
async def server(tmp_path, monkeypatch):
    """Isolated server: fresh DB, seeded auth token, mocked extraction worker."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-placeholder")
    db_path = tmp_path / "smoke.db"
    set_db_path(db_path)
    await init_db()

    # Seed test token before any request hits the auth middleware
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO api_tokens (id, token_hash, namespace, revoked) VALUES (?,?,?,0)",
            (str(uuid.uuid4()), hash_token(TEST_TOKEN), TEST_USER),
        )
        await db.commit()

    mock_worker = AsyncMock()
    # Patch ExtractionWorker so startup() gets our mock (no Anthropic calls in tests)
    with patch("recall.server.ExtractionWorker", return_value=mock_worker):
        app = create_app()

        # Manually drive the ASGI lifespan — httpx ASGITransport doesn't do this automatically
        startup_done = asyncio.Event()
        shutdown_signal = asyncio.Event()

        async def _lifespan_driver() -> None:
            scope: dict = {"type": "lifespan", "asgi": {"spec_version": "2.0"}}
            sent_startup = False

            async def receive() -> dict:
                nonlocal sent_startup
                if not sent_startup:
                    sent_startup = True
                    return {"type": "lifespan.startup"}
                await shutdown_signal.wait()
                return {"type": "lifespan.shutdown"}

            async def send(message: dict) -> None:
                if message["type"] in ("lifespan.startup.complete", "lifespan.startup.failed"):
                    startup_done.set()

            await app(scope, receive, send)

        lifespan_task = asyncio.create_task(_lifespan_driver())
        await startup_done.wait()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        shutdown_signal.set()
        try:
            await asyncio.wait_for(lifespan_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            lifespan_task.cancel()
            try:
                await lifespan_task
            except asyncio.CancelledError:
                pass


# ── Protocol helpers ──────────────────────────────────────────────────────────


def _auth(token: str = TEST_TOKEN) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }


def _parse_mcp_response(resp: httpx.Response) -> dict:
    """Parse a FastMCP response — handles both plain JSON and SSE format."""
    text = resp.text
    # SSE: lines look like "data: {...json...}"
    if "data: " in text:
        for line in text.split("\r\n"):
            if line.startswith("data: "):
                return json.loads(line[6:])
    return resp.json()


async def _initialize(client: httpx.AsyncClient, token: str = TEST_TOKEN) -> str | None:
    """MCP initialize handshake; returns session ID if the server provides one."""
    resp = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": _MCP_PROTO,
                "capabilities": {},
                "clientInfo": {"name": "smoke-test", "version": "0.1"},
            },
        },
        headers=_auth(token),
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    sid = resp.headers.get("mcp-session-id")
    if not sid:
        body = _parse_mcp_response(resp)
        sid = body.get("result", {}).get("sessionId")
    return sid


async def _tool(
    client: httpx.AsyncClient,
    tool: str,
    args: dict[str, Any],
    session_id: str | None = None,
    token: str = TEST_TOKEN,
) -> dict:
    """Call an MCP tool and return the parsed tool output dict."""
    headers = _auth(token)
    if session_id:
        headers["mcp-session-id"] = session_id
    resp = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        },
        headers=headers,
    )
    assert resp.status_code == 200, f"tool call failed ({tool}): {resp.text}"
    body = _parse_mcp_response(resp)
    # FastMCP serializes dict return values as JSON text in content[0].text
    text = body["result"]["content"][0]["text"]
    return json.loads(text)


# ── Auth ──────────────────────────────────────────────────────────────────────


class TestAuth:
    async def test_no_token_returns_401(self, server):
        resp = await server.post(
            "/mcp", json={}, headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 401

    async def test_bad_token_returns_401(self, server):
        resp = await server.post(
            "/mcp",
            json={},
            headers={
                "Authorization": "Bearer not-a-real-token",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 401

    async def test_valid_token_passes_auth(self, server):
        resp = await server.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "auth-check",
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTO,
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0.1"},
                },
            },
            headers=_auth(),
        )
        assert resp.status_code == 200


# ── store_memory ──────────────────────────────────────────────────────────────


class TestStoreMemory:
    async def test_store_returns_queued(self, server):
        sid = await _initialize(server)
        result = await _tool(
            server,
            "store_memory",
            {
                "text": "User prefers dark mode",
                "topic": "prefs",
                "idempotency_key": "k-dark-mode-001",
            },
            session_id=sid,
        )
        assert result["status"] == "ok"
        assert result["data"]["queued"] is True

    async def test_idempotent_key_returns_cached(self, server):
        sid = await _initialize(server)
        args = {
            "text": "User uses vim",
            "topic": "tools",
            "idempotency_key": "k-vim-pref-002",
        }
        await _tool(server, "store_memory", args, session_id=sid)
        result = await _tool(server, "store_memory", args, session_id=sid)
        assert result["data"]["cached"] is True


# ── search_memories ───────────────────────────────────────────────────────────


class TestSearchMemories:
    async def test_search_empty_db_returns_empty(self, server):
        sid = await _initialize(server)
        result = await _tool(server, "search_memories", {"query": "python"}, session_id=sid)
        assert result["status"] == "ok"
        assert result["data"]["results"] == []

    async def test_invalid_recency_weight_returns_error(self, server):
        sid = await _initialize(server)
        result = await _tool(
            server,
            "search_memories",
            {"query": "python", "recency_weight": 2.0},
            session_id=sid,
        )
        assert result["status"] == "error"
        assert result["code"] == "INVALID_PARAM"


# ── inspect_memories ──────────────────────────────────────────────────────────


class TestInspectMemories:
    async def test_inspect_empty_db(self, server):
        sid = await _initialize(server)
        result = await _tool(server, "inspect_memories", {}, session_id=sid)
        assert result["status"] == "ok"
        assert result["data"]["memories"] == []
        assert "has_more" in result["data"]
        assert "total" in result["data"]
        assert result["data"]["total"] == 0


# ── delete_memory ─────────────────────────────────────────────────────────────


class TestDeleteMemory:
    async def test_delete_nonexistent_returns_error(self, server):
        sid = await _initialize(server)
        result = await _tool(
            server,
            "delete_memory",
            {"memory_id": "does-not-exist-abc-xyz"},
            session_id=sid,
        )
        assert result["status"] == "error"
        assert result["code"] == "MEMORY_NOT_FOUND"


# ── get_memory_stats ──────────────────────────────────────────────────────────


class TestGetMemoryStats:
    async def test_stats_shape_on_empty_db(self, server):
        sid = await _initialize(server)
        result = await _tool(server, "get_memory_stats", {}, session_id=sid)
        assert result["status"] == "ok"
        assert "by_type" in result["data"]
        assert "total" in result["data"]
        assert "pending_extractions" in result["data"]
        assert result["data"]["total"] == 0
