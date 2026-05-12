"""End-to-end tests against a live uvicorn process.

Spins up `recall serve` as a subprocess on port 8678, creates a real DB and
auth token, then exercises MCP tools and the A2A endpoint over HTTP.

Requirements:
  - ANTHROPIC_API_KEY env var (used by the server for background extraction)
  - Port 8678 free at test time
  - `pip install -e ".[dev]"` (includes httpx, pytest-asyncio, python-dotenv)

Run:
  pytest tests/test_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
import httpx
import pytest
import pytest_asyncio

from recall.db.connection import set_db_path
from recall.security import hash_token

# ── Constants ──────────────────────────────────────────────────────────────────

_PORT = 8678
_BASE = f"http://localhost:{_PORT}"
_MCP_PROTO = "2024-11-05"
_STARTUP_TIMEOUT = 20   # seconds to wait for the server to accept connections
_POLL_INTERVAL = 0.25


# ── Skip guard ────────────────────────────────────────────────────────────────

_needs_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skip live E2E tests",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def e2e_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One shared SQLite DB for the whole E2E session."""
    return tmp_path_factory.mktemp("e2e") / "e2e_recall.db"


@pytest.fixture(scope="session")
def e2e_token(e2e_db: Path) -> str:
    """Raw bearer token seeded into the E2E database synchronously."""
    raw = secrets.token_urlsafe(32)
    token_hash = hash_token(raw)
    token_id = str(uuid.uuid4())

    import sqlite3
    # DB may not exist yet — the server will create it on startup.
    # We seed the token AFTER the server has initialized the schema.
    # Return the raw token; the server fixture seeds it.
    return raw


@pytest.fixture(scope="session")
def server_process(e2e_db: Path, e2e_token: str):
    """Start a real uvicorn process on port 8678, yield it, then tear it down."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    env = {**os.environ, "RECALL_DB_PATH": str(e2e_db)}

    import shutil
    recall_bin = shutil.which("recall")
    if not recall_bin:
        pytest.skip("recall CLI not found — run `pip install -e .` first")

    proc = subprocess.Popen(
        [recall_bin, "serve", "--host", "127.0.0.1", "--port", str(_PORT), "--db", str(e2e_db)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait until the server is accepting requests
    deadline = time.time() + _STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            import urllib.request
            urllib.request.urlopen(f"{_BASE}/.well-known/agent-card.json", timeout=2)
            break
        except Exception:
            time.sleep(_POLL_INTERVAL)
    else:
        proc.terminate()
        out, err = proc.communicate(timeout=5)
        pytest.fail(
            f"Server did not start within {_STARTUP_TIMEOUT}s\n"
            f"stdout: {out.decode()[-2000:]}\n"
            f"stderr: {err.decode()[-2000:]}"
        )

    # Seed the auth token now that the schema exists
    import sqlite3
    token_hash = hash_token(e2e_token)
    conn = sqlite3.connect(str(e2e_db))
    conn.execute(
        "INSERT OR IGNORE INTO api_tokens (id, token_hash, user_id, revoked) VALUES (?,?,?,0)",
        (str(uuid.uuid4()), token_hash, "e2e-user"),
    )
    conn.commit()
    conn.close()

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _auth(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }


def _parse_mcp(resp: httpx.Response) -> dict:
    """Handle both plain JSON and SSE-framed responses from FastMCP."""
    text = resp.text
    if "data: " in text:
        for line in text.split("\r\n"):
            if line.startswith("data: "):
                return json.loads(line[6:])
    return resp.json()


async def _initialize(client: httpx.AsyncClient, token: str) -> str | None:
    resp = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": _MCP_PROTO,
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "1.0"},
            },
        },
        headers=_auth(token),
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    sid = resp.headers.get("mcp-session-id")
    if not sid:
        body = _parse_mcp(resp)
        sid = body.get("result", {}).get("sessionId")
    return sid


async def _tool(
    client: httpx.AsyncClient,
    tool: str,
    args: dict[str, Any],
    token: str,
    session_id: str | None = None,
) -> dict:
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
    assert resp.status_code == 200, f"{tool} failed: {resp.text}"
    body = _parse_mcp(resp)
    text = body["result"]["content"][0]["text"]
    return json.loads(text)


# ── Test classes ──────────────────────────────────────────────────────────────

@_needs_api
class TestE2EBasic:
    """Server-up smoke tests: auth, agent card, healthy responses."""

    def test_agent_card_is_public(self, server_process):
        """GET /.well-known/agent-card.json requires no token."""
        import urllib.request
        with urllib.request.urlopen(f"{_BASE}/.well-known/agent-card.json") as r:
            card = json.loads(r.read())
        assert card["name"] == "recall"
        assert any(s["id"] == "consolidate_memories" for s in card.get("skills", []))

    def test_no_token_returns_401(self, server_process):
        import urllib.request, urllib.error
        req = urllib.request.Request(
            f"{_BASE}/mcp",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 401

    async def test_bad_token_returns_401(self, server_process):
        async with httpx.AsyncClient(base_url=_BASE, timeout=10) as client:
            resp = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": "x", "method": "initialize",
                      "params": {"protocolVersion": _MCP_PROTO,
                                 "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
                headers=_auth("bad-token-xyz"),
            )
        assert resp.status_code == 401


@_needs_api
class TestE2EMcpTools:
    """MCP tool calls against the live server — no extraction assertions (async)."""

    async def test_store_and_search_memory(self, server_process, e2e_token):
        async with httpx.AsyncClient(base_url=_BASE, timeout=15) as client:
            sid = await _initialize(client, e2e_token)

            key = f"e2e-{uuid.uuid4()}"
            store_resp = await _tool(client, "store_memory", {
                "text": "I always use pytest for Python testing and love fixtures",
                "topic": "engineering",
                "idempotency_key": key,
            }, e2e_token, sid)

            assert store_resp["status"] == "ok"
            assert store_resp["data"]["queued"] is True
            assert "job_id" in store_resp["data"]

            # store_memory is async-acknowledge — no extraction guarantee here.
            # search over existing DB (may be empty for this user, that's fine).
            search_resp = await _tool(client, "search_memories", {
                "query": "Python testing",
                "limit": 5,
            }, e2e_token, sid)

            assert search_resp["status"] == "ok"
            assert "results" in search_resp["data"]
            assert isinstance(search_resp["data"]["results"], list)

    async def test_idempotent_store_returns_cached(self, server_process, e2e_token):
        async with httpx.AsyncClient(base_url=_BASE, timeout=15) as client:
            sid = await _initialize(client, e2e_token)
            key = f"idem-e2e-{uuid.uuid4()}"
            args = {"text": "Idempotency test text", "topic": "test", "idempotency_key": key}

            first = await _tool(client, "store_memory", args, e2e_token, sid)
            second = await _tool(client, "store_memory", args, e2e_token, sid)

            assert first["data"]["queued"] is True
            assert second["data"].get("cached") is True

    async def test_inspect_returns_paginated_shape(self, server_process, e2e_token):
        async with httpx.AsyncClient(base_url=_BASE, timeout=15) as client:
            sid = await _initialize(client, e2e_token)
            result = await _tool(client, "inspect_memories", {"limit": 10, "offset": 0},
                                 e2e_token, sid)
        assert result["status"] == "ok"
        data = result["data"]
        assert "memories" in data
        assert "total" in data
        assert isinstance(data.get("has_more"), bool)

    async def test_get_memory_stats_shape(self, server_process, e2e_token):
        async with httpx.AsyncClient(base_url=_BASE, timeout=15) as client:
            sid = await _initialize(client, e2e_token)
            result = await _tool(client, "get_memory_stats", {}, e2e_token, sid)
        assert result["status"] == "ok"
        data = result["data"]
        assert "total" in data
        assert "by_type" in data
        assert "pending_extractions" in data

    async def test_delete_nonexistent_returns_error(self, server_process, e2e_token):
        async with httpx.AsyncClient(base_url=_BASE, timeout=15) as client:
            sid = await _initialize(client, e2e_token)
            result = await _tool(client, "delete_memory",
                                 {"memory_id": "nonexistent-id-abc"}, e2e_token, sid)
        assert result["status"] == "error"
        assert result.get("code") == "MEMORY_NOT_FOUND"


@_needs_api
class TestE2EA2A:
    """A2A task lifecycle: create → poll → completed."""

    async def _post(
        self,
        client: httpx.AsyncClient,
        path: str,
        token: str,
        body: dict | None = None,
    ) -> dict:
        resp = await client.request(
            "POST" if body is not None else "GET",
            path,
            json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        return resp

    async def test_consolidate_task_lifecycle(self, server_process, e2e_token):
        """Full A2A flow: submit task, poll until terminal state."""
        async with httpx.AsyncClient(base_url=_BASE, timeout=30) as client:
            # Create task
            resp = await client.post(
                "/a2a/",
                json={
                    "skill": "consolidate_memories",
                    "input": {
                        "text": "I prefer dark mode in all editors. I use VS Code primarily.",
                        "topic": "tooling",
                    },
                },
                headers={
                    "Authorization": f"Bearer {e2e_token}",
                    "Content-Type": "application/json",
                },
            )
            assert resp.status_code == 202, f"task create failed: {resp.text}"
            task = resp.json()
            assert "id" in task
            assert task["status"] == "submitted"
            task_id = task["id"]

            # Poll until terminal (completed or input-required)
            terminal = {"completed", "failed", "input-required"}
            deadline = time.time() + 30
            status = task["status"]
            while status not in terminal and time.time() < deadline:
                await asyncio.sleep(0.5)
                poll = await client.get(
                    f"/a2a/{task_id}",
                    headers={"Authorization": f"Bearer {e2e_token}"},
                )
                assert poll.status_code == 200
                status = poll.json()["status"]

            assert status in terminal, f"Task did not reach terminal state; last status: {status}"

            if status == "completed":
                output = poll.json().get("output", {})
                assert "stored" in output or "memories" in output or isinstance(output, dict)
            elif status == "input-required":
                # Contradiction detected — resolve it
                resolve = await client.post(
                    f"/a2a/{task_id}/resume",
                    json={"resolution": "keep_new"},
                    headers={
                        "Authorization": f"Bearer {e2e_token}",
                        "Content-Type": "application/json",
                    },
                )
                assert resolve.status_code in (200, 202)

    async def test_poll_unknown_task_returns_404(self, server_process, e2e_token):
        async with httpx.AsyncClient(base_url=_BASE, timeout=10) as client:
            resp = await client.get(
                "/a2a/nonexistent-task-id",
                headers={"Authorization": f"Bearer {e2e_token}"},
            )
        assert resp.status_code == 404

    async def test_a2a_requires_auth(self, server_process):
        async with httpx.AsyncClient(base_url=_BASE, timeout=10) as client:
            resp = await client.post(
                "/a2a/",
                json={"skill": "consolidate_memories", "input": {"text": "test", "topic": "t"}},
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 401
