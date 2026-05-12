"""Pytest fixtures for Recall tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from recall.client import MemoryClient
from recall.db.connection import set_db_path

# Load .env from the project root so ANTHROPIC_API_KEY is available to live tests.
# override=False means shell exports take precedence over the file.
load_dotenv(Path(__file__).parent.parent / ".env", override=False)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Path to a temporary SQLite database, cleaned up after each test."""
    db_path = tmp_path / "test_recall.db"
    set_db_path(db_path)
    return db_path


@pytest_asyncio.fixture
async def client(tmp_db: Path) -> MemoryClient:
    """Initialized MemoryClient backed by a temporary database."""
    c = MemoryClient(db_path=str(tmp_db))
    await c.initialize()
    return c


@pytest.fixture
def user_id() -> str:
    return "test-user-abc123"


@pytest.fixture
def session_id() -> str:
    return "test-session-xyz789"
