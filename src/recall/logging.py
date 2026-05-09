"""ToolCallRecord dataclass and LoggingMiddleware for Recall."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import aiosqlite

from recall.db.connection import get_db_path

session_id_ctx: ContextVar[str] = ContextVar("session_id", default="")
token_in_ctx: ContextVar[int] = ContextVar("token_in", default=0)
token_out_ctx: ContextVar[int] = ContextVar("token_out", default=0)

_MODEL_RATES: dict[str, tuple[float, float]] = {
    "claude-sonnet": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-haiku": (0.25 / 1_000_000, 1.25 / 1_000_000),
    "claude-opus": (15.00 / 1_000_000, 75.00 / 1_000_000),
}


@dataclass
class ToolCallRecord:
    id: str
    tool_name: str
    user_id: str
    session_id: str
    inputs_hash: str
    status: Literal["success", "error", "timeout"]
    error_code: str | None
    duration_ms: int
    llm_tokens_in: int
    llm_tokens_out: int
    cost_usd: float
    timestamp: datetime


def estimate_cost(tokens_in: int, tokens_out: int, model: str = "claude-sonnet") -> float:
    """Estimate cost from token counts within ~5% of actual billing."""
    rate_in, rate_out = _MODEL_RATES.get(model, _MODEL_RATES["claude-sonnet"])
    return round(tokens_in * rate_in + tokens_out * rate_out, 6)


def hash_inputs(inputs: Any) -> str:
    """16-char prefix of sha256 — detects duplicates without storing content."""
    serialized = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


async def insert_tool_call_record(record: ToolCallRecord) -> None:
    async with aiosqlite.connect(get_db_path()) as db:
        await db.execute(
            """INSERT INTO tool_call_records
               (id, tool_name, user_id, session_id, inputs_hash, status, error_code,
                duration_ms, llm_tokens_in, llm_tokens_out, cost_usd, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.id,
                record.tool_name,
                record.user_id,
                record.session_id,
                record.inputs_hash,
                record.status,
                record.error_code,
                record.duration_ms,
                record.llm_tokens_in,
                record.llm_tokens_out,
                record.cost_usd,
                record.timestamp.isoformat(),
            ),
        )
        await db.commit()


class LoggingMiddleware:
    """Wraps every tool call with a ToolCallRecord. Logs in finally — success AND error."""

    def __init__(self, user_id_ctx: ContextVar[str]) -> None:
        self._user_id_ctx = user_id_ctx

    async def __call__(self, tool_name: str, inputs: Any, call_next: Any) -> Any:
        start = time.monotonic()
        status: Literal["success", "error", "timeout"] = "success"
        error_code: str | None = None

        try:
            result = await call_next()
            return result
        except TimeoutError:
            status = "timeout"
            error_code = "TOOL_TIMEOUT"
            raise
        except Exception as exc:
            status = "error"
            error_code = type(exc).__name__
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            tokens_in = token_in_ctx.get()
            tokens_out = token_out_ctx.get()
            record = ToolCallRecord(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                user_id=self._user_id_ctx.get(),
                session_id=session_id_ctx.get(),
                inputs_hash=hash_inputs(inputs),
                status=status,
                error_code=error_code,
                duration_ms=duration_ms,
                llm_tokens_in=tokens_in,
                llm_tokens_out=tokens_out,
                cost_usd=estimate_cost(tokens_in, tokens_out),
                timestamp=datetime.utcnow(),
            )
            await insert_tool_call_record(record)
