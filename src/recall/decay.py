"""Memory decay scoring — scheduled background job for Recall.

Periodically updates decay_score on active memories using exponential decay
with access-count protection. The decay_score column is then used by
_hybrid_search as a multiplier on the importance component.

Decay formula:
    raw_decay    = exp(-λ · age_since_last_access_days)
    access_boost = min(1.0, log(1+ac) / log(1+BOOST_CAP))
    decay_score  = raw_decay + (1 - raw_decay) * access_boost

Env vars:
    RECALL_DECAY_LAMBDA         float, default 0.02  (~35-day half-life)
    RECALL_DECAY_BOOST_CAP      int,   default 10    (accesses for full protection)
    RECALL_DECAY_JOB_INTERVAL   int,   default 3600  (seconds between runs)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import datetime, timezone

from recall.db.backend import get_backend

logger = logging.getLogger(__name__)

_DEFAULT_LAMBDA = 0.02
_DEFAULT_BOOST_CAP = 10
_DEFAULT_INTERVAL = 3600


class DecayWorker:
    """Scheduled in-process asyncio task that updates decay_score on active memories."""

    def __init__(self) -> None:
        self._lambda = float(os.environ.get("RECALL_DECAY_LAMBDA", _DEFAULT_LAMBDA))
        self._boost_cap = float(os.environ.get("RECALL_DECAY_BOOST_CAP", _DEFAULT_BOOST_CAP))
        self._interval = int(os.environ.get("RECALL_DECAY_JOB_INTERVAL", _DEFAULT_INTERVAL))
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run(), name="recall-decay-worker")
        logger.info("decay_worker_started", extra={"interval_s": self._interval})

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("decay_worker_stopped")

    async def run_once(self) -> int:
        """Apply decay to all active memories. Returns count of memories updated."""
        now = datetime.now(timezone.utc)
        updated = 0

        async with get_backend() as db:
            rows = await db.fetch_all(
                "SELECT id, created_at, last_accessed, access_count "
                "FROM memories WHERE valid_until IS NULL"
            )
            for row_id, created_at, last_accessed, access_count in rows:
                ref_ts = last_accessed or created_at
                try:
                    ref_dt = _parse_ts(ref_ts)
                except (ValueError, TypeError):
                    continue

                age_days = max(0.0, (now - ref_dt).total_seconds() / 86400)
                ac = access_count or 0

                raw_decay = math.exp(-self._lambda * age_days)
                if self._boost_cap > 0:
                    access_boost = min(1.0, math.log(1 + ac) / math.log(1 + self._boost_cap))
                else:
                    access_boost = 0.0
                decay_score = raw_decay + (1 - raw_decay) * access_boost

                await db.execute(
                    "UPDATE memories SET decay_score = ? WHERE id = ? AND valid_until IS NULL",
                    (round(decay_score, 6), row_id),
                )
                updated += 1

            await db.commit()

        logger.info("decay_run_complete", extra={"memories_updated": updated})
        return updated

    async def _run(self) -> None:
        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("decay_run_failed", extra={"error": str(exc)}, exc_info=True)
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
