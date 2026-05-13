"""Data models for Recall — MemoryUnit and MemoryType."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Classification of what a stored memory represents."""

    PREFERENCE = "preference"    # stated likes/dislikes, settings
    FACT = "fact"                # verifiable information about the user
    DECISION = "decision"        # choices the user has made
    PROCEDURE = "procedure"      # steps or workflows the user follows


@dataclass
class MemoryUnit:
    """
    A single extracted memory — the atomic unit of Recall's storage.

    Every field maps directly to a column in the ``memories`` table.
    Fields marked "Memory series" are stored but not computed in v0.1.
    """

    id: str
    namespace: str
    text: str                           # extracted natural-language fact
    type: MemoryType
    topic: str
    importance: float = field(default=0.5)     # 0.0 – 1.0; set by extraction LLM
    confidence: float = field(default=0.8)     # 0.0 – 1.0; set by extraction LLM
    source_session: str = field(default="")    # session ID that produced this memory
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime | None = field(default=None)
    access_count: int = field(default=0)
    decay_score: float | None = field(default=None)      # computed in Memory series
    superseded_by: str | None = field(default=None)      # contradiction resolution in Memory series
    embedding: list[float] | None = field(default=None)  # added when pgvector available

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(
                f"importance must be between 0.0 and 1.0, got {self.importance!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence!r}"
            )
        if isinstance(self.type, str):
            self.type = MemoryType(self.type)

    # ------------------------------------------------------------------ #
    # Representations                                                      #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        accessed = (
            f", last_accessed={self.last_accessed.isoformat()}"
            if self.last_accessed
            else ""
        )
        return (
            f"MemoryUnit(id={self.id!r}, type={self.type.value!r}, "
            f"topic={self.topic!r}, importance={self.importance:.2f}"
            f"{accessed})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON responses or DB insertion."""
        return {
            "id": self.id,
            "namespace": self.namespace,
            "text": self.text,
            "type": self.type.value,
            "topic": self.topic,
            "importance": self.importance,
            "confidence": self.confidence,
            "source_session": self.source_session,
            "created_at": self.created_at.isoformat(),
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
            "access_count": self.access_count,
            "decay_score": self.decay_score,
            "superseded_by": self.superseded_by,
            "embedding": self.embedding,
        }
