"""Unit tests for MemoryUnit and MemoryType."""

from __future__ import annotations

import pytest
from datetime import datetime

from recall.models import MemoryType, MemoryUnit


def make_memory(**kwargs) -> MemoryUnit:
    defaults = {
        "id": "test-id-001",
        "user_id": "user-abc",
        "text": "User prefers dark mode",
        "type": MemoryType.PREFERENCE,
        "topic": "ui_preferences",
    }
    defaults.update(kwargs)
    return MemoryUnit(**defaults)


class TestMemoryType:
    def test_values(self):
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.PROCEDURE.value == "procedure"

    def test_from_string(self):
        assert MemoryType("preference") == MemoryType.PREFERENCE
        assert MemoryType("fact") == MemoryType.FACT


class TestMemoryUnit:
    def test_creation_defaults(self):
        m = make_memory()
        assert m.importance == 0.5
        assert m.confidence == 0.8
        assert m.access_count == 0
        assert m.last_accessed is None
        assert m.embedding is None
        assert isinstance(m.created_at, datetime)

    def test_string_type_coercion(self):
        m = make_memory(type="preference")
        assert m.type is MemoryType.PREFERENCE

    def test_importance_bounds(self):
        with pytest.raises(ValueError, match="importance"):
            make_memory(importance=1.5)
        with pytest.raises(ValueError, match="importance"):
            make_memory(importance=-0.1)

    def test_confidence_bounds(self):
        with pytest.raises(ValueError, match="confidence"):
            make_memory(confidence=1.1)
        with pytest.raises(ValueError, match="confidence"):
            make_memory(confidence=-0.01)

    def test_boundary_values_valid(self):
        m = make_memory(importance=0.0, confidence=1.0)
        assert m.importance == 0.0
        assert m.confidence == 1.0

    def test_str_representation(self):
        m = make_memory()
        s = str(m)
        assert "test-id-001" in s
        assert "preference" in s
        assert "ui_preferences" in s
        assert "0.50" in s

    def test_to_dict_keys(self):
        m = make_memory()
        d = m.to_dict()
        expected_keys = {
            "id", "user_id", "text", "type", "topic", "importance", "confidence",
            "source_session", "created_at", "last_accessed", "access_count",
            "decay_score", "superseded_by", "embedding",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_type_is_string(self):
        m = make_memory()
        d = m.to_dict()
        assert d["type"] == "preference"

    def test_to_dict_datetime_is_iso(self):
        m = make_memory()
        d = m.to_dict()
        assert isinstance(d["created_at"], str)
        datetime.fromisoformat(d["created_at"])

    def test_to_dict_last_accessed_none(self):
        m = make_memory()
        assert m.to_dict()["last_accessed"] is None

    def test_to_dict_last_accessed_set(self):
        now = datetime.utcnow()
        m = make_memory(last_accessed=now)
        d = m.to_dict()
        assert d["last_accessed"] == now.isoformat()
