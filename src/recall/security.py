"""Security utilities for Recall — token hashing and tool description validation."""

from __future__ import annotations

import hashlib
import re


# Patterns that signal a potentially poisoned tool description.
# Applied at startup — server refuses to start if any tool matches.
_POISONING_PATTERNS: list[tuple[str, str]] = [
    (r"https?://", "URL in description"),
    (r"when.*\bask[s]?\b.*\bcall\b", "conditional behavior instruction"),
    (r"also\s+(call|execute|run|invoke)", "chained call instruction"),
    (r"ignore\s+(previous|above|prior)", "prompt injection classic"),
    (r"send.*\bto\b.*\b(url|endpoint|server|webhook)\b", "exfiltration instruction"),
    (r"do\s+not\s+(tell|mention|reveal)", "secrecy instruction"),
]


def hash_token(token: str) -> str:
    """SHA-256 hash of an API token. Store this, never the token itself."""
    return hashlib.sha256(token.encode()).hexdigest()


def validate_tool_descriptions(tools: dict[str, str]) -> None:
    """Validate that no tool description contains a poisoning pattern.

    Args:
        tools: Mapping of tool_name → description string.

    Raises:
        ValueError: If any description matches a poisoning pattern.
    """
    for tool_name, description in tools.items():
        for pattern, label in _POISONING_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                raise ValueError(
                    f"Security: tool '{tool_name}' description failed validation.\n"
                    f"Pattern matched: {label!r}\n"
                    f"Description: {description!r}\n"
                    "Tool descriptions must be purpose statements only. "
                    "No URLs, conditionals, or behavioral instructions."
                )
