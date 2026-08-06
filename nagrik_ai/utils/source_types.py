"""Source type classification and authority weighting for retrieved documents."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from nagrik_ai.config.config_models import AUTHORITY_BONUS as _DEFAULT_AUTHORITY_BONUS

SOURCE_TYPE_PRIORITY: tuple[str, ...] = (
    "act",
    "rules",
    "notification",
    "circular",
    "faq",
    "user_guide",
    "help",
    "other",
)

SOURCE_TYPE_LABELS: dict[str, str] = {
    "act": "Act",
    "rules": "Rules",
    "notification": "Notification",
    "circular": "Circular",
    "faq": "FAQ",
    "user_guide": "User Guide",
    "help": "Help",
    "other": "Other",
}


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def classify_source_type(metadata: dict[str, Any]) -> str:
    """Infer the source type (act, rules, notification, circular, faq, user_guide, help, other).

    An explicit ``source_type`` metadata field takes priority; when absent, fall back to
    title/URL/domain heuristics.
    """
    explicit = str(metadata.get("source_type", "") or "").strip().lower()
    if explicit and explicit in SOURCE_TYPE_PRIORITY:
        return explicit

    title = str(metadata.get("title", "") or "").lower()
    url = str(metadata.get("citation_url", metadata.get("url", "")) or "").lower()
    domain = str(metadata.get("domain", "") or "").lower()
    source_id = str(metadata.get("source_id", "") or "").lower()
    text = f"{title} {url} {domain} {source_id}"

    if title.startswith("faqs") or title.startswith("faq") or title == "faqs" or "faq" in source_id:
        return "faq"
    if title.startswith("manual") or "manual" in source_id or "manual" in title:
        return "user_guide"
    if _contains_word(text, "notification"):
        return "notification"
    if _contains_word(text, "circular"):
        return "circular"
    if _contains_word(text, "rules") or title.startswith("rule"):
        return "rules"
    if _contains_word(text, "act"):
        return "act"
    if any(marker in domain for marker in ("userguide", "tutorial")) or "cbt" in domain:
        return "user_guide"
    if "help" in domain:
        return "help"
    return "other"


def authority_rank(source_type: str) -> int:
    """Lower is more authoritative. Unknown types rank last."""
    if source_type in SOURCE_TYPE_PRIORITY:
        return SOURCE_TYPE_PRIORITY.index(source_type)
    return len(SOURCE_TYPE_PRIORITY) - 1


def authority_bonus(source_type: str, bonuses: Mapping[str, float] | None = None) -> float:
    """Additive score boost applied during retrieval selection."""
    bonuses = bonuses if bonuses is not None else _DEFAULT_AUTHORITY_BONUS
    return bonuses.get(source_type, 0.0)


def source_type_label(source_type: str) -> str:
    """Human-readable label for prompt context and display."""
    return SOURCE_TYPE_LABELS.get(source_type, source_type)
