"""Persistent business facts with explicit, proposal-specific write consent."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field


class BusinessProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    industry: str | None = None
    location: str | None = None
    company_size: str | None = None
    investment: float | None = Field(default=None, ge=0)
    turnover: float | None = Field(default=None, ge=0)
    registrations: list[str] = Field(default_factory=list)
    updated_at: str | None = None
    source: Literal["user_provided", "inferred"] | None = None


class BusinessProfileService:
    """Each operation owns its connection; reads never create user records."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS business_profiles (user_id TEXT PRIMARY KEY, profile TEXT NOT NULL)"
            )

    @staticmethod
    def _validate_user(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("A non-empty user_id is required")

    def get_profile(self, user_id: str) -> BusinessProfile | None:
        self._validate_user(user_id)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            row = connection.execute("SELECT profile FROM business_profiles WHERE user_id = ?", (user_id,)).fetchone()
        return BusinessProfile.model_validate_json(row[0]) if row else None

    def propose_update(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Validate a partial update without persisting it; omitted fields stay untouched."""
        if "updated_at" in updates:
            raise ValueError("updated_at is managed by the service")
        proposal = BusinessProfile.model_validate(updates).model_dump(exclude_unset=True)
        if not set(proposal) - {"source"}:
            raise ValueError("At least one business field is required")
        return proposal

    def confirmation_message(self, updates: dict[str, Any]) -> str:
        """The host presents this exact text for the user to explicitly confirm."""
        proposal = self.propose_update(updates)
        return "Confirm business profile update: " + json.dumps(proposal, sort_keys=True, ensure_ascii=False)

    def commit_update(
        self,
        user_id: str,
        updates: dict[str, Any],
        *,
        user_confirmed: bool,
        messages: Sequence[BaseMessage],
    ) -> BusinessProfile:
        self._validate_user(user_id)
        proposal = self.propose_update(updates)
        if proposal.get("source") != "user_provided":
            raise ValueError("Only user-provided facts can be saved")
        # Only the latest message counts; historical, assistant and tool consent never count.
        if (
            user_confirmed is not True
            or not messages
            or not isinstance(messages[-1], HumanMessage)
            or messages[-1].content != self.confirmation_message(proposal)
        ):
            raise ValueError("Explicit confirmation of this exact profile update is required")
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT profile FROM business_profiles WHERE user_id = ?", (user_id,)).fetchone()
            current = BusinessProfile.model_validate_json(row[0]) if row else BusinessProfile()
            profile = BusinessProfile.model_validate(
                {**current.model_dump(), **proposal, "updated_at": datetime.now(UTC).isoformat()}
            )
            connection.execute(
                "INSERT INTO business_profiles (user_id, profile) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET profile = excluded.profile",
                (user_id, profile.model_dump_json()),
            )
        return profile
