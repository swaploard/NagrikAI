"""Profile mutation is a separate host-controlled path, never an LLM tool."""

from __future__ import annotations

import sqlite3
from typing import Any

from nagrik_ai.agent.react_nodes import BusinessProfileReader, failure, step
from nagrik_ai.models.agent_state import AgentState
from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.services.business_profile_service import BusinessProfileService


def commit_profile_node(state: AgentState, service: BusinessProfileReader | None = None) -> dict[str, Any]:
    updates = state.get("pending_profile_updates")
    output: dict[str, Any] = {"user_confirmed": False, "profile_update_result": None}
    if updates is None:
        return output
    try:
        if not isinstance(service, BusinessProfileService):
            raise ValueError("Profile storage is not available")
        profile = service.commit_update(
            state.get("user_id") or "",
            updates,
            user_confirmed=state.get("user_confirmed", False),
            messages=state.get("messages", []),
        )
    except (ValueError, TypeError) as exc:
        result = failure(str(exc))
    except sqlite3.Error:
        result = failure("Profile storage could not complete the update")
    else:
        result = ToolResult(True, profile.model_dump(), None, None, None, None)
        output.update(business_context=profile.model_dump(), pending_profile_updates=None)
    output["profile_update_result"] = result
    # Profile mutations are observable actions, not documentary/tool evidence.
    output["agent_steps"] = [
        *state.get("agent_steps", []),
        step("commit_profile", "Business profile saved." if result.ok else "Business profile update not saved."),
    ]
    return output
