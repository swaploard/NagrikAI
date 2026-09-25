"""Conservative matching against a small, source-linked MSME scheme catalogue."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.tools.result_utils import failure, tool_result


class _Scheme(TypedDict):
    id: str
    name: str
    url: str
    keywords: set[str]
    condition: str

_SCHEMES: tuple[_Scheme, ...] = (
    {
        "id": "scheme:pmegp",
        "name": "Prime Minister's Employment Generation Programme",
        "url": "https://www.msme.gov.in/sites/default/files/Revisedguidelines07.12.2023.pdf",
        "keywords": {"startup", "new", "employment", "loan", "subsidy", "pmegp"},
        "condition": "New project; further applicant, sector and project conditions apply.",
    },
    {
        "id": "scheme:zed",
        "name": "MSME Sustainable (ZED) Certification",
        "url": "https://msme.gov.in/sites/default/files/MSMESchemebooklet2024.pdf",
        "keywords": {"msme", "quality", "zed", "certification", "manufacturing"},
        "condition": "Udyam registration is required.",
    },
    {
        "id": "scheme:lean",
        "name": "MSME Competitive (Lean) Scheme",
        "url": "https://lean.msme.gov.in/Home/Eligibility",
        "keywords": {"msme", "lean", "productivity", "manufacturing"},
        "condition": "Udyam registration is required.",
    },
)


@tool_result
def scheme_discovery(query: str, business_context: object | None = None) -> ToolResult:
    """Return plausible schemes; a match is never a final eligibility decision."""
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        return failure("Provide a scheme search query of 1 to 500 characters.")
    if business_context is not None and not isinstance(business_context, dict):
        return failure("business_context must be an object.")
    context = cast(dict[str, Any], business_context or {})
    stage = context.get("business_stage")
    if stage is not None and stage not in {"new", "existing"}:
        return failure("business_stage must be new or existing when supplied.")
    registrations = context.get("registrations") or []
    if not isinstance(registrations, list) or any(not isinstance(r, str) for r in registrations):
        return failure("registrations must be a list of strings.")
    udyam = any("udyam" in r.lower() for r in registrations)
    industry = str(context.get("industry") or "").lower()
    terms = set((query + " " + industry).lower().split())
    broad = bool(terms & {"scheme", "schemes", "support", "msme", "benefits"})
    matches: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for scheme in _SCHEMES:
        if not broad and not terms & scheme["keywords"]:
            continue
        if scheme["id"] == "scheme:pmegp" and stage == "existing":
            continue
        missing = []
        if scheme["id"] == "scheme:pmegp" and stage is None:
            missing.append("whether this is a new project")
        if scheme["id"] in {"scheme:zed", "scheme:lean"} and not udyam:
            missing.append("Udyam registration")
        matches.append(
            {
                "source_id": scheme["id"],
                "name": scheme["name"],
                "status": "potential_match",
                "condition": scheme["condition"],
                "facts_to_verify": missing,
            }
        )
        sources.append(
            {
                "source_id": scheme["id"],
                "citation_id": len(sources) + 1,
                "title": scheme["name"] + " guidance",
                "url": scheme["url"],
            }
        )
    if not matches:
        return failure("No catalogue schemes matched the supplied query and profile.", "NOT_FOUND")
    return ToolResult(
        True,
        {
            "schemes": matches,
            "sources": sources,
            "notice": "These are candidates; verify current eligibility against each linked scheme.",
        },
        None,
        None,
        "secondary",
        None,
        tuple(row["source_id"] for row in matches),
    )
