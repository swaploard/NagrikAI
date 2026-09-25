"""Standard compliance dates derived from linked official guidance.

Dates are baseline rules, not a live notification feed. The return dashboard or
regulator notice must be checked for extensions affecting a particular period.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.tools.result_utils import failure, tool_result

_SOURCES = {
    "gstr1": ("GST portal GSTR-1 guide", "https://tutorial.gst.gov.in/userguide/returns/GSTR_1.htm"),
    "gstr3b": ("CBIC CGST Act, section 39", "https://cbic-gst.gov.in/hindi/CGST-bill-e.html"),
    "qrmp": ("GST portal QRMP advisory", "https://tutorial.gst.gov.in/offlineutilities/returns/QRMP_Advisory.pdf"),
    "aoc4": ("MCA Companies Act, section 137", "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf"),
    "pf": (
        "EPFO employer booklet",
        "https://www.epfindia.gov.in/site_docs/PDFs/MiscPDFs/Employer_Information_Booklet.pdf",
    ),
    "esic": ("ESIC contribution due-date notice", "https://esic.gov.in/attachments/files/ESIC%20AD.pdf"),
}

_QRMP_22 = frozenset(
    {
        "chhattisgarh",
        "madhya pradesh",
        "gujarat",
        "maharashtra",
        "karnataka",
        "goa",
        "kerala",
        "tamil nadu",
        "telangana",
        "andhra pradesh",
        "daman and diu",
        "dadra and nagar haveli",
        "puducherry",
        "andaman and nicobar islands",
        "lakshadweep",
    }
)
_QRMP_24 = frozenset(
    {
        "himachal pradesh",
        "punjab",
        "uttarakhand",
        "haryana",
        "rajasthan",
        "uttar pradesh",
        "bihar",
        "sikkim",
        "arunachal pradesh",
        "nagaland",
        "manipur",
        "mizoram",
        "tripura",
        "meghalaya",
        "assam",
        "west bengal",
        "jharkhand",
        "odisha",
        "jammu and kashmir",
        "ladakh",
        "chandigarh",
        "delhi",
    }
)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


@tool_result
def compliance_calendar(
    compliance: str,
    period: str,
    filing_frequency: str = "monthly",
    state: str = "",
    company_type: str = "",
) -> ToolResult:
    """Return a standard date and an official source for a specified tax period."""
    if compliance not in {"gstr1", "gstr3b", "aoc4", "pf", "esic"}:
        return failure("Unsupported compliance type.")
    if compliance == "aoc4":
        if company_type != "ordinary":
            return failure(
                "Specify company_type=ordinary for the standard AOC-4 rule; special company rules need review."
            )
        try:
            if not isinstance(period, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", period):
                raise ValueError("Invalid AGM date")
            agm = date.fromisoformat(period)
        except (TypeError, ValueError):
            return failure("For AOC-4, period must be the AGM date YYYY-MM-DD.")
        due = agm + timedelta(days=30)
        source_key = "aoc4"
        basis = "30 days after the AGM for an ordinary company"
    else:
        if not isinstance(period, str) or not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", period):
            return failure("period must be YYYY-MM for the tax or wage month.")
        year, month = map(int, period.split("-"))
        if year < 2017 or year > 2100:
            return failure("period year is out of range.")
        if compliance in {"pf", "esic"}:
            if filing_frequency != "monthly" or state:
                return failure("PF and ESIC use a wage month; omit filing frequency and state.")
            next_year, next_month = _next_month(year, month)
            due = date(next_year, next_month, 15)
            source_key = compliance
            basis = "15th of the following month"
        elif filing_frequency == "monthly":
            next_year, next_month = _next_month(year, month)
            due = date(next_year, next_month, 11 if compliance == "gstr1" else 20)
            source_key = compliance
            basis = "monthly return, following month"
        elif filing_frequency == "quarterly":
            if month not in {3, 6, 9, 12}:
                return failure("For quarterly GST, period must be the quarter-end month.")
            next_year, next_month = _next_month(year, month)
            if compliance == "gstr1":
                due = date(next_year, next_month, 13)
                source_key = "gstr1"
            else:
                location = state.strip().lower() if isinstance(state, str) else ""
                if location not in _QRMP_22 | _QRMP_24:
                    return failure(
                        "A recognized principal-place-of-business state or UT is required for quarterly GSTR-3B."
                    )
                due = date(next_year, next_month, 22 if location in _QRMP_22 else 24)
                source_key = "qrmp"
            basis = "quarterly return, following month"
        else:
            return failure("filing_frequency must be monthly or quarterly.")
    sid = f"calendar:{source_key}"
    title, url = _SOURCES[source_key]
    return ToolResult(
        True,
        {
            "compliance": compliance,
            "period": period,
            "standard_due_date": due.isoformat(),
            "basis": basis,
            "notice": "Check the regulator portal for period-specific extensions or special rules.",
            "sources": [{"source_id": sid, "citation_id": 1, "title": title, "url": url}],
        },
        None,
        None,
        "authoritative",
        None,
        (sid,),
    )
