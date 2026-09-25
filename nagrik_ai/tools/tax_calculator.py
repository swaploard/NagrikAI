"""Decimal GST arithmetic; legal eligibility and rates must come from evidence."""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.tools.result_utils import failure, source_id, tool_result


def _money(value: str, label: str) -> Decimal:
    if not isinstance(value, str) or not value.strip() or len(value) > 40:
        raise ValueError(f"{label} must be a decimal string.")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a decimal string.") from exc
    if not number.is_finite() or number < 0 or number > Decimal("100000000000000"):
        raise ValueError(f"{label} must be a finite nonnegative amount within range.")
    return number


@tool_result
def tax_calculator(
    operation: str,
    amount: str,
    rate_percent: str = "",
    eligible_itc: str = "0",
    blocked_itc: str = "0",
) -> ToolResult:
    """Calculate a supplied GST rate, supplied eligible ITC, or composition turnover.

    The caller must establish Section 10/16/17 applicability and the rate using
    authoritative evidence. This tool only performs the resulting arithmetic.
    """
    if operation not in {"gst_liability", "itc", "composition"}:
        return failure("Choose gst_liability, itc, or composition.")
    try:
        base = _money(amount, "amount")
        if operation == "itc":
            if rate_percent or eligible_itc != "0":
                return failure("ITC uses amount and blocked_itc only.")
            blocked = _money(blocked_itc, "blocked_itc")
            if blocked > base:
                return failure("blocked_itc cannot exceed input tax.")
            value = base - blocked
        else:
            if not rate_percent:
                return failure("A verified rate_percent is required.")
            rate = _money(rate_percent, "rate_percent")
            if rate > 100:
                return failure("rate_percent cannot exceed 100.")
            if operation == "composition" and (eligible_itc != "0" or blocked_itc != "0"):
                return failure("Composition calculation cannot apply ITC.")
            if operation == "gst_liability" and blocked_itc != "0":
                return failure("Use the itc operation to exclude blocked credit first.")
            credit = _money(eligible_itc, "eligible_itc")
            gross = base * rate / 100
            if credit > gross:
                return failure("eligible_itc exceeds calculated GST; calculate carry-forward separately.")
            value = gross - credit
        canonical = format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")
        if "." in canonical:
            canonical = canonical.rstrip("0").rstrip(".")
        canonical = canonical or "0"
        identity = json.dumps(
            {
                "operation": operation,
                "amount": str(base),
                "rate_percent": rate_percent,
                "eligible_itc": eligible_itc,
                "blocked_itc": blocked_itc,
            },
            sort_keys=True,
        )
        return ToolResult(True, canonical, None, None, "deterministic", None, (source_id("tax_calc", identity),))
    except (ValueError, InvalidOperation) as exc:
        return failure(str(exc))
