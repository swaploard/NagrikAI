"""Bounded arithmetic interpreter using decimal values and an AST allowlist."""

from __future__ import annotations

import ast
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

from nagrik_ai.models.tool_result import ToolResult
from nagrik_ai.tools.result_utils import failure, source_id, tool_result


@tool_result
def calculator(query: str) -> ToolResult:
    """Evaluate arithmetic only. round(value, digits) uses decimal half-even rounding."""
    if not isinstance(query, str) or not query.strip() or len(query) > 512:
        return failure("Provide an arithmetic expression of 1 to 512 characters.")
    expression = query.strip()
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        return failure("Expression is too complex.")

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Decimal(ast.get_source_segment(expression, node) or "")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            match node.op:
                case ast.Add():
                    value = left + right
                case ast.Sub():
                    value = left - right
                case ast.Mult():
                    value = left * right
                case ast.Div():
                    value = left / right
                case ast.Mod():
                    value = left % right
                case ast.Pow():
                    if right != right.to_integral_value() or abs(right) > 100:
                        raise ValueError("Exponent must be an integer between -100 and 100")
                    value = left ** int(right)
                case _:
                    raise ValueError("Unsupported operator")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "round"
            and len(node.args) == 2
            and not node.keywords
        ):
            operand, digits = evaluate(node.args[0]), evaluate(node.args[1])
            if digits != digits.to_integral_value() or abs(digits) > 28:
                raise ValueError("Rounding precision out of bounds")
            value = operand.quantize(Decimal(1).scaleb(-int(digits)))
        else:
            raise ValueError("Only numeric arithmetic and round(value, digits) are allowed")
        if not value.is_finite() or (value and abs(value.adjusted()) > 100):
            raise ValueError("Result out of bounds")
        return value

    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        value = evaluate(tree.body)
        canonical = format(value, "f")
        if "." in canonical:
            canonical = canonical.rstrip("0").rstrip(".")
        if not value:
            canonical = "0"
    return ToolResult(True, canonical, None, None, "deterministic", None, (source_id("calculator", expression),))
