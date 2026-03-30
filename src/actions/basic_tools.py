import ast
import logging
import operator as op
from datetime import datetime

from langchain_core.tools import tool


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}
logger = logging.getLogger("chat.tools")


def _safe_eval_expr(expression: str) -> float:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))

        raise ValueError("Unsupported expression. Use only basic arithmetic.")

    parsed = ast.parse(expression, mode="eval")
    return _eval(parsed.body)


@tool
def get_current_time() -> str:
    """Get the current local date and time."""
    logger.info("[TOOL_CALL] name=get_current_time")
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("[TOOL_RESULT] name=get_current_time result=%s", now_text)
    return now_text


@tool
def calculate(expression: str) -> str:
    """Calculate a math expression, e.g. '(12 + 3) * 4 / 2'."""
    logger.info("[TOOL_CALL] name=calculate expression=%s", expression)
    try:
        result = _safe_eval_expr(expression)
    except Exception as exc:
        error_text = f"Calculation error: {exc}"
        logger.warning("[TOOL_RESULT] name=calculate error=%s", error_text)
        return error_text

    if result.is_integer():
        result_text = str(int(result))
    else:
        result_text = str(result)

    logger.info("[TOOL_RESULT] name=calculate result=%s", result_text)
    return result_text


def get_actions():
    return [get_current_time, calculate]
