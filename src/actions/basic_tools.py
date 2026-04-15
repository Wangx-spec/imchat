import ast
import logging
import operator as op
from datetime import datetime

from langchain_core.tools import tool

from actions.knowledge_base_tools import search_knowledge_base
from actions.dishes.recommend_dishes_tools import recommend_dishes
from prompts.skills import get_tool_names_for_skills

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
    """Get the current local date and time with weekday."""
    logger.info("[TOOL_CALL] name=get_current_time")

    weekday_map = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    now = datetime.now()
    weekday = weekday_map[now.weekday()]
    now_text = f"{now.year}年{now.month}月{now.day}日（{weekday}）{now:%H:%M:%S}"

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

_ALL_TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
    "recommend_dishes": recommend_dishes,
}


def get_actions(enabled_skills: list[str] | None = None) -> list:
    if enabled_skills is None:
        return list(_ALL_TOOLS.values())
    needed = get_tool_names_for_skills(enabled_skills)
    return [t for name, t in _ALL_TOOLS.items() if name in needed]
