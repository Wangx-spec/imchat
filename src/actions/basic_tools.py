import ast
import logging
import operator as op
from datetime import datetime

from langchain_core.tools import tool
from typing import Any


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

_rag_service: Any = None


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

def set_rag_service(service: Any) -> None:
    global _rag_service
    _rag_service = service
    logger.info("[RAG_BIND] service=%s", type(service).__name__ if service else "None")

def _format_kb_output(answer: str, sources: list[str] | None) -> str:
    answer_text = (answer or "").strip() or "未检索到有效答案。"
    lines = [answer_text]
    if sources:
        lines.append("")
        lines.append("来源：")
        for src in sources[:3]:
            lines.append(f"- {src}")
    return "\n".join(lines)

@tool
def search_knowledge_base(query: str) -> str:
    """Search local knowledge base and return answer with sources."""
    q = (query or "").strip()
    logger.info("[TOOL_CALL] name=search_knowledge_base query=%s", q)
    if not q:
        return "请提供要检索的问题。"
    if _rag_service is None:
        return "知识检索服务未初始化。"
    try:
        result = _rag_service.answer(q)
        text = _format_kb_output(
            answer=getattr(result, "answer", ""),
            sources=getattr(result, "sources", []),
        )
        logger.info("[TOOL_RESULT] name=search_knowledge_base ok")
        return text
    except Exception as exc:
        logger.exception("[TOOL_RESULT] name=search_knowledge_base error=%s", exc)
        return f"知识检索暂时不可用：{exc}"


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
    return [get_current_time, calculate, search_knowledge_base]
