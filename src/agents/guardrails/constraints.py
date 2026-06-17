from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str


# 绝对化诊断 / 越权结论措辞
_FORBIDDEN_DIAGNOSIS = [
    "确诊",
    "明确良恶性",
    "明确良性",
    "明确恶性",
    "确定良恶性",
    "确定分期",
    "可以确诊",
    "可确诊为",
    "可以诊断为",
    "确定为癌",
    "一定是癌",
]

# 必备免责声明命中提示（任一即可）
_DISCLAIMER_HINTS = [
    "仅供参考",
    "不能替代",
    "不能代替",
    "请咨询",
    "请及时就医",
    "就医",
    "专业医生",
    "面诊",
]

# 系统提示词 / 工具实现细节泄露
_LEAK_PATTERNS = [
    "系统提示词",
    "system prompt",
    "我的提示词",
    "我的系统提示",
    "工具调用",
    "function call",
    "tool_call",
]


def evaluate(text: str, *, require_disclaimer: bool = True) -> list[Violation]:
    """对输出文本做确定性规则校验，返回结构化违规项（无违规返回空列表）。"""
    body = (text or "").strip()
    if not body:
        return []

    low = body.lower()
    violations: list[Violation] = []

    for kw in _FORBIDDEN_DIAGNOSIS:
        if kw in body:
            violations.append(Violation("forbidden_diagnosis", f"出现绝对化诊断措辞：{kw}"))

    for kw in _LEAK_PATTERNS:
        if kw.lower() in low:
            violations.append(Violation("prompt_or_tool_leak", f"疑似泄露系统/工具细节：{kw}"))

    if require_disclaimer and not any(h in body for h in _DISCLAIMER_HINTS):
        violations.append(Violation("missing_disclaimer", "缺少必要的医学免责声明"))

    return violations


def format_reason(violations: list[Violation]) -> str:
    """把违规项汇总成一句话原因，供修复链使用。"""
    if not violations:
        return ""
    return "；".join(f"{v.code}:{v.detail}" for v in violations)
