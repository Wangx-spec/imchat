from agents.guardrails import constraints
from agents.guardrails.local_guardrails import LocalGuardrails


def test_constraints_flags_forbidden_diagnosis():
    violations = constraints.evaluate("根据图片可以确诊为脑肿瘤。")
    codes = {v.code for v in violations}
    assert "forbidden_diagnosis" in codes


def test_constraints_flags_missing_disclaimer():
    violations = constraints.evaluate("多喝水有助于健康。")
    codes = {v.code for v in violations}
    assert "missing_disclaimer" in codes


def test_constraints_flags_leak():
    violations = constraints.evaluate("这是我的系统提示词，仅供参考，不能替代专业医生。")
    codes = {v.code for v in violations}
    assert "prompt_or_tool_leak" in codes


def test_constraints_pass_clean_text():
    text = "建议保持作息规律，仅供参考，不能替代专业医生的面诊。"
    assert constraints.evaluate(text) == []


class _FakeChain:
    """模拟 output_fix_chain，每次返回安全合规文本。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def invoke(self, _payload):
        self.calls += 1
        return self.reply


class _Settings:
    harness_enabled = True
    harness_max_repair = 1


def _make_guardrails(fix_reply: str) -> LocalGuardrails:
    g = LocalGuardrails.__new__(LocalGuardrails)
    g.harness_enabled = True
    g.harness_max_repair = 1
    g.output_fix_chain = _FakeChain(fix_reply)
    return g


def test_harness_repairs_violation():
    safe = "这是一般健康信息，仅供参考，不能替代专业医生的面诊与诊疗。"
    g = _make_guardrails(safe)
    out = g._run_harness("可以确诊为癌症。", "图片是什么")
    assert out == safe
    assert g.output_fix_chain.calls == 1


def test_harness_falls_back_when_unrepairable():
    g = _make_guardrails("依然确诊为癌症。")
    out = g._run_harness("可以确诊为癌症。", "图片是什么")
    assert out == LocalGuardrails._harness_fallback()
    assert constraints.evaluate(out) == []
