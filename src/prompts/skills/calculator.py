from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: calculate】
触发条件：用户需要计算数学表达式时调用。
输入要求：传入合法算术表达式字符串，如 "(12 + 3) * 4 / 2"。
输出格式：返回计算结果字符串。
使用规则：仅支持加减乘除、幂运算和取模，不支持函数调用。\
"""

register(SkillDef(
    name="calculator",
    tool_names=["calculate"],
    prompt=SKILL_PROMPT,
))