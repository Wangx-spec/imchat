from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: get_current_time】
触发条件：用户询问当前时间、日期、星期等时间相关问题时调用。
输入要求：无参数。
输出格式：返回 "YYYY-MM-DD HH:MM:SS" 格式的字符串。\
"""

register(SkillDef(
    name="time",
    tool_names=["get_current_time"],
    prompt=SKILL_PROMPT,
))