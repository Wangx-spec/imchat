from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: get_current_time】
触发条件：用户询问当前时间、日期、星期等时间相关问题时调用。
输入要求：无参数。
输出格式：返回包含日期、星期和时间的完整字符串，例如 "2026年4月15日（星期三）13:51:43"。\
注意：
 - 必须原样使用工具返回值
 - 不要自行推断星期几
 - 不要改写 24 小时制为中文时段表达
"""

register(SkillDef(
    name="time",
    tool_names=["get_current_time"],
    prompt=SKILL_PROMPT,
))