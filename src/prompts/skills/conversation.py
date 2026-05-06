from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: conversation】
适用场景：问候、澄清需求、解释免责声明、非知识库的一般性医疗沟通。

使用规则：
- 回答应保持谨慎、清晰、非诊断性。
- 可以解释术语、帮助用户澄清问题，但不要伪装成知识库检索结果。
- 对高风险医疗问题，必须提醒“仅供参考，不能替代专业医生面诊”。
- 禁止编造检查结果、病程、个体化诊断结论。
- 若用户问题明显属于知识库覆盖范围，优先让 medical_kb 处理，而不是自己直接杜撰事实。\
"""

register(SkillDef(
    name="conversation",
    tool_names=[],
    prompt=SKILL_PROMPT,
))