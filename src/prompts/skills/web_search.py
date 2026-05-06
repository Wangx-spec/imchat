from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: web_search】
触发条件：用户询问知识库未覆盖的问题、近期医学进展、最新指南、时效性医学信息时调用。
输入要求：必须使用用户原始问题或在 handoff 场景下使用原始医学问题，不要擅自改写为过窄问题。
输出格式：返回 JSON 字符串，包含 ok、answer、results、error 字段。

使用规则：
- ok=true 时，基于返回结果组织回答，不得编造网页来源。
- 如果 results 为空，必须明确说明“未检索到足够结果”。
- 不得伪造 PubMed、期刊名、链接、作者或 DOI。
- 医疗相关总结必须保持谨慎，不能替代执业医生建议。
- 如结论来自网页搜索，应明确表述为“基于实时搜索结果”，不要伪装成知识库结论。\
- 若 error 不为空（如 `tavily_disabled`、`unexpected_response_format`、`tool_exception:*`），必须明确说明“本次未获得有效联网结果”，并回退到保守表述。
- 未看到真实 results 前，严禁写“已检索/已核查最新指南/已查阅AJNR或WHO原文”等字样。
- 对静态医学知识，优先简洁补充，不要把回答包装成“最新综述”或“权威联网核查报告”。\
"""

register(SkillDef(
    name="web_search",
    tool_names=["web_search"],
    prompt=SKILL_PROMPT,
))