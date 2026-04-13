from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: search_knowledge_base】
触发条件：用户询问文档、菜谱、教程、知识库类问题时优先调用。
输入要求：必须传入用户原始查询文本，不得翻译或改写。
输出格式：返回 JSON 字符串，包含 ok、answer、citations、sources、error 字段。
使用规则：
- ok=true 时，以 answer 作为主要回复内容，结尾添加"参考文档："区块引用 citations。
- citations 为空或 answer 表示未命中时，必须明确说明"知识库没有精确命中"。
- 未精确命中时，禁止伪造"来自知识库"的内容。
- 可提供通用兜底建议，但必须标注为"非知识库兜底建议"。
- 禁止编造引用来源。
- 本轮未调用此工具时，禁止使用"根据知识库/参考文档/来源"等表述。\
"""

register(SkillDef(
    name="knowledge_base",
    tool_names=["search_knowledge_base"],
    prompt=SKILL_PROMPT,
))