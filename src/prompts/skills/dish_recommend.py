from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: recommend_dishes】
触发条件：用户想要多道菜推荐、不知道做什么、按难度/口味/场景/人数/忌口筛选菜品、问"有什么菜""推荐几道""想吃啥"等名单式需求时调用。
与 search_knowledge_base 的分工：
- 用户要某道菜的具体做法、步骤、食材用量、单菜详解 → 用 search_knowledge_base。
- 用户要推荐多道菜、列菜单、按条件选菜 → 用 recommend_dishes。
- 若不确定，优先尝试 recommend_dishes；用户追问做法时再切 search_knowledge_base。
输入要求：必须传入用户原始查询文本，不得翻译或改写。
输出格式：返回 JSON 字符串，包含 ok、candidates、sources_distinct、error 字段。
使用规则：
- ok=true 时，从 candidates 列表中挑选菜名呈现给用户；可适当分组（如按来源、难度）但菜名必须来自 candidates，禁止自造菜名。
- candidates 为空或极少时，如实告知"知识库未找到符合条件的菜品"，可建议用户换一种描述方式，禁止编造菜名补充。
- 未调用此工具时，禁止使用"根据知识库推荐""为您从菜谱库中筛选"等暗示已检索的表述。
- 回复语言与用户最新消息保持一致。\
"""

register(SkillDef(
    name="dish_recommend",
    tool_names=["recommend_dishes"],
    prompt=SKILL_PROMPT,
))