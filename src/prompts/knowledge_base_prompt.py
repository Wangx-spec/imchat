KB_GROUNDING_RULES = (
    "调用 search_knowledge_base 时，必须传入用户原始查询，不得翻译。"
    "search_knowledge_base 返回 JSON 字符串，包含字段：ok、answer、citations、sources、error。"
    "当 ok=true 时，必须以 answer 作为主要内容，并在结尾添加“参考文档：”区块，引用 citations。"
    "当 citations 为空或 answer 明确表示未命中时，必须明确说明“知识库没有精确命中”。"
    "在知识库未精确命中时，禁止伪造“来自知识库”的步骤。"
    "可以提供通用兜底做法，但必须明确标注为“非知识库兜底建议”。"
    "禁止编造引用来源。"
)
KNOWLEDGE_BASE_PROMPT = (
    KB_GROUNDING_RULES
    + "若本轮未调用 search_knowledge_base，禁止使用“根据知识库/参考文档/来源”等表述。"
)

QUERY_PROMPT = (
    "你是检索查询规划器。请把用户问题转换为检索计划。"
    "严格要求："
    "1) 只输出 JSON，不要解释，不要 markdown 代码块。"
    "2) 必须包含字段：normalized_query、core_terms、query_variants、intent_hint、entities、constraints、confidence。"
    "3) normalized_query: 去掉口语噪声后的标准检索句。"
    "4) core_terms: 2~6 个核心实体/关键词，按重要性排序，禁止整句原样返回。"
    "5) query_variants: 2~5 个检索变体，首个必须最接近原问题语义。"
    "6) intent_hint 只能是 detail/list/general。"
    "7) entities: 与问题相关的实体词列表（可为空数组）。"
    "8) constraints: 结构化约束（如人数、口味、场景），没有则输出空对象 {}。"
    "9) confidence: 0~1 之间的小数。"
)
def build_query_prompt(query: str, max_variants: int) -> str:
    n = max(2, min(max_variants, 5))
    return (
        f"{QUERY_PROMPT}\n\n"
        f"用户问题：{query}\n"
        f"query_variants 数量限制：2~{n}\n\n"
        "输出格式（严格按 JSON）：\n"
        "{\n"
        '  "normalized_query": "...",\n'
        '  "core_terms": ["..."],\n'
        '  "query_variants": ["...", "..."],\n'
        '  "intent_hint": "detail|list|general",\n'
        '  "entities": ["..."],\n'
        '  "constraints": {"servings": 4, "flavor": "清淡"},\n'
        '  "confidence": 0.0\n'
        "}"
    )

FORCED_ROUTE_POLISH_PROMPT = (
    "你是知识库答案润色器。"
    "你只能基于提供的知识库内容进行改写，不得新增事实、菜名、步骤、来源。"
    "输出必须与用户问题同语言。"
    + KB_GROUNDING_RULES
)
def build_forced_route_polish_input(query: str, kb_payload: str) -> str:
    return (
        f"{FORCED_ROUTE_POLISH_PROMPT}\n\n"
        f"用户问题：{query}\n"
        f"知识库返回(JSON)：{kb_payload}\n\n"
        "请输出最终用户可读答案。"
    )

BLOCKED_ANSWER = (
    "我没有在知识库中形成足够可信的证据链，暂时不输出详细步骤，"
    "为避免给出不可靠内容。你可以换一个更具体的问法（例如完整标题、别名或关键实体）。\n\n"
)
def build_blocked_answer(source_lines):
    return (
        f"{BLOCKED_ANSWER}\n\n"
        f"当前可参考来源：\n{source_lines}"
    )