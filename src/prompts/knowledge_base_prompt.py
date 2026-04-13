
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

BLOCKED_ANSWER = (
    "我没有在知识库中形成足够可信的证据链，暂时不输出详细步骤，"
    "为避免给出不可靠内容。你可以换一个更具体的问法（例如完整标题、别名或关键实体）。\n\n"
)
def build_blocked_answer(source_lines):
    return (
        f"{BLOCKED_ANSWER}\n\n"
        f"当前可参考来源：\n{source_lines}"
    )