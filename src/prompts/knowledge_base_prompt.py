
QUERY_PROMPT = (
    "你是医学知识库检索查询规划器。请把用户问题转换为检索计划。"
    "严格要求："
    "1) 只输出 JSON，不要解释，不要 markdown 代码块。"
    "2) 必须包含字段：normalized_query、core_terms、query_variants、"
    "intent_hint、entities、constraints、confidence。"
    "3) normalized_query: 去掉口语噪声后的标准医学检索句，保留医学实体原文。"
    "4) core_terms: 2~6 个核心医学实体/关键词，按重要性排序；"
    "优先疾病名、解剖部位、检查手段、药物、指南名；禁止整句原样返回；"
    "遇到缩写应同时给出全称（如 'AMI' 与 '急性心肌梗死' 视为同一组中的两项）。"
    "5) query_variants: 2~5 个检索变体，首个最接近原问题语义，"
    "其余覆盖中英文同义、缩写展开、指南/综述表述方式（如'诊疗指南'/'clinical guideline'）。"
    "6) intent_hint 只能是 detail/list/general。"
    "7) entities: 与问题相关的医学实体（疾病、症状、药物、检查、基因、部位等），可为空数组。"
    "8) constraints: 结构化医学约束对象，只能使用以下字段，不确定一律留空："
    "condition(string)、body_system(枚举: neuro|cardio|respiratory|gi|endocrine|"
    "dermatology|oncology|infectious|musculoskeletal|other)、"
    "modality(枚举: ct|mri|xray|ultrasound|pathology|ecg|lab|none)、"
    "intent(枚举: diagnosis|treatment|screening|prognosis|mechanism|guideline|definition|epidemiology)、"
    "population(对象: age_group=adult|pediatric|elderly|neonate, sex=male|female|any, pregnancy=true|false)、"
    "severity(枚举: acute|chronic|mild|moderate|severe|critical)、"
    "evidence_level(枚举: guideline|meta_analysis|rct|review|case_report|any)、"
    "time_range(字符串，如 '近5年'、'2020-2025'，不确定留空)。"
    "枚举字段不能出现表外值；任何不确定的字段必须留空字符串或空对象。"
    "9) confidence: 0~1 之间的小数，表达你对 constraints 判断的置信度。"
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
        '  "constraints": {\n'
        '    "condition": "",\n'
        '    "body_system": "",\n'
        '    "modality": "",\n'
        '    "intent": "",\n'
        '    "population": {"age_group": "", "sex": "any", "pregnancy": false},\n'
        '    "severity": "",\n'
        '    "evidence_level": "any",\n'
        '    "time_range": ""\n'
        '  },\n'
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