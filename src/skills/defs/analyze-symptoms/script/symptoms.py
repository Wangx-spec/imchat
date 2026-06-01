import json

SYMPTOM_CATEGORIES = {
    "respiratory": {
        "keywords": ["咳嗽", "呼吸", "鼻塞", "喉咙", "气短", "痰", "胸闷"],
        "name": "呼吸系统",
    },
    "digestive": {
        "keywords": ["腹痛", "腹泻", "恶心", "呕吐", "胃痛", "便秘", "消化"],
        "name": "消化系统",
    },
    "neurological": {
        "keywords": ["头痛", "头晕", "眩晕", "失眠", "麻木", "乏力"],
        "name": "神经系统",
    },
    "cardiovascular": {
        "keywords": ["胸痛", "心悸", "气短", "心慌", "血压"],
        "name": "心血管系统",
    },
    "musculoskeletal": {
        "keywords": ["关节", "肌肉", "骨骼", "疼痛", "肿胀", "僵硬"],
        "name": "骨骼肌肉系统",
    },
}

DISEASE_MAP = {
    "respiratory": ["感冒", "支气管炎", "肺炎"],
    "digestive": ["胃炎", "肠炎", "消化不良"],
    "cardiovascular": ["心绞痛", "高血压", "心律不齐"],
    "neurological": ["偏头痛", "神经衰弱", "脑供血不足"],
}


def analyze_symptoms(symptoms: str) -> str:
    raw = (symptoms or "").strip()
    if not raw:
        return json.dumps({"ok": False, "error": "empty_symptoms"}, ensure_ascii=False)

    symptom_list = [s.strip() for s in raw.split(",") if s.strip()] or [raw]

    detected: list[dict[str, str]] = []
    seen: set[str] = set()
    for cid, cdata in SYMPTOM_CATEGORIES.items():
        for symptom in symptom_list:
            if any(keyword in symptom for keyword in cdata["keywords"]):
                if cid not in seen:
                    seen.add(cid)
                    detected.append({"id": cid, "name": cdata["name"]})
                break

    patterns: list[str] = []
    if detected:
        patterns.append("症状涉及：" + ", ".join(c["name"] for c in detected))
    if len(detected) > 1:
        patterns.append("涉及多个身体系统，建议全面检查")

    possible: list[str] = []
    for c in detected:
        possible.extend(DISEASE_MAP.get(c["id"], []))
    possible = list(dict.fromkeys(possible))[:5]

    return json.dumps(
        {
            "ok": True,
            "symptoms": raw,
            "patterns": patterns,
            "possible_diseases": possible,
            "disclaimer": "以上仅为模式分析，不能作为诊断依据，请咨询专业医生。",
        },
        ensure_ascii=False,
    )
