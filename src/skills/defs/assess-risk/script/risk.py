import json
import logging

from actions.knowledge_base_tools import get_rag_service

logger = logging.getLogger("skills.assess_risk")

HIGH_RISK_SYMPTOMS = [
    "胸痛", "呼吸困难", "意识模糊", "严重出血", "剧烈头痛",
    "持续呕吐", "高热不退", "突然晕厥", "剧烈腹痛", "面部下垂",
]
MEDIUM_RISK_KEYWORDS = ["持续", "加重", "反复", "严重", "剧烈"]

RECOMMENDATION = {
    "high": "建议立即就医或拨打急救电话120",
    "medium": "建议尽快就医，不要拖延，必要时前往医院",
    "low": "建议密切观察症状变化，如症状加重或持续不缓解应及时就医",
}


def assess_risk(symptoms: str) -> str:
    raw = (symptoms or "").strip()
    if not raw:
        return json.dumps({"ok": False, "error": "empty_symptoms"}, ensure_ascii=False)

    symptom_list = [s.strip() for s in raw.split(",") if s.strip()] or [raw]

    risk_level = "low"
    reasons: list[str] = []
    for symptom in symptom_list:
        if any(high in symptom for high in HIGH_RISK_SYMPTOMS):
            risk_level = "high"
            reasons.append(f"检测到高风险症状：{symptom}")

    if risk_level == "low":
        for symptom in symptom_list:
            if any(keyword in symptom for keyword in MEDIUM_RISK_KEYWORDS):
                risk_level = "medium"
                reasons.append(f"症状描述提示需要关注：{symptom}")

    kb_advice = None
    service = get_rag_service()
    if service is not None:
        try:
            result = service.answer(f"{raw} 紧急程度 风险评估 就医建议")
            kb_advice = getattr(result, "answer", "") or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SKILL] assess_risk kb enrich failed: %s", exc)

    return json.dumps(
        {
            "ok": True,
            "symptoms": raw,
            "risk_level": risk_level,
            "reasons": reasons,
            "recommendation": RECOMMENDATION[risk_level],
            "kb_advice": kb_advice,
        },
        ensure_ascii=False,
    )
