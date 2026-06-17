import json
from skills.runtime import get_vlm_client

def analyze_image(
    image_path: str | None = None,
    image_url: str | None = None,
    question: str | None = None,
) -> str:
    if not image_path and not image_url:
        return json.dumps(
            {
                "ok": False,
                "error": "no_image_input",
                "answer": "未提供可分析的图片，请重试。"
            },
            ensure_ascii=False
        )
    
    vlm = get_vlm_client()
    if vlm is None:
        return json.dumps(
            {
                "ok": False,
                "error": "vlm_not_initialized",
                "answer": "视觉大模型未初始化，请稍后再试。"
            },
            ensure_ascii=False
        )
    
    cap = vlm.summarize_image(
        image_path=image_path,
        image_url=image_url,
        user_question=question or "请分析这张图片中的内容"
    )

    if not cap.ok:
        return json.dumps(
            {
                "ok": False,
                "error": cap.error,
                "answer": "图片分析失败，请重试。"
            },
            ensure_ascii=False
        )
    
    return json.dumps(
        {
            "ok": True,
            "caption": cap.caption,
            "image_type": cap.image_type,
            "is_medical": cap.is_medical,
            "is_diagnostic_request": cap.is_diagnostic_request,
            "requires_hitl": bool(cap.is_diagnostic_request),
            "hitl_reason": "diagnostic_image_request" if cap.is_diagnostic_request else None,
            "uncertain_points": cap.uncertain_points,
            "answer": _format_answer(cap),
        },
        ensure_ascii=False
    )

def _format_answer(cap) -> str:
    lines = [
        "【图像分析结果】",
        f"图片类型：{cap.image_type}",
        f"是否医学相关：{'是' if cap.is_medical else '否'}",
        "",
        "可见内容摘要：",
        cap.caption or "未能生成有效描述。",
    ]

    if cap.uncertain_points:
        lines.append("")
        lines.append("【可能存在歧义的点】")
        lines.extend(f"- {x}" for x in cap.uncertain_points)

    if cap.is_diagnostic_request:
        lines.append("")
        lines.append("提醒：以上仅为图像可见内容分析，不能替代医生诊断。若涉及疾病判断，请咨询专业医生。")
    
    return "\n".join(lines)