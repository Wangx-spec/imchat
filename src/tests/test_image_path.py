import json

from skills import registry
from skills.runtime import set_vlm_client


class FakeCaption:
    ok = True
    caption = "胸片可见右下肺片状阴影。"
    image_type = "medical_imaging"
    is_medical = True
    is_diagnostic_request = True
    uncertain_points = ["需要医生结合病史复核"]
    error = None


class FakeVLM:
    def summarize_image(self, *, image_path=None, image_url=None, user_question=""):
        assert image_path or image_url
        assert user_question
        return FakeCaption()


def test_analyze_image_returns_hitl_marker():
    skill = registry.get("analyze_image")
    assert skill is not None

    set_vlm_client(FakeVLM())
    try:
        payload = json.loads(skill.function(image_path="/tmp/example.png", question="这张图是否异常？"))
    finally:
        set_vlm_client(None)

    assert payload["ok"] is True
    assert payload["is_medical"] is True
    assert payload["is_diagnostic_request"] is True
    assert payload["requires_hitl"] is True
    assert payload["hitl_reason"] == "diagnostic_image_request"
    assert "不能替代医生诊断" in payload["answer"]


def test_analyze_image_gracefully_handles_missing_image():
    skill = registry.get("analyze_image")
    assert skill is not None

    payload = json.loads(skill.function())

    assert payload["ok"] is False
    assert payload["error"] == "no_image_input"
