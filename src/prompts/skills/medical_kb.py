from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: search_medical_kb】
触发条件：用户询问脑肿瘤、胸片影像、皮肤病变、糖尿病等已入库医学主题时优先调用。
输入要求：必须传入用户原始问题，不得翻译、删改或自行补充诊断假设。
输出格式：返回 JSON 字符串，包含 ok、answer、citations、sources、route、debug、error 字段。

使用规则：
- ok=true 且 answer 非空时，以 answer 为主组织回复。
- 若 citations / sources 非空，必须基于返回内容引用，不得自行编造论文、作者或出处。
- 若未精确命中，必须明确说明“医学知识库未精确命中”或“检索置信度较低”。
- 不得把通用常识伪装成“医学知识库结论”。
- 不得给出个体化诊断、处方剂量或替代执业医师判断。
- 若工具返回 low_confidence_blocked 或明显低置信，应明确提示用户需要进一步咨询专业医生。
- 本轮未调用此工具时，禁止使用“根据医学知识库/参考文档/文献”之类表述。\
- 若上下文中包含 “[随附图片摘要]”，请把这些摘要视为用户已经描述过的图片信息：
  - 可以基于摘要做知识检索和解释。
  - 不得宣称自己看到了图片，不得做诊断结论。
  - 若摘要标注 image_type=unsupported 或 ok=false，应主动询问用户重新上传。
"""

register(SkillDef(
    name="medical_kb",
    tool_names=["search_medical_kb"],
    prompt=SKILL_PROMPT,
))