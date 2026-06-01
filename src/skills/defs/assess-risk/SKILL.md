---
name: assess_risk
description: 评估症状的风险等级（low/medium/high）并给出就医建议。适用于判断就医紧急程度。
script: risk
function: assess_risk
parameters:
  - name: symptoms
    type: string
    description: 症状描述，多个症状可用逗号分隔
    required: true
---

# Assess Risk（风险评估）

## When to Use
- 用户描述症状，需要评估严重程度
- 判断是否需要紧急就医

## 底层实现
- 风险规则引擎（高风险症状表 + 中风险关键词）
- 可选从 RAGService 检索风险相关医学建议增强（缺失时优雅降级）
