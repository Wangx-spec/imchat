---
name: analyze_symptoms
description: 分析症状模式、涉及的身体系统与潜在疾病关联。适用于多症状的鉴别分析。
script: symptoms
function: analyze_symptoms
parameters:
  - name: symptoms
    type: string
    description: 症状描述，多个症状可用逗号分隔
    required: true
---

# Analyze Symptoms（症状分析）

## When to Use
- 用户描述多个症状，需要模式分析
- 需要鉴别诊断方向的提示

## 底层实现
- 症状分类规则引擎（按身体系统归类 + 关联疾病映射）
- 仅作模式分析，不构成诊断依据
