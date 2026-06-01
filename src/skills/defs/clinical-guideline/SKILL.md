---
name: clinical_guideline
description: 检索疾病或治疗主题的临床诊疗指南与规范。适用于需要权威诊疗依据的场景。
script: guideline
function: clinical_guideline
parameters:
  - name: query
    type: string
    description: 查询内容（疾病名称或治疗主题）
    required: true
---

# Clinical Guideline（临床指南检索）

## When to Use
- 需要某疾病的临床诊疗指南
- 需要规范化的诊疗建议依据

## 底层实现
- 用定制 query（`{query} 临床指南 诊疗规范`）调用 RAGService
- 现有 RAG 无 metadata filter，按查询词检索；未命中时优雅降级
