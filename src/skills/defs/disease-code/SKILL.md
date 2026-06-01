---
name: disease_code
description: 查询疾病的 ICD-10 编码与疾病分类信息。适用于需要规范疾病编码的场景。
script: code
function: disease_code
parameters:
  - name: disease_name
    type: string
    description: 疾病名称
    required: true
---

# Disease Code（疾病编码查询）

## When to Use
- 需要查询某疾病的 ICD-10 编码
- 需要疾病的标准分类信息

## 底层实现
- 用定制 query（`{disease_name} ICD-10编码 疾病分类`）调用 RAGService
- 现有 RAG 无 metadata filter，按查询词检索；未命中时优雅降级
