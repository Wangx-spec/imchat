---
name: search_knowledge
description: 检索本地医学知识库，返回答案、来源与调试信息。适用于一般医学知识问答。
script: search
function: search_knowledge
parameters:
  - name: query
    type: string
    description: 医学问题或检索关键词
    required: true
---

# Search Knowledge（知识库检索）

## When to Use
- 用户提出一般医学知识问题
- 需要带来源引用的本地知识库答案

## 底层实现
- 复用 `src/actions/knowledge_base_tools.py` 的 `get_rag_service().answer()`
- service 未初始化时返回 `{"ok": false, "error": "service_not_initialized"}`
