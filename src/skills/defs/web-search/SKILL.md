---
name: web_search
description: 使用 Tavily 检索当前医学/网络信息，返回结果列表与摘要。适用于需要实时或外部信息的场景。
script: search
function: web_search
parameters:
  - name: query
    type: string
    description: 检索关键词或问题
    required: true
---

# Web Search（联网检索）

## When to Use
- 本地知识库无法覆盖的实时或最新信息
- 需要外部网页来源佐证

## 底层实现
- 包装 `src/actions/web_search_tools.py` 的 `web_search`
- Tavily 未启用时返回 `{"ok": false, "error": "tavily_disabled"}`
