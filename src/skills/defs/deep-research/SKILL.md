---
name: deep_research
description: 对一个主题进行较深入的资料检索（本期复用联网检索，后续升级为证据综合）。
script: research
function: deep_research
parameters:
  - name: query
    type: string
    description: 研究主题或问题
    required: true
---

# Deep Research（深度研究）

## When to Use
- 需要围绕主题做较系统的资料检索
- 研究型 Agent 的高级检索能力

## 底层实现
- 本期复用 `src/actions/web_search_tools.py` 的 `web_search`
- Phase 7 升级为 web_search + RAG + 证据综合
