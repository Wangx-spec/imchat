# Multi-Agent 联调问题修复清单

---

## 一、背景

当前多智能体主链路已经跑通，已验证以下能力：

- `AGENT_MODE=multi` 已生效
- `langgraph-multi` runtime 已启用
- `Supervisor + knowledge/recommend/chat` 三个 specialist 已构建成功
- 推荐类问题可路由到 `recommend`
- 具体做法类问题可路由到 `knowledge`
- 时间/计算类问题可路由到 `chat`

但从联调日志看，仍存在若干功能与稳定性问题，需要继续修复。

---

## 二、问题总览

| 优先级 | 问题 | 现象 | 影响 |
|------|------|------|------|
| P0 | 推荐工具低置信度下乱回答 | “来几道高难度一点的菜”返回了工具候选之外的内容 | 用户看到不受控结果，可信度下降 |
| P0 | 时间工具结果被模型错误改写 | “现在几点了”星期几错误 | 直接影响正确性 |
| P1 | Supervisor JSON 路由不稳定 | `parse_failed fallback=chat raw=''` | 路由偶发失败，误入 `chat` |
| P1 | `TOOL_USED` 日志混入历史消息 | 当前轮日志打印出整段会话的工具调用 | 干扰测试判断，也可能影响 sanitize 逻辑 |

---

## 三、P0 修复项

### 1. 推荐工具增加低置信度门控

#### 问题

`recommend_dishes` 当前只做检索和去重，不判断召回结果是否足够可靠；即使 `confidence_score` 很低，也会继续返回 `ok=true`，交给 LLM 自由组织答案。

在测试日志中，高难度推荐场景已经出现：

- `confidence_score=0.396`
- `is_confident=False`
- 但工具仍返回 `ok candidates=4`

这会导致模型围绕不稳定候选继续扩写，出现“乱推荐”。

#### 修改文件

- `src/actions/dishes/recommend_dishes_tools.py`

#### 修改目标

在 `svc.retrieve(expanded)` 之后读取 `ret.debug` 中的置信度信息：

- `confidence_score`
- `is_confident`

当满足以下任一条件时，不继续返回推荐结果：

- `is_confident is False`
- `confidence_score < 0.45`
- `candidates` 为空或极少

#### 建议处理方式

返回：

- `ok=false`
- `error="low_confidence_recommendation"`
- `candidates=[]`

并把以下信息写入 `debug`：

- `expanded_query`
- `confidence_score`
- `is_confident`
- `parent_hits`
- `deduped`

#### 验收标准

- 问题：`来几道高难度一点的菜`
- 预期：
  - 若召回低置信度，不再输出大量自由扩写内容
  - 回复承认“知识库未找到足够可靠的推荐结果”或建议换一种问法

---

### 2. 过滤非真实菜名候选

#### 问题

高难度推荐很可能命中了“星级索引页”或“概念页”，例如：

- `7 星难度菜品`
- `8 星难度菜品`
- `0 星难度菜品`
- `示例菜的做法`

这些标题不适合作为最终推荐项展示给用户。

#### 修改文件

- `src/actions/dishes/recommend_dishes_tools.py`

#### 修改目标

在 `_dedup_parents()` 中增加一层候选标题过滤，只保留“真实菜名”。

#### 建议过滤规则

可先用简单规则拦截明显无效标题：

- 包含 `星难度菜品`
- 包含 `示例`
- 包含 `模板`
- 标题过短且无明确菜名语义

#### 验收标准

- 问题：`来几道高难度一点的菜`
- 预期：
  - 返回的 `candidates` 不再出现星级占位标题
  - 模型只能围绕真实菜名回答

---

### 3. 时间工具直接返回完整中文时间

#### 问题

当前 `get_current_time()` 只返回：

- `YYYY-MM-DD HH:MM:SS`

星期几和“上午/下午”是由 LLM 二次推断的，因此会出现：

- 日期正确，但星期几错误
- `下午13:51:43` 这种不自然格式

#### 修改文件

- `src/actions/basic_tools.py`
- 可选：`src/prompts/skills/time.py`

#### 修改目标

让工具层直接返回完整的最终时间文本，不让模型推理星期几。

#### 建议返回格式

例如：

- `2026年4月15日（星期三）13:51:43`

或：

- `2026年4月15日（星期三）下午1:51:43`

建议优先使用统一、少歧义的 24 小时制格式。

#### 实现要点

- 使用 `datetime.now().weekday()` 计算星期
- 在 Python 中完成格式化
- 工具日志打印最终字符串

#### Prompt 补充（可选）

在 `src/prompts/skills/time.py` 中补充约束：

- 回复应直接使用工具返回值
- 不要自行推算星期几
- 不要改写时间格式

#### 验收标准

- 问题：`现在几点了？`
- 预期：
  - 最终回复中的星期与实际日期一致
  - 不再出现模型猜错星期的情况

---

## 四、P1 修复项

### 4. 提高 Supervisor 路由稳定性

#### 问题

当前 supervisor 使用自然语言 prompt，并通过 `json.loads(response.content)` 手动解析返回值。  
测试中已出现多次：

- `parse_failed fallback=chat raw=''`

这会导致本应进入 `knowledge` 或 `chat` 的请求，统一兜底进入 `chat`。

#### 修改文件

- `src/graphs/multi_agent_graph.py`

#### 修改目标

让 supervisor 的路由输出更稳定，不依赖模型“自觉”返回 JSON。

#### 方案建议

优先级从高到低：

1. **规则路由优先，LLM 路由兜底**
   - 时间/日期类关键词 → `chat`
   - 数学表达式 → `chat`
   - “推荐几道 / 有什么菜 / 想吃啥” → `recommend`
   - “怎么做 / 做法 / 食材 / 步骤” → `knowledge`

2. **若规则未命中，再调用 LLM 路由**

3. **若仍解析失败，再 fallback 到 `chat`**

#### 验收标准

以下问题不应再出现 `parse_failed`：

- `红烧肉需要哪些食材？`
- `现在几点了？`
- `(12 + 8) * 3 / 2 等于多少？`

---

### 5. 让 `TOOL_USED` 日志只记录本轮调用

#### 问题

当前 `extract_tool_calls(result)` 会遍历 `result["messages"]` 中的全部消息，而 LangGraph + checkpointer 通常返回的是整个线程历史。  
因此当前日志里会把前几轮所有工具都重复打印出来。

#### 修改文件

- `src/services/chat_service.py`

#### 修改目标

只记录“当前这一轮”的 tool calls，而不是整个会话历史。

#### 建议处理方式

对于 `stream()`：

- 优先使用当前已经收集到的 `collected_tool_calls`

对于 `invoke()`：

- 仅提取本轮新增消息中的 tool calls
- 或增加一个辅助函数，只从尾部 AI/tool message 中回溯本轮调用

#### 风险说明

如果继续使用“全历史 tool call”，不仅测试日志会混乱，还可能让：

- `sanitize_ungrounded_kb_claim()`

误以为本轮调用过知识库工具。

#### 验收标准

每发送一条问题：

- `TOOL_USED` 只出现本轮实际调用的工具
- 不再重复打印历史工具

---

## 五、推荐修复顺序

### 第一阶段：保证回答正确性

1. 修 `get_current_time()` 输出格式
2. 修 `recommend_dishes` 低置信度门控
3. 过滤推荐候选中的伪菜名/占位标题

### 第二阶段：保证路由稳定性

4. 给 Supervisor 增加规则路由优先策略
5. 保留 LLM 路由作为兜底

### 第三阶段：提升测试与观察性

6. 修正 `TOOL_USED` 只记录本轮
7. 测试时尽量每轮使用新 `session_id`

---

## 六、回归测试清单

| 问题 | 预期路由 | 预期工具 | 验收点 |
|------|----------|----------|--------|
| `推荐几道适合晚餐做的菜` | `recommend` | `recommend_dishes` | 正常推荐具体菜名 |
| `来几道高难度一点的菜` | `recommend` | `recommend_dishes` | 低置信度时不乱编 |
| `告诉我芹菜拌茶树菇的做法` | `knowledge` | `search_knowledge_base` | 命中知识库详情 |
| `红烧肉需要哪些食材？` | `knowledge` | `search_knowledge_base` | 不再 `parse_failed` |
| `现在几点了？` | `chat` | `get_current_time` | 星期几正确 |
| `(12 + 8) * 3 / 2 等于多少？` | `chat` | `calculate` | 不再 `parse_failed` |

---

## 七、完成标志

满足以下条件后，可认为 multi-agent 方案进入“可稳定联调”状态：

- supervisor 对核心问题类型路由稳定
- recommend / knowledge / chat 三类 specialist 都能按预期工作
- 时间类回答不再出现星期错误
- 推荐类低置信度场景不再乱编
- `TOOL_USED` 日志能够准确反映本轮行为

