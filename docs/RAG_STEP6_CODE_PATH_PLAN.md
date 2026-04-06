# Step 6 技术代码路径方案（main + web/app 启动接入）

目标：在 Step5（工具已可被 Agent 调用）基础上，完成“应用启动接入闭环”：

- CLI 与 Web 启动时按配置初始化 RAG
- 初始化成功后注入工具层（`set_rag_service`）
- 初始化失败可降级，不阻塞主程序
- 输出可诊断日志，便于排障

对应主文档：`docs/RAG_IMPLEMENTATION.md` 的 Step 6。

---

## 1. 本阶段范围

仅实现：

- `src/main.py`
- `src/web/app.py`

依赖输入（来自 Step1~5）：

- `load_settings`
- `build_rag_config` / `sanitize_rag_config` / `validate_rag_config`
- `RAGService`
- `set_rag_service`

不在本阶段实现：

- 依赖补齐与自动化测试收口（Step7）

---

## 2. 代码路径（调用关系）

Step6 完成后的最小调用链：

1. 启动入口读取 `settings = load_settings()`
2. 判断 `settings.rag_enabled`
3. 若启用：
   - 构建并校验 RAG 配置
   - 创建 `RAGService(cfg)` 并 `initialize(...)`
   - 调用 `set_rag_service(service)` 注入工具层
4. 若异常：
   - 记录 `rag_bootstrap_failed` 日志
   - 继续主流程（CLI/Web 正常可用）
5. 构建 Agent 并进入正常请求处理

---

## 3. 文件级实现方案

## 3.1 `src/main.py`（CLI）

### 3.1.1 推荐新增函数

- `bootstrap_rag(settings) -> tuple[bool, str]`
  - 返回 `(ok, reason)`，用于日志和状态展示

### 3.1.2 启动接入顺序

1. 加载 settings。
2. `if settings.rag_enabled: bootstrap_rag(settings)`。
3. 无论 bootstrap 成败，都继续构建并启动 CLI 对话。

### 3.1.3 日志建议

- 成功：`rag_bootstrap_ok`
- 跳过：`rag_bootstrap_skipped`（未启用）
- 失败：`rag_bootstrap_failed`（附错误摘要）

---

## 3.2 `src/web/app.py`（FastAPI）

### 3.2.1 推荐接入点

- 放在应用启动事件（如 `startup`）中执行 `bootstrap_rag(settings)`。

### 3.2.2 要点

- 不应因 RAG 初始化失败导致 FastAPI 启动失败。
- 可在全局状态记录 `rag_ready`，用于健康检查或调试接口。

### 3.2.3 可选增强

- 提供 `/health` 或已有健康接口中增加 `rag_ready/rag_reason` 字段。

---

## 4. `bootstrap_rag` 统一逻辑建议

统一逻辑（CLI/Web 共用）：

1. `cfg = build_rag_config(settings)`
2. `cfg = sanitize_rag_config(cfg)`
3. `validate_rag_config(cfg)`
4. `service = RAGService(cfg)`
5. `service.initialize(force_rebuild=cfg.rebuild)`
6. `set_rag_service(service)`
7. 返回成功状态

建议将这段逻辑抽为公共函数，避免 `main.py` 与 `web/app.py` 复制粘贴。

---

## 5. 关键伪代码（可直接映射）

```python
# src/main.py / src/web/app.py (shared idea)
def bootstrap_rag(settings):
    if not settings.rag_enabled:
        return False, "disabled"

    cfg = build_rag_config(settings)
    cfg = sanitize_rag_config(cfg)
    validate_rag_config(cfg)

    service = RAGService(cfg)
    service.initialize(force_rebuild=cfg.rebuild)
    set_rag_service(service)
    return True, "ok"
```

```python
# src/main.py
def startup():
    settings = load_settings()
    try:
        ok, reason = bootstrap_rag(settings)
        logger.info("rag_bootstrap", extra={"ok": ok, "reason": reason})
    except Exception as exc:
        logger.warning("rag_bootstrap_failed", extra={"error": str(exc)})
    # continue normal CLI flow
```

```python
# src/web/app.py
@app.on_event("startup")
def on_startup():
    settings = load_settings()
    try:
        ok, reason = bootstrap_rag(settings)
        app.state.rag_ready = ok
        app.state.rag_reason = reason
    except Exception as exc:
        app.state.rag_ready = False
        app.state.rag_reason = str(exc)
        logger.warning("rag_bootstrap_failed", extra={"error": str(exc)})
    # continue starting API server
```

---

## 6. 最小可运行自测（Step6）

## 6.1 建议脚本

- `src/tests/tmp_step6_cli_bootstrap_smoke.py`
- `src/tests/tmp_step6_web_bootstrap_smoke.py`

## 6.2 CLI 自测流程

1. `RAG_ENABLED=false` 启动：应可进入对话（且不报错）。
2. `RAG_ENABLED=true` 且配置正确：应打印 `rag_bootstrap_ok`。
3. `RAG_ENABLED=true` 且故意给错路径/API 配置：应打印 `rag_bootstrap_failed`，但 CLI 仍可启动。

## 6.3 Web 自测流程

1. 启动 FastAPI 服务。
2. 观察 startup 日志中的 RAG bootstrap 状态。
3. 若有健康接口，确认 `rag_ready` 与预期一致。
4. 发起一次文档类提问，确认可触发 `search_knowledge_base`。

## 6.4 运行命令（示例）

```bash
PYTHONPATH=src python src/main.py
```

```bash
PYTHONPATH=src uvicorn src.web.app:app --reload
```

---

## 7. 验收标准（DoD）

- [ ] CLI 启动路径支持 `RAG_ENABLED` 开关。
- [ ] Web 启动路径支持 `RAG_ENABLED` 开关。
- [ ] RAG 初始化成功时完成 `set_rag_service` 注入。
- [ ] RAG 初始化失败不会阻断 CLI/Web 主流程。
- [ ] 启动日志能区分“成功/跳过/失败”状态。

---

## 8. 风险与规避

- **启动时长上升**：首次建索引较慢，建议保留索引复用并打印耗时。
- **重复实现风险**：CLI/Web 各写一份 bootstrap 容易漂移，建议抽公共函数。
- **异常吞没风险**：不能只 `except` 不记录，至少保留 error 摘要日志。
- **未注入误判风险**：初始化失败后工具会走“未初始化”降级文案，需在日志中明确原因。

---

## 9. 与 Step7 的接口约定

Step7 测试应基于 Step6 的稳定行为编写：

- `RAG_ENABLED=false`：工具可降级但系统正常
- `RAG_ENABLED=true` + 正常配置：工具可用并可返回来源
- `RAG_ENABLED=true` + 异常配置：系统可继续运行且日志可追踪

建议在 Step7 集成测试中覆盖 CLI 与 Web 两条启动链路。

