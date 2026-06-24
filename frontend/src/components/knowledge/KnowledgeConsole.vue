<template>
  <section class="knowledge-console">
    <header class="console-header">
      <div>
        <h2>知识库控制台</h2>
        <p>查看 RAG 状态、上传文档、重建索引，并调试召回链路。</p>
      </div>
      <button type="button" class="primary-btn" :disabled="knowledge.loadingStats" @click="knowledge.refreshAll">
        刷新
      </button>
    </header>

    <p v-if="knowledge.error" class="error-box">{{ knowledge.error }}</p>

    <div class="stats-grid">
      <article class="card">
        <span class="label">RAG 状态</span>
        <strong>{{ knowledge.stats?.ready ? "Ready" : "Not Ready" }}</strong>
        <small>{{ knowledge.stats?.reason || knowledge.stats?.index_meta_reason || "暂无原因" }}</small>
      </article>
      <article class="card">
        <span class="label">父文档 / 子块</span>
        <strong>{{ knowledge.stats?.parent_count ?? 0 }} / {{ knowledge.stats?.child_count ?? 0 }}</strong>
        <small>{{ knowledge.stats?.vector_db_provider || "unknown" }}</small>
      </article>
      <article class="card">
        <span class="label">索引</span>
        <strong>{{ knowledge.stats?.index_loaded ? "Loaded" : knowledge.stats?.index_rebuilt ? "Rebuilt" : "Unknown" }}</strong>
        <small>{{ knowledge.stats?.index_dir || "无 index_dir" }}</small>
      </article>
      <article class="card">
        <span class="label">优化能力</span>
        <strong>并行: {{ yesNo(knowledge.stats?.parallel_recall) }}</strong>
        <small>Rerank: {{ yesNo(knowledge.stats?.rerank_enabled) }} / Cache: {{ yesNo(knowledge.stats?.cache_enabled) }}</small>
      </article>
    </div>

    <div class="panel-grid">
      <section class="panel">
        <div class="panel-title">
          <h3>源文件</h3>
          <button type="button" class="ghost-btn" :disabled="knowledge.loadingSources" @click="knowledge.refreshSources">
            刷新列表
          </button>
        </div>
        <div class="source-list">
          <div v-for="item in knowledge.sources.slice(0, 80)" :key="item.path" class="source-item">
            <strong>{{ item.name }}</strong>
            <small>{{ formatSize(item.size) }} · {{ formatTime(item.modified_at) }}</small>
            <code>{{ item.path }}</code>
          </div>
          <p v-if="!knowledge.sources.length" class="empty">暂无源文件或 RAG_SOURCE_DIRS 未配置。</p>
        </div>
      </section>

      <section class="panel">
        <h3>上传与重建</h3>
        <label class="check-row">
          <input v-model="rebuildAfterUpload" type="checkbox" />
          上传后立即重建索引
        </label>
        <label class="field-label">
          <span>PDF 解析方式</span>
          <select v-model="pdfParser" :disabled="knowledge.uploading">
            <option value="light">快速解析（PyMuPDF）</option>
            <option value="marker">深度解析（marker-pdf）</option>
          </select>
        </label>
        <label class="field-label">
          <span>Topic</span>
          <input v-model="uploadTopic" type="text" :disabled="knowledge.uploading" placeholder="uploaded" />
        </label>
        <input type="file" accept=".md,.txt,.json,.pdf" :disabled="knowledge.uploading" @change="handleUpload" />
        <button type="button" class="primary-btn wide" :disabled="knowledge.rebuilding" @click="knowledge.rebuild">
          手动重建索引
        </button>
        <p class="hint">
          支持 .md/.txt/.json/.pdf。文本型 PDF 推荐快速解析；扫描件快速解析可能失败或文本过少；深度解析首次会下载模型，耗时较长。
        </p>
      </section>
    </div>

    <section class="panel">
      <h3>召回调试</h3>
      <div class="search-row">
        <input v-model="query" type="text" placeholder="输入检索问题，例如：持续头痛伴呕吐需要注意什么" />
        <input v-model.number="topK" type="number" min="1" max="20" />
        <label class="check-row inline">
          <input v-model="includeAnswer" type="checkbox" />
          生成答案
        </label>
        <button type="button" class="primary-btn" :disabled="knowledge.searching" @click="submitSearch">
          搜索
        </button>
      </div>

      <div v-if="knowledge.searchResult" class="result-grid">
        <article class="result-card">
          <h4>召回结果</h4>
          <p class="hint">耗时 {{ knowledge.searchResult.elapsed_ms }} ms，来源 {{ knowledge.searchResult.sources.length }} 个。</p>
          <div v-for="(parent, idx) in knowledge.searchResult.parents" :key="`${parent.source}-${idx}`" class="parent-item">
            <strong>{{ idx + 1 }}. {{ parent.title || "未命名片段" }}</strong>
            <code>{{ parent.source || "无来源路径" }}</code>
            <p>{{ parent.content }}</p>
          </div>
          <p v-if="!knowledge.searchResult.parents.length" class="empty">没有召回父文档。</p>
        </article>

        <article class="result-card">
          <h4>调试信息</h4>
          <div class="debug-kv">
            <span>vector_hits</span><strong>{{ debugValue("vector_hits") }}</strong>
            <span>bm25_hits</span><strong>{{ debugValue("bm25_hits") }}</strong>
            <span>fused_hits</span><strong>{{ debugValue("fused_hits") }}</strong>
            <span>recall_parallel</span><strong>{{ debugValue("recall_parallel") }}</strong>
            <span>recall_elapsed_ms</span><strong>{{ debugValue("recall_elapsed_ms") }}</strong>
            <span>confidence_score</span><strong>{{ debugValue("confidence_score") }}</strong>
            <span>qwen_rerank_ok</span><strong>{{ debugValue("qwen_rerank_ok") }}</strong>
            <span>cache_hit</span><strong>{{ debugValue("cache_hit") }}</strong>
          </div>
          <details open>
            <summary>完整 debug JSON</summary>
            <pre>{{ prettyDebug }}</pre>
          </details>
        </article>

        <article v-if="knowledge.searchResult.answer" class="result-card wide-result">
          <h4>答案预览</h4>
          <p class="answer-preview">{{ knowledge.searchResult.answer }}</p>
        </article>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

import { useKnowledgeStore } from "@/stores/knowledge";

const knowledge = useKnowledgeStore();
const query = ref("");
const topK = ref(5);
const includeAnswer = ref(false);
const rebuildAfterUpload = ref(false);
const pdfParser = ref<"light" | "marker">("light");
const uploadTopic = ref("uploaded");

const prettyDebug = computed(() => {
  return JSON.stringify(knowledge.searchResult?.debug || {}, null, 2);
});

onMounted(() => {
  void knowledge.refreshAll();
});

function yesNo(value: unknown): string {
  return value ? "开" : "关";
}

function formatSize(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(ts?: number | null): string {
  if (!ts) return "未知时间";
  return new Date(ts * 1000).toLocaleString();
}

function debugValue(key: string): string {
  const value = knowledge.searchResult?.debug?.[key];
  if (value === undefined || value === null) return "-";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

async function submitSearch() {
  const trimmed = query.value.trim();
  if (!trimmed) {
    knowledge.error = "请输入检索问题";
    return;
  }
  await knowledge.runSearch(trimmed, topK.value, includeAnswer.value);
}

async function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  await knowledge.upload(file, rebuildAfterUpload.value, pdfParser.value, uploadTopic.value.trim() || "uploaded");
  input.value = "";
}
</script>

<style scoped>
.knowledge-console {
  height: calc(100vh - 36px);
  overflow: auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 18px;
}

.console-header,
.panel-title,
.search-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.console-header {
  justify-content: space-between;
  margin-bottom: 16px;
}

.console-header h2,
.panel h3,
.result-card h4 {
  margin: 0;
  color: var(--primary-dark);
}

.console-header p,
.hint,
.empty {
  color: var(--muted);
  font-size: 0.86rem;
}

.stats-grid,
.panel-grid,
.result-grid {
  display: grid;
  gap: 12px;
}

.stats-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-bottom: 12px;
}

.panel-grid,
.result-grid {
  grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
  margin-bottom: 12px;
}

.card,
.panel,
.result-card {
  border: 1px solid var(--border);
  border-radius: 14px;
  background: #f8fafc;
  padding: 14px;
}

.card {
  display: grid;
  gap: 6px;
}

.label,
.debug-kv span {
  color: var(--muted);
  font-size: 0.78rem;
}

.source-list {
  display: grid;
  gap: 8px;
  max-height: 340px;
  overflow: auto;
}

.source-item,
.parent-item {
  display: grid;
  gap: 4px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
}

code {
  color: #475569;
  font-size: 0.75rem;
  overflow-wrap: anywhere;
}

input[type="text"],
input[type="number"] {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
}

input[type="text"] {
  flex: 1;
}

input[type="number"] {
  width: 88px;
}

.primary-btn,
.ghost-btn {
  border: 1px solid transparent;
  border-radius: 10px;
  cursor: pointer;
  padding: 9px 12px;
}

.primary-btn {
  background: var(--primary);
  color: #fff;
}

.ghost-btn {
  background: #fff;
  border-color: var(--border);
  color: #334155;
}

.wide {
  display: block;
  margin-top: 12px;
  width: 100%;
}

.check-row {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #334155;
  font-size: 0.9rem;
}

.inline {
  white-space: nowrap;
}

.error-box {
  border: 1px solid #fecdd3;
  border-radius: 10px;
  background: #fff1f2;
  color: #be123c;
  padding: 10px 12px;
}

.debug-kv {
  display: grid;
  grid-template-columns: 160px minmax(0, 1fr);
  gap: 6px 10px;
  margin-bottom: 12px;
}

pre {
  max-height: 360px;
  overflow: auto;
  background: #0f172a;
  border-radius: 10px;
  color: #e2e8f0;
  font-size: 0.78rem;
  padding: 12px;
}

.wide-result {
  grid-column: 1 / -1;
}

.answer-preview {
  white-space: pre-wrap;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

@media (max-width: 1100px) {
  .stats-grid,
  .panel-grid,
  .result-grid {
    grid-template-columns: 1fr;
  }

  .search-row {
    align-items: stretch;
    flex-direction: column;
  }
}
</style>
