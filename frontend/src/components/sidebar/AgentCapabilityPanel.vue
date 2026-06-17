<template>
  <section class="cap-card">
    <h2>Available Agents</h2>
    <ul>
      <li>💬 医学咨询 Agent</li>
      <li>🧭 诊断辅助 Agent</li>
      <li>📚 医学 RAG Agent</li>
      <li>🌐 Web Search Agent</li>
      <li>🖼 多模态图像分析</li>
    </ul>

    <h2>Current Capabilities</h2>
    <ul>
      <li>LangGraph 多 Agent 编排</li>
      <li>FAISS/Qdrant 可切换向量检索</li>
      <li>Guardrails + Harness 输出约束</li>
      <li>Human-in-the-loop 复核</li>
      <li>ElevenLabs STT/TTS</li>
      <li>Mem0 长期记忆预留</li>
    </ul>

    <p class="disclaimer">输出仅供学习与信息参考，不能替代专业医生面诊或诊疗。</p>
  </section>
</template>

<style scoped>
.cap-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 16px;
  overflow-y: auto;
}

h2 {
  margin: 0 0 10px;
  color: var(--primary-dark);
  font-size: 0.95rem;
}

h2:not(:first-child) {
  margin-top: 18px;
}

ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

li {
  color: #334155;
  margin-bottom: 8px;
  line-height: 1.4;
  font-size: 0.9rem;
}

.disclaimer {
  margin: 18px 0 0;
  border: 1px solid #fde68a;
  border-radius: 12px;
  background: #fffbeb;
  color: #92400e;
  padding: 10px;
  font-size: 0.82rem;
  line-height: 1.45;
}
</style>
