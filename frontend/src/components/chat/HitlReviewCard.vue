<template>
  <section v-if="chat.activeHitlInterrupt" class="hitl-card">
    <div>
      <strong>需要人工复核</strong>
      <p>原因：{{ chat.activeHitlInterrupt.reason || "needs_human_validation" }}</p>
    </div>
    <textarea v-model="note" rows="3" placeholder="可选：填写复核意见或修订建议" />
    <div class="actions">
      <button class="primary-btn" type="button" :disabled="chat.isStreaming" @click="submit('approve')">批准</button>
      <button class="ghost-btn" type="button" :disabled="chat.isStreaming" @click="submit('revise')">带意见恢复</button>
      <button class="danger-btn" type="button" :disabled="chat.isStreaming" @click="submit('reject')">拒绝</button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";

import { useChatStore } from "@/stores/chat";

const chat = useChatStore();
const note = ref("");

async function submit(decision: "approve" | "reject" | "revise") {
  await chat.submitHitl(decision, note.value);
  note.value = "";
}
</script>

<style scoped>
.hitl-card {
  align-self: stretch;
  border: 1px solid var(--warning-border);
  border-radius: 14px;
  background: var(--warning-bg);
  color: #92400e;
  padding: 14px;
}

p {
  margin: 6px 0 0;
  font-size: 0.9rem;
}

textarea {
  width: 100%;
  margin-top: 12px;
  border: 1px solid #fcd34d;
  border-radius: 10px;
  padding: 10px;
  resize: vertical;
  background: #fff;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
</style>
