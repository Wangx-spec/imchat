<template>
  <main class="chat-card">
    <header class="chat-header">
      <div>
        <h2>医学多 Agent 助手</h2>
        <p>内容仅供参考，不构成诊疗建议</p>
      </div>
      <div class="badges">
        <StatusBadge tone="muted">{{ chat.healthStatus?.runtime || "runtime unknown" }}</StatusBadge>
        <StatusBadge :tone="chat.healthStatus?.rag_ready ? 'ok' : 'warn'">
          RAG {{ chat.healthStatus?.rag_ready ? "ready" : chat.healthStatus?.rag_reason || "unknown" }}
        </StatusBadge>
        <StatusBadge tone="muted">{{ chat.sessionLabel }}</StatusBadge>
      </div>
    </header>

    <section ref="scrollBox" class="chat-body">
      <MessageBubble v-for="message in chat.messages" :key="message.id" :message="message" />
      <HitlReviewCard />
    </section>

    <Composer />
  </main>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";

import StatusBadge from "@/components/common/StatusBadge.vue";
import Composer from "@/components/chat/Composer.vue";
import HitlReviewCard from "@/components/chat/HitlReviewCard.vue";
import MessageBubble from "@/components/chat/MessageBubble.vue";
import { useChatStore } from "@/stores/chat";

const chat = useChatStore();
const scrollBox = ref<HTMLElement | null>(null);

onMounted(() => {
  void chat.initialize();
});

watch(
  () => [
    chat.messages.length,
    chat.messages.length ? chat.messages[chat.messages.length - 1].content : "",
    chat.activeHitlInterrupt,
  ],
  async () => {
    await nextTick();
    if (scrollBox.value) scrollBox.value.scrollTop = scrollBox.value.scrollHeight;
  },
  { deep: true },
);
</script>

<style scoped>
.chat-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--border);
  background: #f8fafc;
  padding: 14px 18px;
}

h2 {
  margin: 0;
  color: #334155;
  font-size: 1rem;
}

p {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 0.78rem;
}

.badges {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.chat-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 16px;
  background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

@media (max-width: 720px) {
  .chat-header {
    align-items: flex-start;
    flex-direction: column;
  }

  .badges {
    justify-content: flex-start;
  }
}
</style>
