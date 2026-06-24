<template>
  <div class="app-root">
    <div class="mode-tabs">
      <button type="button" :class="{ active: mode === 'chat' }" @click="mode = 'chat'">
        智能对话
      </button>
      <button type="button" :class="{ active: mode === 'knowledge' }" @click="mode = 'knowledge'">
        知识库控制台
      </button>
    </div>

    <div v-if="mode === 'chat'" class="page-shell">
      <div class="sidebar-col">
        <ConversationSidebar class="conversation-panel" />
        <AgentCapabilityPanel class="capability-panel" />
      </div>
      <ChatWindow class="chat-col" />
    </div>
    <KnowledgeConsole v-else />
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";

import ChatWindow from "@/components/chat/ChatWindow.vue";
import KnowledgeConsole from "@/components/knowledge/KnowledgeConsole.vue";
import AgentCapabilityPanel from "@/components/sidebar/AgentCapabilityPanel.vue";
import ConversationSidebar from "@/components/sidebar/ConversationSidebar.vue";

const mode = ref<"chat" | "knowledge">("chat");
</script>

<style scoped>
.app-root {
  min-height: 100vh;
  padding: 18px;
}

.mode-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.mode-tabs button {
  border: 1px solid var(--border);
  border-radius: 999px;
  background: #fff;
  color: #334155;
  cursor: pointer;
  padding: 8px 14px;
}

.mode-tabs button.active {
  background: var(--primary);
  border-color: var(--primary);
  color: #fff;
}

.page-shell {
  display: grid;
  grid-template-columns: minmax(260px, 320px) minmax(0, 1fr);
  gap: 16px;
}

.sidebar-col {
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(260px, 1fr) auto;
  gap: 16px;
}

.chat-col {
  min-width: 0;
}

.conversation-panel,
.chat-col {
  height: calc(100vh - 84px);
}

.capability-panel {
  max-height: 40vh;
}

@media (max-width: 960px) {
  .page-shell {
    grid-template-columns: 1fr;
    height: auto;
  }

  .conversation-panel,
  .chat-col {
    height: auto;
    min-height: 420px;
  }
}
</style>
