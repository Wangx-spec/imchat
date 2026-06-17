<template>
  <div class="page-shell">
    <div class="sidebar-col">
      <ConversationSidebar class="conversation-panel" />
      <AgentCapabilityPanel class="capability-panel" />
    </div>
    <ChatWindow class="chat-col" />
  </div>
</template>

<script setup lang="ts">
import ChatWindow from "@/components/chat/ChatWindow.vue";
import AgentCapabilityPanel from "@/components/sidebar/AgentCapabilityPanel.vue";
import ConversationSidebar from "@/components/sidebar/ConversationSidebar.vue";
</script>

<style scoped>
.page-shell {
  min-height: 100vh;
  padding: 18px;
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
  height: calc(100vh - 36px);
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
