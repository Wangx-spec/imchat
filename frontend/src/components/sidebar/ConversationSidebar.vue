<template>
  <aside class="sidebar-card">
    <div class="sidebar-header">
      <h1 class="sidebar-title">⚕ Medical Assistant</h1>
      <button class="ghost-btn new-chat" type="button" :disabled="chat.isStreaming" @click="chat.startNewConversation">
        + 新建会话
      </button>
    </div>

    <div class="conv-list">
      <button
        v-for="conversation in chat.conversations"
        :key="conversation.session_id"
        type="button"
        class="conv-item"
        :class="{ active: conversation.session_id === chat.sessionId }"
        @click="chat.switchSession(conversation.session_id)"
      >
        <span class="conv-title">{{ conversation.title || "新会话" }}</span>
        <span class="conv-time">{{ formatTime(conversation.updated_at) }}</span>
      </button>
      <p v-if="!chat.conversations.length" class="empty">暂无会话</p>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { useChatStore } from "@/stores/chat";

const chat = useChatStore();

function formatTime(value?: string | null): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}
</script>

<style scoped>
.sidebar-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid var(--border);
  background: #f8fafc;
}

.sidebar-title {
  margin: 0 0 12px;
  font-size: 1rem;
  color: var(--primary-dark);
}

.new-chat {
  width: 100%;
}

.conv-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.conv-item {
  display: block;
  width: 100%;
  text-align: left;
  border: 1px solid transparent;
  border-radius: 10px;
  background: #f8fafc;
  color: #334155;
  padding: 10px 12px;
  margin-bottom: 6px;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.conv-item:hover {
  border-color: #cbd5e1;
  background: #f1f5f9;
}

.conv-item.active {
  border-color: #93c5fd;
  background: var(--primary-soft);
  color: #1e3a8a;
}

.conv-title {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conv-time {
  display: block;
  margin-top: 2px;
  font-size: 0.72rem;
  color: var(--muted);
}

.empty {
  margin: 16px 8px;
  color: var(--muted);
  font-size: 0.86rem;
}
</style>
