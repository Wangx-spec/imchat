<template>
  <article class="message" :class="[message.role, message.status]">
    <div class="agent-tag">{{ tag }}</div>
    <ThinkingIndicator v-if="message.status === 'streaming' && !message.content" />
    <div v-else class="markdown-body" v-html="html" />

    <div v-if="message.images?.length" class="attached-images">
      <img v-for="image in message.images" :key="image.id" :src="image.objectUrl" :alt="image.file.name" />
    </div>

    <TtsButton v-if="message.role === 'assistant' && message.content" :text="plainText" />
  </article>
</template>

<script setup lang="ts">
import { marked } from "marked";
import { computed } from "vue";

import type { ChatMessage } from "@/api/types";
import TtsButton from "@/components/speech/TtsButton.vue";
import ThinkingIndicator from "./ThinkingIndicator.vue";

const props = defineProps<{
  message: ChatMessage;
}>();

const tag = computed(() => {
  if (props.message.role === "user") return "User";
  if (props.message.role === "system") return "System";
  if (props.message.status === "hitl_waiting") return "HITL Required";
  return "Assistant";
});

const html = computed(() => marked.parse(props.message.content || "") as string);
const plainText = computed(() => props.message.content.replace(/[#*_>`\[\]()]/g, "").trim());
</script>

<style scoped>
.message {
  max-width: min(82%, 880px);
  border: 1px solid transparent;
  border-radius: 16px;
  padding: 10px 13px;
  line-height: 1.55;
  word-break: break-word;
}

.message.user {
  align-self: flex-end;
  background: var(--user);
  border-color: #bae6fd;
  color: #0c4a6e;
  border-bottom-right-radius: 4px;
}

.message.assistant,
.message.system {
  align-self: flex-start;
  background: var(--assistant);
  border-color: #e2e8f0;
  color: #0f172a;
  border-bottom-left-radius: 4px;
}

.message.error {
  border-color: #fecaca;
  background: #fef2f2;
}

.message.hitl_waiting {
  border-color: var(--warning-border);
  background: var(--warning-bg);
}

.agent-tag {
  display: inline-block;
  margin-bottom: 6px;
  border-radius: 999px;
  background: #ebf5ff;
  color: var(--primary-dark);
  padding: 2px 8px;
  font-size: 0.76rem;
  font-weight: 800;
}

.attached-images {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.attached-images img {
  max-width: 180px;
  max-height: 180px;
  border-radius: 10px;
  border: 1px solid #dbeafe;
  object-fit: cover;
}
</style>
