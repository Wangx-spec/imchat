<template>
  <footer class="composer">
    <ImagePreviewList />
    <div class="input-row">
      <input ref="fileInput" type="file" accept="image/*" multiple hidden @change="handleFiles" />
      <button class="icon-btn" type="button" title="上传图片" :disabled="chat.isStreaming" @click="fileInput?.click()">
        📎
      </button>
      <VoiceRecorder @transcript="applyTranscript" />
      <textarea
        ref="textInput"
        v-model="draft"
        rows="1"
        placeholder="请输入医学问题（如：脑肿瘤的常见影像学表现有哪些？）"
        :disabled="chat.isStreaming"
        @keydown.enter.exact.prevent="send"
        @input="autoResize"
      />
      <button class="primary-btn send" type="button" :disabled="!canSend || chat.isStreaming" @click="send">发送</button>
    </div>
    <div class="helper-text">支持最多 3 张图片；有图片时自动调用多模态接口。语音功能关闭时会显示后端错误提示。</div>
  </footer>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";

import ImagePreviewList from "@/components/chat/ImagePreviewList.vue";
import VoiceRecorder from "@/components/speech/VoiceRecorder.vue";
import { useChatStore } from "@/stores/chat";

const chat = useChatStore();
const draft = ref("");
const fileInput = ref<HTMLInputElement | null>(null);
const textInput = ref<HTMLTextAreaElement | null>(null);

const canSend = computed(() => draft.value.trim().length > 0 || chat.pendingImages.length > 0);

function handleFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  chat.addImages(Array.from(input.files || []));
  input.value = "";
}

function applyTranscript(text: string) {
  draft.value = text;
  textInput.value?.focus();
}

async function send() {
  if (!canSend.value) return;
  const message = draft.value;
  draft.value = "";
  resetHeight();
  await chat.send(message);
}

function autoResize() {
  const el = textInput.value;
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
}

function resetHeight() {
  const el = textInput.value;
  if (el) el.style.height = "auto";
}
</script>

<style scoped>
.composer {
  border-top: 1px solid var(--border);
  background: #f8fafc;
  padding-bottom: 12px;
}

.input-row {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding: 12px 14px 0;
}

textarea {
  flex: 1;
  min-height: 40px;
  max-height: 150px;
  border: 1px solid #cbd5e1;
  border-radius: 18px;
  background: #fff;
  padding: 10px 14px;
  resize: none;
  line-height: 1.45;
}

textarea:focus {
  outline: none;
  border-color: #93c5fd;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.18);
}

.send {
  min-height: 40px;
}

.helper-text {
  margin: 8px 16px 0;
  color: var(--muted);
  font-size: 0.75rem;
}

@media (max-width: 640px) {
  .input-row {
    align-items: center;
    flex-wrap: wrap;
  }

  textarea {
    order: -1;
    flex-basis: 100%;
  }
}
</style>
