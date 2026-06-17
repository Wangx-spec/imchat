<template>
  <button
    class="icon-btn voice-btn"
    :class="{ recording }"
    type="button"
    :disabled="processing"
    :title="recording ? '停止录音' : '语音输入'"
    @click="toggleRecording"
  >
    {{ icon }}
  </button>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";

import { speechToText } from "@/api/speech";
import { useChatStore } from "@/stores/chat";

const emit = defineEmits<{
  transcript: [text: string];
}>();

const chat = useChatStore();
const recording = ref(false);
const processing = ref(false);
let recorder: MediaRecorder | null = null;
let chunks: Blob[] = [];

const icon = computed(() => {
  if (processing.value) return "...";
  return recording.value ? "■" : "🎙";
});

async function toggleRecording() {
  if (recording.value) {
    stopRecording();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onstop = async () => {
      processing.value = true;
      stream.getTracks().forEach((track) => track.stop());
      try {
        const blob = new Blob(chunks, { type: recorder?.mimeType || "audio/webm" });
        const text = await speechToText(blob);
        if (text) emit("transcript", text);
      } catch (error) {
        chat.pushSystemMessage(`语音识别失败：${error instanceof Error ? error.message : String(error)}`);
      } finally {
        processing.value = false;
        recorder = null;
        chunks = [];
      }
    };
    recorder.start();
    recording.value = true;
  } catch {
    chat.pushSystemMessage("无法访问麦克风，请检查浏览器权限。");
  }
}

function stopRecording() {
  if (!recorder || recorder.state === "inactive") return;
  recording.value = false;
  recorder.stop();
}
</script>

<style scoped>
.voice-btn {
  background: #fff7ed;
  border-color: #fed7aa;
  color: #c2410c;
}

.voice-btn.recording {
  background: #fee2e2;
  border-color: #fecaca;
  color: #b91c1c;
}
</style>
