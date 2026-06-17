<template>
  <button class="tts-btn" type="button" :disabled="!text.trim() || loading" @click="toggle">
    {{ label }}
  </button>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";

import { textToSpeech } from "@/api/speech";
import { useChatStore } from "@/stores/chat";

const props = defineProps<{
  text: string;
}>();

const chat = useChatStore();
const loading = ref(false);
const playing = ref(false);
let audioUrl = "";
let localAudio: HTMLAudioElement | null = null;

const label = computed(() => {
  if (loading.value) return "生成语音中...";
  if (playing.value) return "暂停语音";
  return "播放语音";
});

async function toggle() {
  if (localAudio && !localAudio.paused) {
    localAudio.pause();
    playing.value = false;
    return;
  }

  if (localAudio && localAudio.paused) {
    await localAudio.play();
    playing.value = true;
    return;
  }

  loading.value = true;
  try {
    const text = props.text.length > 1000 ? `${props.text.slice(0, 1000)}...` : props.text;
    audioUrl = await textToSpeech(text);
    localAudio = new Audio(audioUrl);
    chat.setCurrentAudio(localAudio);
    localAudio.onended = cleanup;
    localAudio.onpause = () => {
      playing.value = false;
    };
    localAudio.onplay = () => {
      playing.value = true;
    };
    await localAudio.play();
  } catch (error) {
    chat.pushSystemMessage(`语音播放失败：${error instanceof Error ? error.message : String(error)}`);
  } finally {
    loading.value = false;
  }
}

function cleanup() {
  playing.value = false;
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = "";
  localAudio = null;
}
</script>

<style scoped>
.tts-btn {
  margin-top: 10px;
  border: 1px solid var(--primary-border);
  border-radius: 999px;
  background: #fff;
  color: var(--primary-dark);
  font-size: 0.78rem;
  font-weight: 700;
  padding: 6px 10px;
}
</style>
