<template>
  <div v-if="chat.pendingImages.length" class="image-preview">
    <div v-for="image in chat.pendingImages" :key="image.id" class="image-chip">
      <img :src="image.objectUrl" :alt="image.file.name" />
      <span class="image-name" :title="image.file.name">{{ image.file.name }}</span>
      <button type="button" @click="chat.removeImage(image.id)">移除</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useChatStore } from "@/stores/chat";

const chat = useChatStore();
</script>

<style scoped>
.image-preview {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  border-top: 1px solid #f1f5f9;
  background: #fff;
  padding: 10px 16px 0;
}

.image-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  max-width: 280px;
  border: 1px solid #dbeafe;
  border-radius: 999px;
  background: var(--primary-soft);
  padding: 4px 8px 4px 4px;
}

img {
  width: 34px;
  height: 34px;
  object-fit: cover;
  border-radius: 999px;
  border: 1px solid var(--primary-border);
}

.image-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #1e3a8a;
  font-size: 0.78rem;
}

button {
  border: 0;
  border-radius: 999px;
  background: #bfdbfe;
  color: #1e3a8a;
  padding: 2px 7px;
  font-size: 0.74rem;
}
</style>
