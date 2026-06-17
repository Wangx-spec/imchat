<template>
  <span class="status-badge" :class="tone">
    <span class="dot" />
    <slot />
  </span>
</template>

<script setup lang="ts">
defineProps<{
  tone: "ok" | "warn" | "muted";
}>();
</script>

<style scoped>
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  border: 1px solid var(--border);
  padding: 5px 9px;
  font-size: 0.78rem;
  color: var(--muted);
  background: #fff;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: currentColor;
}

.ok {
  color: #047857;
  background: #ecfdf5;
  border-color: #a7f3d0;
}

.warn {
  color: #b45309;
  background: #fffbeb;
  border-color: #fde68a;
}

.muted {
  color: #64748b;
  background: #f8fafc;
  border-color: #e2e8f0;
}
</style>
