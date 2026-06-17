<template>
  <span class="thinking" aria-label="正在思考">
    <span />
    <span />
    <span />
  </span>
</template>

<style scoped>
.thinking {
  display: inline-flex;
  gap: 5px;
  align-items: center;
  min-height: 22px;
}

.thinking span {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--primary);
  animation: pulse 1.35s infinite ease-in-out;
}

.thinking span:nth-child(2) {
  animation-delay: 0.2s;
}

.thinking span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes pulse {
  0%,
  100% {
    transform: scale(0.82);
    opacity: 0.45;
  }
  50% {
    transform: scale(1.18);
    opacity: 1;
  }
}
</style>
