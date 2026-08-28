<script setup lang="ts">
defineProps<{ text: string }>()
</script>

<template>
  <span class="mk-tooltip">
    <slot />
    <span class="mk-tooltip-bubble" role="tooltip">{{ text }}</span>
  </span>
</template>

<style scoped>
.mk-tooltip {
  position: relative;
  display: inline-flex;
}

.mk-tooltip-bubble {
  position: absolute;
  bottom: calc(100% + 0.45rem);
  left: 50%;
  z-index: 40;
  border: 1px solid var(--mk-border-strong);
  border-radius: 0.6rem;
  background: var(--mk-float);
  color: var(--mk-ink-dim);
  padding: 0.28rem 0.55rem;
  font-size: 0.68rem;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transform: translate(-50%, 4px);
  transition: opacity 140ms ease, transform 140ms ease;
}

.mk-tooltip:hover .mk-tooltip-bubble,
.mk-tooltip:focus-within .mk-tooltip-bubble {
  opacity: 1;
  transform: translate(-50%, 0);
}

@media (prefers-reduced-motion: reduce) {
  .mk-tooltip-bubble { transition: none; }
}
</style>
