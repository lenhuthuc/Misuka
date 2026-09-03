<script setup lang="ts">
export interface QuickAction {
  id: string
  emoji: string
  label: string
  /** Sent verbatim as the user's message when the chip is picked. */
  prompt: string
}

defineProps<{ actions: QuickAction[], disabled?: boolean }>()

defineEmits<{ (e: 'pick', action: QuickAction): void }>()
</script>

<template>
  <div class="quick-actions">
    <button
      v-for="action in actions"
      :key="action.id"
      class="quick-chip"
      type="button"
      :disabled="disabled"
      @click="$emit('pick', action)"
    >
      <span aria-hidden="true">{{ action.emoji }}</span>
      {{ action.label }}
    </button>
  </div>
</template>

<style scoped>
.quick-actions {
  display: flex;
  flex-wrap: nowrap;
  overflow-x: auto;
  padding: 0 0.95rem 0.5rem;
  gap: 0.4rem;
  scrollbar-width: none;
  -webkit-overflow-scrolling: touch;
}

.quick-actions::-webkit-scrollbar {
  display: none;
}

.quick-chip {
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
  white-space: nowrap;
  border: 1px solid var(--mk-border);
  border-radius: 999px;
  background: var(--mk-raise);
  color: var(--mk-ink-dim);
  cursor: pointer;
  padding: 0.32rem 0.65rem;
  font: inherit;
  font-size: 0.72rem;
  font-weight: 600;
  gap: 0.32rem;
  transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
}

.quick-chip:hover:not(:disabled) {
  border-color: rgb(255 121 193 / 0.34);
  background: rgb(255 111 181 / 0.14);
  transform: translateY(-1px);
}

.quick-chip:disabled { cursor: default; opacity: 0.4; }

@media (max-width: 768px) {
  .quick-actions {
    padding: 0 0.5rem 0.4rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .quick-chip { transition: none; }
  .quick-chip:hover:not(:disabled) { transform: none; }
}
</style>
