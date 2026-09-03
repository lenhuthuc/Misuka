<script setup lang="ts">
import type { LocalConvState } from '@proj-airi/stage-ui/composables/local-conversation'
import type { EmotionState } from '@proj-airi/stage-ui/stores/modules/emotion'

import { computed } from 'vue'

const props = defineProps<{
  state: LocalConvState
  emotion: EmotionState
}>()

const STATE_LABEL: Record<LocalConvState, string> = {
  idle: 'Sẵn sàng',
  listening: 'Đang nghe…',
  transcribing: 'Đang nhận diện…',
  thinking: 'Đang suy nghĩ…',
  speaking: 'Đang nói…',
}

const stateLabel = computed(() => STATE_LABEL[props.state] || 'Sẵn sàng')
</script>

<template>
  <div class="character-status-badge" :class="`state-${state}`">
    <span class="status-dot" :class="`status-dot--${state}`" />
    <span class="status-text">{{ stateLabel }}</span>
  </div>
</template>

<style scoped>
.character-status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  background: var(--mk-hud, rgba(20, 10, 32, 0.75));
  border: 1px solid var(--mk-border, rgba(255, 255, 255, 0.12));
  backdrop-filter: blur(18px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
  user-select: none;
  transition: all 250ms ease;
}

.status-dot {
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 0.5rem #10b981;
  transition: all 250ms ease;
}

.status-dot--idle {
  background: #10b981;
  box-shadow: 0 0 0.4rem #10b981;
}

.status-dot--listening {
  background: #38bdf8;
  box-shadow: 0 0 0.6rem #38bdf8;
  animation: pulse-ring 1.4s infinite;
}

.status-dot--transcribing {
  background: #fbbf24;
  box-shadow: 0 0 0.6rem #fbbf24;
}

.status-dot--thinking {
  background: #c084fc;
  box-shadow: 0 0 0.6rem #a855f7;
  animation: pulse-ring 1.2s infinite;
}

.status-dot--speaking {
  background: #f43f5e;
  box-shadow: 0 0 0.7rem #f43f5e;
  animation: pulse-ring 1s infinite;
}

.status-text {
  font-size: 0.74rem;
  font-weight: 600;
  color: var(--mk-ink-dim, #e2e8f0);
  letter-spacing: 0.01em;
}

@keyframes pulse-ring {
  0% { transform: scale(0.95); opacity: 0.8; }
  50% { transform: scale(1.25); opacity: 1; }
  100% { transform: scale(0.95); opacity: 0.8; }
}
</style>
