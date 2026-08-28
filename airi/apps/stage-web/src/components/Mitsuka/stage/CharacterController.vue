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
  listening: 'Đang lắng nghe',
  transcribing: 'Đang nhận giọng nói',
  thinking: 'Đang suy nghĩ',
  speaking: 'Đang nói',
}

const stateLabel = computed(() => STATE_LABEL[props.state])

/** V/A/D arrive on the signed [-1, 1] scale — map to a 0–100 bar. */
function toPercent(value: number) {
  return Math.round(((Math.min(1, Math.max(-1, value)) + 1) / 2) * 100)
}

const meters = computed(() => [
  { key: 'v', label: 'Cảm xúc', value: props.emotion.v },
  { key: 'a', label: 'Năng lượng', value: props.emotion.a },
])
</script>

<template>
  <div class="character-controller">
    <div class="cc-state">
      <span class="cc-dot" :class="`cc-dot--${state}`" />
      <span>{{ stateLabel }}</span>
    </div>
    <div v-for="meter in meters" :key="meter.key" class="cc-meter">
      <span class="cc-meter-label">{{ meter.label }}</span>
      <span class="cc-meter-track">
        <i :style="{ width: `${toPercent(meter.value)}%` }" />
      </span>
      <span class="cc-meter-value">{{ (meter.value).toFixed(2) }}</span>
    </div>
  </div>
</template>

<style scoped>
.character-controller {
  display: flex;
  min-width: 11.5rem;
  flex-direction: column;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius);
  background: var(--mk-hud);
  padding: 0.6rem 0.7rem;
  gap: 0.45rem;
  backdrop-filter: blur(18px);
}

.cc-state {
  display: flex;
  align-items: center;
  color: var(--mk-ink-dim);
  font-size: 0.72rem;
  font-weight: 700;
  gap: 0.4rem;
}

.cc-dot {
  width: 0.42rem;
  height: 0.42rem;
  border-radius: 50%;
  background: var(--mk-muted);
}

.cc-dot--listening { background: #7dd3fc; box-shadow: 0 0 0.5rem #38bdf8; }
.cc-dot--transcribing { background: #fcd34d; box-shadow: 0 0 0.5rem #fbbf24; }
.cc-dot--thinking { background: var(--mk-purple-soft); box-shadow: 0 0 0.5rem var(--mk-purple); }
.cc-dot--speaking { background: var(--mk-pink); box-shadow: var(--mk-glow-pink); }

.cc-meter {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.cc-meter-label {
  width: 4.1rem;
  flex: 0 0 auto;
  color: var(--mk-muted);
  font-size: 0.65rem;
}

.cc-meter-track {
  position: relative;
  overflow: hidden;
  height: 0.28rem;
  flex: 1;
  border-radius: 999px;
  background: rgb(255 255 255 / 0.1);
}

.cc-meter-track i {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: var(--mk-accent);
  transition: width 320ms ease;
}

.cc-meter-value {
  flex: 0 0 auto;
  color: var(--mk-ink-dim);
  font-size: 0.62rem;
  font-variant-numeric: tabular-nums;
  opacity: 0.75;
}

@media (prefers-reduced-motion: reduce) {
  .cc-meter-track i { transition: none; }
}
</style>
