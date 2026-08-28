<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  label?: string
  min?: number
  max?: number
  step?: number
  /** Rendered next to the track; omit to show the raw value. */
  display?: string
}>(), {
  min: 0,
  max: 1,
  step: 0.01,
})

const model = defineModel<number>({ default: 0 })

const percent = computed(() => {
  const span = props.max - props.min
  if (span <= 0)
    return 0
  return Math.min(100, Math.max(0, ((model.value - props.min) / span) * 100))
})
</script>

<template>
  <label class="mk-slider">
    <span v-if="label" class="mk-slider-label">
      {{ label }}
      <em>{{ display ?? model }}</em>
    </span>
    <input
      v-model.number="model"
      type="range"
      :min="min"
      :max="max"
      :step="step"
      :aria-label="label"
      :style="{ '--mk-slider-fill': `${percent}%` }"
    >
  </label>
</template>

<style scoped>
.mk-slider { display: block; }

.mk-slider-label {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 0.35rem;
  color: var(--mk-muted);
  font-size: 0.72rem;
}

.mk-slider-label em {
  color: var(--mk-ink-dim);
  font-size: 0.7rem;
  font-style: normal;
  font-variant-numeric: tabular-nums;
}

input[type='range'] {
  width: 100%;
  height: 0.32rem;
  border-radius: 999px;
  outline: none;
  appearance: none;
  background: linear-gradient(90deg, #b57cff 0%, #ff6fb5 var(--mk-slider-fill), var(--mk-raise) var(--mk-slider-fill));
  cursor: pointer;
}

input[type='range']::-webkit-slider-thumb {
  width: 0.85rem;
  height: 0.85rem;
  border: 0;
  border-radius: 50%;
  appearance: none;
  background: #fff;
  box-shadow: 0 2px 8px rgb(0 0 0 / 0.35);
}

input[type='range']::-moz-range-thumb {
  width: 0.85rem;
  height: 0.85rem;
  border: 0;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 2px 8px rgb(0 0 0 / 0.35);
}
</style>
