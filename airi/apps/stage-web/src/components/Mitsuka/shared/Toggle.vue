<script setup lang="ts">
const props = defineProps<{ label?: string, disabled?: boolean }>()
const model = defineModel<boolean>({ default: false })

function toggle() {
  if (!props.disabled)
    model.value = !model.value
}
</script>

<template>
  <button
    class="mk-toggle"
    :class="{ 'mk-toggle--on': model }"
    type="button"
    role="switch"
    :aria-checked="model"
    :aria-label="label"
    :disabled="disabled"
    @click="toggle"
  >
    <span class="mk-toggle-knob" />
  </button>
</template>

<style scoped>
.mk-toggle {
  position: relative;
  width: 2.65rem;
  height: 1.5rem;
  flex: 0 0 auto;
  border: 1px solid var(--mk-border);
  border-radius: 999px;
  background: var(--mk-raise);
  cursor: pointer;
  transition: background 180ms ease, border-color 180ms ease;
}

.mk-toggle--on {
  border-color: transparent;
  background: var(--mk-accent);
}

.mk-toggle-knob {
  position: absolute;
  top: 50%;
  left: 0.16rem;
  width: 1.05rem;
  height: 1.05rem;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 2px 6px rgb(0 0 0 / 0.3);
  transform: translateY(-50%);
  transition: left 180ms ease;
}

.mk-toggle--on .mk-toggle-knob { left: calc(100% - 1.21rem); }
.mk-toggle:disabled { cursor: default; opacity: 0.4; }

@media (prefers-reduced-motion: reduce) {
  .mk-toggle, .mk-toggle-knob { transition: none; }
}
</style>
