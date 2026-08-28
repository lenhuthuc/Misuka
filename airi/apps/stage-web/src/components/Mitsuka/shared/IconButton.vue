<script setup lang="ts">
withDefaults(defineProps<{
  /** UnoCSS icon class, e.g. `i-solar:microphone-3-outline`. */
  icon: string
  label: string
  active?: boolean
  disabled?: boolean
  size?: 'sm' | 'md'
  variant?: 'default' | 'accent'
}>(), {
  active: false,
  disabled: false,
  size: 'md',
  variant: 'default',
})
</script>

<template>
  <button
    class="icon-button"
    :class="[`icon-button--${size}`, `icon-button--${variant}`, { 'icon-button--active': active }]"
    type="button"
    :title="label"
    :aria-label="label"
    :aria-pressed="active"
    :disabled="disabled"
  >
    <span :class="icon" />
  </button>
</template>

<style scoped>
.icon-button {
  display: grid;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius-sm);
  background: var(--mk-raise);
  color: var(--mk-muted);
  cursor: pointer;
  transition: transform 150ms ease, background 150ms ease, color 150ms ease, border-color 150ms ease;
}

.icon-button--sm { width: 1.9rem; height: 1.9rem; font-size: 0.92rem; }
.icon-button--md { width: 2.25rem; height: 2.25rem; font-size: 1.05rem; }

.icon-button:hover:not(:disabled) {
  border-color: var(--mk-border-strong);
  background: var(--mk-hover);
  color: var(--mk-ink);
  transform: translateY(-1px);
}

.icon-button--active {
  border-color: rgb(255 121 193 / 0.42);
  background: rgb(255 111 181 / 0.2);
  color: var(--mk-pink-soft);
  box-shadow: var(--mk-glow-pink);
}

.icon-button--accent {
  border-color: transparent;
  background: var(--mk-accent);
  color: #fff;
}

.icon-button:disabled { cursor: default; opacity: 0.35; }

@media (prefers-reduced-motion: reduce) {
  .icon-button { transition: none; }
  .icon-button:hover:not(:disabled) { transform: none; }
}
</style>
