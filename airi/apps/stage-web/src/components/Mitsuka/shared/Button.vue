<script setup lang="ts">
withDefaults(defineProps<{
  variant?: 'primary' | 'ghost' | 'soft'
  size?: 'sm' | 'md'
  type?: 'button' | 'submit'
  disabled?: boolean
}>(), {
  variant: 'primary',
  size: 'md',
  type: 'button',
  disabled: false,
})
</script>

<template>
  <button
    class="mk-button"
    :class="[`mk-button--${variant}`, `mk-button--${size}`]"
    :type="type"
    :disabled="disabled"
  >
    <slot />
  </button>
</template>

<style scoped>
.mk-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  border: 1px solid transparent;
  border-radius: 999px;
  cursor: pointer;
  font: inherit;
  font-weight: 700;
  transition: transform 150ms ease, background 150ms ease, box-shadow 150ms ease, opacity 150ms ease;
}

.mk-button--sm { padding: 0.34rem 0.75rem; font-size: 0.74rem; }
.mk-button--md { padding: 0.5rem 1rem; font-size: 0.82rem; }

.mk-button--primary {
  background: var(--mk-accent);
  color: #fff;
  box-shadow: 0 10px 26px rgb(146 68 200 / 0.3);
}

.mk-button--soft {
  border-color: var(--mk-border-strong);
  background: var(--mk-raise);
  color: var(--mk-ink-dim);
}

.mk-button--ghost {
  background: transparent;
  color: var(--mk-muted);
}

.mk-button:hover:not(:disabled) { transform: translateY(-1px); }
.mk-button--soft:hover:not(:disabled),
.mk-button--ghost:hover:not(:disabled) { background: var(--mk-hover); color: var(--mk-ink); }
.mk-button:disabled { cursor: default; opacity: 0.4; }

@media (prefers-reduced-motion: reduce) {
  .mk-button { transition: none; }
  .mk-button:hover:not(:disabled) { transform: none; }
}
</style>
