<script setup lang="ts">
withDefaults(defineProps<{
  /** `plain` drops the inner top highlight — used for nested/quiet cards. */
  variant?: 'default' | 'plain' | 'accent'
  padded?: boolean
}>(), {
  variant: 'default',
  padded: true,
})
</script>

<template>
  <div class="glass-card" :class="[`glass-card--${variant}`, { 'glass-card--padded': padded }]">
    <slot />
  </div>
</template>

<style scoped>
.glass-card {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius);
  background: var(--mk-panel);
  backdrop-filter: blur(22px) saturate(1.1);
}

.glass-card--padded { padding: 1rem; }

.glass-card--default::before,
.glass-card--accent::before {
  position: absolute;
  top: 0;
  left: 10%;
  width: 46%;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--mk-pink), var(--mk-purple), transparent);
  content: '';
  opacity: 0.55;
}

.glass-card--accent {
  border-color: rgb(255 121 193 / 0.22);
  background:
    linear-gradient(150deg, rgb(171 125 255 / 0.16), rgb(255 111 181 / 0.1)),
    var(--mk-panel);
}
</style>
