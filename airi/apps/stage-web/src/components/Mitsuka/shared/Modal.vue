<script setup lang="ts">
import { useTheme } from '@proj-airi/ui'
import { onKeyStroke } from '@vueuse/core'

withDefaults(defineProps<{
  title?: string
  /** Widen for the multi-column settings panel. */
  wide?: boolean
}>(), { wide: false })

const open = defineModel<boolean>({ default: false })

// The panel is teleported out of `.mitsuka-app`, so it has to re-declare both
// the token scope and the light variant itself.
const { isDark } = useTheme()

onKeyStroke('Escape', () => {
  if (open.value)
    open.value = false
})
</script>

<template>
  <Teleport to="body">
    <Transition name="mk-modal">
      <div
        v-if="open"
        class="mk-modal-root mitsuka-app"
        :class="{ 'mitsuka-app--light': !isDark }"
        role="dialog"
        aria-modal="true"
        :aria-label="title"
      >
        <div class="mk-modal-scrim" @click="open = false" />
        <div class="mk-modal-panel" :class="{ 'mk-modal-panel--wide': wide }">
          <header v-if="title || $slots.header" class="mk-modal-head">
            <slot name="header">
              <h2>{{ title }}</h2>
            </slot>
            <button class="mk-modal-close" type="button" aria-label="Đóng" @click="open = false">
              <span class="i-solar:close-circle-outline" />
            </button>
          </header>

          <div class="mk-modal-body mk-scroll">
            <slot />
          </div>

          <footer v-if="$slots.footer" class="mk-modal-foot">
            <slot name="footer" />
          </footer>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.mk-modal-root {
  position: fixed;
  z-index: 120;
  display: grid;
  inset: 0;
  place-items: center;
  padding: 1rem;
}

.mk-modal-scrim {
  position: absolute;
  inset: 0;
  background: rgb(6 2 12 / 0.62);
  backdrop-filter: blur(6px);
}

.mk-modal-panel {
  position: relative;
  display: flex;
  overflow: hidden;
  width: min(30rem, 100%);
  max-height: min(38rem, 88dvh);
  flex-direction: column;
  border: 1px solid var(--mk-border-strong);
  border-radius: var(--mk-radius-lg);
  background: var(--mk-modal);
  box-shadow: var(--mk-shadow);
}

.mk-modal-panel--wide { width: min(52rem, 100%); }

.mk-modal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--mk-border);
  padding: 0.95rem 1.1rem;
  gap: 1rem;
}

.mk-modal-head h2 {
  margin: 0;
  font-size: 0.98rem;
  font-weight: 800;
}

.mk-modal-close {
  display: grid;
  width: 1.9rem;
  height: 1.9rem;
  flex: 0 0 auto;
  place-items: center;
  border: 0;
  border-radius: var(--mk-radius-sm);
  background: transparent;
  color: var(--mk-muted);
  cursor: pointer;
  font-size: 1.15rem;
  line-height: 1;
}

.mk-modal-close:hover { color: var(--mk-ink); }

.mk-modal-body { overflow-y: auto; flex: 1; min-height: 0; }

.mk-modal-foot {
  display: flex;
  justify-content: flex-end;
  border-top: 1px solid var(--mk-border);
  padding: 0.8rem 1.1rem;
  gap: 0.5rem;
}

.mk-modal-enter-active, .mk-modal-leave-active { transition: opacity 180ms ease; }
.mk-modal-enter-from, .mk-modal-leave-to { opacity: 0; }
.mk-modal-enter-active .mk-modal-panel, .mk-modal-leave-active .mk-modal-panel { transition: transform 180ms ease; }
.mk-modal-enter-from .mk-modal-panel, .mk-modal-leave-to .mk-modal-panel { transform: translateY(10px) scale(0.98); }

@media (prefers-reduced-motion: reduce) {
  .mk-modal-enter-active, .mk-modal-leave-active,
  .mk-modal-enter-active .mk-modal-panel, .mk-modal-leave-active .mk-modal-panel { transition: none; }
}
</style>
