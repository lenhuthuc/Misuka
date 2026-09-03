<script setup lang="ts">
import { onClickOutside } from '@vueuse/core'
import { computed, ref } from 'vue'

import IconButton from '../shared/IconButton.vue'

defineProps<{
  dark: boolean
}>()

const emit = defineEmits<{
  (e: 'toggleTheme' | 'newConversation' | 'openSettings'): void
}>()

const now = new Date()
const menuOpen = ref(false)
const menu = ref<HTMLElement | null>(null)

onClickOutside(menu, () => {
  menuOpen.value = false
})

const greeting = computed(() => {
  const hour = now.getHours()
  if (hour < 11)
    return { emoji: '🌅', text: 'Chào buổi sáng!' }
  if (hour < 14)
    return { emoji: '☀️', text: 'Chào buổi trưa!' }
  if (hour < 18)
    return { emoji: '🌤️', text: 'Chào buổi chiều!' }
  return { emoji: '🌙', text: 'Chào buổi tối!' }
})

function runAndClose(action: 'newConversation' | 'openSettings' | 'toggleTheme') {
  menuOpen.value = false
  emit(action)
}
</script>

<template>
  <header class="chat-header">
    <div class="chat-title-group">
      <span class="chat-brand">Mitsuka</span>
      <span class="chat-greeting">
        <span aria-hidden="true">{{ greeting.emoji }}</span> {{ greeting.text }}
      </span>
    </div>

    <div class="chat-header-actions">
      <IconButton
        icon="i-solar:pen-new-square-outline"
        label="Cuộc trò chuyện mới"
        size="sm"
        @click="emit('newConversation')"
      />
      <IconButton
        :icon="dark ? 'i-solar:sun-outline' : 'i-solar:moon-outline'"
        :label="dark ? 'Chuyển sang giao diện sáng' : 'Chuyển sang giao diện tối'"
        size="sm"
        @click="emit('toggleTheme')"
      />

      <div ref="menu" class="chat-menu">
        <IconButton
          icon="i-solar:menu-dots-bold"
          label="Tùy chọn"
          size="sm"
          :active="menuOpen"
          @click="menuOpen = !menuOpen"
        />
        <div v-if="menuOpen" class="chat-menu-pop" role="menu">
          <button type="button" role="menuitem" @click="runAndClose('newConversation')">
            <span class="i-solar:pen-new-square-outline" /> Cuộc trò chuyện mới
          </button>
          <button type="button" role="menuitem" @click="runAndClose('openSettings')">
            <span class="i-solar:settings-outline" /> Cài đặt
          </button>
        </div>
      </div>
    </div>
  </header>
</template>

<style scoped>
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--mk-border);
  padding: 0.65rem 0.85rem;
  gap: 0.5rem;
}

.chat-title-group {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
}

.chat-brand {
  font-weight: 700;
  font-size: 0.88rem;
  background: linear-gradient(135deg, var(--mk-pink, #f43f5e), var(--mk-purple, #a855f7));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.chat-greeting {
  border: 1px solid rgb(255 121 193 / 0.22);
  border-radius: 999px;
  background: rgb(255 111 181 / 0.12);
  color: var(--mk-pink-soft);
  padding: 0.2rem 0.55rem;
  font-size: 0.7rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chat-header-actions {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  gap: 0.25rem;
}

.chat-menu { position: relative; }

.chat-menu-pop {
  position: absolute;
  top: calc(100% + 0.4rem);
  right: 0;
  z-index: 40;
  display: flex;
  width: 12rem;
  flex-direction: column;
  border: 1px solid var(--mk-border-strong);
  border-radius: var(--mk-radius-sm);
  background: var(--mk-float);
  box-shadow: var(--mk-shadow);
  padding: 0.25rem;
}

.chat-menu-pop button {
  display: flex;
  align-items: center;
  border: 0;
  border-radius: 0.5rem;
  background: transparent;
  color: var(--mk-ink-dim);
  cursor: pointer;
  padding: 0.45rem 0.5rem;
  font: inherit;
  font-size: 0.76rem;
  gap: 0.45rem;
  text-align: left;
}

.chat-menu-pop button:hover { background: var(--mk-hover); color: var(--mk-ink); }

@media (max-width: 480px) {
  .chat-greeting { display: none; }
}
</style>
