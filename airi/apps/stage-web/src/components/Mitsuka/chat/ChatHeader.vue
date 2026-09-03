<script setup lang="ts">
import { onClickOutside } from '@vueuse/core'
import { computed, ref } from 'vue'

import IconButton from '../shared/IconButton.vue'

defineProps<{
  listening: boolean
  dark: boolean
}>()

const emit = defineEmits<{
  (e: 'toggleListening' | 'toggleTheme' | 'newConversation' | 'openSettings'): void
}>()

const now = new Date()
const showFullDate = ref(false)
const menuOpen = ref(false)
const menu = ref<HTMLElement | null>(null)

onClickOutside(menu, () => {
  menuOpen.value = false
})

const fullDate = computed(() => now.toLocaleDateString('vi-VN', {
  weekday: 'long',
  day: '2-digit',
  month: '2-digit',
  year: 'numeric',
}))

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

function runAndClose(action: 'newConversation' | 'openSettings') {
  menuOpen.value = false
  emit(action)
}
</script>

<template>
  <header class="chat-header">
    <button class="chat-day" type="button" :aria-expanded="showFullDate" @click="showFullDate = !showFullDate">
      <span class="i-solar:home-smile-outline" />
      {{ showFullDate ? fullDate : 'Hôm nay' }}
      <span class="i-solar:alt-arrow-down-linear chat-day-caret" :class="{ 'chat-day-caret--open': showFullDate }" />
    </button>

    <span class="chat-greeting">
      <span aria-hidden="true">{{ greeting.emoji }}</span> {{ greeting.text }}
    </span>

    <div class="chat-header-actions">
      <IconButton
        :icon="listening ? 'i-solar:soundwave-outline' : 'i-solar:microphone-3-outline'"
        :label="listening ? 'Tắt lắng nghe' : 'Bật lắng nghe'"
        :active="listening"
        size="sm"
        @click="emit('toggleListening')"
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
  overflow-x: auto;
  align-items: center;
  border-bottom: 1px solid var(--mk-border);
  padding: 0.7rem 0.85rem;
  gap: 0.5rem;
}

.chat-day {
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  border: 1px solid var(--mk-border);
  border-radius: 999px;
  background: var(--mk-raise);
  color: var(--mk-ink-dim);
  cursor: pointer;
  padding: 0.3rem 0.6rem;
  font: inherit;
  font-size: 0.74rem;
  font-weight: 700;
  gap: 0.32rem;
  white-space: nowrap;
}

.chat-day:hover { background: var(--mk-hover); }
.chat-day-caret { font-size: 0.8rem; opacity: 0.7; transition: transform 160ms ease; }
.chat-day-caret--open { transform: rotate(180deg); }

.chat-greeting {
  border: 1px solid rgb(255 121 193 / 0.22);
  border-radius: 999px;
  background: rgb(255 111 181 / 0.12);
  color: var(--mk-pink-soft);
  padding: 0.3rem 0.62rem;
  font-size: 0.72rem;
  font-weight: 700;
  white-space: nowrap;
}

.chat-header-actions {
  display: flex;
  flex-shrink: 0;
  margin-left: auto;
  align-items: center;
  gap: 0.3rem;
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

@media (max-width: 1180px) {
  .chat-greeting { display: none; }
}

@media (prefers-reduced-motion: reduce) {
  .chat-day-caret { transition: none; }
}
</style>
