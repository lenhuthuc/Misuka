<script setup lang="ts">
import type { ChatMessage } from './types'

import { computed } from 'vue'

import Button from '../shared/Button.vue'

const props = defineProps<{ messages: ChatMessage[] }>()

defineEmits<{ (e: 'open' | 'clear'): void }>()

/** History lists the user's turns — the questions are what people scan for. */
const turns = computed(() => props.messages
  .filter(message => message.role === 'user')
  .slice()
  .reverse())

function time(at: number) {
  return new Date(at).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
}
</script>

<template>
  <section class="history-panel" aria-label="Lịch sử trò chuyện">
    <header class="history-head">
      <div>
        <h2>Lịch sử</h2>
        <p>Các câu bạn đã hỏi trong phiên này</p>
      </div>
      <Button v-if="turns.length" size="sm" variant="soft" @click="$emit('clear')">
        Xóa
      </Button>
    </header>

    <div v-if="turns.length" class="history-list mk-scroll">
      <span class="history-day">Hôm nay</span>
      <button
        v-for="turn in turns"
        :key="turn.id"
        class="history-item"
        type="button"
        @click="$emit('open')"
      >
        <span class="history-item-text">{{ turn.content }}</span>
        <span class="history-item-time">{{ time(turn.at) }}</span>
      </button>
    </div>

    <div v-else class="history-empty">
      <span class="i-solar:history-outline" />
      <p>Chưa có câu hỏi nào trong phiên này.</p>
      <Button size="sm" @click="$emit('open')">
        Bắt đầu trò chuyện
      </Button>
    </div>
  </section>
</template>

<style scoped>
.history-panel {
  display: flex;
  overflow: hidden;
  min-height: 0;
  height: 100%;
  flex-direction: column;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius-lg);
  background: var(--mk-panel);
  box-shadow: var(--mk-shadow);
  backdrop-filter: blur(22px) saturate(1.1);
}

.history-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--mk-border);
  padding: 0.85rem 0.95rem;
  gap: 1rem;
}

.history-head h2 { margin: 0; font-size: 0.95rem; font-weight: 800; }
.history-head p { margin: 0.1rem 0 0; color: var(--mk-muted); font-size: 0.7rem; }

.history-list {
  display: flex;
  overflow-y: auto;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  padding: 0.8rem 0.85rem;
  gap: 0.35rem;
}

.history-day {
  margin-bottom: 0.2rem;
  color: var(--mk-muted);
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.history-item {
  display: flex;
  align-items: center;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius-sm);
  background: var(--mk-raise);
  color: var(--mk-ink-dim);
  cursor: pointer;
  padding: 0.55rem 0.6rem;
  font: inherit;
  font-size: 0.78rem;
  gap: 0.6rem;
  text-align: left;
}

.history-item:hover { border-color: rgb(171 125 255 / 0.32); background: var(--mk-hover); }

.history-item-text {
  overflow: hidden;
  flex: 1;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-item-time {
  flex: 0 0 auto;
  color: var(--mk-muted);
  font-size: 0.66rem;
}

.history-empty {
  display: flex;
  flex: 1;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--mk-muted);
  gap: 0.6rem;
  padding: 1.5rem;
  text-align: center;
}

.history-empty span { font-size: 1.6rem; opacity: 0.5; }
.history-empty p { margin: 0; font-size: 0.8rem; }
</style>
