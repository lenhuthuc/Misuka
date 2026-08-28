<script setup lang="ts">
import type { ChatMessage } from './types'

import { computed } from 'vue'

const props = defineProps<{ message: ChatMessage }>()

const isUser = computed(() => props.message.role === 'user')

const time = computed(() => new Date(props.message.at).toLocaleTimeString('vi-VN', {
  hour: '2-digit',
  minute: '2-digit',
}))
</script>

<template>
  <div class="bubble-row" :class="isUser ? 'bubble-row--user' : 'bubble-row--mitsuka'">
    <div class="bubble" :class="[isUser ? 'bubble--user' : 'bubble--mitsuka', { 'bubble--error': message.error }]">
      <span class="bubble-name">{{ isUser ? 'Bạn' : 'Mitsuka' }}</span>

      <p v-if="message.error" class="bubble-text bubble-text--error">
        {{ message.error }}
      </p>
      <p v-else class="bubble-text">
        {{ message.content }}<span v-if="message.streaming" class="bubble-caret" />
      </p>

      <div class="bubble-foot">
        <span>{{ time }}</span>
        <span v-if="isUser" class="i-solar:check-read-outline bubble-read" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.bubble-row { display: flex; }
.bubble-row--user { justify-content: flex-end; }
.bubble-row--mitsuka { justify-content: flex-start; }

.bubble {
  position: relative;
  max-width: 86%;
  border: 1px solid var(--mk-border);
  padding: 0.6rem 0.8rem 0.5rem;
}

.bubble--mitsuka {
  border-color: rgb(255 121 193 / 0.2);
  border-radius: var(--mk-radius) var(--mk-radius) var(--mk-radius) 0.4rem;
  background: var(--mk-bubble-ai);
}

.bubble--user {
  border-color: rgb(171 125 255 / 0.28);
  border-radius: var(--mk-radius) var(--mk-radius) 0.4rem var(--mk-radius);
  background: var(--mk-bubble-user);
}

.bubble--error { border-color: rgb(255 143 163 / 0.4); }

.bubble-name {
  display: block;
  margin-bottom: 0.18rem;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.02em;
}

.bubble--mitsuka .bubble-name { color: var(--mk-pink-soft); }
.bubble--user .bubble-name { color: var(--mk-purple-soft); }

.bubble-text {
  margin: 0;
  color: var(--mk-ink);
  font-size: 0.83rem;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  text-wrap: pretty;
}

.bubble-text--error { color: var(--mk-danger); }

.bubble-caret {
  display: inline-block;
  width: 0.42rem;
  height: 0.85rem;
  margin-left: 0.12rem;
  border-radius: 1px;
  background: var(--mk-pink);
  animation: bubble-caret 900ms steps(2, start) infinite;
  vertical-align: text-bottom;
}

@keyframes bubble-caret { 50% { opacity: 0.15; } }

.bubble-foot {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  margin-top: 0.2rem;
  color: var(--mk-muted);
  font-size: 0.62rem;
  gap: 0.22rem;
  opacity: 0.75;
}

.bubble-read { font-size: 0.72rem; }

@media (prefers-reduced-motion: reduce) {
  .bubble-caret { animation: none; }
}
</style>
