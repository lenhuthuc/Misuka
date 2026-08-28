<script setup lang="ts">
import type { ChatMessage } from './types'

import { nextTick, ref, watch } from 'vue'

import MessageBubble from './MessageBubble.vue'

const props = defineProps<{
  messages: ChatMessage[]
  /** Shows the three-dot bubble while the reply has not started streaming. */
  thinking: boolean
}>()

const scroller = ref<HTMLElement | null>(null)
const pinnedToBottom = ref(true)

function onScroll() {
  const el = scroller.value
  if (!el)
    return
  // 48px of slack so a near-bottom position still counts as "following".
  pinnedToBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 48
}

async function scrollToBottom(force = false) {
  if (!force && !pinnedToBottom.value)
    return
  await nextTick()
  const el = scroller.value
  if (el)
    el.scrollTop = el.scrollHeight
}

// The last message's content mutates in place while the reply streams, so
// watching the array alone would miss every token after the first.
watch(
  () => [props.messages.length, props.messages.at(-1)?.content, props.thinking],
  () => void scrollToBottom(),
  { flush: 'post' },
)

defineExpose({ scrollToBottom })
</script>

<template>
  <div ref="scroller" class="message-list mk-scroll" @scroll.passive="onScroll">
    <MessageBubble
      v-for="message in messages"
      :key="message.id"
      :message="message"
    />

    <div v-if="thinking" class="typing-row">
      <div class="typing-bubble">
        <span class="typing-name">Mitsuka</span>
        <span class="typing-dots"><i /><i /><i /></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.message-list {
  display: flex;
  overflow-y: auto;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  padding: 0.9rem 0.95rem;
  gap: 0.7rem;
}

.typing-row { display: flex; }

.typing-bubble {
  border: 1px solid rgb(255 121 193 / 0.2);
  border-radius: var(--mk-radius) var(--mk-radius) var(--mk-radius) 0.4rem;
  background: var(--mk-bubble-ai);
  padding: 0.55rem 0.8rem;
}

.typing-name {
  display: block;
  margin-bottom: 0.25rem;
  color: var(--mk-pink-soft);
  font-size: 0.68rem;
  font-weight: 800;
}

.typing-dots { display: flex; align-items: center; gap: 0.22rem; }

.typing-dots i {
  width: 0.32rem;
  height: 0.32rem;
  border-radius: 50%;
  background: var(--mk-pink-soft);
  animation: typing 1.1s ease-in-out infinite;
}

.typing-dots i:nth-child(2) { animation-delay: 160ms; }
.typing-dots i:nth-child(3) { animation-delay: 320ms; }

@keyframes typing {
  0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
  30% { opacity: 1; transform: translateY(-2px); }
}

@media (prefers-reduced-motion: reduce) {
  .typing-dots i { animation: none; opacity: 0.7; }
}
</style>
