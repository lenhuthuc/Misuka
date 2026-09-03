<script setup lang="ts">
import type { QuickAction } from './QuickActions.vue'
import type { ChatMessage } from './types'

import ChatHeader from './ChatHeader.vue'
import MessageComposer from './MessageComposer.vue'
import MessageList from './MessageList.vue'
import QuickActions from './QuickActions.vue'

defineProps<{
  messages: ChatMessage[]
  quickActions: QuickAction[]
  thinking: boolean
  busy: boolean
  listening: boolean
  dark: boolean
  error?: string
}>()

const emit = defineEmits<{
  (e: 'send', text: string, image?: File): void
  (e: 'pickQuickAction', action: QuickAction): void
  (e: 'toggleListening' | 'toggleTheme' | 'newConversation' | 'openSettings'): void
}>()

const draft = defineModel<string>('draft', { default: '' })
</script>

<template>
  <section class="chat-panel" aria-label="Khung trò chuyện">
    <ChatHeader
      :listening="listening"
      :dark="dark"
      @toggle-listening="emit('toggleListening')"
      @toggle-theme="emit('toggleTheme')"
      @new-conversation="emit('newConversation')"
      @open-settings="emit('openSettings')"
    />

    <MessageList :messages="messages" :thinking="thinking" />

    <p v-if="error" class="chat-error">
      <span class="i-solar:danger-triangle-outline" />
      {{ error }}
    </p>

    <QuickActions
      :actions="quickActions"
      :disabled="busy"
      @pick="action => emit('pickQuickAction', action)"
    />

    <MessageComposer
      v-model="draft"
      :listening="listening"
      :busy="busy"
      @send="(text, image) => emit('send', text, image)"
      @toggle-listening="emit('toggleListening')"
    />
  </section>
</template>

<style scoped>
.chat-panel {
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

.chat-error {
  display: flex;
  align-items: center;
  margin: 0 0.95rem 0.5rem;
  border: 1px solid rgb(255 143 163 / 0.3);
  border-radius: var(--mk-radius-sm);
  background: rgb(255 143 163 / 0.1);
  color: var(--mk-danger);
  padding: 0.45rem 0.6rem;
  font-size: 0.72rem;
  gap: 0.4rem;
}
</style>
