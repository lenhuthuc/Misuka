<script setup lang="ts">
import { computed } from 'vue'

import GlassCard from '../shared/GlassCard.vue'
import IconButton from '../shared/IconButton.vue'

const props = defineProps<{
  name: string
  /** null while the first health probe is still in flight. */
  online: boolean | null
}>()

defineEmits<{ (e: 'openSettings'): void }>()

const statusLabel = computed(() => {
  if (props.online === null)
    return 'Đang kết nối'
  return props.online ? 'Online' : 'Mất kết nối'
})
</script>

<template>
  <GlassCard class="mk-user" :padded="false">
    <div class="mk-user-inner">
      <div class="mk-user-avatar" aria-hidden="true">
        🐱
      </div>
      <div class="mk-user-meta">
        <strong>{{ name }}</strong>
        <span class="mk-user-status" :class="{ 'mk-user-status--on': online, 'mk-user-status--pending': online === null }">
          <i />{{ statusLabel }}
        </span>
      </div>
      <IconButton
        icon="i-solar:settings-outline"
        label="Cài đặt"
        size="sm"
        @click="$emit('openSettings')"
      />
    </div>
  </GlassCard>
</template>

<style scoped>
.mk-user-inner {
  display: flex;
  align-items: center;
  padding: 0.6rem 0.65rem;
  gap: 0.6rem;
}

.mk-user-avatar {
  display: grid;
  width: 2.1rem;
  height: 2.1rem;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid rgb(255 255 255 / 0.14);
  border-radius: 50%;
  background: linear-gradient(150deg, rgb(255 176 216 / 0.5), rgb(171 125 255 / 0.42));
  font-size: 0.95rem;
}

.mk-user-meta { display: flex; min-width: 0; flex: 1; flex-direction: column; }
.mk-user-meta strong { font-size: 0.82rem; font-weight: 700; }

.mk-user-status {
  display: flex;
  align-items: center;
  color: var(--mk-muted);
  font-size: 0.66rem;
  gap: 0.3rem;
}

.mk-user-status i {
  width: 0.4rem;
  height: 0.4rem;
  border-radius: 50%;
  background: var(--mk-danger);
  box-shadow: 0 0 0.5rem var(--mk-danger);
}

.mk-user-status--on { color: var(--mk-online); }
.mk-user-status--on i { background: var(--mk-online); box-shadow: 0 0 0.5rem var(--mk-online); }
.mk-user-status--pending i { background: var(--mk-purple-soft); box-shadow: none; }
</style>
