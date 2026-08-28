<script setup lang="ts">
import type { NavItem } from './Navigation.vue'

import MitsukaIntroCard from './MitsukaIntroCard.vue'
import MitsukaLogo from './MitsukaLogo.vue'
import Navigation from './Navigation.vue'
import UserCard from './UserCard.vue'

defineProps<{
  items: NavItem[]
  userName: string
  online: boolean | null
}>()

defineEmits<{ (e: 'learnMore' | 'openSettings'): void }>()

const active = defineModel<string>({ required: true })
</script>

<template>
  <aside class="mk-sidebar mk-scroll">
    <MitsukaLogo />
    <Navigation v-model="active" :items="items" />
    <div class="mk-sidebar-spacer" />
    <MitsukaIntroCard @learn-more="$emit('learnMore')" />
    <UserCard :name="userName" :online="online" @open-settings="$emit('openSettings')" />
  </aside>
</template>

<style scoped>
.mk-sidebar {
  display: flex;
  overflow-y: auto;
  min-height: 0;
  height: 100%;
  flex-direction: column;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius-lg);
  background: var(--mk-panel);
  box-shadow: var(--mk-shadow);
  padding: 0.9rem 0.8rem;
  gap: 0.85rem;
  backdrop-filter: blur(22px) saturate(1.1);
}

.mk-sidebar-spacer { flex: 1; min-height: 0.5rem; }
</style>
