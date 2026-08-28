<script setup lang="ts">
export interface NavItem {
  id: string
  label: string
  icon: string
  badge?: string
}

defineProps<{ items: NavItem[] }>()
const active = defineModel<string>({ required: true })
</script>

<template>
  <nav class="mk-nav" aria-label="Điều hướng Mitsuka">
    <button
      v-for="item in items"
      :key="item.id"
      class="mk-nav-item"
      :class="{ 'mk-nav-item--active': active === item.id }"
      type="button"
      :aria-current="active === item.id ? 'page' : undefined"
      @click="active = item.id"
    >
      <span class="mk-nav-icon" :class="item.icon" />
      <span class="mk-nav-label">{{ item.label }}</span>
      <span v-if="item.badge" class="mk-nav-badge">{{ item.badge }}</span>
      <span v-if="active === item.id" class="mk-nav-bloom">✿</span>
    </button>
  </nav>
</template>

<style scoped>
.mk-nav {
  display: flex;
  flex-direction: column;
  gap: 0.18rem;
}

.mk-nav-item {
  position: relative;
  display: flex;
  align-items: center;
  border: 1px solid transparent;
  border-radius: var(--mk-radius-sm);
  background: transparent;
  color: var(--mk-muted);
  cursor: pointer;
  padding: 0.6rem 0.7rem;
  font: inherit;
  font-size: 0.85rem;
  font-weight: 600;
  gap: 0.7rem;
  text-align: left;
  transition: background 150ms ease, color 150ms ease;
}

.mk-nav-item:hover { background: var(--mk-raise); color: var(--mk-ink-dim); }

.mk-nav-item--active {
  border-color: rgb(255 121 193 / 0.24);
  background: linear-gradient(100deg, rgb(171 125 255 / 0.24), rgb(255 111 181 / 0.16));
  color: var(--mk-ink);
  box-shadow: inset 0 1px rgb(255 255 255 / 0.08);
}

.mk-nav-icon { flex: 0 0 auto; font-size: 1.05rem; }
.mk-nav-label { flex: 1; }

.mk-nav-badge {
  border-radius: 999px;
  background: var(--mk-accent);
  color: #fff;
  padding: 0.1rem 0.42rem;
  font-size: 0.58rem;
  font-weight: 800;
  letter-spacing: 0.06em;
}

.mk-nav-bloom {
  position: absolute;
  top: -0.3rem;
  right: -0.25rem;
  color: var(--mk-pink);
  font-size: 0.7rem;
  opacity: 0.85;
  pointer-events: none;
}
</style>
