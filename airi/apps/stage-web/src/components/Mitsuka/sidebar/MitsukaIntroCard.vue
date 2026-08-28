<script setup lang="ts">
import Button from '../shared/Button.vue'
import GlassCard from '../shared/GlassCard.vue'

defineEmits<{ (e: 'learnMore'): void }>()
</script>

<template>
  <GlassCard variant="accent" class="mk-intro">
    <div class="mk-intro-title">
      Mitsuka <span>✦</span>
    </div>
    <p class="mk-intro-body">
      Trợ lý AI luôn sẵn sàng lắng nghe và hỗ trợ bạn mọi lúc.
    </p>
    <Button size="sm" @click="$emit('learnMore')">
      Tìm hiểu thêm <span class="i-solar:arrow-right-linear" />
    </Button>
    <span class="mk-intro-petal">✿</span>
  </GlassCard>
</template>

<style scoped>
.mk-intro { padding: 0.9rem; }

.mk-intro-title {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--mk-pink-soft);
  font-size: 0.92rem;
  font-weight: 800;
}

.mk-intro-title span { font-size: 0.7rem; }

.mk-intro-body {
  margin: 0.4rem 0 0.75rem;
  color: var(--mk-ink-dim);
  font-size: 0.76rem;
  line-height: 1.5;
  opacity: 0.85;
}

.mk-intro-petal {
  position: absolute;
  right: 0.6rem;
  bottom: 0.5rem;
  color: var(--mk-pink);
  font-size: 0.85rem;
  opacity: 0.45;
}
</style>
