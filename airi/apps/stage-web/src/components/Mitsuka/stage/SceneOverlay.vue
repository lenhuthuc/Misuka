<script setup lang="ts">
import IconButton from '../shared/IconButton.vue'

defineProps<{
  listening?: boolean
  paused: boolean
  viewControls?: boolean
  fullscreen: boolean
}>()

defineEmits<{
  (e: 'toggleListening' | 'togglePaused' | 'toggleViewControls' | 'toggleFullscreen' | 'pickBackground' | 'openSettings'): void
}>()
</script>

<template>
  <div class="scene-overlay">
    <IconButton
      icon="i-solar:gallery-outline"
      label="Đổi ảnh nền"
      size="sm"
      @click="$emit('pickBackground')"
    />
    <IconButton
      :icon="paused ? 'i-solar:play-outline' : 'i-solar:pause-outline'"
      :label="paused ? 'Tiếp tục hoạt ảnh' : 'Tạm dừng hoạt ảnh'"
      size="sm"
      @click="$emit('togglePaused')"
    />
    <IconButton
      :icon="fullscreen ? 'i-solar:quit-full-screen-outline' : 'i-solar:full-screen-outline'"
      :label="fullscreen ? 'Thoát toàn màn hình' : 'Toàn màn hình'"
      size="sm"
      @click="$emit('toggleFullscreen')"
    />
  </div>
</template>

<style scoped>
.scene-overlay {
  display: flex;
  align-items: center;
  border: 1px solid var(--mk-border);
  border-radius: 999px;
  background: var(--mk-hud, rgba(20, 10, 32, 0.75));
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
  padding: 0.25rem 0.4rem;
  gap: 0.25rem;
  backdrop-filter: blur(18px);
  transition: opacity 200ms ease;
}

@media (max-width: 768px) {
  .scene-overlay {
    padding: 0.2rem 0.35rem;
    gap: 0.2rem;
    transform: scale(0.92);
  }
}
</style>
