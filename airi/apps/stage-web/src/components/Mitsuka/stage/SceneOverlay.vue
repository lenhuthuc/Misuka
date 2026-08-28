<script setup lang="ts">
import IconButton from '../shared/IconButton.vue'

defineProps<{
  listening: boolean
  paused: boolean
  viewControls: boolean
  fullscreen: boolean
}>()

defineEmits<{
  (e: 'toggleListening' | 'togglePaused' | 'toggleViewControls' | 'toggleFullscreen' | 'pickBackground' | 'openSettings'): void
}>()
</script>

<template>
  <div class="scene-overlay">
    <IconButton
      :icon="listening ? 'i-solar:microphone-3-bold' : 'i-solar:microphone-3-outline'"
      :label="listening ? 'Tắt lắng nghe' : 'Bật lắng nghe'"
      :active="listening"
      size="sm"
      @click="$emit('toggleListening')"
    />
    <IconButton
      :icon="paused ? 'i-solar:play-outline' : 'i-solar:pause-outline'"
      :label="paused ? 'Tiếp tục hoạt ảnh' : 'Tạm dừng hoạt ảnh'"
      size="sm"
      @click="$emit('togglePaused')"
    />
    <IconButton
      icon="i-solar:gallery-outline"
      label="Đổi ảnh nền"
      size="sm"
      @click="$emit('pickBackground')"
    />
    <IconButton
      icon="i-solar:tuning-2-outline"
      label="Điều khiển góc nhìn"
      :active="viewControls"
      size="sm"
      @click="$emit('toggleViewControls')"
    />
    <IconButton
      :icon="fullscreen ? 'i-solar:quit-full-screen-outline' : 'i-solar:full-screen-outline'"
      :label="fullscreen ? 'Thoát toàn màn hình' : 'Toàn màn hình'"
      size="sm"
      @click="$emit('toggleFullscreen')"
    />
    <IconButton
      icon="i-solar:settings-outline"
      label="Cài đặt"
      size="sm"
      @click="$emit('openSettings')"
    />
  </div>
</template>

<style scoped>
.scene-overlay {
  display: flex;
  align-items: center;
  border: 1px solid var(--mk-border);
  border-radius: 999px;
  background: var(--mk-hud);
  box-shadow: 0 12px 32px rgb(6 2 14 / 0.42);
  padding: 0.32rem;
  gap: 0.28rem;
  backdrop-filter: blur(18px);
}
</style>
