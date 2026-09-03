<script setup lang="ts">
import type { EmotionPayload } from '@proj-airi/stage-ui/constants/emotions'

import { ViewControlSlider, WidgetStage } from '@proj-airi/stage-ui/components/scenes'
import { computed } from 'vue'

import CharacterController from './CharacterController.vue'
import SceneOverlay from './SceneOverlay.vue'

const props = withDefaults(defineProps<{
  state?: 'pending' | 'loading' | 'mounted'
  cursorPosition?: { x: number, y: number }
  enableOrbitControls?: boolean
  paused?: boolean
  viewControls?: boolean
  fullscreen?: boolean
  live2d?: boolean
  emotion?: EmotionPayload
}>(), {
  cursorPosition: () => ({ x: 0, y: 0 }),
  enableOrbitControls: true,
  paused: false,
  viewControls: false,
  fullscreen: false,
  live2d: true,
  state: 'pending',
})

defineEmits<{
  (e: 'togglePaused'): void
  (e: 'toggleViewControls'): void
  (e: 'toggleFullscreen'): void
  (e: 'pickBackground'): void
  (e: 'openSettings'): void
}>()

const viewControlClass = computed(() => {
  return props.live2d ? 'stage-view-controls--live2d' : 'stage-view-controls--three'
})
</script>

<template>
  <section class="live2d-stage" aria-label="Sân khấu Mitsuka">
    <div class="stage-canvas">
      <WidgetStage
        h-full
        w-full
        :cursor-position="cursorPosition"
        :enable-orbit-controls="enableOrbitControls"
        :paused="paused"
      />
    </div>

    <div class="stage-hud stage-hud--top">
      <CharacterController :state="state" :emotion="emotion" />
    </div>

    <div v-if="viewControls" class="stage-view-controls" :class="viewControlClass">
      <ViewControlSlider />
    </div>

    <div class="stage-hud stage-hud--bottom">
      <SceneOverlay
        :paused="paused"
        :fullscreen="fullscreen"
        @toggle-paused="$emit('togglePaused')"
        @toggle-fullscreen="$emit('toggleFullscreen')"
        @pick-background="$emit('pickBackground')"
      />
    </div>
  </section>
</template>

<style scoped>
.live2d-stage {
  position: relative;
  overflow: hidden;
  height: 100%;
  width: 100%;
}

.stage-canvas {
  height: 100%;
  width: 100%;
}

/* Hide duplicate background inside WidgetStage so canvas stays 100% transparent */
.stage-canvas :deep(.absolute.left-0.top-0.z-0) {
  display: none !important;
}

.stage-hud {
  position: absolute;
  z-index: 20;
  display: flex;
  pointer-events: none;
}

.stage-hud > * { pointer-events: auto; }

.stage-hud--top {
  top: 1.2rem;
  left: 4.2rem;
}

.stage-hud--bottom {
  bottom: 1.2rem;
  left: 1.2rem;
}

.stage-view-controls {
  position: absolute;
  z-index: 20;
  top: 50%;
  left: 0.6rem;
  transform: translateY(-50%);
}

.stage-view-controls--live2d {
  top: 20%;
  height: 45%;
  transform: none;
}

@media (max-width: 860px) {
  .stage-view-controls { display: none; }

  .stage-hud--top {
    top: 0.5rem;
    left: 0.6rem;
  }

  .stage-hud--bottom {
    bottom: 0.5rem;
    right: 0.6rem;
    left: auto;
  }
}
</style>
