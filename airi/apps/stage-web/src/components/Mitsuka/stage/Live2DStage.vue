<script setup lang="ts">
import type { BackgroundItem } from '@proj-airi/stage-layouts/stores/background'
import type { LocalConvState } from '@proj-airi/stage-ui/composables/local-conversation'
import type { EmotionState } from '@proj-airi/stage-ui/stores/modules/emotion'

import { ViewControlSlider, WidgetStage } from '@proj-airi/stage-ui/components/scenes'
import { computed, useTemplateRef } from 'vue'

import CharacterController from './CharacterController.vue'
import SceneBackground from './SceneBackground.vue'
import SceneOverlay from './SceneOverlay.vue'

const props = defineProps<{
  background: BackgroundItem
  topColor?: string
  cursorPosition: { x: number, y: number }
  enableOrbitControls: boolean
  paused: boolean
  listening: boolean
  viewControls: boolean
  fullscreen: boolean
  live2d: boolean
  state: LocalConvState
  emotion: EmotionState
}>()

defineEmits<{
  (e: 'toggleListening' | 'togglePaused' | 'toggleViewControls' | 'toggleFullscreen' | 'pickBackground' | 'openSettings'): void
}>()

const scene = useTemplateRef<InstanceType<typeof SceneBackground>>('scene')

// Live2D models are anchored differently from VRM/Spine ones, so the slider
// column sits higher for them — same rule the previous stage page used.
const viewControlClass = computed(() => props.live2d ? 'stage-view-controls--live2d' : '')

defineExpose({
  get surfaceEl() {
    return scene.value?.surfaceEl
  },
})
</script>

<template>
  <section class="live2d-stage" aria-label="Sân khấu Mitsuka">
    <SceneBackground ref="scene" :background="background" :top-color="topColor">
      <div class="stage-canvas">
        <WidgetStage
          h-full
          w-full
          :cursor-position="cursorPosition"
          :enable-orbit-controls="enableOrbitControls"
          :paused="paused"
        />
      </div>
    </SceneBackground>

    <div class="stage-hud stage-hud--top">
      <CharacterController :state="state" :emotion="emotion" />
    </div>

    <div v-if="viewControls" class="stage-view-controls" :class="viewControlClass">
      <ViewControlSlider />
    </div>

    <div class="stage-hud stage-hud--bottom">
      <SceneOverlay
        :listening="listening"
        :paused="paused"
        :view-controls="viewControls"
        :fullscreen="fullscreen"
        @toggle-listening="$emit('toggleListening')"
        @toggle-paused="$emit('togglePaused')"
        @toggle-view-controls="$emit('toggleViewControls')"
        @toggle-fullscreen="$emit('toggleFullscreen')"
        @pick-background="$emit('pickBackground')"
        @open-settings="$emit('openSettings')"
      />
    </div>
  </section>
</template>

<style scoped>
.live2d-stage {
  position: relative;
  overflow: hidden;
  min-height: 0;
  height: 100%;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius-lg);
  background: var(--mk-panel-solid);
  box-shadow: var(--mk-shadow);
}

.stage-canvas { height: 100%; width: 100%; }

.stage-hud {
  position: absolute;
  z-index: 20;
  display: flex;
  pointer-events: none;
}

.stage-hud > * { pointer-events: auto; }

.stage-hud--top { top: 0.85rem; left: 0.85rem; }

.stage-hud--bottom {
  bottom: 0.85rem;
  left: 50%;
  transform: translateX(-50%);
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

@media (max-width: 900px) {
  .live2d-stage { min-height: 15rem; }
  .stage-view-controls { display: none; }
}
</style>
