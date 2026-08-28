<script setup lang="ts">
import type { BackgroundItem } from '@proj-airi/stage-layouts/stores/background'

import { BackgroundProvider } from '@proj-airi/stage-layouts/components/Backgrounds'
import { computed, useTemplateRef } from 'vue'

defineProps<{
  background: BackgroundItem
  topColor?: string
}>()

const provider = useTemplateRef<InstanceType<typeof BackgroundProvider>>('provider')

// `useBackgroundThemeColor` samples this element to derive the page's theme
// colour. It reads `.surfaceEl` off whatever ref it is handed, so re-exposing
// the provider's own element under the same name lets the shell keep passing
// its template ref straight into the composable.
const surfaceEl = computed(() => provider.value?.surfaceEl)

defineExpose({ surfaceEl })
</script>

<template>
  <div class="scene-background">
    <BackgroundProvider ref="provider" :background="background" :top-color="topColor">
      <slot />
    </BackgroundProvider>
    <div class="scene-vignette" />
    <div class="scene-glow scene-glow--top" />
    <div class="scene-glow scene-glow--bottom" />
  </div>
</template>

<style scoped>
.scene-background {
  position: relative;
  overflow: hidden;
  height: 100%;
  width: 100%;
}

/* BackgroundProvider is written as a full-page wrapper (`min-h-100dvh`);
   inside the stage card it has to fill the card instead. */
.scene-background :deep(.customized-background) {
  min-height: 100%;
  height: 100%;
}

.scene-background :deep(.customized-background > .relative) {
  height: 100%;
}

.scene-vignette,
.scene-glow {
  position: absolute;
  z-index: 5;
  pointer-events: none;
}

.scene-vignette {
  inset: 0;
  background:
    linear-gradient(180deg, rgb(9 4 18 / 0.5) 0%, transparent 24%, transparent 56%, rgb(8 3 17 / 0.78) 100%),
    radial-gradient(circle at 50% 42%, transparent 32%, rgb(11 5 24 / 0.38) 100%);
}

.scene-glow {
  width: 26rem;
  height: 26rem;
  border-radius: 999px;
  filter: blur(90px);
  opacity: 0.22;
}

.scene-glow--top { top: -13rem; right: -11rem; background: #ec4899; }
.scene-glow--bottom { bottom: -14rem; left: -12rem; background: #8b5cf6; }
</style>
