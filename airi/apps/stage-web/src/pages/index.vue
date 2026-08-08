<script setup lang="ts">
import type { ChatProvider } from '@xsai-ext/providers/utils'

import Header from '@proj-airi/stage-layouts/components/Layouts/Header.vue'
import InteractiveArea from '@proj-airi/stage-layouts/components/Layouts/InteractiveArea.vue'
import MobileHeader from '@proj-airi/stage-layouts/components/Layouts/MobileHeader.vue'
import MobileInteractiveArea from '@proj-airi/stage-layouts/components/Layouts/MobileInteractiveArea.vue'
import workletUrl from '@proj-airi/stage-ui/workers/vad/process.worklet?worker&url'

import { BackgroundProvider } from '@proj-airi/stage-layouts/components/Backgrounds'
import { useBackgroundThemeColor } from '@proj-airi/stage-layouts/composables/theme-color'
import { useBackgroundStore } from '@proj-airi/stage-layouts/stores/background'
import { HoloCoupon } from '@proj-airi/stage-ui/components'
import { ViewControlSlider, WidgetStage } from '@proj-airi/stage-ui/components/scenes'
import { useAudioRecorder } from '@proj-airi/stage-ui/composables/audio/audio-recorder'
import { encodeWav, normalizeSpeechSamples } from '@proj-airi/stage-ui/composables/audio/wav-encoder'
import { useLocalConversation } from '@proj-airi/stage-ui/composables/local-conversation'
import { useVAD } from '@proj-airi/stage-ui/stores/ai/models/vad'
import { useChatOrchestratorStore } from '@proj-airi/stage-ui/stores/chat'
import { useConsciousnessStore } from '@proj-airi/stage-ui/stores/modules/consciousness'
import { useEmotionStore } from '@proj-airi/stage-ui/stores/modules/emotion'
import { useHearingSpeechInputPipeline } from '@proj-airi/stage-ui/stores/modules/hearing'
import { useProvidersStore } from '@proj-airi/stage-ui/stores/providers'
import { useSettings, useSettingsAudioDevice } from '@proj-airi/stage-ui/stores/settings'
import { breakpointsTailwind, useBreakpoints, useLocalStorage, useMouse } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import { computed, onMounted, onUnmounted, ref, useTemplateRef, watch } from 'vue'

const paused = ref(false)

function handleSettingsOpen(open: boolean) {
  paused.value = open
}

// ── Local conversation mode (Whisper → RAG Chat → Piper TTS) ─────────────────
// Default true — this project runs the local Python stack; set to false to use cloud providers instead.
const localMode = useLocalStorage('settings/local-conversation/enabled', true)

const emotionStore = useEmotionStore()

const localConv = useLocalConversation({
  language: 'en',
  onEmotion: (e) => { emotionStore.emotion = e },
})
const { state: localState, transcript: localTranscript, reply: localReply, error: localError } = localConv

const LOCAL_STATE_LABEL: Record<string, string> = {
  idle: 'Đang lắng nghe',
  listening: 'Đang nghe...',
  transcribing: 'Đang nhận dạng...',
  thinking: 'Đang suy nghĩ...',
  speaking: 'Đang nói...',
}

const breakpoints = useBreakpoints(breakpointsTailwind)
const isMobile = breakpoints.smaller('md')

const backgroundStore = useBackgroundStore()
const { selectedOption, sampledColor } = storeToRefs(backgroundStore)
const backgroundSurface = useTemplateRef<InstanceType<typeof BackgroundProvider>>('backgroundSurface')
const { stageModelRenderer } = storeToRefs(useSettings())

const { syncBackgroundTheme } = useBackgroundThemeColor({ backgroundSurface, selectedOption, sampledColor })
onMounted(() => syncBackgroundTheme())

// Audio + transcription pipeline (mirrors stage-tamagotchi)
const settingsAudioDeviceStore = useSettingsAudioDevice()
const { stream, enabled } = storeToRefs(settingsAudioDeviceStore)
const { startRecord, stopRecord, onStopRecord } = useAudioRecorder(stream)
const hearingPipeline = useHearingSpeechInputPipeline()
const { transcribeForRecording } = hearingPipeline
const { supportsStreamInput } = storeToRefs(hearingPipeline)
const providersStore = useProvidersStore()
const consciousnessStore = useConsciousnessStore()
const { activeProvider: activeChatProvider, activeModel: activeChatModel } = storeToRefs(consciousnessStore)
const chatStore = useChatOrchestratorStore()

const shouldUseStreamInput = computed(() => supportsStreamInput.value && !!stream.value)

const {
  init: initVAD,
  dispose: disposeVAD,
  start: startVAD,
  loaded: vadLoaded,
} = useVAD(workletUrl, {
  threshold: ref(0.45),
  minSilenceDurationMs: ref(900),
  onSpeechStart: () => {
    if (localMode.value)
      localConv.onSpeechStart()
    handleSpeechStart()
  },
  onSpeechEnd: () => handleSpeechEnd(),
  onSpeechReady: (buffer, _duration) => {
    console.info('[VAD] speech-ready | localMode:', localMode.value, '| samples:', buffer.length)
    if (!localMode.value)
      return

    const normalized = normalizeSpeechSamples(buffer)
    const wav = encodeWav(normalized, 16000)
    console.info('[VAD] normalized wav blob size:', wav.size)
    void localConv.process(wav)
  },
})

let stopOnStopRecord: (() => void) | undefined

async function startAudioInteraction() {
  try {
    await initVAD()
    if (stream.value)
      await startVAD(stream.value)

    // Hook once
    stopOnStopRecord = onStopRecord(async (recording) => {
      if (localMode.value) {
        // Local mode uses the serialized VAD segment above because it includes
        // pre-speech padding. The recorder starts only after detection and
        // therefore loses the beginning of short utterances.
        return
      }

      // Normal pipeline: external STT provider → LLM provider
      const text = await transcribeForRecording(recording)
      if (!text || !text.trim())
        return

      // Analyse emotion in parallel — non-blocking, failure is safe to ignore
      void emotionStore.analyzeText(text)

      try {
        const provider = await providersStore.getProviderInstance(activeChatProvider.value)
        if (!provider || !activeChatModel.value)
          return

        await chatStore.ingest(text, { model: activeChatModel.value, chatProvider: provider as ChatProvider })
      }
      catch (err) {
        console.error('Failed to send chat from voice:', err)
      }
    })
  }
  catch (e) {
    console.error('Audio interaction init failed:', e)
  }
}

async function handleSpeechStart() {
  // For streaming providers, ChatArea component handles transcription manually
  // The main page should not start automatic transcription to avoid duplicate sessions
  if (shouldUseStreamInput.value) {
    return
  }

  startRecord()
}

async function handleSpeechEnd() {
  if (shouldUseStreamInput.value) {
    // Keep streaming session alive; idle timer in pipeline will handle teardown.
    return
  }

  stopRecord()
}

function stopAudioInteraction() {
  try {
    stopOnStopRecord?.()
    stopOnStopRecord = undefined
    disposeVAD()
  }
  catch {}
}

watch(enabled, async (val) => {
  if (val) {
    await startAudioInteraction()
  }
  else {
    stopAudioInteraction()
  }
}, { immediate: true })

onUnmounted(() => {
  stopAudioInteraction()
})

watch([stream, () => vadLoaded.value], async ([s, loaded]) => {
  if (enabled.value && loaded && s) {
    try {
      await startVAD(s)
    }
    catch (e) {
      console.error('Failed to start VAD with stream:', e)
    }
  }
})

const { x: mouseX, y: mouseY } = useMouse()
const cursorPosition = computed(() => ({
  x: mouseX.value,
  y: mouseY.value,
}))
</script>

<template>
  <BackgroundProvider
    ref="backgroundSurface"
    class="widgets top-widgets"
    :background="selectedOption"
    :top-color="sampledColor"
  >
    <div relative flex="~ col" z-2 h-100dvh w-100vw of-hidden>
      <!-- header -->
      <div class="px-0 py-1 md:px-3 md:py-3" w-full gap-2>
        <Header class="hidden md:flex" />
        <MobileHeader class="flex md:hidden" />
      </div>
      <!-- page -->
      <div relative flex="~ 1 row gap-y-0 gap-x-2 <md:col">
        <div relative flex-1 min-w="1/2">
          <div
            absolute left-0 z-15 px-3
            :class="[
              stageModelRenderer === 'live2d' ? 'top-0 h-full py-[20vh]' : 'top-1/2 -translate-y-1/2',
            ]"
          >
            <ViewControlSlider />
          </div>
          <WidgetStage
            h-full w-full
            :cursor-position="cursorPosition"
            :enable-orbit-controls="!isMobile"
            :paused="paused"
          />
        </div>
        <InteractiveArea v-if="!isMobile" h="85dvh" absolute right-4 flex flex-1 flex-col max-w="500px" min-w="30%" />
        <MobileInteractiveArea v-if="isMobile" @settings-open="handleSettingsOpen" />
      </div>
      <HoloCoupon />

      <!-- Local conversation mode toggle + status -->
      <div
        style="position:absolute;bottom:1rem;left:50%;transform:translateX(-50%);z-index:50;display:flex;flex-direction:column;align-items:center;gap:0.5rem;"
      >
        <!-- Toggle button -->
        <button
          style="padding:0.35rem 0.9rem;border-radius:999px;font-size:0.78rem;font-weight:600;border:none;cursor:pointer;transition:all 0.2s;"
          :style="localMode
            ? 'background:rgba(99,102,241,0.85);color:#fff;box-shadow:0 0 0 2px #818cf8'
            : 'background:rgba(0,0,0,0.35);color:rgba(255,255,255,0.7)'"
          @click="localMode = !localMode"
        >
          {{ localMode ? '🎙 Local mode BẬT' : '🎙 Local mode TẮT' }}
        </button>

        <!-- Status pill (only shown in local mode) -->
        <div
          v-if="localMode"
          style="display:flex;flex-direction:column;align-items:center;gap:0.25rem;"
        >
          <span
            style="padding:0.25rem 0.75rem;border-radius:999px;font-size:0.72rem;backdrop-filter:blur(8px);background:rgba(0,0,0,0.5);color:#e2e8f0;"
          >
            {{ LOCAL_STATE_LABEL[localState] ?? localState }}
          </span>
          <span
            v-if="localTranscript"
            style="max-width:320px;padding:0.25rem 0.6rem;border-radius:8px;font-size:0.68rem;background:rgba(0,0,0,0.4);color:#94a3b8;text-align:center;word-break:break-word;"
          >
            👤 {{ localTranscript }}
          </span>
          <span
            v-if="localReply"
            style="max-width:320px;padding:0.25rem 0.6rem;border-radius:8px;font-size:0.68rem;background:rgba(99,102,241,0.35);color:#c7d2fe;text-align:center;word-break:break-word;"
          >
            🤖 {{ localReply }}
          </span>
          <span
            v-if="localError"
            style="max-width:320px;padding:0.25rem 0.6rem;border-radius:8px;font-size:0.68rem;background:rgba(239,68,68,0.35);color:#fecaca;text-align:center;word-break:break-word;"
          >
            ⚠️ {{ localError }} — hãy thử nói lại
          </span>
        </div>
      </div>
    </div>
  </BackgroundProvider>
</template>

<route lang="yaml">
name: IndexScenePage
meta:
  layout: stage
  stageTransition:
    name: bubble-wave-out
</route>
