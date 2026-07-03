<script setup lang="ts">
import { SpeechProviderSettings } from '@proj-airi/stage-ui/components'
import { useProvidersStore } from '@proj-airi/stage-ui/stores/providers'
import { Button } from '@proj-airi/ui'
import { onUnmounted, ref } from 'vue'

const providerId = 'vieneu-tts'
const defaultModel = 'vieneu'

const providersStore = useProvidersStore()

const testText = ref('Xin chào! Tôi là Mitsuka, trợ lý AI của cậu. Rất vui được gặp cậu hôm nay!')
const isPlaying = ref(false)
const statusMessage = ref('')
const errorMessage = ref('')
const sentenceCount = ref(0)

let audioCtx: AudioContext | null = null
let nextPlayTime = 0
let stopRequested = false

function getBaseURL(): string {
  const config = providersStore.getProviderConfig(providerId) as Record<string, unknown>
  return ((config?.baseUrl as string) || 'http://localhost:8000/v1').replace(/\/$/, '')
}

async function togglePlayback() {
  if (isPlaying.value) {
    stopRequested = true
    isPlaying.value = false
    statusMessage.value = 'Đã dừng.'
    return
  }

  errorMessage.value = ''
  sentenceCount.value = 0
  stopRequested = false
  isPlaying.value = true
  statusMessage.value = 'Đang kết nối...'

  try {
    if (!audioCtx || audioCtx.state === 'closed') {
      audioCtx = new AudioContext()
    }
    if (audioCtx.state === 'suspended') {
      await audioCtx.resume()
    }
    nextPlayTime = audioCtx.currentTime + 0.2

    const response = await fetch(`${getBaseURL()}/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: testText.value, voice: 'default', model: defaultModel }),
    })

    if (!response.ok || !response.body) {
      throw new Error(`Lỗi máy chủ TTS: ${response.status} ${response.statusText}`)
    }

    statusMessage.value = 'Đang tổng hợp câu đầu...'

    const reader = response.body.getReader()
    const SAMPLE_RATE = 24000
    const WAV_HEADER = 44
    let headerSkipped = 0

    while (!stopRequested) {
      const { done, value } = await reader.read()
      if (done)
        break

      // Skip WAV header bytes (server sends 44-byte header first)
      let pcm = value
      if (headerSkipped < WAV_HEADER) {
        const skip = Math.min(WAV_HEADER - headerSkipped, value.length)
        headerSkipped += skip
        if (skip >= value.length)
          continue
        pcm = value.slice(skip)
      }

      if (pcm.length < 2)
        continue

      // Decode PCM16-LE to Float32
      const samples = pcm.length >> 1
      const f32 = new Float32Array(samples)
      const dv = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength)
      for (let i = 0; i < samples; i++) {
        f32[i] = dv.getInt16(i * 2, true) / 32768.0
      }

      // Schedule this chunk for gapless playback
      const buf = audioCtx.createBuffer(1, samples, SAMPLE_RATE)
      buf.getChannelData(0).set(f32)
      const src = audioCtx.createBufferSource()
      src.buffer = buf
      src.connect(audioCtx.destination)

      const startAt = Math.max(nextPlayTime, audioCtx.currentTime + 0.05)
      src.start(startAt)
      nextPlayTime = startAt + samples / SAMPLE_RATE

      sentenceCount.value++
      statusMessage.value = `Đang phát câu ${sentenceCount.value}...`
    }

    statusMessage.value = stopRequested ? 'Đã dừng.' : 'Hoàn thành!'
  }
  catch (err) {
    errorMessage.value = err instanceof Error ? err.message : String(err)
    statusMessage.value = ''
  }
  finally {
    isPlaying.value = false
    stopRequested = false
  }
}

onUnmounted(() => {
  stopRequested = true
  audioCtx?.close()
})
</script>

<template>
  <SpeechProviderSettings :provider-id="providerId" :default-model="defaultModel">
    <template #playground>
      <div flex="~ col gap-4">
        <h2 class="text-lg text-neutral-500 md:text-2xl dark:text-neutral-400">
          Thử giọng nói (streaming)
        </h2>

        <div class="border border-blue-200 rounded-lg bg-blue-50 p-3 dark:border-blue-800 dark:bg-blue-900/20">
          <div class="flex items-center gap-2 text-blue-700 dark:text-blue-400">
            <div i-solar:info-circle-line-duotone class="text-sm" />
            <span class="text-xs">
              Audio phát từng câu khi VieNeu hoàn thành tổng hợp — mỗi câu ~5-7s trên CPU.
            </span>
          </div>
        </div>

        <textarea
          v-model="testText"
          rows="4"
          class="border border-neutral-200 rounded-lg bg-white p-3 text-sm dark:border-neutral-700 dark:bg-neutral-900 resize-y focus:outline-none focus:ring-2 focus:ring-primary-500"
          placeholder="Nhập văn bản tiếng Việt..."
        />

        <div class="flex items-center gap-3">
          <Button
            :disabled="false"
            class="flex items-center gap-2"
            @click="togglePlayback"
          >
            <div v-if="isPlaying" i-solar:stop-circle-bold-duotone class="text-lg" />
            <div v-else i-solar:play-circle-bold-duotone class="text-lg" />
            {{ isPlaying ? 'Dừng' : 'Phát thử' }}
          </Button>

          <div v-if="statusMessage" class="flex items-center gap-2 text-sm text-neutral-600 dark:text-neutral-400">
            <div
              v-if="isPlaying"
              class="animate-spin"
              i-solar:spinner-line-duotone
            />
            {{ statusMessage }}
          </div>
        </div>

        <div
          v-if="errorMessage"
          class="border border-red-200 rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400"
        >
          {{ errorMessage }}
        </div>
      </div>
    </template>
  </SpeechProviderSettings>
</template>

<route lang="yaml">
meta:
  layout: settings
  stageTransition:
    name: slide
</route>
