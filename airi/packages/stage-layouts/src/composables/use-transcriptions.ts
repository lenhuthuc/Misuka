import type { MaybeRefOrGetter, Ref } from 'vue'

import { useHearingSpeechInputPipeline, useHearingStore } from '@proj-airi/stage-ui/stores/modules/hearing'
import { useProvidersStore } from '@proj-airi/stage-ui/stores/providers'
import { useSettingsAudioDevice } from '@proj-airi/stage-ui/stores/settings'
import { until } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import { nextTick, onScopeDispose, ref, toValue, watch } from 'vue'

interface TranscriptionOptions {
  messageInputRef: Ref<string>
  sendMessage: () => void
  isStageTamagotchi: MaybeRefOrGetter<boolean>
}

// VAD thresholds for silence detection used by the file-based recording loop.
const VAD_SILENCE_THRESHOLD = 0.01  // RMS volume below this = silence
const VAD_SILENCE_DURATION_MS = 1500 // silence this long after speech = utterance end

export function useTranscriptions(options: TranscriptionOptions) {
  const { messageInputRef: messageInput, sendMessage, isStageTamagotchi } = options

  const hearingStore = useHearingStore()
  const audioDeviceSettingsStore = useSettingsAudioDevice()
  const hearingPipeline = useHearingSpeechInputPipeline()
  const { transcribeForMediaStream, stopStreamingTranscription, transcribeForRecording } = hearingPipeline
  const { supportsStreamInput } = storeToRefs(hearingPipeline)
  const { configured: hearingConfigured, autoSendEnabled, autoSendDelay } = storeToRefs(hearingStore)
  const { enabled: hearingEnabled, stream } = storeToRefs(audioDeviceSettingsStore)
  const providersStore = useProvidersStore()
  const { askPermission, startStream } = audioDeviceSettingsStore

  const isListening = ref(false)

  // Tracks the file-based VAD recording loop so stopStreaming can clean it up.
  let vadSession: {
    audioCtx: AudioContext
    analyser: AnalyserNode
    mediaRecorder: MediaRecorder
    pollInterval: ReturnType<typeof setInterval>
    silenceTimer?: ReturnType<typeof setTimeout>
  } | null = null

  // Auto-send logic
  let autoSendTimeout: ReturnType<typeof setTimeout> | undefined
  function clearPendingAutoSend() {
    if (autoSendTimeout) {
      clearTimeout(autoSendTimeout)
      autoSendTimeout = undefined
    }
  }
  async function debouncedAutoSend() {
    // Double-check auto-send is enabled before proceeding
    if (!autoSendEnabled.value) {
      clearPendingAutoSend()
      return
    }
    if (autoSendTimeout) {
      clearTimeout(autoSendTimeout)
    }

    autoSendTimeout = setTimeout(async () => {
      // Final check before sending - auto-send might have been disabled while waiting
      if (!autoSendEnabled.value) {
        clearPendingAutoSend()
        return
      }
      sendMessage()
      autoSendTimeout = undefined
    }, autoSendDelay.value)
  }

  function stopVadSession() {
    if (!vadSession)
      return
    clearInterval(vadSession.pollInterval)
    if (vadSession.silenceTimer)
      clearTimeout(vadSession.silenceTimer)
    try {
      if (vadSession.mediaRecorder.state !== 'inactive')
        vadSession.mediaRecorder.stop()
    }
    catch {}
    vadSession.audioCtx.close().catch(() => {})
    vadSession = null
  }

  const stopStreaming = async () => {
    if (!isListening.value)
      return

    try {
      console.info('Stopping transcription...', { source: 'useTranscriptions' })
      clearPendingAutoSend()
      stopVadSession()
      await stopStreamingTranscription(true)
      isListening.value = false
      console.info('Transcription stopped', { source: 'useTranscriptions' })
    }
    catch (err) {
      console.error('Error stopping transcription:', err, { source: 'useTranscriptions' })
      isListening.value = false
    }
  }

  // File-based VAD loop for non-streaming providers (e.g., local Whisper).
  // Records audio, detects silence after speech, sends the clip for transcription,
  // appends the result to the message input, then restarts for the next utterance.
  async function startVadRecordingLoop(micStream: MediaStream) {
    const audioCtx = new AudioContext()
    const analyser = audioCtx.createAnalyser()
    analyser.fftSize = 512
    const source = audioCtx.createMediaStreamSource(micStream)
    source.connect(analyser)

    const fftBuffer = new Float32Array(analyser.fftSize)
    const getRms = () => {
      analyser.getFloatTimeDomainData(fftBuffer)
      let sum = 0
      for (const v of fftBuffer)
        sum += v * v
      return Math.sqrt(sum / fftBuffer.length)
    }

    let mediaRecorder: MediaRecorder
    let chunks: Blob[] = []
    let hasSpeech = false
    let silenceTimer: ReturnType<typeof setTimeout> | undefined

    const startRecorder = () => {
      chunks = []
      hasSpeech = false
      silenceTimer = undefined
      mediaRecorder = new MediaRecorder(micStream)
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0)
          chunks.push(e.data)
      }
      mediaRecorder.onstop = async () => {
        if (!isListening.value)
          return
        if (hasSpeech && chunks.length) {
          const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' })
          try {
            const text = await transcribeForRecording(blob)
            if (text?.trim()) {
              const current = messageInput.value.trim()
              messageInput.value = current ? `${current} ${text.trim()}` : text.trim()
              debouncedAutoSend()
            }
          }
          catch (err) {
            console.error('VAD transcription error:', err, { source: 'useTranscriptions' })
          }
        }
        // Restart loop for the next utterance unless mic was turned off.
        if (isListening.value)
          startRecorder()
      }
      mediaRecorder.start(100) // collect in 100 ms chunks
    }

    startRecorder()

    const pollInterval = setInterval(() => {
      if (!isListening.value) {
        clearInterval(pollInterval)
        return
      }
      const rms = getRms()
      if (rms > VAD_SILENCE_THRESHOLD) {
        hasSpeech = true
        if (silenceTimer) {
          clearTimeout(silenceTimer)
          silenceTimer = undefined
        }
      }
      else if (hasSpeech && !silenceTimer) {
        // Start silence countdown — stop recorder after sustained silence.
        silenceTimer = setTimeout(() => {
          silenceTimer = undefined
          if (mediaRecorder.state === 'recording')
            mediaRecorder.stop()
        }, VAD_SILENCE_DURATION_MS)
      }
    }, 50)

    vadSession = { audioCtx, analyser, mediaRecorder: mediaRecorder!, pollInterval }
  }

  const startStreaming = async () => {
    console.info('Starting streaming transcription', {
      enabled: hearingEnabled.value,
      hasStream: !!stream.value,
      supportsStreamInput: supportsStreamInput.value,
      hearingConfigured: hearingConfigured.value,
    }, { source: 'useTranscriptions' })

    // Auto-configure Web Speech API as default if no provider is configured
    if (!hearingConfigured.value) {
      console.info('No transcription provider configured. Auto-configuring Web Speech API as default', { source: 'useTranscriptions' })
      // Check if Web Speech API is available in the browser
      // Web Speech API is NOT available in Electron (stage-tamagotchi) - it requires Google's embedded API keys
      // which are not available in Electron, causing it to fail at runtime
      const isWebSpeechAvailable = typeof window !== 'undefined'
        && !toValue(isStageTamagotchi) // Explicitly exclude Electron
        && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)

      if (!isWebSpeechAvailable) {
        // TODO: also propagate to user
        const errorMsg = 'Web Speech API is not available and no transcription provider is configured. Please go to Settings > Modules > Hearing to configure a transcription provider. '
        console.error(errorMsg, 'Browser support:', {
          hasWindow: typeof window !== 'undefined',
          hasWebkitSpeechRecognition: typeof window !== 'undefined' && 'webkitSpeechRecognition' in window,
          hasSpeechRecognition: typeof window !== 'undefined' && 'SpeechRecognition' in window,
        }, { source: 'useTranscriptions' })
        isListening.value = false
        return
      }

      // Initialize the provider in the providers store first
      try {
        providersStore.initializeProvider('browser-web-speech-api')
        hearingStore.activeTranscriptionProvider = 'browser-web-speech-api'
      }
      catch (err) {
        console.warn('Error initializing Web Speech API provider:', err, { source: 'useTranscriptions' })
      }
      // Wait for reactivity to update
      await nextTick()

      // Verify the provider was set to Web Speech API
      if (hearingStore.activeTranscriptionProvider !== 'browser-web-speech-api') {
        console.error('Failed to set Web Speech API as default provider', { source: 'useTranscriptions' })
        isListening.value = false
        return
      }
      console.info('Web Speech API configured as default provider', { source: 'useTranscriptions' })
    }

    // Web Speech API manages its own microphone — it does not use the MediaStream
    // from getUserMedia. Skip the stream requirement so it can start even when the
    // browser has denied getUserMedia (e.g., user clicked "Block" on the prompt).
    const isWebSpeechAPI = hearingStore.activeTranscriptionProvider === 'browser-web-speech-api'

    if (!isWebSpeechAPI) {
      try {
        // Request microphone permission if needed
        if (!stream.value) {
          console.info('Requesting microphone permission', { source: 'useTranscriptions' })
          await askPermission()

          if (!stream.value && hearingEnabled.value) {
            console.info('Attempting to start stream manually', { source: 'useTranscriptions' })
            startStream()
            try {
              await until(stream).toBeTruthy({ timeout: 3000, throwOnTimeout: true })
            }
            catch {
              console.error('Timed out waiting for audio stream. Stopping transcription.', { source: 'useTranscriptions' })
              isListening.value = false
              return
            }
          }
        }
      }
      catch (err) {
        console.error('Failed to request microphone permission:', err, { source: 'useTranscriptions' })
        isListening.value = false
      }

      if (!stream.value) {
        const errorMsg = 'Failed to get audio stream for transcription. Please check microphone permissions and ensure a device is selected.'
        console.error(errorMsg, { source: 'useTranscriptions' })
        isListening.value = false
        return
      }
    }

    // For file-based providers (e.g., local Whisper): use VAD-based recording loop
    // instead of the streaming path. Records utterances, sends each as a file, appends text.
    if (!supportsStreamInput.value) {
      if (!stream.value) {
        console.error('No audio stream available for VAD recording.', { source: 'useTranscriptions' })
        isListening.value = false
        return
      }
      console.info('Provider does not support streaming — starting VAD recording loop', { source: 'useTranscriptions' })
      isListening.value = true
      try {
        await startVadRecordingLoop(stream.value)
      }
      catch (err) {
        console.error('VAD recording loop failed to start:', err, { source: 'useTranscriptions' })
        isListening.value = false
      }
      return
    }

    console.info('Starting streaming transcription with stream:', stream.value?.id ?? '(web-speech-api)', { source: 'useTranscriptions' })

    const mediaStream = stream.value ?? new MediaStream()
    try {
      await transcribeForMediaStream(mediaStream, {
        onSentenceEnd: (delta) => {
          if (delta && delta.trim()) {
            console.info('Received transcription delta:', delta, { source: 'useTranscriptions' })
            const currentText = messageInput.value.trim()
            messageInput.value = currentText ? `${currentText} ${delta}` : delta
            debouncedAutoSend()
          }
        },
        // Omit onSpeechEnd to avoid re-adding user-deleted text; use sentence deltas only.
      })

      isListening.value = true
      console.info('Streaming transcription initiated successfully', { source: 'useTranscriptions' })
    }
    catch (err) {
      console.error('Transcription error:', err, { source: 'useTranscriptions' })
      isListening.value = false
      throw err
    }
  }

  // Watch for auto-send setting changes and clear pending sends if disabled
  watch(autoSendEnabled, (enabled) => {
    if (!enabled) {
      clearPendingAutoSend()
      console.info('Auto-send disabled', { source: 'useTranscriptions' })
    }
  })

  watch(hearingEnabled, async (enabled) => {
    if (!enabled) {
      await stopStreaming()
      console.info('Stopping streaming transcription because hearing is disabled.', { source: 'useTranscriptions' })
    }
  })

  onScopeDispose(() => {
    clearPendingAutoSend()
    stopStreaming()
  })

  return {
    startStreamingTranscription: startStreaming,
    stopStreamingTranscription: stopStreaming,
    isListening,
    autoSendEnabled,
  }
}
