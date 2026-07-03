import { useLocalStorageManualReset } from '@proj-airi/stage-shared/composables'
import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface EmotionState {
  /** Valence: positive/negative dimension, range [-1, 1] */
  v: number
  /** Arousal: active/passive dimension, range [-1, 1] */
  a: number
  /** Dominance: dominant/submissive dimension, range [-1, 1] */
  d: number
}

const NEUTRAL_EMOTION: EmotionState = { v: 0, a: 0, d: 0 }

/**
 * Stores the current emotional state derived from the VAD (Valence-Arousal-Dominance)
 * sentiment analysis service and exposes `analyzeText` to run inference.
 *
 * Use when:
 * - You want to analyse the emotional content of a transcribed utterance
 * - You want to read the latest V/A/D scores to drive character expressions
 *
 * Expects:
 * - The local VAD Python service to be running at `baseUrl` (default http://localhost:8000)
 *
 * Returns:
 * - `emotion` – latest V/A/D values (updated after each successful call)
 * - `analyzeText(text)` – sends text to the service and updates `emotion`
 */
export const useEmotionStore = defineStore('modules:emotion', () => {
  const baseUrl = useLocalStorageManualReset<string>(
    'settings/emotion/vad-base-url',
    'http://localhost:8000',
  )

  const emotion = ref<EmotionState>({ ...NEUTRAL_EMOTION })
  const loading = ref(false)
  const error = ref<string>()

  /**
   * Send `text` to the VAD sentiment service and update `emotion` with the result.
   * Silently swallows network errors so a service outage never breaks the main flow.
   */
  async function analyzeText(text: string): Promise<EmotionState | undefined> {
    if (!text.trim())
      return undefined

    loading.value = true
    error.value = undefined

    try {
      const base = baseUrl.value.replace(/\/+$/, '')
      const res = await fetch(`${base}/vad`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      if (!res.ok) {
        error.value = `VAD service returned ${res.status}`
        return undefined
      }

      const data = await res.json() as { v: number, a: number, d: number }
      emotion.value = { v: data.v, a: data.a, d: data.d }
      return emotion.value
    }
    catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
      return undefined
    }
    finally {
      loading.value = false
    }
  }

  /**
   * Send a WAV audio blob to the `/emotion-vad` endpoint (audio + text fusion).
   * Updates `emotion` with the fused VAD scores. Used in local conversation mode
   * where we have the raw audio buffer from the VAD worklet.
   *
   * Silently swallows errors so a service outage never breaks the main flow.
   */
  async function analyzeAudio(audioBlob: Blob): Promise<EmotionState | undefined> {
    if (!audioBlob || audioBlob.size === 0)
      return undefined

    loading.value = true
    error.value = undefined

    try {
      const base = baseUrl.value.replace(/\/+$/, '')
      const form = new FormData()
      // Field name must be "audio" — matches the FastAPI alias in /emotion-vad
      form.append('audio', audioBlob, 'audio.wav')

      const res = await fetch(`${base}/emotion-vad`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        error.value = `emotion-vad returned ${res.status}`
        return undefined
      }

      const data = await res.json() as {
        transcript: string
        audio: { valence: number, arousal: number, dominance: number }
        text: { valence: number, arousal: number, dominance: number }
        fused: { valence: number, arousal: number, dominance: number }
      }
      emotion.value = {
        v: data.fused.valence,
        a: data.fused.arousal,
        d: data.fused.dominance,
      }
      return emotion.value
    }
    catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
      return undefined
    }
    finally {
      loading.value = false
    }
  }

  function reset() {
    emotion.value = { ...NEUTRAL_EMOTION }
    error.value = undefined
    baseUrl.reset()
  }

  return {
    baseUrl,
    emotion,
    loading,
    error,
    analyzeText,
    analyzeAudio,
    reset,
  }
})
