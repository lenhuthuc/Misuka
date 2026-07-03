import type { EmotionVADResult, VADScores } from '../modules/emotion-vad'

import { useLocalStorage } from '@vueuse/core'
import { ref } from 'vue'

import { formatVADSystemPrompt, float32ToWav } from '../modules/emotion-vad'

/**
 * Composable for emotion Valence-Arousal-Dominance analysis.
 *
 * Use when:
 * - A user audio segment is ready (from VAD speech-ready event) and should be
 *   analyzed before the next LLM turn
 * - The fused emotional state needs to be injected into the system prompt
 *
 * Expects:
 * - `settings/server/baseUrl` localStorage key points to the running AIRI server
 * - The server has `HF_API_KEY` configured
 * - The user is authenticated (Bearer token in `settings/llm/apiKey`)
 *
 * Returns:
 * - Reactive `fusedVAD`, individual `audioVAD` / `textVAD`, and `transcript`
 * - `analyzeAudio(Float32Array, sampleRate)` to trigger server-side analysis
 * - `getSystemPromptContext()` to get the formatted injection string
 *
 * Call stack:
 *
 * analyzeAudio (useEmotionVAD)
 *   -> float32ToWav            (modules/emotion-vad/wav-encoder)
 *   -> POST /api/v1/audio/emotion-vad  (server)
 *        Promise.all([
 *          wav2vec2 emotion  → audioVAD
 *          Whisper ASR       → transcript → DistilRoBERTa → textVAD
 *        ])
 *        -> fuseVAD 70 % + 30 %
 *   <- { transcript, audio, text, fused }
 *   -> formatVADSystemPrompt   (modules/emotion-vad/fusion)
 */
export function useEmotionVAD() {
  const serverBaseUrl = useLocalStorage('settings/server/baseUrl', 'http://localhost:3000')
  const apiKey = useLocalStorage('settings/llm/apiKey', '')

  const transcript = ref<string>('')
  const audioVAD = ref<VADScores | null>(null)
  const textVAD = ref<VADScores | null>(null)
  const fusedVAD = ref<VADScores | null>(null)
  const isAnalyzing = ref(false)
  const error = ref<string | null>(null)

  /**
   * Sends a mono PCM Float32 audio segment to the server for parallel emotion
   * analysis. Updates all reactive refs on success.
   */
  async function analyzeAudio(audio: Float32Array, sampleRate: number = 16000): Promise<void> {
    if (!serverBaseUrl.value)
      return

    try {
      isAnalyzing.value = true
      error.value = null

      const wav = float32ToWav(audio, sampleRate)
      const formData = new FormData()
      formData.append('audio', new File([wav], 'segment.wav', { type: 'audio/wav' }))

      const base = serverBaseUrl.value.replace(/\/$/, '')
      const response = await fetch(`${base}/api/v1/audio/emotion-vad`, {
        method: 'POST',
        headers: apiKey.value ? { Authorization: `Bearer ${apiKey.value}` } : {},
        body: formData,
      })

      if (!response.ok) {
        const body = await response.text()
        throw new Error(`Server ${response.status}: ${body}`)
      }

      const result: EmotionVADResult = await response.json()
      transcript.value = result.transcript
      audioVAD.value = result.audio
      textVAD.value = result.text
      fusedVAD.value = result.fused
    }
    catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
    }
    finally {
      isAnalyzing.value = false
    }
  }

  /**
   * Returns the formatted VAD block for system prompt injection, or an empty
   * string if no analysis has been run yet.
   */
  function getSystemPromptContext(): string {
    if (!fusedVAD.value)
      return ''
    return formatVADSystemPrompt(fusedVAD.value)
  }

  return {
    serverBaseUrl,
    transcript,
    audioVAD,
    textVAD,
    fusedVAD,
    isAnalyzing,
    error,
    analyzeAudio,
    getSystemPromptContext,
  }
}
