import { errorMessageFrom } from '@moeru/std'
import { storeToRefs } from 'pinia'
import { ref } from 'vue'

import { useAudioContext, useSpeakingStore } from '../stores/audio'
import { describeHttpFailure, extractSentences, MIN_SENTENCE_LEN, parseChatStreamLine } from './local-conversation-sse'
import { createTtsPlaybackQueue } from './local-conversation-tts-queue'

export type LocalConvState = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'speaking'

export interface EmotionVAD {
  v: number
  a: number
  d: number
}

export interface UseLocalConversationOptions {
  /** Base URL of the Python VAD/Brain service. Default: http://localhost:8000 */
  baseUrl?: string
  /** BCP-47 language code for Whisper STT. Default: vi (Vietnamese) */
  language?: string
  /** Called after emotion analysis completes — use to update avatar expression. */
  onEmotion?: (emotion: EmotionVAD) => void
}

/**
 * Full local conversation pipeline:
 *   User speaks → Whisper STT → streaming RAG Chat → sentence-by-sentence Piper TTS
 *
 * TTS starts as soon as the first sentence boundary arrives from the LLM stream —
 * the user hears the response before generation is complete.
 *
 * Barge-in: calling onSpeechStart() cancels all queued/playing TTS immediately.
 */
export function useLocalConversation(options: UseLocalConversationOptions = {}) {
  const baseUrl = (options.baseUrl ?? 'http://localhost:8000').replace(/\/+$/, '')
  const language = options.language ?? 'vi'
  const onEmotion = options.onEmotion

  const state = ref<LocalConvState>('idle')
  const transcript = ref('')
  const reply = ref('')
  const error = ref<string>()

  // Local mode does not go through the cloud speech pipeline, so nothing else
  // populates the speaking store — without this the avatar stays mute-faced
  // while Piper audio plays.
  const { audioContext } = useAudioContext()
  const { mouthOpenSize, mouthOpenSource, nowSpeaking } = storeToRefs(useSpeakingStore())

  const ttsQueue = createTtsPlaybackQueue(baseUrl, {
    audioContext: () => audioContext,
    onSpeakingChange: (speaking) => { nowSpeaking.value = speaking },
    onMouthOpen: (value) => { mouthOpenSize.value = value },
  })

  // Per-turn abort controller — cancelled on barge-in or new turn start
  let _abort: AbortController | null = null
  // Bumped at the start of every turn. A turn resuming from an `await` compares
  // its own snapshot against this to tell whether a newer turn has superseded
  // it, instead of relying only on AbortController cancellation timing — an
  // aborted fetch can still let one already-buffered chunk through a `read()`
  // that was in flight before `abort()` took effect.
  let _generation = 0

  function _stopAll() {
    _abort?.abort()
    _abort = null
    // Aborting an in-flight sentence already closes the mouth through the
    // queue's abort handler; this covers the case where the fetch was cancelled
    // before playback ever started.
    nowSpeaking.value = false
    mouthOpenSize.value = 0
  }

  /** Call when VAD detects speech start — interrupts any in-progress TTS (barge-in). */
  function onSpeechStart() {
    if (state.value === 'speaking' || state.value === 'thinking') {
      _stopAll()
      state.value = 'listening'
    }
  }

  /** Call with the WAV blob when user finishes speaking. Runs full STT → Chat → TTS pipeline. */
  async function process(audioBlob: Blob | undefined) {
    console.info('[localConv] process() | blob size:', audioBlob?.size ?? 'undefined')
    if (!audioBlob || audioBlob.size === 0)
      return

    _stopAll()
    mouthOpenSource.value = 'local-conversation'
    const myGeneration = ++_generation
    const abort = new AbortController()
    _abort = abort
    // True once either a newer turn has started, or this turn was itself
    // barge-in cancelled — either way, this turn must stop touching state.
    const superseded = () => myGeneration !== _generation || abort.signal.aborted

    error.value = undefined

    // ── 1. emotion-vad: Whisper STT + audio emotion analysis (one call) ──────
    // /emotion-vad runs Whisper + wav2vec2 + PhoBERT concurrently and returns
    // both the transcript and fused VAD scores — avoids a second Whisper call.
    state.value = 'transcribing'
    console.info('[localConv] calling /emotion-vad at', baseUrl)

    const form = new FormData()
    form.append('audio', audioBlob, 'audio.wav')
    form.append('language', language)

    let text = ''
    try {
      const r = await fetch(`${baseUrl}/emotion-vad`, {
        method: 'POST',
        body: form,
        signal: abort.signal,
      })
      if (!r.ok)
        throw new Error(describeHttpFailure('emotion-vad failed', r.status, r.headers.get('X-Request-Id')))
      const data = await r.json() as {
        transcript: string
        fused: { valence: number, arousal: number, dominance: number }
      }
      text = (data.transcript ?? '').trim()
      console.info('[localConv] transcript:', text, '| fused VAD:', data.fused)
      // Notify caller so they can update the avatar expression
      if (!superseded() && data.fused && onEmotion) {
        onEmotion({ v: data.fused.valence, a: data.fused.arousal, d: data.fused.dominance })
      }
    }
    catch (e) {
      if (!superseded()) {
        error.value = errorMessageFrom(e) ?? 'emotion-vad request failed'
        state.value = 'idle'
      }
      return
    }

    if (superseded())
      return
    if (!text) {
      state.value = 'idle'
      return
    }
    transcript.value = text

    // ── 2. Streaming RAG Chat + sentence-by-sentence TTS ─────────────────────
    state.value = 'thinking'
    ttsQueue.reset()

    let response = ''
    let sentenceBuffer = ''
    let firstSentenceQueued = false

    try {
      const r = await fetch(`${baseUrl}/v1/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text }),
        signal: abort.signal,
      })
      if (!r.ok)
        throw new Error(describeHttpFailure('Chat failed', r.status, r.headers.get('X-Request-Id')))
      if (!r.body)
        throw new Error('Chat stream response has no body')

      const reader = r.body.getReader()
      const decoder = new TextDecoder()
      // SSE line buffer — a single network read may contain partial or multiple events
      let lineBuffer = ''
      let streamDone = false

      while (!streamDone) {
        const { done, value } = await reader.read()
        // A newer turn (or a barge-in) may have superseded this one while
        // `read()` was in flight — a chunk that was already buffered can
        // still resolve here even after `abort()` was called.
        if (superseded())
          return
        if (done)
          break

        lineBuffer += decoder.decode(value, { stream: true })

        // Process all complete lines (SSE events end with \n\n, lines separated by \n)
        const lines = lineBuffer.split('\n')
        // Keep the last (potentially incomplete) segment in buffer
        lineBuffer = lines.pop() ?? ''

        for (const line of lines) {
          const event = parseChatStreamLine(line)
          if (!event)
            continue

          if (event.type === 'done') {
            streamDone = true
            break
          }

          if (event.type === 'error') {
            // Backend still sends a `done` event after an in-band error —
            // keep reading so the loop terminates normally instead of hanging.
            error.value = event.message
            continue
          }

          if (event.type === 'emotion') {
            // The brain's own emotional state for this reply — it supersedes the
            // user-turn /emotion-vad reading above, so the avatar mirrors the
            // speaker while listening and then acts out its own answer.
            // `continue` (rather than falling through to `.content`) is what
            // stops this event from appending a literal "undefined" to `response`.
            onEmotion?.({ v: event.state.valence, a: event.state.arousal, d: event.state.dominance })
            continue
          }

          response += event.content
          reply.value = response
          sentenceBuffer += event.content

          const [sentences, remaining] = extractSentences(sentenceBuffer)
          sentenceBuffer = remaining

          for (const sentence of sentences) {
            ttsQueue.enqueue(sentence, abort)
            if (!firstSentenceQueued) {
              firstSentenceQueued = true
              state.value = 'speaking'
            }
          }
        }
      }
    }
    catch (e) {
      if (!superseded()) {
        error.value = errorMessageFrom(e) ?? 'chat stream request failed'
        state.value = 'idle'
      }
      return
    }

    if (superseded())
      return

    // Send any remaining text that didn't end with a terminator
    const tail = sentenceBuffer.trim()
    if (tail.length >= MIN_SENTENCE_LEN)
      ttsQueue.enqueue(tail, abort)

    if (!response.trim()) {
      state.value = 'idle'
      return
    }
    reply.value = response.trim()

    // ── 3. Wait for TTS queue to drain ───────────────────────────────────────
    if (!firstSentenceQueued) {
      // No sentence boundary hit — send entire response at once
      state.value = 'speaking'
      ttsQueue.enqueue(response.trim(), abort)
    }

    await ttsQueue.drain()

    if (!superseded() && state.value === 'speaking')
      state.value = 'idle'
  }

  function reset() {
    _stopAll()
    _generation++
    state.value = 'idle'
    transcript.value = ''
    reply.value = ''
    error.value = undefined
  }

  return {
    state,
    transcript,
    reply,
    error,
    onSpeechStart,
    process,
    reset,
  }
}
