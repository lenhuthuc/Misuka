import { ref } from 'vue'

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

const MIN_SENTENCE_LEN = 5

/**
 * Extract complete sentences from a streaming text buffer.
 *
 * Before: "Hello. How are you? Fine"
 * After:  sentences=["Hello.", "How are you?"], remaining="Fine"
 */
function extractSentences(buffer: string): [sentences: string[], remaining: string] {
  // Split on sentence terminators followed by whitespace
  const parts = buffer.split(/(?<=[.!?])\s+/)
  if (parts.length <= 1)
    return [[], buffer]
  const sentences = parts.slice(0, -1).map(s => s.trim()).filter(s => s.length >= MIN_SENTENCE_LEN)
  return [sentences, parts[parts.length - 1]]
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

  // Per-turn abort controller — cancelled on barge-in or new turn start
  let _abort: AbortController | null = null
  // Serial TTS playback queue — sentences chain as promises so they play in order
  let _ttsChain: Promise<void> = Promise.resolve()

  function _stopAll() {
    _abort?.abort()
    _abort = null
  }

  /**
   * Fetch TTS audio for one sentence and append it to the playback chain.
   * Fetches run in parallel (pipelined), but audio.play() is gated by the chain.
   */
  function _queueSentence(sentence: string, abort: AbortController) {
    // Start the TTS fetch immediately so it's ready when it's this sentence's turn
    const audioPromise = fetch(`${baseUrl}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: sentence, voice: 'default' }),
      signal: abort.signal,
    })
      .then(r => (r.ok ? r.blob() : null))
      .then(blob => blob ? URL.createObjectURL(blob) : null)
      .catch(() => null)

    _ttsChain = _ttsChain.then(async () => {
      if (abort.signal.aborted)
        return
      const url = await audioPromise
      if (!url || abort.signal.aborted)
        return
      await new Promise<void>((resolve) => {
        const audio = new Audio(url)
        const cleanup = () => { URL.revokeObjectURL(url); resolve() }
        audio.onended = cleanup
        audio.onerror = cleanup
        abort.signal.addEventListener('abort', () => { audio.pause(); cleanup() }, { once: true })
        audio.play().catch(cleanup)
      })
    })
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
    console.log('[localConv] process() | blob size:', audioBlob?.size ?? 'undefined')
    if (!audioBlob || audioBlob.size === 0)
      return

    _stopAll()
    error.value = undefined
    const abort = new AbortController()
    _abort = abort

    // ── 1. emotion-vad: Whisper STT + audio emotion analysis (one call) ──────
    // /emotion-vad runs Whisper + wav2vec2 + PhoBERT concurrently and returns
    // both the transcript and fused VAD scores — avoids a second Whisper call.
    state.value = 'transcribing'
    console.log('[localConv] calling /emotion-vad at', baseUrl)

    const form = new FormData()
    form.append('audio', audioBlob, 'audio.wav')

    let text = ''
    try {
      const r = await fetch(`${baseUrl}/emotion-vad`, {
        method: 'POST',
        body: form,
        signal: abort.signal,
      })
      if (!r.ok)
        throw new Error(`emotion-vad failed (${r.status})`)
      const data = await r.json() as {
        transcript: string
        fused: { valence: number, arousal: number, dominance: number }
      }
      text = (data.transcript ?? '').trim()
      console.log('[localConv] transcript:', text, '| fused VAD:', data.fused)
      // Notify caller so they can update the avatar expression
      if (data.fused && onEmotion) {
        onEmotion({ v: data.fused.valence, a: data.fused.arousal, d: data.fused.dominance })
      }
    }
    catch (e) {
      console.error('[localConv] emotion-vad error:', e)
      state.value = 'idle'
      return
    }

    if (!text || abort.signal.aborted) {
      state.value = 'idle'
      return
    }
    transcript.value = text

    // ── 2. Streaming RAG Chat + sentence-by-sentence TTS ─────────────────────
    state.value = 'thinking'
    _ttsChain = Promise.resolve()

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
        throw new Error(`Chat failed (${r.status})`)

      const reader = r.body!.getReader()
      const decoder = new TextDecoder()
      // SSE line buffer — a single network read may contain partial or multiple events
      let lineBuffer = ''

      outer: while (true) {
        const { done, value } = await reader.read()
        if (done)
          break

        lineBuffer += decoder.decode(value, { stream: true })

        // Process all complete lines (SSE events end with \n\n, lines separated by \n)
        const lines = lineBuffer.split('\n')
        // Keep the last (potentially incomplete) segment in buffer
        lineBuffer = lines.pop() ?? ''

        for (const line of lines) {
          // Skip SSE comments (": ping") and blank lines
          if (!line.startsWith('data: '))
            continue
          const payload = line.slice(6).trim()
          if (payload === '[DONE]')
            break outer
          if (!payload)
            continue

          let chunk: string
          try {
            chunk = (JSON.parse(payload) as { content: string }).content
          }
          catch {
            continue
          }

          response += chunk
          reply.value = response
          sentenceBuffer += chunk

          const [sentences, remaining] = extractSentences(sentenceBuffer)
          sentenceBuffer = remaining

          for (const sentence of sentences) {
            _queueSentence(sentence, abort)
            if (!firstSentenceQueued) {
              firstSentenceQueued = true
              state.value = 'speaking'
            }
          }
        }
      }
    }
    catch {
      if (!abort.signal.aborted)
        state.value = 'idle'
      return
    }

    // Send any remaining text that didn't end with a terminator
    const tail = sentenceBuffer.trim()
    if (tail.length >= MIN_SENTENCE_LEN)
      _queueSentence(tail, abort)

    if (!response.trim()) {
      state.value = 'idle'
      return
    }
    reply.value = response.trim()

    // ── 3. Wait for TTS queue to drain ───────────────────────────────────────
    if (!firstSentenceQueued) {
      // No sentence boundary hit — send entire response at once
      state.value = 'speaking'
      _queueSentence(response.trim(), abort)
    }

    await _ttsChain

    if (!abort.signal.aborted && state.value === 'speaking')
      state.value = 'idle'
  }

  function reset() {
    _stopAll()
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
