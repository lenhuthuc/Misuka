import { errorMessageFrom } from '@moeru/std'
import { storeToRefs } from 'pinia'
import { ref } from 'vue'

import { useAudioContext, useSpeakingStore } from '../stores/audio'
import { describeHttpFailure, parseChatStreamLine } from './local-conversation-sse'
import { createTtsPlayer, extractSpeakableChunks, notifySpeechFinished } from './local-conversation-tts'

export type LocalConvState = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'speaking'

export interface EmotionVAD {
  v: number
  a: number
  d: number
}

export interface UseLocalConversationOptions {
  /** Base URL of the Python VAD/Brain service. Default: http://127.0.0.1:8010 */
  baseUrl?: string
  /** BCP-47 language code sent with the audio. Default: vi (Vietnamese) */
  language?: string
  /** Called after emotion analysis completes — use to update avatar expression. */
  onEmotion?: (emotion: EmotionVAD) => void
}

/**
 * Full local conversation pipeline:
 *   User speaks → Sherpa STT → streaming RAG Chat → Piper, sentence by sentence
 *
 * Both the text and the speech stream. Waiting for `done` before speaking
 * anything meant the user heard nothing for the whole of generation — 12s of
 * decode on this machine, on top of a 6s time-to-first-token — so the reply is
 * now spoken a sentence at a time, starting as soon as the first sentence
 * boundary arrives.
 *
 * The one invariant that keeps this correct is that `pendingSpeech` holds
 * every character of the reply that has not been handed to the player, and
 * nothing else is ever enqueued from — not `response`, not `reply.value`. The
 * previous sentence-splitting implementation spoke single-sentence replies
 * twice precisely because it enqueued the tail buffer *and* the full response
 * (see `local-conversation-tts.ts`).
 *
 * Barge-in: calling onSpeechStart() aborts synthesis and playback immediately.
 */
export function useLocalConversation(options: UseLocalConversationOptions = {}) {
  const baseUrl = (options.baseUrl ?? 'http://127.0.0.1:8010').replace(/\/+$/, '')
  const language = options.language ?? 'vi'
  const onEmotion = options.onEmotion

  const state = ref<LocalConvState>('idle')
  const transcript = ref('')
  const reply = ref('')
  const error = ref<string>()
  // Incremented at the start of every turn, and paired with clearing
  // `transcript`/`reply` there. A UI rendering a thread needs to tell two turns
  // apart, and the text refs alone cannot do that: asking the same question
  // twice in a row leaves `transcript` on the same string, so a watcher on it
  // never fires for the second turn.
  const turn = ref(0)

  // Local mode does not go through the cloud speech pipeline, so nothing else
  // populates the speaking store — without this the avatar stays mute-faced
  // while Piper audio plays.
  const { audioContext } = useAudioContext()
  const { mouthOpenSize, mouthOpenSource, nowSpeaking } = storeToRefs(useSpeakingStore())

  const tts = createTtsPlayer(baseUrl, {
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
    // The queue's chains still hold the aborted turn's slots; a new turn must
    // not wait behind them (they resolve immediately, but `drain()` would then
    // be reporting on the wrong turn).
    tts.reset()
    // Aborting mid-playback already closes the mouth through the player's
    // abort handler; this covers the case where the synthesis fetch was
    // cancelled before playback ever started.
    nowSpeaking.value = false
    mouthOpenSize.value = 0
  }

  /**
   * Release the server-side wait for this turn's reply to be spoken.
   *
   * `/emotion-vad` raised that flag the moment the recording was uploaded, so
   * every path out of a turn has to lower it — a turn that dies before it can
   * speak owes the server the same signal a spoken one does, or the work
   * waiting behind it sits there until the server's own timeout.
   */
  function _releaseSpeechFlag() {
    void notifySpeechFinished(baseUrl)
  }

  /** Call when VAD detects speech start — interrupts any in-progress TTS (barge-in). */
  function onSpeechStart() {
    if (state.value === 'speaking' || state.value === 'thinking') {
      _stopAll()
      state.value = 'listening'
    }
  }

  /** Run the shared Chat -> TTS half of a speech or typed turn. */
  async function respond(text: string, abort: AbortController, myGeneration: number) {
    const superseded = () => myGeneration !== _generation || abort.signal.aborted

    transcript.value = text
    state.value = 'thinking'

    let response = ''
    let pendingSpeech = ''
    let spokeAnything = false

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
      let lineBuffer = ''
      let streamDone = false

      while (!streamDone) {
        const { done, value } = await reader.read()
        if (superseded())
          return
        if (done)
          break

        lineBuffer += decoder.decode(value, { stream: true })
        const lines = lineBuffer.split('\n')
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
            error.value = event.message
            continue
          }
          if (event.type === 'emotion') {
            onEmotion?.({ v: event.state.valence, a: event.state.arousal, d: event.state.dominance })
            continue
          }

          response += event.content
          reply.value = response
          pendingSpeech += event.content
          const [chunks, remaining] = extractSpeakableChunks(pendingSpeech)
          pendingSpeech = remaining

          for (const chunk of chunks) {
            tts.enqueue(chunk, undefined, abort)
            if (!spokeAnything) {
              spokeAnything = true
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
        _releaseSpeechFlag()
      }
      return
    }

    if (superseded())
      return
    if (!response.trim()) {
      state.value = 'idle'
      _releaseSpeechFlag()
      return
    }

    reply.value = response.trim()
    const tail = pendingSpeech.trim()
    if (tail) {
      tts.enqueue(tail, undefined, abort)
      if (!spokeAnything)
        state.value = 'speaking'
    }

    await tts.drain()
    if (superseded())
      return

    _releaseSpeechFlag()
    if (state.value === 'speaking')
      state.value = 'idle'
  }

  /** Start a local-api conversation turn from text typed in the Mitsuka UI. */
  async function processText(input: string) {
    const text = input.trim()
    if (!text)
      return

    _stopAll()
    mouthOpenSource.value = 'local-conversation'
    const myGeneration = ++_generation
    turn.value++
    const abort = new AbortController()
    _abort = abort
    error.value = undefined
    transcript.value = ''
    reply.value = ''

    await respond(text, abort, myGeneration)
  }

  /** Call with the WAV blob when user finishes speaking. Runs full STT → Chat → TTS pipeline. */
  async function process(audioBlob: Blob | undefined) {
    console.info('[localConv] process() | blob size:', audioBlob?.size ?? 'undefined')
    if (!audioBlob || audioBlob.size === 0)
      return

    _stopAll()
    mouthOpenSource.value = 'local-conversation'
    const myGeneration = ++_generation
    turn.value++
    const abort = new AbortController()
    _abort = abort
    // True once either a newer turn has started, or this turn was itself
    // barge-in cancelled — either way, this turn must stop touching state.
    const superseded = () => myGeneration !== _generation || abort.signal.aborted

    error.value = undefined
    // A voice turn only learns its transcript once STT returns; leaving the
    // previous turn's text in place until then would show it against this one.
    transcript.value = ''
    reply.value = ''

    // ── 1. emotion-vad: Sherpa STT + audio emotion analysis (one call) ───────
    // /emotion-vad runs Sherpa-ONNX + WavLM + PhoBERT over the same recording
    // and returns both the transcript and the fused V/A/D — one upload, not two.
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
        user_vad?: { valence: number, arousal: number, dominance: number }
      }
      text = (data.transcript ?? '').trim()
      console.info('[localConv] transcript:', text, '| user VAD:', data.user_vad)
      // Notify caller so they can update the avatar expression. The field is
      // `user_vad` (VAD/schemas/vad.py) — this used to read a `fused` key that
      // the endpoint has never sent, so the avatar simply never mirrored the
      // speaker. It is also in the checkpoints' [0, 1] range, while the Live2D
      // driver reads the signed [-1, 1] convention.
      if (!superseded() && data.user_vad && onEmotion) {
        onEmotion({
          v: data.user_vad.valence * 2 - 1,
          a: data.user_vad.arousal * 2 - 1,
          d: data.user_vad.dominance * 2 - 1,
        })
      }
    }
    catch (e) {
      if (!superseded()) {
        error.value = errorMessageFrom(e) ?? 'emotion-vad request failed'
        state.value = 'idle'
        _releaseSpeechFlag()
      }
      return
    }

    if (superseded())
      return
    if (!text) {
      // Silence, or nothing the recogniser could make out. There is no turn
      // to speak, so nothing should be waiting on one.
      state.value = 'idle'
      _releaseSpeechFlag()
      return
    }
    transcript.value = text

    // ── 2. Streaming RAG Chat, spoken as it arrives ──────────────────────────
    state.value = 'thinking'

    let response = ''
    // The reply's text that has not yet been handed to the TTS queue. Every
    // character leaves here exactly once — see this module's header.
    let pendingSpeech = ''
    let spokeAnything = false

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
            // `event.agentVad` — the reply's own reading — is deliberately not
            // used for speech: it only exists once the whole reply does, by
            // which point every sentence has already been queued. The server
            // reads each sentence's own emotion instead (`auto_prosody`).
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

          pendingSpeech += event.content
          const [chunks, remaining] = extractSpeakableChunks(pendingSpeech)
          pendingSpeech = remaining

          for (const chunk of chunks) {
            tts.enqueue(chunk, undefined, abort)
            if (!spokeAnything) {
              spokeAnything = true
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
        _releaseSpeechFlag()
      }
      return
    }

    if (superseded())
      return

    if (!response.trim()) {
      state.value = 'idle'
      _releaseSpeechFlag()
      return
    }
    reply.value = response.trim()

    // ── 3. Speak whatever the last boundary left behind ──────────────────────
    // The stream ends mid-buffer far more often than not: a reply whose final
    // sentence ends in "." has no whitespace after it, so no boundary closed
    // it. This is the whole of the unspoken remainder and the only place it is
    // read — emptying it here is what makes speaking it twice impossible.
    const tail = pendingSpeech.trim()
    pendingSpeech = ''
    if (tail) {
      tts.enqueue(tail, undefined, abort)
      if (!spokeAnything) {
        spokeAnything = true
        state.value = 'speaking'
      }
    }

    await tts.drain()

    if (superseded())
      return

    // The reply has been heard. Releases the server-side work that was staying
    // out of synthesis's way (VAD/core/llm_priority.py).
    _releaseSpeechFlag()

    if (state.value === 'speaking')
      state.value = 'idle'
  }

  function reset() {
    _stopAll()
    _generation++
    turn.value++
    state.value = 'idle'
    transcript.value = ''
    reply.value = ''
    error.value = undefined
  }

  return {
    state,
    turn,
    transcript,
    reply,
    error,
    onSpeechStart,
    process,
    processText,
    reset,
  }
}
