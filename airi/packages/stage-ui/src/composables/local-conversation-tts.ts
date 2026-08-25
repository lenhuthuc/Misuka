/**
 * Sentence-paced TTS playback for the local-conversation pipeline.
 *
 * Use when: a chat reply is arriving as a token stream and should be heard
 * while it is still being written.
 *
 * Speaking only after the whole reply was generated left a measured 12s of
 * silence between the last word the user said and the first word they heard —
 * generation time the user simply waited through. `enqueue` speaks each
 * sentence as its boundary arrives: synthesis of the next sentence overlaps
 * playback of the current one, so after the first sentence the queue is
 * always ahead of the ear.
 *
 * The previous sentence-by-sentence implementation was removed over two
 * failures, both addressed here rather than avoided:
 *
 * - **Speaking the reply twice.** Its splitter only closed a sentence on a
 *   terminator *followed by whitespace*, so a reply ending in "." kept its
 *   whole text in the tail buffer; the caller enqueued that tail and then,
 *   seeing no sentence had ever been queued, enqueued the full reply behind
 *   it. `extractSpeakableChunks` returns the unconsumed remainder instead of
 *   a flag, and that remainder is the *only* text the caller may flush — so
 *   "spoken twice" is not a state this module can be put into.
 * - **Prosody with no clause to plan across.** A per-sentence request carried
 *   no V/A/D, because the reply's own reading only exists once the reply
 *   does. `speechRequestBody` now asks the server to read each sentence's
 *   emotion off that sentence (`auto_prosody`), and a sentence *is* the clause
 *   the Fujisaki contour is planned over (VAD/service/prosody.py).
 *
 * Playback goes through the Web Audio graph rather than an `<audio>` element
 * so the waveform is observable: an `AnalyserNode` tap turns the reply into a
 * per-frame mouth opening, which is what drives the avatar's lip sync. An
 * `<audio>` element would play the same sound but leave the mouth shut.
 */

/**
 * Sentences shorter than this are folded into the next one instead of being
 * spoken alone. A two-word "Vâng." is both a worse prosody unit than the
 * clause it belongs to and a whole HTTP round trip for a third of a second of
 * audio. Folding, not dropping: the old splitter filtered short fragments out
 * and silently lost them from the spoken reply.
 */
export const MIN_CHUNK_LEN = 12

/**
 * Boundary of a speakable chunk: sentence-final punctuation that some
 * whitespace follows, or a line break.
 *
 * Requiring the whitespace is what keeps "3.14" and a mid-stream "..." from
 * splitting — at the end of the buffer the terminator may still be growing,
 * and the caller's flush covers the genuine end of the reply. Closing quotes
 * and brackets are pulled in so `Cô ấy nói "xong."` closes after the quote.
 */
const CHUNK_BOUNDARY = /[.!?…]+[)\]"'”’]*(?=\s)|\n+/g

/**
 * Split a growing text buffer into what can be spoken now and what cannot yet.
 *
 * Before: `extractSpeakableChunks('Xin chào bạn. Mình khoẻ')`
 * After:  `[['Xin chào bạn.'], ' Mình khoẻ']`
 *
 * Expects: `buffer` is the text not yet returned as a chunk — feed the
 * remainder back in with the next tokens appended, never the whole reply.
 *
 * Returns: complete chunks in speaking order, and the unconsumed remainder.
 * Every character of `buffer` ends up in exactly one of the two, which is the
 * invariant that makes double-speaking impossible.
 */
export function extractSpeakableChunks(buffer: string): [chunks: string[], remaining: string] {
  const chunks: string[] = []
  // Everything before this index has been handed out; the rest is remainder.
  let consumed = 0

  CHUNK_BOUNDARY.lastIndex = 0
  let match = CHUNK_BOUNDARY.exec(buffer)
  while (match !== null) {
    const end = match.index + match[0].length
    const candidate = buffer.slice(consumed, end).trim()
    // Too short to speak on its own: leave `consumed` where it is so this
    // fragment is carried into the next chunk rather than lost.
    if (candidate.length >= MIN_CHUNK_LEN) {
      chunks.push(candidate)
      consumed = end
    }
    match = CHUNK_BOUNDARY.exec(buffer)
  }

  return [chunks, buffer.slice(consumed)]
}

/** Agent V/A/D in [0, 1] — the checkpoints' native range, as sent by `done`. */
export interface AgentVad {
  valence: number
  arousal: number
  dominance: number
}

/** Per-utterance playback signals — wire these to the speaking store. */
export interface TtsPlaybackHooks {
  /**
   * Speech boundary: `true` the moment audio starts, `false` when it ends or
   * is aborted. The avatar closes its mouth on `false`.
   */
  onSpeakingChange?: (speaking: boolean) => void
  /** Mouth opening in `[0, 1]`, emitted once per animation frame while sounding. */
  onMouthOpen?: (value: number) => void
}

export interface TtsPlayerOptions extends TtsPlaybackHooks {
  /**
   * Shared `AudioContext`. Omitted (or returning undefined) falls back to
   * `<audio>` playback, which still speaks but cannot drive lip sync.
   */
  audioContext?: () => AudioContext | undefined
}

/** Amplifies speech RMS (typically 0.05–0.3) into a usable mouth opening. */
const MOUTH_GAIN = 4.2
/** Softens peaks so the jaw doesn't slam fully open on every stressed vowel. */
const MOUTH_EXPONENT = 0.7
/** Per-frame approach rate toward the measured opening. */
const MOUTH_SMOOTHING = 0.35

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value))
}

/**
 * Reads the analyser's current waveform and turns it into a mouth opening.
 *
 * Before: a 1024-sample window centred on 128 (silence).
 * After:  0 for silence, ~0.6–0.9 for a spoken vowel.
 */
export function mouthOpenFromWaveform(samples: Uint8Array): number {
  if (samples.length === 0)
    return 0

  let sumOfSquares = 0
  for (let i = 0; i < samples.length; i++) {
    const centred = (samples[i] - 128) / 128
    sumOfSquares += centred * centred
  }

  const rms = Math.sqrt(sumOfSquares / samples.length)
  return clamp01((rms * MOUTH_GAIN) ** MOUTH_EXPONENT)
}

/**
 * Body for `POST /v1/audio/speech`.
 *
 * A partial reading is never sent: the keys are all present or all absent, so
 * the backend cannot treat a missing axis as neutral. With none of them, the
 * request opts into server-derived prosody instead.
 */
export function speechRequestBody(text: string, vad?: AgentVad): Record<string, unknown> {
  return {
    input: text,
    voice: 'default',
    ...(vad
      ? { valence: vad.valence, arousal: vad.arousal, dominance: vad.dominance }
      // No reading to send — a sentence spoken mid-stream never has one, since
      // the reply's own V/A/D is only inferred once the reply is complete. Ask
      // the server to read this sentence's emotion off its own words instead
      // of falling back to the voice's flat default delivery.
      : { auto_prosody: true }),
  }
}

export function createTtsPlayer(baseUrl: string, options: TtsPlayerOptions = {}) {
  const { audioContext, onSpeakingChange, onMouthOpen } = options

  function endSpeaking() {
    onMouthOpen?.(0)
    onSpeakingChange?.(false)
  }

  /**
   * Plays one decoded utterance, resolving when it finishes or is aborted.
   * Runs the mouth-open loop for as long as it is sounding.
   */
  function playThroughAudioGraph(context: AudioContext, buffer: AudioBuffer, abort: AbortController) {
    return new Promise<void>((resolve) => {
      const source = context.createBufferSource()
      source.buffer = buffer

      const analyser = context.createAnalyser()
      analyser.fftSize = 1024
      const samples = new Uint8Array(analyser.fftSize)

      source.connect(analyser)
      analyser.connect(context.destination)

      let frameId: number | undefined
      let smoothed = 0
      let finished = false

      const finish = () => {
        if (finished)
          return
        finished = true

        if (frameId !== undefined)
          cancelAnimationFrame(frameId)

        try {
          source.stop()
        }
        catch {
          // Already stopped — `stop()` on a finished source throws.
        }
        source.disconnect()
        analyser.disconnect()

        endSpeaking()
        resolve()
      }

      const tick = () => {
        analyser.getByteTimeDomainData(samples)
        const target = mouthOpenFromWaveform(samples)
        smoothed += (target - smoothed) * MOUTH_SMOOTHING
        onMouthOpen?.(smoothed)
        frameId = requestAnimationFrame(tick)
      }

      source.onended = finish
      abort.signal.addEventListener('abort', finish, { once: true })

      onSpeakingChange?.(true)
      source.start()
      frameId = requestAnimationFrame(tick)
    })
  }

  /** Fallback for environments without a usable AudioContext. No lip sync. */
  function playThroughAudioElement(bytes: ArrayBuffer, abort: AbortController) {
    return new Promise<void>((resolve) => {
      const url = URL.createObjectURL(new Blob([bytes]))
      const audio = new Audio(url)
      const cleanup = () => {
        URL.revokeObjectURL(url)
        endSpeaking()
        resolve()
      }

      audio.onended = cleanup
      audio.onerror = cleanup
      abort.signal.addEventListener('abort', () => {
        audio.pause()
        cleanup()
      }, { once: true })

      onSpeakingChange?.(true)
      audio.play().catch(cleanup)
    })
  }

  /**
   * Fetch one chunk's audio. Resolves to null on any failure — a TTS error
   * must not fail the turn, the user has already read the reply on screen.
   */
  async function synthesize(text: string, vad: AgentVad | undefined, abort: AbortController) {
    if (abort.signal.aborted)
      return null

    return fetch(`${baseUrl}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(speechRequestBody(text, vad)),
      signal: abort.signal,
    })
      .then(r => (r.ok ? r.arrayBuffer() : null))
      .catch(() => null)
  }

  /** Play one already-fetched chunk to completion, or skip it if aborted. */
  async function play(bytes: ArrayBuffer | null, abort: AbortController): Promise<void> {
    if (!bytes || abort.signal.aborted)
      return

    const context = audioContext?.()
    if (!context) {
      await playThroughAudioElement(bytes, abort)
      return
    }

    // A context created before any user gesture starts suspended; without this
    // the source plays silently and the mouth flaps over nothing.
    if (context.state === 'suspended')
      await context.resume().catch(() => {})

    // decodeAudioData detaches the buffer, so decode a copy — a retry or a
    // second consumer would otherwise see a zero-length ArrayBuffer.
    const decoded = await context.decodeAudioData(bytes.slice()).catch(() => null)
    if (!decoded || abort.signal.aborted)
      return

    await playThroughAudioGraph(context, decoded, abort)
  }

  // Two chains, deliberately. Playback is serial because speech is; synthesis
  // is serial too, one chunk ahead of the ear, because Piper renders on the
  // same CPU that is still decoding the rest of the reply — firing every
  // sentence's request at once (which the old queue did) turns one Ollama
  // stream plus five concurrent renders into a machine that finishes nothing.
  let synthesisChain: Promise<ArrayBuffer | null> = Promise.resolve(null)
  let playbackChain: Promise<void> = Promise.resolve()

  /** Start a fresh queue. Call at the start of every turn. */
  function reset(): void {
    synthesisChain = Promise.resolve(null)
    playbackChain = Promise.resolve()
  }

  /**
   * Queue one chunk to be spoken after everything already queued.
   *
   * Expects: chunks in speaking order, each with the turn's `AbortController`
   * — aborting it drops this chunk's fetch and playback, and every chunk
   * behind it, since each queue slot re-checks the signal before sounding.
   */
  function enqueue(text: string, vad: AgentVad | undefined, abort: AbortController): void {
    if (!text.trim())
      return

    const audio = synthesisChain.then(() => synthesize(text, vad, abort))
    // A rejected link would break every later `.then` in the chain, and one
    // sentence that failed to render must not silence the rest of the reply.
    synthesisChain = audio.catch(() => null)
    playbackChain = playbackChain
      .then(() => audio.catch(() => null))
      .then(bytes => play(bytes, abort))
      .catch(() => {})
  }

  /** Resolves once everything queued so far has finished playing or been skipped. */
  function drain(): Promise<void> {
    return playbackChain
  }

  /**
   * Speak `text` as a single utterance, start to finish.
   *
   * Use when: the whole text is already known — there is nothing to pipeline.
   */
  async function speak(text: string, vad: AgentVad | undefined, abort: AbortController): Promise<void> {
    reset()
    enqueue(text, vad, abort)
    await drain()
  }

  return { speak, reset, enqueue, drain }
}

/**
 * Tell the server the reply has been heard.
 *
 * Use when: the playback queue has drained at the end of a turn.
 *
 * The server holds background work (embedding the finished exchange) out of
 * the way of synthesis until this arrives — see VAD/core/llm_priority.py.
 * Failures are swallowed: the server has its own timeout behind this signal,
 * and a turn must never fail over a notification.
 */
export function notifySpeechFinished(baseUrl: string): Promise<void> {
  return fetch(`${baseUrl}/v1/audio/speech/finished`, { method: 'POST', keepalive: true })
    .then(() => {})
    .catch(() => {})
}
