/**
 * Serial TTS playback queue for the local-conversation pipeline.
 *
 * Use when: sentences from a streaming chat response need to be spoken in
 * the order they arrive, with the next sentence's audio fetched ahead of
 * time (pipelined) while playback itself stays strictly one-at-a-time.
 *
 * Expects: `enqueue` is called with sentences in speaking order, each with
 * the `AbortController` for the turn that produced them — aborting it stops
 * that sentence's fetch/playback (and everything queued behind it, since a
 * queue slot that finds `abort.signal.aborted` skips its own playback too).
 *
 * Returns: `drain()` resolves once every currently-queued sentence has
 * finished playing (or been skipped via abort).
 *
 * Playback goes through the Web Audio graph rather than an `<audio>` element
 * so the waveform is observable: an `AnalyserNode` tap turns the sentence into
 * a per-frame mouth opening, which is what drives the avatar's lip sync. An
 * `<audio>` element would play the same sound but leave the mouth shut.
 */

/** Per-sentence playback signals — wire these to the speaking store. */
export interface TtsPlaybackHooks {
  /**
   * Speech boundary for one sentence: `true` the moment audio starts, `false`
   * when it ends or is aborted. The avatar closes its mouth on `false`, so this
   * flipping between sentences is intended, not a glitch.
   */
  onSpeakingChange?: (speaking: boolean) => void
  /** Mouth opening in `[0, 1]`, emitted once per animation frame while sounding. */
  onMouthOpen?: (value: number) => void
}

export interface TtsPlaybackQueueOptions extends TtsPlaybackHooks {
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

export function createTtsPlaybackQueue(baseUrl: string, options: TtsPlaybackQueueOptions = {}) {
  const { audioContext, onSpeakingChange, onMouthOpen } = options

  let chain: Promise<void> = Promise.resolve()

  /** Start a fresh queue — call at the beginning of a new turn. */
  function reset(): void {
    chain = Promise.resolve()
  }

  function endSpeaking() {
    onMouthOpen?.(0)
    onSpeakingChange?.(false)
  }

  /**
   * Plays one decoded sentence, resolving when it finishes or is aborted.
   * Runs the mouth-open loop for as long as the sentence is sounding.
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

  function enqueue(sentence: string, abort: AbortController): void {
    // Start the TTS fetch immediately so it's ready when it's this sentence's turn
    const audioPromise = fetch(`${baseUrl}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: sentence, voice: 'default' }),
      signal: abort.signal,
    })
      .then(r => (r.ok ? r.arrayBuffer() : null))
      .catch(() => null)

    chain = chain.then(async () => {
      if (abort.signal.aborted)
        return
      const bytes = await audioPromise
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
    })
  }

  function drain(): Promise<void> {
    return chain
  }

  return { reset, enqueue, drain }
}
