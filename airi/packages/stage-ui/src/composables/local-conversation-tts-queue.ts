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
 */
export function createTtsPlaybackQueue(baseUrl: string) {
  let chain: Promise<void> = Promise.resolve()

  /** Start a fresh queue — call at the beginning of a new turn. */
  function reset(): void {
    chain = Promise.resolve()
  }

  function enqueue(sentence: string, abort: AbortController): void {
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

    chain = chain.then(async () => {
      if (abort.signal.aborted)
        return
      const url = await audioPromise
      if (!url || abort.signal.aborted)
        return
      await new Promise<void>((resolve) => {
        const audio = new Audio(url)
        const cleanup = () => {
          URL.revokeObjectURL(url)
          resolve()
        }
        audio.onended = cleanup
        audio.onerror = cleanup
        abort.signal.addEventListener('abort', () => {
          audio.pause()
          cleanup()
        }, { once: true })
        audio.play().catch(cleanup)
      })
    })
  }

  function drain(): Promise<void> {
    return chain
  }

  return { reset, enqueue, drain }
}
