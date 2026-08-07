/**
 * Pure parsing helpers for the local-conversation SSE stream and TTS sentence
 * buffering. Split out from `local-conversation.ts` so the wire-format and
 * text-chunking logic can be unit tested without touching `fetch`/`Audio`.
 *
 * The event shapes here mirror the backend's versioned envelope
 * (VAD/schemas/chat.py `ChatStreamEvent`): every event carries a
 * discriminating `type` and a `turnId`, so the client can route on shape
 * instead of guessing from which fields happen to be present.
 */

/** Sentences shorter than this are folded into the next chunk instead of being spoken on their own. */
export const MIN_SENTENCE_LEN = 5

/**
 * Extract complete sentences from a streaming text buffer.
 *
 * Before: "Hello. How are you? Fine"
 * After:  sentences=["Hello.", "How are you?"], remaining="Fine"
 */
export function extractSentences(buffer: string): [sentences: string[], remaining: string] {
  // Split on sentence terminators followed by whitespace
  const parts = buffer.split(/(?<=[.!?])\s+/)
  if (parts.length <= 1)
    return [[], buffer]
  const sentences = parts.slice(0, -1).map(s => s.trim()).filter(s => s.length >= MIN_SENTENCE_LEN)
  return [sentences, parts[parts.length - 1]]
}

/**
 * Formats an HTTP failure message, appending the server's request id (from
 * the `X-Request-Id` response header — see VAD/core/http_middleware.py) when
 * present, so a user-visible error can be correlated to a server log line.
 *
 * Before:
 * - describeHttpFailure('emotion-vad failed', 404, null)
 *   -> "emotion-vad failed (404)"
 * - describeHttpFailure('Chat failed', 500, 'req-abc123')
 *   -> "Chat failed (500) [request req-abc123]"
 */
export function describeHttpFailure(action: string, status: number, requestId: string | null): string {
  const suffix = requestId ? ` [request ${requestId}]` : ''
  return `${action} (${status})${suffix}`
}

/** A single decoded `/v1/chat/stream` SSE event, discriminated by `type`. */
export type ChatStreamEvent
  = | { type: 'delta', turnId: string, content: string }
    | { type: 'emotion', turnId: string, emotion: string, state: { valence: number, arousal: number, dominance: number } }
    | { type: 'error', turnId: string, code: string, message: string, retryable: boolean }
    | { type: 'done', turnId: string }

function isVADState(value: unknown): value is { valence: number, arousal: number, dominance: number } {
  if (typeof value !== 'object' || value === null)
    return false
  const state = value as Record<string, unknown>
  return typeof state.valence === 'number' && typeof state.arousal === 'number' && typeof state.dominance === 'number'
}

/**
 * Parse one raw SSE line from `/v1/chat/stream` into a typed event.
 *
 * Use when: reading lines out of the stream's decoded text buffer, one at a
 * time, in order.
 *
 * Expects: a single line (no trailing newline required either way).
 *
 * Returns: `null` for SSE comments (`: ping`), blank lines, and lines whose
 * `data:` payload isn't JSON or doesn't match the envelope's `type` — callers
 * should skip these rather than treat them as errors, so an older/newer
 * server version degrades gracefully instead of crashing the client.
 */
export function parseChatStreamLine(line: string): ChatStreamEvent | null {
  if (!line.startsWith('data: '))
    return null

  const payload = line.slice('data: '.length).trim()
  if (!payload)
    return null

  let parsed: unknown
  try {
    parsed = JSON.parse(payload)
  }
  catch {
    return null
  }

  if (typeof parsed !== 'object' || parsed === null)
    return null

  const obj = parsed as Record<string, unknown>
  if (typeof obj.turn_id !== 'string')
    return null
  const turnId = obj.turn_id

  switch (obj.type) {
    case 'delta':
      return typeof obj.content === 'string' ? { type: 'delta', turnId, content: obj.content } : null

    case 'emotion':
      return typeof obj.emotion === 'string' && isVADState(obj.state)
        ? { type: 'emotion', turnId, emotion: obj.emotion, state: obj.state }
        : null

    case 'error': {
      const err = obj.error as Record<string, unknown> | undefined
      return err && typeof err.code === 'string' && typeof err.message === 'string'
        ? { type: 'error', turnId, code: err.code, message: err.message, retryable: err.retryable !== false }
        : null
    }

    case 'done':
      return { type: 'done', turnId }

    default:
      return null
  }
}
