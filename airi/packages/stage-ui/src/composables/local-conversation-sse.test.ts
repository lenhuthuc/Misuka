import { describe, expect, it } from 'vitest'

import { describeHttpFailure, extractSentences, MIN_SENTENCE_LEN, parseChatStreamLine } from './local-conversation-sse'

describe('extractSentences', () => {
  it('splits complete sentences off a streaming buffer, keeping the tail unterminated', () => {
    const [sentences, remaining] = extractSentences('Hello. How are you? Fine')
    expect(sentences).toEqual(['Hello.', 'How are you?'])
    expect(remaining).toBe('Fine')
  })

  it('returns no sentences when the buffer has no terminator yet', () => {
    const [sentences, remaining] = extractSentences('Hello there')
    expect(sentences).toEqual([])
    expect(remaining).toBe('Hello there')
  })

  it(`drops sentences shorter than MIN_SENTENCE_LEN (${MIN_SENTENCE_LEN})`, () => {
    const [sentences] = extractSentences('Hi. How are you? Fine')
    expect(sentences).toEqual(['How are you?'])
  })
})

describe('parseChatStreamLine', () => {
  it('parses a content delta', () => {
    const event = parseChatStreamLine('data: {"type":"delta","turn_id":"t1","content":"xin chao"}')
    expect(event).toEqual({ type: 'delta', turnId: 't1', content: 'xin chao' })
  })

  it('parses a done event', () => {
    expect(parseChatStreamLine('data: {"type":"done","turn_id":"t1"}')).toEqual({ type: 'done', turnId: 't1' })
  })

  it('parses an in-band error payload with code/message/retryable', () => {
    const event = parseChatStreamLine(
      'data: {"type":"error","turn_id":"t1","error":{"code":"LLM_UNAVAILABLE","message":"ollama unreachable","retryable":true}}',
    )
    expect(event).toEqual({
      type: 'error',
      turnId: 't1',
      code: 'LLM_UNAVAILABLE',
      message: 'ollama unreachable',
      retryable: true,
    })
  })

  it('defaults retryable to true when the server omits it', () => {
    const event = parseChatStreamLine(
      'data: {"type":"error","turn_id":"t1","error":{"code":"LLM_UNAVAILABLE","message":"down"}}',
    )
    expect(event).toMatchObject({ retryable: true })
  })

  it('parses an emotion/state payload', () => {
    const event = parseChatStreamLine(
      'data: {"type":"emotion","turn_id":"t1","emotion":"joy","state":{"valence":0.5,"arousal":0.2,"dominance":0.1}}',
    )
    expect(event).toEqual({
      type: 'emotion',
      turnId: 't1',
      emotion: 'joy',
      state: { valence: 0.5, arousal: 0.2, dominance: 0.1 },
    })
  })

  it('does not misparse an emotion payload as a delta with an "undefined" content field', () => {
    // ROOT CAUSE:
    //
    // The pre-Phase-1 parser did `(JSON.parse(payload) as { content: string
    // }).content` unconditionally for every `data:` line. An
    // `{"emotion":...}` payload has no `.content` field, so this evaluated to
    // `undefined`, and the caller did `response += chunk` — silently
    // appending the literal string "undefined" into the assistant's
    // spoken/displayed reply.
    //
    // We fixed this by discriminating on the envelope's `type` field before
    // extracting any payload field, so an emotion event can never be read as
    // a delta.
    const event = parseChatStreamLine(
      'data: {"type":"emotion","turn_id":"t1","emotion":"joy","state":{"valence":0.5,"arousal":0.2,"dominance":0.1}}',
    )
    expect(event?.type).not.toBe('delta')
  })

  it('ignores SSE comments (": ping")', () => {
    expect(parseChatStreamLine(': ping')).toBeNull()
  })

  it('ignores blank lines', () => {
    expect(parseChatStreamLine('')).toBeNull()
  })

  it('ignores unparseable JSON instead of throwing', () => {
    expect(parseChatStreamLine('data: not json')).toBeNull()
  })

  it('ignores a payload missing turn_id', () => {
    expect(parseChatStreamLine('data: {"type":"delta","content":"hi"}')).toBeNull()
  })

  it('ignores a payload with an unrecognized type', () => {
    expect(parseChatStreamLine('data: {"type":"ping","turn_id":"t1"}')).toBeNull()
  })
})

describe('describeHttpFailure', () => {
  it('formats a failure without a request id', () => {
    expect(describeHttpFailure('emotion-vad failed', 404, null)).toBe('emotion-vad failed (404)')
  })

  it('appends the request id when present, for correlating to server logs', () => {
    expect(describeHttpFailure('Chat failed', 500, 'req-abc123')).toBe('Chat failed (500) [request req-abc123]')
  })
})
