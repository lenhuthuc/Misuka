import { describe, expect, it } from 'vitest'

import { describeHttpFailure, parseChatStreamLine } from './local-conversation-sse'
import { speechRequestBody } from './local-conversation-tts'

describe('parseChatStreamLine', () => {
  it('parses a content delta', () => {
    const event = parseChatStreamLine('data: {"type":"delta","turn_id":"t1","content":"xin chao"}')
    expect(event).toEqual({ type: 'delta', turnId: 't1', content: 'xin chao' })
  })

  it('parses a done event', () => {
    expect(parseChatStreamLine('data: {"type":"done","turn_id":"t1"}')).toEqual({ type: 'done', turnId: 't1' })
  })

  it('carries agent_vad off the done event, since that is what shapes TTS prosody', () => {
    const event = parseChatStreamLine(
      'data: {"type":"done","turn_id":"t1","agent_vad":{"mode":"text","valence":0.7,"arousal":0.6,"dominance":0.4}}',
    )
    expect(event).toEqual({
      type: 'done',
      turnId: 't1',
      agentVad: { mode: 'text', valence: 0.7, arousal: 0.6, dominance: 0.4 },
    })
  })

  it('omits agentVad when the server could not infer it, rather than inventing a neutral one', () => {
    const event = parseChatStreamLine('data: {"type":"done","turn_id":"t1","agent_vad":null}')
    expect(event).toEqual({ type: 'done', turnId: 't1' })
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

describe('speechRequestBody', () => {
  it('asks the server to derive prosody when the caller has no reading', () => {
    // A sentence spoken mid-stream never has one: the reply's own V/A/D is
    // only inferred once the whole reply exists, which is after every sentence
    // of it has already been queued. Without this the reply would be spoken in
    // the voice's flat default delivery.
    expect(speechRequestBody('Xin chào bạn.')).toEqual({
      input: 'Xin chào bạn.',
      voice: 'default',
      auto_prosody: true,
    })
  })

  it('passes V/A/D through when the turn produced a reading', () => {
    expect(speechRequestBody('xin chào', { valence: 0.7, arousal: 0.6, dominance: 0.4 })).toEqual({
      input: 'xin chào',
      voice: 'default',
      valence: 0.7,
      arousal: 0.6,
      dominance: 0.4,
    })
  })

  it('never mixes a measured reading with the derived one', () => {
    expect(speechRequestBody('xin chào', { valence: 0.7, arousal: 0.6, dominance: 0.4 }))
      .not
      .toHaveProperty('auto_prosody')
  })
})
