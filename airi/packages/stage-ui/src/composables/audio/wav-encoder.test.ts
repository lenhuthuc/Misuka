import { describe, expect, it } from 'vitest'

import { normalizeSpeechSamples } from './wav-encoder'

describe('normalizeSpeechSamples', () => {
  it('raises a quiet signal without clipping', () => {
    const result = normalizeSpeechSamples(new Float32Array([-0.04, 0, 0.04]))

    expect(Math.max(...result.map(Math.abs))).toBeCloseTo(0.9, 5)
    expect(result.every(sample => Math.abs(sample) <= 1)).toBe(true)
  })

  it('does not amplify digital silence', () => {
    expect([...normalizeSpeechSamples(new Float32Array(32))]).toEqual([...new Float32Array(32)])
  })

  it('caps gain for extremely low-level noise', () => {
    const result = normalizeSpeechSamples(new Float32Array([-0.00001, 0.00001]))

    expect(Math.max(...result.map(Math.abs))).toBeLessThanOrEqual(0.00024)
  })
})
