import { describe, expect, it } from 'vitest'

import { classifyMitsukaTouch } from './mitsuka-touch'

describe('classifyMitsukaTouch', () => {
  it.each([
    [{ x: 0.5, y: 0.12 }, 'head'],
    [{ x: 0.5, y: 0.48 }, 'chest'],
    [{ x: 0.18, y: 0.32 }, 'leftArm'],
    [{ x: 0.82, y: 0.32 }, 'rightArm'],
    [{ x: 0.1, y: 0.5 }, 'leftArm'],
    [{ x: 0.9, y: 0.5 }, 'rightArm'],
    [{ x: 0.5, y: 0.65 }, 'torso'],
    [{ x: 0.5, y: 0.9 }, 'legs'],
  ] as const)('maps %o to %s', (point, expected) => {
    expect(classifyMitsukaTouch(point)).toBe(expected)
  })

  it('clamps points outside the model bounds', () => {
    expect(classifyMitsukaTouch({ x: -2, y: -1 })).toBe('head')
    expect(classifyMitsukaTouch({ x: 3, y: 4 })).toBe('legs')
  })
})
