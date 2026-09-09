import type { EmotionVAD } from './emotion-vad'
import type { MotionManagerPluginContext } from './motion-manager'

import parameters from '../../assets/live2d/parameters.json'
import { describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import { createLive2DEmotionDriver } from './emotion-vad'

const byId = new Map(parameters.map(parameter => [parameter.id, parameter]))
const owned = new Set(['ParamMouthForm', 'ParamCheek', 'ParamBrowLY', 'ParamBrowRY'])

function createContext(deltaTime = 1) {
  const values = new Map<string, number>()
  const writes: Array<{ id: string, value: number }> = []
  return {
    model: {
      setParameterValueById: vi.fn((id: string, value: number) => {
        values.set(id, value)
        writes.push({ id, value })
      }),
    },
    timeDelta: deltaTime,
    modelParameters: ref({}),
    values,
    writes,
  } as unknown as MotionManagerPluginContext & { values: Map<string, number>, writes: Array<{ id: string, value: number }> }
}

function driver(vad: EmotionVAD) {
  return createLive2DEmotionDriver({ source: () => vad })
}

describe('Mitsuka VAD controller', () => {
  it('loads ranges and defaults from the runtime parameter registry', () => {
    const ctx = createContext()
    const subject = driver({ v: 0, a: 0, d: 0 })
    subject.resetToNeutral(ctx)
    for (const id of owned)
      expect(ctx.values.get(id)).toBe(byId.get(id)?.default)
  })

  it('maps positive and negative valence to meaningful mouth polarity', () => {
    const happy = createContext()
    driver({ v: 1, a: 1, d: 1 }).update(happy)
    expect(happy.values.get('ParamMouthForm')).toBeGreaterThan(0)
    expect(happy.values.get('ParamCheek')).toBeGreaterThan(0)

    const sad = createContext()
    driver({ v: -1, a: 1, d: 0 }).update(sad)
    expect(sad.values.get('ParamMouthForm')).toBeLessThan(0)
  })

  it('clamps supplied VAD values and every emitted parameter value', () => {
    const ctx = createContext()
    driver({ v: 999, a: -999, d: 5 }).update(ctx)
    for (const { id, value } of ctx.writes) {
      const parameter = byId.get(id)!
      expect(value).toBeGreaterThanOrEqual(parameter.min)
      expect(value).toBeLessThanOrEqual(parameter.max)
    }
  })

  it('writes only VAD-owned parameters', () => {
    const ctx = createContext()
    driver({ v: 0.8, a: 0.6, d: -0.4 }).update(ctx)
    expect(ctx.writes.every(({ id }) => owned.has(id))).toBe(true)
  })

  it('settles to equivalent values at 30, 60, and 120 FPS', () => {
    const settled = [30, 60, 120].map((fps) => {
      const ctx = createContext(1 / fps)
      const subject = driver({ v: 0.7, a: -0.3, d: 0.5 })
      for (let frame = 0; frame < fps; frame++)
        subject.update(ctx)
      return Object.fromEntries([...owned].map(id => [id, ctx.values.get(id)!]))
    })

    for (const id of owned) {
      expect(settled[0][id]).toBeCloseTo(settled[1][id], 6)
      expect(settled[1][id]).toBeCloseTo(settled[2][id], 6)
    }
  })

  it('rejects non-finite VAD input', () => {
    const ctx = createContext()
    expect(() => driver({ v: Number.NaN, a: 0, d: 0 }).update(ctx)).toThrow(TypeError)
  })
})
