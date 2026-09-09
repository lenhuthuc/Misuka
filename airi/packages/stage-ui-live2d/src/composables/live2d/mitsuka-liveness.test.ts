import type { MotionManagerPluginContext } from './motion-manager'

import parameters from '../../assets/live2d/parameters.json'
import { describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import { createMitsukaLivenessDriver } from './mitsuka-liveness'

const byId = new Map(parameters.map(parameter => [parameter.id, parameter]))
const owned = new Set(['ParamBreath', 'ParamBrowLX', 'ParamBrowRX', 'ParamBrowLAngle', 'ParamBrowRAngle', 'ParamBrowLForm', 'ParamBrowRForm', 'ParamBodyAngleX', 'ParamBodyAngleY', 'ParamBodyAngleZ'])

function context(now: number) {
  const writes: Array<{ id: string, value: number }> = []
  return {
    now,
    timeDelta: 1 / 60,
    modelParameters: ref({}),
    model: {
      setParameterValueById: vi.fn((id: string, value: number) => writes.push({ id, value })),
    },
    writes,
  } as unknown as MotionManagerPluginContext & { writes: Array<{ id: string, value: number }> }
}

describe('Mitsuka procedural liveness', () => {
  it('owns only breath and hair, with values clamped to the JSON registry', () => {
    const ctx = context(2.5)
    createMitsukaLivenessDriver(() => ({ v: 0, a: 1, d: 0 }))(ctx)

    expect(new Set(ctx.writes.map(write => write.id))).toEqual(owned)
    for (const write of ctx.writes) {
      const parameter = byId.get(write.id)!
      expect(write.value).toBeGreaterThanOrEqual(parameter.min)
      expect(write.value).toBeLessThanOrEqual(parameter.max)
    }
  })

  it('varies the fallback body pose over time without touching physics-owned hair', () => {
    const early = context(0)
    const late = context(1.2)
    const driver = createMitsukaLivenessDriver(() => ({ v: 0, a: 0, d: 0 }))
    driver(early)
    driver(late)

    const earlyBody = early.writes.find(write => write.id === 'ParamBodyAngleX')!.value
    const lateBody = late.writes.find(write => write.id === 'ParamBodyAngleX')!.value
    expect(lateBody).not.toBeCloseTo(earlyBody, 6)
    expect(early.writes.every(write => !write.id.startsWith('ParamHair'))).toBe(true)
  })
})
