import type { EmotionVAD } from './emotion-vad'
import type { MotionManagerPluginContext } from './motion-manager'

import { describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import { createLive2DEmotionDriver } from './emotion-vad'

function createModel(initialValues: Record<string, number> = {}) {
  const values = new Map(Object.entries(initialValues))
  return {
    getParameterValueById: vi.fn((id: string) => values.get(id) ?? 1),
    setParameterValueById: vi.fn((id: string, value: number) => {
      values.set(id, value)
    }),
    values,
  }
}

function createContext(model = createModel()) {
  return {
    model,
    now: 0,
    timeDelta: 1 / 60,
    modelParameters: ref({
      leftEyeSmile: 0,
      rightEyeSmile: 0,
      leftEyebrowY: 0,
      rightEyebrowY: 0,
      leftEyebrowAngle: 0,
      rightEyebrowAngle: 0,
      leftEyebrowForm: 0,
      rightEyebrowForm: 0,
      leftEyebrowLR: 0,
      rightEyebrowLR: 0,
      mouthOpen: 0,
      mouthForm: 0,
      cheek: 0,
      bodyAngleX: 0,
      bodyAngleY: 0,
      bodyAngleZ: 0,
    }),
  } as unknown as MotionManagerPluginContext & { model: ReturnType<typeof createModel> }
}

function createDriver(vad: EmotionVAD, intensity = 1) {
  // responseTime 0 makes a single update land the full target, so assertions
  // read the mapping instead of the easing curve.
  return createLive2DEmotionDriver({ source: () => vad, intensity: () => intensity, responseTime: 0 })
}

describe('live2d emotion vad driver', () => {
  /**
   * @example
   * expect(model.values.get('ParamMouthForm')).toBeGreaterThan(0)
   */
  it('curves the mouth up for pleasant valence and down for unpleasant', () => {
    const happy = createContext()
    createDriver({ v: 0.8, a: 0.2, d: 0.2 }).update(happy)

    const sad = createContext()
    createDriver({ v: -0.8, a: -0.2, d: -0.4 }).update(sad)

    expect(happy.model.values.get('ParamMouthForm')!).toBeGreaterThan(0.5)
    expect(sad.model.values.get('ParamMouthForm')!).toBeLessThan(-0.5)
  })

  /**
   * @example
   * expect(model.values.get('ParamMouthOpenY')).toBe(0)
   */
  it('leaves the resting mouth closed so lip sync releases to a shut mouth', () => {
    const ctx = createContext()
    // Loudly excited: the axis most likely to talk a naive mapping into
    // hanging the jaw open between sentences.
    createDriver({ v: 0.9, a: 1, d: 0.9 }).update(ctx)

    expect(ctx.model.values.get('ParamMouthOpenY')).toBe(0)
  })

  /**
   * @example
   * expect(model.values.get('ParamEyeLOpen')).toBeLessThan(1)
   */
  it('droops the eyes when arousal is low', () => {
    const drowsy = createContext(createModel({ ParamEyeLOpen: 1, ParamEyeROpen: 1 }))
    createDriver({ v: 0, a: -1, d: 0 }).update(drowsy)

    const alert = createContext(createModel({ ParamEyeLOpen: 1, ParamEyeROpen: 1 }))
    createDriver({ v: 0, a: 1, d: 0 }).update(alert)

    expect(drowsy.model.values.get('ParamEyeLOpen')!).toBeLessThan(0.7)
    expect(alert.model.values.get('ParamEyeLOpen')!).toBe(1)
  })

  /**
   * @example
   * expect(model.values.get('ParamEyeLOpen')).toBe(0)
   */
  it('scales the eyes multiplicatively so a blink still closes them fully', () => {
    // The auto-blink plugin runs first and leaves 0 mid-blink; an absolute
    // write here would re-open the eyes on every blinked frame.
    const ctx = createContext(createModel({ ParamEyeLOpen: 0, ParamEyeROpen: 0 }))
    createDriver({ v: 0.5, a: 1, d: 0.5 }).update(ctx)

    expect(ctx.model.values.get('ParamEyeLOpen')).toBe(0)
    expect(ctx.model.values.get('ParamEyeROpen')).toBe(0)
  })

  /**
   * @example
   * expect(driver.headAngle.y.value).toBeLessThan(0)
   */
  it('drops the head and shoulders for submissive distress', () => {
    const ctx = createContext()
    const driver = createDriver({ v: -0.9, a: -0.3, d: -0.8 })
    driver.update(ctx)

    expect(driver.headAngle.y.value).toBeLessThan(-4)
    expect(ctx.model.values.get('ParamBodyAngleY')!).toBeLessThan(0)
  })

  /**
   * @example
   * expect(model.values.get('ParamBrowLAngle')).not.toBe(model.values.get('ParamBrowLAngle'))
   */
  it('splits negative valence into angry and sad brows by dominance', () => {
    const angry = createContext()
    createDriver({ v: -0.9, a: 0.9, d: 0.9 }).update(angry)

    const sad = createContext()
    createDriver({ v: -0.9, a: -0.2, d: -0.9 }).update(sad)

    // Same valence, opposite brows — the reason V/A/D is combined into affect
    // terms rather than mapped one parameter per axis.
    expect(angry.model.values.get('ParamBrowLY')!).toBeLessThan(0)
    expect(sad.model.values.get('ParamBrowLY')!).toBeGreaterThan(0)
    expect(angry.model.values.get('ParamBrowLAngle')!).toBeLessThan(0)
    expect(sad.model.values.get('ParamBrowLAngle')!).toBeGreaterThan(0)
  })

  /**
   * @example
   * expect(model.values.get('ParamMouthForm')).toBe(0)
   */
  it('collapses to the model base pose at zero intensity', () => {
    const ctx = createContext()
    createDriver({ v: 1, a: 1, d: 1 }, 0).update(ctx)

    expect(ctx.model.values.get('ParamMouthForm')).toBe(0)
    expect(ctx.model.values.get('ParamCheek')).toBe(0)
    expect(ctx.model.values.get('ParamBrowLY')).toBe(0)
  })

  /**
   * @example
   * expect(driver.current().v).toBeLessThan(1)
   */
  it('eases toward a new emotion instead of snapping to it', () => {
    const ctx = createContext()
    const driver = createLive2DEmotionDriver({ source: () => ({ v: 1, a: 1, d: 1 }), responseTime: 0.45 })

    driver.update(ctx)
    const afterOneFrame = driver.current().v

    expect(afterOneFrame).toBeGreaterThan(0)
    expect(afterOneFrame).toBeLessThan(0.2)

    // ~1s of frames gets most of the way there.
    for (let i = 0; i < 60; i++)
      driver.update(ctx)

    expect(driver.current().v).toBeGreaterThan(0.85)
  })

  /**
   * @example
   * expect(driver.current().v).toBeCloseTo(0)
   */
  it('relaxes back to neutral when disabled', () => {
    const ctx = createContext()
    const enabled = ref(true)
    const driver = createLive2DEmotionDriver({
      source: () => ({ v: 1, a: 1, d: 1 }),
      enabled: () => enabled.value,
      responseTime: 0,
    })

    driver.update(ctx)
    expect(driver.current().v).toBe(1)

    enabled.value = false
    driver.update(ctx)
    expect(driver.current().v).toBe(0)
  })
})
