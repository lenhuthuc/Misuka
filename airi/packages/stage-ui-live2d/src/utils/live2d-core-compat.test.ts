import { beforeEach, describe, expect, it, vi } from 'vitest'

import { applyLive2DCoreCompat } from './live2d-core-compat'

/**
 * `drawables` as Cubism Core 6 (SDK 5-r.5) actually returns it — measured
 * against `TiredGirl_V1.moc3`. Note `drawOrders` is present but is the authored
 * draw order (all 500 for that model), *not* a renamed `renderOrders`.
 */
function createCore6Model() {
  return {
    renderOrders: Int32Array.from([2, 0, 1]),
    drawables: {
      count: 3,
      drawOrders: Int32Array.from([500, 500, 500]),
      constantFlags: Uint8Array.from([0, 0, 0]),
      dynamicFlags: Uint8Array.from([1, 1, 1]),
    },
  }
}

function installCore(model: unknown) {
  const fromMoc = vi.fn(() => model)
  ;(globalThis as any).Live2DCubismCore = { Model: { fromMoc } }
  return fromMoc
}

describe('live2d cubism core compat', () => {
  beforeEach(() => {
    // The module applies itself once on import; each test needs a fresh module
    // instance so `applied` starts false again.
    vi.resetModules()
    delete (globalThis as any).Live2DCubismCore
  })

  /**
   * @example
   * expect(model.drawables.renderOrders).toEqual(model.renderOrders)
   */
  it('re-exposes Core 6 renderOrders on drawables for the Cubism 4 framework', async () => {
    const model = createCore6Model()
    installCore(model)

    const { applyLive2DCoreCompat: apply } = await import('./live2d-core-compat')
    expect(apply()).toBe(true)

    const created = (globalThis as any).Live2DCubismCore.Model.fromMoc(new ArrayBuffer(0))

    // ROOT CAUSE:
    //
    // Core 6 moved renderOrders from `drawables` to the model root. The bundled
    // Cubism 4 framework still reads `_model.drawables.renderOrders`, so
    // `doDrawModel` indexed into undefined on the very first rendered frame.
    expect(created.drawables.renderOrders).toBe(model.renderOrders)
    expect(Array.from(created.drawables.renderOrders)).toEqual([2, 0, 1])
    // The alias must track the live array, not a snapshot — the Core rewrites
    // it in place on every `model.update()`.
    model.renderOrders.set([1, 2, 0])
    expect(Array.from(created.drawables.renderOrders)).toEqual([1, 2, 0])
  })

  /**
   * @example
   * expect(model.drawables.renderOrders).toBe(original)
   */
  it('leaves a Core 5 model untouched', async () => {
    const original = Int32Array.from([0, 1, 2])
    const model = { drawables: { count: 3, renderOrders: original } }
    installCore(model)

    const { applyLive2DCoreCompat: apply } = await import('./live2d-core-compat')
    apply()

    const created = (globalThis as any).Live2DCubismCore.Model.fromMoc(new ArrayBuffer(0))
    expect(created.drawables.renderOrders).toBe(original)
  })

  /**
   * @example
   * expect(applyLive2DCoreCompat()).toBe(false)
   */
  it('reports failure while the core global is missing so callers can retry', async () => {
    const { applyLive2DCoreCompat: apply } = await import('./live2d-core-compat')
    expect(apply()).toBe(false)

    const model = createCore6Model()
    installCore(model)
    expect(apply()).toBe(true)
  })

  /**
   * @example
   * expect(fromMoc).toHaveBeenCalledTimes(1)
   */
  it('wraps fromMoc only once no matter how often it is called', async () => {
    const model = createCore6Model()
    const fromMoc = installCore(model)

    const { applyLive2DCoreCompat: apply } = await import('./live2d-core-compat')
    apply()
    const wrapped = (globalThis as any).Live2DCubismCore.Model.fromMoc
    apply()
    apply()

    expect((globalThis as any).Live2DCubismCore.Model.fromMoc).toBe(wrapped)

    wrapped(new ArrayBuffer(0))
    expect(fromMoc).toHaveBeenCalledTimes(1)
  })

  /**
   * @example
   * expect(created).toBeNull()
   */
  it('passes a null model straight through', async () => {
    installCore(null)

    const { applyLive2DCoreCompat: apply } = await import('./live2d-core-compat')
    apply()

    expect((globalThis as any).Live2DCubismCore.Model.fromMoc(new ArrayBuffer(0))).toBeNull()
  })
})

// Guards the import-time side effect that the tests above bypass via resetModules.
describe('live2d cubism core compat (import-time)', () => {
  it('is safe to import with no core present', () => {
    expect(typeof applyLive2DCoreCompat).toBe('function')
  })
})
