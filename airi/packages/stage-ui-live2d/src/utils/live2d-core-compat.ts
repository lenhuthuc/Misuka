/**
 * Bridges Live2D Cubism **Core 6** to the Cubism **4** Framework that
 * `pixi-live2d-display@0.4.0` bundles.
 *
 * Core 6 (SDK 5-r.5) is required to load `moc3` v6 models — Core 5.x rejects
 * them outright, see `vite/download-cubism-core.mjs`. But Core 6 also relocated
 * one member the old Framework reads:
 *
 *   Core 5: `model.drawables.renderOrders`
 *   Core 6: `model.renderOrders`   (`drawables.drawOrders` is the *authored*
 *                                   order, a different array — not a rename)
 *
 * The Framework reads it in exactly one place (`CubismModel.getDrawableRenderOrders`)
 * and never writes it, so re-exposing it on `drawables` is enough. Without this
 * the model loads and then dies on the first frame with
 * `TypeError: Cannot read properties of undefined (reading '0')` inside
 * `doDrawModel`.
 *
 * Everything else the Framework touches — `parameters`, `parts`, `canvasinfo`,
 * the `Utils.has*Bit` flag helpers, `constantFlags` blend bits — is unchanged in
 * Core 6 and needs no shim. Drop this module once pixi-live2d-display ships a
 * Cubism 5 Framework.
 */

interface CubismCoreDrawables {
  renderOrders?: Int32Array
}

interface CubismCoreModel {
  drawables?: CubismCoreDrawables
  renderOrders?: Int32Array
}

interface CubismCore {
  Model?: {
    fromMoc?: (...args: unknown[]) => CubismCoreModel | null
  }
}

let applied = false

/**
 * Idempotent, and a no-op when the Core global is absent or already exposes
 * `drawables.renderOrders` (Core 5), so it is safe to call eagerly and again
 * right before a model is created.
 */
export function applyLive2DCoreCompat(): boolean {
  if (applied)
    return true

  const core = (globalThis as { Live2DCubismCore?: CubismCore }).Live2DCubismCore
  const fromMoc = core?.Model?.fromMoc
  if (!core?.Model || !fromMoc)
    return false

  core.Model.fromMoc = function patchedFromMoc(this: unknown, ...args: unknown[]) {
    const model = fromMoc.apply(this, args)

    if (model?.drawables && model.drawables.renderOrders === undefined && model.renderOrders !== undefined) {
      Object.defineProperty(model.drawables, 'renderOrders', {
        get: () => model.renderOrders,
        configurable: true,
      })
    }

    return model
  }

  applied = true
  return true
}

applyLive2DCoreCompat()
