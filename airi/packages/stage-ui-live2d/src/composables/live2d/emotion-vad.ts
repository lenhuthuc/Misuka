import type { Ref } from 'vue'

import type { MotionManagerPlugin, MotionManagerPluginContext } from './motion-manager'

import { ref } from 'vue'

import { clampMitsukaParameter, getMitsukaParameter } from './mitsuka-parameters'

/** Valence, arousal, and dominance from the local emotion service. */
export interface EmotionVAD {
  v: number
  a: number
  d: number
}

const VAD_PARAMETER_IDS = ['ParamMouthForm', 'ParamCheek', 'ParamBrowLY', 'ParamBrowRY'] as const
const NEUTRAL: EmotionVAD = { v: 0, a: 0, d: 0 }
const TAU_SECONDS = 0.12

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export interface Live2DEmotionDriverOptions {
  source: () => EmotionVAD | undefined | null
  enabled?: () => boolean
  /** Retained for callers that expose an emotion on/off intensity control. */
  intensity?: () => number
  /** Retained for API compatibility; Mitsuka uses the documented 120 ms tau. */
  responseTime?: number
}

export interface Live2DEmotionDriver {
  /** Kept at zero: head tracking, not VAD, owns head parameters. */
  headAngle: { x: Ref<number>, y: Ref<number>, z: Ref<number> }
  current: () => EmotionVAD
  update: (ctx: MotionManagerPluginContext) => void
  resetToNeutral: (ctx: MotionManagerPluginContext) => void
}

/**
 * Mitsuka's VAD expression layer. Parameter ranges and defaults come solely
 * from `assets/live2d/parameters.json`, copied from the runtime package.
 */
export function createLive2DEmotionDriver(options: Live2DEmotionDriverOptions): Live2DEmotionDriver {
  const { source, enabled = () => true } = options
  const current: EmotionVAD = { ...NEUTRAL }
  const values = Object.fromEntries(VAD_PARAMETER_IDS.map(id => [id, getMitsukaParameter(id).default])) as Record<typeof VAD_PARAMETER_IDS[number], number>
  const headAngle = { x: ref(0), y: ref(0), z: ref(0) }

  function write(ctx: MotionManagerPluginContext, id: typeof VAD_PARAMETER_IDS[number], value: number) {
    values[id] = clampMitsukaParameter(id, value)
    ctx.model.setParameterValueById(id, values[id])
  }

  function resetToNeutral(ctx: MotionManagerPluginContext) {
    current.v = 0
    current.a = 0
    current.d = 0
    for (const id of VAD_PARAMETER_IDS)
      write(ctx, id, getMitsukaParameter(id).default)
  }

  function update(ctx: MotionManagerPluginContext) {
    const input = enabled() ? (source() ?? NEUTRAL) : NEUTRAL
    if (![input.v, input.a, input.d].every(Number.isFinite))
      throw new TypeError('V, A, and D must be finite')

    if (!Number.isFinite(ctx.timeDelta) || ctx.timeDelta < 0)
      throw new TypeError('deltaTime must be finite and non-negative')

    const v = clamp(input.v, -1, 1)
    const a = clamp(input.a, -1, 1)
    const d = clamp(input.d, -1, 1)
    const deltaTime = ctx.timeDelta
    const alpha = 1 - Math.exp(-deltaTime / TAU_SECONDS)

    current.v = v
    current.a = a
    current.d = d

    const nA = (a + 1) / 2
    const nD = (d + 1) / 2
    const positiveV = Math.max(v, 0)
    const negativeV = Math.max(-v, 0)
    const mouthForm = 0.55 * positiveV * (0.35 + 0.65 * nD) - 0.55 * negativeV * (0.45 + 0.55 * nA)
    const cheek = positiveV * (0.20 + 0.50 * nA)
    const browBase = 0.30 * positiveV * (0.25 + 0.75 * nD) - 0.45 * negativeV * (0.35 + 0.65 * nA)

    write(ctx, 'ParamMouthForm', values.ParamMouthForm + (clampMitsukaParameter('ParamMouthForm', mouthForm) - values.ParamMouthForm) * alpha)
    write(ctx, 'ParamCheek', values.ParamCheek + (clampMitsukaParameter('ParamCheek', cheek) - values.ParamCheek) * alpha)
    write(ctx, 'ParamBrowLY', values.ParamBrowLY + (clampMitsukaParameter('ParamBrowLY', browBase - 0.08 * d) - values.ParamBrowLY) * alpha)
    write(ctx, 'ParamBrowRY', values.ParamBrowRY + (clampMitsukaParameter('ParamBrowRY', browBase + 0.08 * d) - values.ParamBrowRY) * alpha)
  }

  return { headAngle, current: () => ({ ...current }), update, resetToNeutral }
}

/** Registers the VAD layer after blink and before lip-sync. */
export function useMotionUpdatePluginEmotionVAD(driver: Live2DEmotionDriver): MotionManagerPlugin {
  return ctx => driver.update(ctx)
}
