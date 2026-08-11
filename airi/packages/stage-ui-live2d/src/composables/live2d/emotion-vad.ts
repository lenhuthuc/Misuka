import type { Ref } from 'vue'

import type { MotionManagerPlugin, MotionManagerPluginContext } from './motion-manager'

import { ref } from 'vue'

/**
 * Valence / Arousal / Dominance, each in `[-1, 1]` — the output of the local
 * brain's emotion model (`apps/local-api/model/vad_model.py`, `tanh` head).
 */
export interface EmotionVAD {
  /** Unpleasant (-1) → pleasant (+1). */
  v: number
  /** Calm / drowsy (-1) → excited (+1). */
  a: number
  /** Submissive (-1) → in-control (+1). */
  d: number
}

const NEUTRAL: EmotionVAD = { v: 0, a: 0, d: 0 }

const TAU = Math.PI * 2

/**
 * Sign conventions for these two parameters are a rigging choice, not a Cubism
 * standard — a model whose brows read "angry" when they should read "sad" only
 * needs these flipped to -1.
 */
const BROW_ANGLE_SIGN = 1
const BROW_FORM_SIGN = 1

/** Cubism standard ranges for the parameters this driver writes. */
const RANGE_BROW = 1
const RANGE_MOUTH_FORM = 1
const RANGE_BODY_ANGLE = 10

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export interface Live2DEmotionDriverOptions {
  /** Latest V/A/D from the brain. Returning `undefined` is read as neutral. */
  source: () => EmotionVAD | undefined | null
  /** Master switch — when false the driver relaxes back to the neutral pose. */
  enabled?: () => boolean
  /** Scales every emotion-driven offset. 1 = as authored, 0 = neutral pose. */
  intensity?: () => number
  /**
   * Seconds for the rendered V/A/D to cover ~63% of the gap to a new target.
   * Emotion should land as a visible shift, not a snap.
   */
  responseTime?: number
}

export interface Live2DEmotionDriver {
  /**
   * Head pose offset in Cubism angle units, to be folded into the beat-sync
   * controller's base angles.
   *
   * `ParamAngleX/Y/Z` are re-written from the beat-sync spring every frame in
   * the `pre` stage, so a `final`-stage write here would be sprung back to the
   * base within a few frames. Moving the *base* instead is what makes the head
   * pose survive — and it inherits the spring's easing for free.
   */
  headAngle: { x: Ref<number>, y: Ref<number>, z: Ref<number> }
  /** The smoothed V/A/D currently being rendered. */
  current: () => EmotionVAD
  /** Writes one frame worth of parameters. Called by the motion-manager plugin. */
  update: (ctx: MotionManagerPluginContext) => void
}

/**
 * Drives a Live2D model's face and posture continuously from V/A/D, for models
 * that ship no motions or expressions (`TiredGirl_V1` has neither) and would
 * otherwise be frozen in their rest pose.
 *
 * The three axes are not mapped one-parameter-per-axis; they are first combined
 * into affect terms (`joy`, `anger`, `sorrow`, `energy`) because the same
 * negative valence reads as anger when dominant and as sadness when submissive,
 * and those want opposite brows.
 */
export function createLive2DEmotionDriver(options: Live2DEmotionDriverOptions): Live2DEmotionDriver {
  const {
    source,
    enabled = () => true,
    intensity = () => 1,
    responseTime = 0.45,
  } = options

  const smoothed: EmotionVAD = { ...NEUTRAL }
  const headAngle = { x: ref(0), y: ref(0), z: ref(0) }

  function set(ctx: MotionManagerPluginContext, id: string, value: number) {
    ctx.model.setParameterValueById(id, value)
  }

  function multiply(ctx: MotionManagerPluginContext, id: string, factor: number) {
    const current = ctx.model.getParameterValueById(id) as number
    ctx.model.setParameterValueById(id, current * factor)
  }

  function update(ctx: MotionManagerPluginContext) {
    // A backgrounded tab resumes with a multi-second delta; letting that through
    // would teleport the pose instead of easing into it.
    const dt = clamp(ctx.timeDelta, 0, 0.1)
    const target = enabled() ? (source() ?? NEUTRAL) : NEUTRAL
    const alpha = responseTime > 0 ? 1 - Math.exp(-dt / responseTime) : 1

    smoothed.v += (clamp(target.v, -1, 1) - smoothed.v) * alpha
    smoothed.a += (clamp(target.a, -1, 1) - smoothed.a) * alpha
    smoothed.d += (clamp(target.d, -1, 1) - smoothed.d) * alpha

    const k = clamp(intensity(), 0, 2)
    const { v, a, d } = smoothed
    const base = ctx.modelParameters.value ?? {}

    // --- Affect terms -------------------------------------------------------
    const joy = Math.max(0, v)
    const upset = Math.max(0, -v)
    /** 0 = drowsy, 1 = wired. */
    const energy = (a + 1) / 2
    /** Negative *and* in control *and* activated. */
    const anger = upset * Math.max(0, d) * energy
    /** Negative *and* not in control. */
    const sorrow = upset * (1 - Math.max(0, d))

    // --- Eyes ---------------------------------------------------------------
    // Multiplicative: the auto-blink plugin runs before this one and owns the
    // blink itself, so scaling its output keeps a blink fully closing the eyes
    // no matter how wide the emotion wants them.
    const eyeOpen = clamp(1 - k * (0.4 * (1 - energy) + 0.3 * anger + 0.2 * sorrow), 0.05, 1)
    multiply(ctx, 'ParamEyeLOpen', eyeOpen)
    multiply(ctx, 'ParamEyeROpen', eyeOpen)

    const eyeSmile = k * joy * (0.35 + 0.65 * energy)
    set(ctx, 'ParamEyeLSmile', clamp((base.leftEyeSmile ?? 0) + eyeSmile, 0, 1))
    set(ctx, 'ParamEyeRSmile', clamp((base.rightEyeSmile ?? 0) + eyeSmile, 0, 1))

    // --- Brows --------------------------------------------------------------
    // Raised by delight and by distress, pulled down and together by anger.
    const browY = k * (0.3 * joy + 0.45 * sorrow + 0.25 * joy * Math.max(0, a) - 0.8 * anger)
    const browAngle = BROW_ANGLE_SIGN * k * (0.7 * sorrow - 0.8 * anger)
    const browForm = BROW_FORM_SIGN * k * 0.6 * v
    // Brows squeeze inward when angry, spread when at ease.
    const browX = k * (0.35 * anger - 0.15 * joy)

    for (const side of ['L', 'R'] as const) {
      const mirror = side === 'L' ? 1 : -1
      const prefix = side === 'L' ? 'left' : 'right'
      set(ctx, `ParamBrow${side}Y`, clamp((base[`${prefix}EyebrowY`] ?? 0) + browY, -RANGE_BROW, RANGE_BROW))
      set(ctx, `ParamBrow${side}Angle`, clamp((base[`${prefix}EyebrowAngle`] ?? 0) + browAngle, -RANGE_BROW, RANGE_BROW))
      set(ctx, `ParamBrow${side}Form`, clamp((base[`${prefix}EyebrowForm`] ?? 0) + browForm, -RANGE_BROW, RANGE_BROW))
      set(ctx, `ParamBrow${side}X`, clamp((base[`${prefix}EyebrowLR`] ?? 0) + browX * mirror, -RANGE_BROW, RANGE_BROW))
    }

    // --- Mouth --------------------------------------------------------------
    set(ctx, 'ParamMouthForm', clamp((base.mouthForm ?? 0) + k * (0.85 * v - 0.25 * anger), -RANGE_MOUTH_FORM, RANGE_MOUTH_FORM))
    // Emotion deliberately contributes nothing to mouth *opening*: this is the
    // rest value the lip-sync plugin releases to, and a resting mouth is a
    // closed mouth. Everything expressive happens through ParamMouthForm.
    set(ctx, 'ParamMouthOpenY', clamp(base.mouthOpen ?? 0, 0, 1))

    // --- Cheek --------------------------------------------------------------
    set(ctx, 'ParamCheek', clamp((base.cheek ?? 0) + k * joy * (0.3 + 0.7 * energy), 0, 1))

    // --- Posture ------------------------------------------------------------
    // Breathing and sway never stop, but they widen with arousal — a calm model
    // barely drifts, an excited one is visibly restless.
    const t = ctx.now
    const life = 0.35 + 0.65 * energy
    const swayA = Math.sin(t * 0.11 * TAU)
    const swayB = Math.sin(t * 0.19 * TAU + 1.7)
    const swayC = Math.sin(t * 0.07 * TAU + 0.9)

    const bodyX = (base.bodyAngleX ?? 0) + k * 2.5 * d + life * (1.6 * swayA + 0.7 * swayB)
    // Shoulders drop when downcast, chest lifts when in control.
    const bodyY = (base.bodyAngleY ?? 0) + k * (2.5 * d - 3.5 * sorrow) + life * 0.8 * swayC
    const bodyZ = (base.bodyAngleZ ?? 0) + k * (1.5 * joy - 1.5 * sorrow) + life * 1.4 * swayB

    set(ctx, 'ParamBodyAngleX', clamp(bodyX, -RANGE_BODY_ANGLE, RANGE_BODY_ANGLE))
    set(ctx, 'ParamBodyAngleY', clamp(bodyY, -RANGE_BODY_ANGLE, RANGE_BODY_ANGLE))
    set(ctx, 'ParamBodyAngleZ', clamp(bodyZ, -RANGE_BODY_ANGLE, RANGE_BODY_ANGLE))

    // Head pitch/roll go through the beat-sync base instead of the parameter —
    // see `headAngle`. Yaw is left to the focus controller, which already owns
    // it for eye tracking.
    headAngle.x.value = 0
    headAngle.y.value = k * (6 * d - 9 * sorrow) + life * 2 * swayC
    headAngle.z.value = k * (2.5 * joy - 2 * sorrow) + life * 2.5 * swayA
  }

  return {
    headAngle,
    current: () => ({ ...smoothed }),
    update,
  }
}

/**
 * Final-stage plugin that applies {@link createLive2DEmotionDriver}.
 *
 * Must be registered *after* the auto-blink plugin (whose eye writes it scales)
 * and *before* the lip-sync plugin (whose release target is the resting
 * `ParamMouthOpenY` this plugin writes).
 */
export function useMotionUpdatePluginEmotionVAD(driver: Live2DEmotionDriver): MotionManagerPlugin {
  return ctx => driver.update(ctx)
}
