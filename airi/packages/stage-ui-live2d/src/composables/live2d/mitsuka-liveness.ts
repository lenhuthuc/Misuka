import type { EmotionVAD } from './emotion-vad'
import type { MotionManagerPlugin, MotionManagerPluginContext } from './motion-manager'

import { clampMitsukaParameter } from './mitsuka-parameters'

/**
 * Procedural movement for the brow and fallback body idle motion. Mitsuka's
 * exported Physics asset owns hair; tracking, blink, gaze, lip-sync, and the
 * four primary VAD expression parameters retain their dedicated owners.
 */
export function createMitsukaLivenessDriver(source: () => EmotionVAD | undefined | null): MotionManagerPlugin {
  return (ctx: MotionManagerPluginContext) => {
    const emotion = source()
    const arousal = Number.isFinite(emotion?.a) ? Math.min(1, Math.max(-1, emotion!.a)) : 0
    const energy = (arousal + 1) / 2
    const t = Number.isFinite(ctx.now) ? ctx.now : 0
    const wave = (frequency: number, phase = 0) => Math.sin(t * Math.PI * 2 * frequency + phase)

    const positiveV = Math.max(emotion?.v ?? 0, 0)
    const negativeV = Math.max(-(emotion?.v ?? 0), 0)
    const base = ctx.modelParameters.value ?? {}

    // A calm idle breath remains visible; arousal makes it marginally quicker
    // and deeper without ever exceeding the model's authored range.
    const breath = 0.5 + (0.16 + 0.10 * energy) * wave(0.18 + 0.05 * energy)
    ctx.model.setParameterValueById('ParamBreath', clampMitsukaParameter('ParamBreath', breath))

    // Secondary brow controls have their own owner, as required by the
    // parameter contract. They add subtle tension/relief without rewriting the
    // VAD-owned BrowLY / BrowRY parameters.
    const browX = 0.12 * negativeV - 0.06 * positiveV
    const browAngle = -0.28 * negativeV + 0.12 * positiveV
    const browForm = 0.22 * ((emotion?.v ?? 0))
    ctx.model.setParameterValueById('ParamBrowLX', clampMitsukaParameter('ParamBrowLX', (base.leftEyebrowLR ?? 0) + browX))
    ctx.model.setParameterValueById('ParamBrowRX', clampMitsukaParameter('ParamBrowRX', (base.rightEyebrowLR ?? 0) - browX))
    ctx.model.setParameterValueById('ParamBrowLAngle', clampMitsukaParameter('ParamBrowLAngle', (base.leftEyebrowAngle ?? 0) + browAngle))
    ctx.model.setParameterValueById('ParamBrowRAngle', clampMitsukaParameter('ParamBrowRAngle', (base.rightEyebrowAngle ?? 0) + browAngle))
    ctx.model.setParameterValueById('ParamBrowLForm', clampMitsukaParameter('ParamBrowLForm', (base.leftEyebrowForm ?? 0) + browForm))
    ctx.model.setParameterValueById('ParamBrowRForm', clampMitsukaParameter('ParamBrowRForm', (base.rightEyebrowForm ?? 0) + browForm))

    // A low-amplitude fallback body sway keeps the model alive when no body
    // tracker is connected. It composes with the manually configured baseline.
    ctx.model.setParameterValueById('ParamBodyAngleX', clampMitsukaParameter('ParamBodyAngleX', (base.bodyAngleX ?? 0) + (0.35 + 0.45 * energy) * wave(0.12, 0.5)))
    ctx.model.setParameterValueById('ParamBodyAngleY', clampMitsukaParameter('ParamBodyAngleY', (base.bodyAngleY ?? 0) + (0.22 + 0.30 * energy) * wave(0.16, 2.2)))
    ctx.model.setParameterValueById('ParamBodyAngleZ', clampMitsukaParameter('ParamBodyAngleZ', (base.bodyAngleZ ?? 0) + (0.30 + 0.35 * energy) * wave(0.10, 4.1)))
  }
}
