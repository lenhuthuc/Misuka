import type { MotionManagerPlugin, MotionManagerPluginContext } from './motion-manager'

import { clampMitsukaParameter } from './mitsuka-parameters'

export type MitsukaTouchArea = 'head' | 'chest' | 'leftArm' | 'rightArm' | 'torso' | 'legs'

export interface NormalizedTouchPoint {
  x: number
  y: number
}

/**
 * Maps a point inside the rendered model bounds to a stable semantic area.
 * The bundled model has no Cubism HitAreas, so this geometric fallback is
 * also what makes touch work on the local preset and its packed live copy.
 */
export function classifyMitsukaTouch(point: NormalizedTouchPoint): MitsukaTouchArea {
  const x = Math.min(1, Math.max(0, point.x))
  const y = Math.min(1, Math.max(0, point.y))

  if (y < 0.23)
    return 'head'

  // Shoulders sit above and outside the bust. Checking them first keeps a
  // sleeve/shoulder tap from triggering the embarrassed chest response.
  if (y < 0.42 && x < 0.34)
    return 'leftArm'
  if (y < 0.42 && x > 0.66)
    return 'rightArm'
  if (y < 0.61 && x >= 0.29 && x <= 0.71)
    return 'chest'
  if (y < 0.76 && x < 0.29)
    return 'leftArm'
  if (y < 0.76 && x > 0.71)
    return 'rightArm'
  if (y < 0.78)
    return 'torso'
  return 'legs'
}

type TouchPoseParameter
  = | 'ParamAngleX'
    | 'ParamAngleY'
    | 'ParamAngleZ'
    | 'ParamEyeLOpen'
    | 'ParamEyeROpen'
    | 'ParamEyeLSmile'
    | 'ParamEyeRSmile'
    | 'ParamBrowLY'
    | 'ParamBrowRY'
    | 'ParamBrowLAngle'
    | 'ParamBrowRAngle'
    | 'ParamMouthForm'
    | 'ParamCheek'
    | 'ParamBodyAngleX'
    | 'ParamBodyAngleY'
    | 'ParamBodyAngleZ'

type TouchPose = Partial<Record<TouchPoseParameter, number>>

const TOUCH_POSES: Record<MitsukaTouchArea, TouchPose> = {
  head: {
    ParamEyeLOpen: 0.16,
    ParamEyeROpen: 0.16,
    ParamEyeLSmile: 0.9,
    ParamEyeRSmile: 0.9,
    ParamMouthForm: 0.62,
    ParamCheek: 0.38,
    ParamAngleY: 6,
  },
  // A startled, embarrassed anime pose: blush, squeezed smiling eyes and a
  // small recoil. It deliberately does not modify mouth-open, which remains
  // owned by real audio lip sync.
  chest: {
    ParamEyeLOpen: 0.48,
    ParamEyeROpen: 0.48,
    ParamEyeLSmile: 0.7,
    ParamEyeRSmile: 0.7,
    ParamBrowLY: 0.34,
    ParamBrowRY: 0.34,
    ParamBrowLAngle: -0.42,
    ParamBrowRAngle: -0.42,
    ParamMouthForm: -0.2,
    ParamCheek: 1,
    ParamAngleX: -9,
    ParamAngleY: -5,
    ParamAngleZ: 5,
    ParamBodyAngleX: -5,
    ParamBodyAngleZ: 4,
  },
  leftArm: { ParamAngleX: 8, ParamBodyAngleX: 4, ParamBodyAngleZ: -3, ParamCheek: 0.24 },
  rightArm: { ParamAngleX: -8, ParamBodyAngleX: -4, ParamBodyAngleZ: 3, ParamCheek: 0.24 },
  torso: { ParamAngleY: 5, ParamBodyAngleY: -3, ParamMouthForm: 0.25, ParamCheek: 0.2 },
  legs: { ParamAngleY: -7, ParamBodyAngleY: 4, ParamBrowLY: 0.18, ParamBrowRY: 0.18, ParamCheek: 0.3 },
}

export interface MitsukaTouchReactionDriver extends MotionManagerPlugin {
  activate: (area: MitsukaTouchArea) => void
  activeArea: () => MitsukaTouchArea | undefined
}

export function createMitsukaTouchReactionDriver(): MitsukaTouchReactionDriver {
  let area: MitsukaTouchArea | undefined
  let elapsed = 0
  // Keep a quick phone tap visible long enough to read as a deliberate pose.
  const holdSeconds = 2
  const fadeSeconds = 0.7

  const driver = ((ctx: MotionManagerPluginContext) => {
    if (!area)
      return

    elapsed += Number.isFinite(ctx.timeDelta) ? Math.max(0, ctx.timeDelta) : 0
    const fade = elapsed <= holdSeconds ? 1 : Math.max(0, 1 - (elapsed - holdSeconds) / fadeSeconds)
    if (fade <= 0) {
      area = undefined
      return
    }

    const pose = TOUCH_POSES[area]
    for (const [id, target] of Object.entries(pose)) {
      const current = ctx.model.getParameterValueById(id)
      const value = current + (target! - current) * fade
      ctx.model.setParameterValueById(id, clampMitsukaParameter(id, value))
    }
  }) as MitsukaTouchReactionDriver

  driver.activate = (nextArea) => {
    area = nextArea
    elapsed = 0
  }
  driver.activeArea = () => area
  return driver
}
