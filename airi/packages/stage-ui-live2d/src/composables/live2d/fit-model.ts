import type { MaybeRefOrGetter } from 'vue'

import { isStageWeb } from '@proj-airi/stage-shared'
import { computed, toValue } from 'vue'

/**
 * Default framing, applied before the user's own view controls
 * (`settings/live2d/{position,scale}`) are layered on top.
 *
 * `fitFactor` is how many canvas heights tall the model is drawn at and
 * `anchorY` where its centre lands as a fraction of the canvas height. The
 * tamagotchi pet window keeps the original framing — twice the canvas height
 * anchored to the bottom edge, which crops the model to its upper half — while
 * the web stage shows the whole figure centred, because there it sits inside a
 * bordered panel rather than floating over the desktop.
 */
const framing = computed(() => isStageWeb()
  ? { fitFactor: 1, anchorY: 0.5 }
  : { fitFactor: 2, anchorY: 1 })

/**
 *  Normalizes the model so that user `scale == 1` fits the model to the canvas
 *  (twice the viewport height on non-web targets), and the model is centered
 *  horizontally when `position.x == 0`, centered vertically when
 *  `position.y == 0` on web / showing the upper half of the body elsewhere.
 */
export function useFitModel(
  canvasDim: MaybeRefOrGetter<{ width: number, height: number }>,
  modelDim: MaybeRefOrGetter<{ width: number, height: number }>,
) {
  const normalizedParam = computed(() => {
    const canvas = toValue(canvasDim)
    const model = toValue(modelDim)
    const { fitFactor, anchorY } = framing.value

    const heightScale = (canvas.height / model.height * fitFactor)
    const widthScale = (canvas.width / model.width * fitFactor)
    let minScale = Math.min(heightScale, widthScale)

    if (Number.isNaN(minScale) || minScale <= 0) {
      minScale = 1e-6
    }
    return {
      scale: minScale,
      x: canvas.width / 2,
      y: canvas.height * anchorY,
    }
  })

  return normalizedParam
}
