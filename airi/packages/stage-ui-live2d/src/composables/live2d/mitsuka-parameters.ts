import parameters from '../../assets/live2d/parameters.json'

export interface MitsukaParameter {
  id: string
  min: number
  max: number
  default: number
}

export const mitsukaParameters = parameters as MitsukaParameter[]
const byId = new Map(mitsukaParameters.map(parameter => [parameter.id, parameter]))

export function getMitsukaParameter(id: string) {
  const parameter = byId.get(id)
  if (!parameter)
    throw new Error(`Missing Live2D parameter metadata for ${id}`)
  return parameter
}

export function clampMitsukaParameter(id: string, value: number) {
  const parameter = getMitsukaParameter(id)
  return Math.min(parameter.max, Math.max(parameter.min, value))
}
