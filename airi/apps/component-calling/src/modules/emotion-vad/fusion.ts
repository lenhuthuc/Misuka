import type { VADScores } from './types'

const AUDIO_WEIGHT = 0.7
const TEXT_WEIGHT = 0.3

/**
 * Fuses audio-derived and text-derived VAD scores using a fixed 70/30 split.
 *
 * Use when:
 * - Both an audio and a text VAD estimate are available for the same utterance
 *
 * Expects:
 * - Both inputs are in [-1, 1] on all three dimensions
 *
 * Returns:
 * - A single VADScores in [-1, 1] representing the weighted combination
 */
export function fuseVAD(audioVAD: VADScores, textVAD: VADScores): VADScores {
  return {
    valence: audioVAD.valence * AUDIO_WEIGHT + textVAD.valence * TEXT_WEIGHT,
    arousal: audioVAD.arousal * AUDIO_WEIGHT + textVAD.arousal * TEXT_WEIGHT,
    dominance: audioVAD.dominance * AUDIO_WEIGHT + textVAD.dominance * TEXT_WEIGHT,
  }
}

/**
 * Formats a VADScores object as an XML-delimited block for system prompt injection.
 *
 * Use when:
 * - Injecting the fused emotional state into an LLM system prompt so the model
 *   can adapt its tone accordingly
 *
 * Expects:
 * - `vad` is in [-1, 1] on all three dimensions
 *
 * Returns:
 * - A multi-line string wrapped in <user_emotional_state> tags
 */
export function formatVADSystemPrompt(vad: VADScores): string {
  const f = (n: number) => (n >= 0 ? `+${n.toFixed(2)}` : n.toFixed(2))
  return [
    `<user_emotional_state>`,
    `Valence=${f(vad.valence)} Arousal=${f(vad.arousal)} Dominance=${f(vad.dominance)}`,
    `(scale: -1 to +1, inferred from voice and speech)`,
    `</user_emotional_state>`,
  ].join('\n')
}
