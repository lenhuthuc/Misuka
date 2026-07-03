/**
 * Valence-Arousal-Dominance emotion dimensions, scaled to [-1, 1].
 *
 * Use when:
 * - Representing the emotional state of a speaker derived from audio or text
 * - Fusing audio and text emotion signals
 *
 * Expects:
 * - All three fields are in the range [-1, 1]
 *
 * Returns:
 * - A point in the PAD (Pleasure-Arousal-Dominance) emotional space
 */
export interface VADScores {
  /** Positive–negative affect. -1 = very negative, +1 = very positive. */
  valence: number
  /** Energy / activation level. -1 = very calm, +1 = very excited / agitated. */
  arousal: number
  /** Sense of control. -1 = feeling controlled, +1 = feeling in control. */
  dominance: number
}

export interface EmotionVADResult {
  transcript: string
  audio: VADScores
  text: VADScores
  fused: VADScores
}
