export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  /** Epoch millis — rendered as the bubble's timestamp. */
  at: number
  /** True while the assistant reply is still streaming in. */
  streaming?: boolean
  /** Set when the turn failed; rendered in place of the content. */
  error?: string
  /**
   * Object URL for an image attached to a user message. Local-api's chat
   * model is text-only, so this never reaches the AI — it is a client-side
   * attachment shown in the bubble, not something Mitsuka can see.
   */
  imageUrl?: string
}
