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
}
