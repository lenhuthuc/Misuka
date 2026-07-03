import type { Env } from '../../libs/env'
import type { AuthInstance } from '../../libs/auth'
import type { HonoEnv } from '../../types/hono'

import { Hono } from 'hono'
import { bodyLimit } from 'hono/body-limit'

import { resolveRequestAuth } from '../../libs/request-auth'
import {
  createBadGatewayError,
  createBadRequestError,
  createUnauthorizedError,
} from '../../utils/error'

interface EmotionVadDeps {
  env: Env
  auth: AuthInstance
}

/**
 * POST /api/v1/audio/emotion-vad
 *
 * Auth-gated proxy to the Python emotion-VAD microservice (services/vad).
 * Accepts a WAV audio segment as multipart `audio` field (up to 10 MB) and
 * forwards it verbatim to `VAD_SERVICE_URL/emotion-vad`.
 *
 * The Python service runs two branches in parallel:
 *   1. Audio → audeering/wav2vec2 emotion model → audioVAD
 *   2. Audio → Whisper ASR → transcript → DistilRoBERTa → textVAD
 * Then returns fused = 70 % audioVAD + 30 % textVAD.
 *
 * NOTICE: Registered on `app` BEFORE the global bodyLimit(1 MB) so the
 * per-route bodyLimit(10 MB) below wins for large audio uploads.
 */
export function createEmotionVadRoutes(deps: EmotionVadDeps) {
  return new Hono<HonoEnv>()
    .use('*', bodyLimit({ maxSize: 10 * 1024 * 1024 }))

    .post('/', async (c) => {
      // Manual auth: this route is registered before the global sessionMiddleware.
      const session = await resolveRequestAuth(deps.auth, deps.env, c.req.raw.headers)
      if (!session?.user) {
        throw createUnauthorizedError()
      }

      const body = await c.req.parseBody()
      const audioFile = body.audio

      if (!audioFile || typeof audioFile === 'string') {
        throw createBadRequestError('Missing `audio` file field in multipart body', 'MISSING_AUDIO')
      }

      // Forward the original file bytes to the Python service.
      const formData = new FormData()
      formData.append('audio', audioFile as File)

      const vadUrl = `${deps.env.VAD_SERVICE_URL.replace(/\/$/, '')}/emotion-vad`
      let upstream: Response

      try {
        upstream = await fetch(vadUrl, { method: 'POST', body: formData })
      }
      catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        throw createBadGatewayError(`VAD service unreachable: ${msg}`)
      }

      if (!upstream.ok) {
        const detail = await upstream.text()
        throw createBadGatewayError(`VAD service ${upstream.status}: ${detail}`)
      }

      const result = await upstream.json()
      return c.json(result)
    })
}
