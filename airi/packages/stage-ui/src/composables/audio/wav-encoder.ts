/**
 * Encode a Float32Array of mono PCM samples to a standard 16-bit WAV Blob.
 *
 * Use when:
 * - You have raw audio samples from Web Audio API / VAD worklet
 * - You need a Blob that Whisper (or any STT API) can accept without re-encoding
 *
 * Expects:
 * - samples: mono channel, values in [-1, 1]
 * - sampleRate: matches the AudioContext rate used to capture (e.g. 16000)
 *
 * Returns:
 * - Blob with MIME type "audio/wav", RIFF/PCM-Int16 format
 */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const int16 = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
  }

  const dataBytes = int16.byteLength
  const buffer = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buffer)

  const write = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++)
      view.setUint8(offset + i, str.charCodeAt(i))
  }

  // RIFF header
  write(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)   // file size - 8
  write(8, 'WAVE')

  // fmt chunk
  write(12, 'fmt ')
  view.setUint32(16, 16, true)              // chunk size
  view.setUint16(20, 1, true)               // PCM = 1
  view.setUint16(22, 1, true)               // mono
  view.setUint32(24, sampleRate, true)      // sample rate
  view.setUint32(28, sampleRate * 2, true)  // byte rate (rate × channels × bitsPerSample/8)
  view.setUint16(32, 2, true)               // block align (channels × bitsPerSample/8)
  view.setUint16(34, 16, true)              // bits per sample

  // data chunk
  write(36, 'data')
  view.setUint32(40, dataBytes, true)
  new Int16Array(buffer, 44).set(int16)

  return new Blob([buffer], { type: 'audio/wav' })
}

/** Raise quiet microphone PCM to a useful level without clipping. */
export function normalizeSpeechSamples(
  samples: Float32Array,
  targetPeak = 0.9,
  maxGain = 24,
): Float32Array {
  if (samples.length === 0)
    return new Float32Array()

  let mean = 0
  for (const sample of samples)
    mean += sample
  mean /= samples.length

  let peak = 0
  for (const sample of samples)
    peak = Math.max(peak, Math.abs(sample - mean))

  if (peak < 1e-6)
    return new Float32Array(samples.length)

  const gain = Math.min(maxGain, targetPeak / peak)
  const normalized = new Float32Array(samples.length)
  for (let i = 0; i < samples.length; i++)
    normalized[i] = Math.max(-1, Math.min(1, (samples[i] - mean) * gain))

  return normalized
}
