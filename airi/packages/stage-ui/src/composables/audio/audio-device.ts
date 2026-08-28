import { errorMessageFrom } from '@moeru/std'
import { useDevicesList, useUserMedia } from '@vueuse/core'
import { computed, nextTick, ref, watch } from 'vue'

export function useAudioDevice(requestPermission: boolean = false) {
  const { audioInputs, permissionGranted, ensurePermissions } = useDevicesList({ constraints: { audio: true }, requestPermissions: requestPermission })
  const selectedAudioInput = ref<string>(audioInputs.value.find(device => device.deviceId === 'default')?.deviceId || '')
  const deviceConstraints = computed<MediaStreamConstraints>(() => {
    const audio: MediaTrackConstraints = {
      autoGainControl: true,
      echoCancellation: true,
      noiseSuppression: true,
    }

    // `deviceId: { exact }` is a *hard* constraint: a miss makes getUserMedia
    // reject with OverconstrainedError and hand back no stream at all. Two
    // everyday cases hit it. Before microphone permission is granted,
    // enumerateDevices() reports every deviceId as an empty string, so this
    // asked for `{ exact: '' }` — a device that cannot exist. And an id
    // persisted in `settings/audio/input` outlives the device it names, so
    // unplugging a headset leaves the same impossible request behind.
    //
    // Pin the device only while the browser is actually offering it; otherwise
    // let it pick the default, which is what the user wanted either way.
    if (selectedAudioInput.value && audioInputs.value.some(device => device.deviceId === selectedAudioInput.value))
      audio.deviceId = { exact: selectedAudioInput.value }

    return { audio }
  })
  const { stream, stop: stopStream, start: startUserMedia } = useUserMedia({ constraints: deviceConstraints, enabled: false, autoSwitch: true })

  /** Why the microphone last refused to open, for UI that wants to say so. */
  const streamError = ref<string>()

  /**
   * `useUserMedia().start()` lets getUserMedia's rejection straight through,
   * and callers so far invoked it fire-and-forget. A denied or over-constrained
   * microphone therefore surfaced as nothing but an unhandled rejection, while
   * the rest of the app sat waiting on a stream that was never coming. Name the
   * failure here so every caller inherits a usable error.
   */
  async function startStream() {
    streamError.value = undefined
    try {
      const started = await startUserMedia()
      if (!started) {
        streamError.value = 'getUserMedia resolved without a stream'
        console.error('[audio-device] getUserMedia resolved without a stream', deviceConstraints.value)
      }

      return started
    }
    catch (error) {
      streamError.value = errorMessageFrom(error) ?? String(error)
      console.error('[audio-device] getUserMedia failed', { error, constraints: deviceConstraints.value })
      throw error
    }
  }

  // The browser is free to silently ignore `echoCancellation`/`noiseSuppression`
  // requested above — Chrome on Windows in particular drops them for some
  // device/driver combinations without throwing. Logging what the track
  // actually negotiated is the only way to tell "the flag did nothing" apart
  // from "the flag worked and the mic still picked up speaker bleed anyway".
  watch(stream, (s) => {
    const settings = s?.getAudioTracks()[0]?.getSettings()
    if (settings)
      console.info('[audio-device] mic track settings (actual, not requested):', settings)
  })

  watch(audioInputs, () => {
    if (selectedAudioInput.value === '' && audioInputs.value.length > 0) {
      selectedAudioInput.value = audioInputs.value.find(input => input.deviceId === 'default')?.deviceId || audioInputs.value[0].deviceId
    }
  })

  function askPermission() {
    return ensurePermissions()
      .then(() => nextTick())
      .then(() => {
        if (audioInputs.value.length > 0 && !selectedAudioInput.value) {
          selectedAudioInput.value = audioInputs.value.find(input => input.deviceId === 'default')?.deviceId || audioInputs.value[0].deviceId
        }
      })
      .catch((error) => {
        console.error('Error ensuring permissions:', error)
        throw error // Re-throw so callers can handle the error
      })
  }

  return {
    audioInputs,
    selectedAudioInput,
    stream,
    streamError,
    deviceConstraints,
    permissionGranted,

    askPermission,
    startStream,
    stopStream,
  }
}
