<script setup lang="ts">
import type { QuickAction } from '../components/Mitsuka/chat/QuickActions.vue'
import type { ChatMessage } from '../components/Mitsuka/chat/types'
import type { NavItem } from '../components/Mitsuka/sidebar/Navigation.vue'

import workletUrl from '@proj-airi/stage-ui/workers/vad/process.worklet?worker&url'

import { errorMessageFrom } from '@moeru/std'
import { BackgroundDialogPicker } from '@proj-airi/stage-layouts/components/Backgrounds'
import { useBackgroundThemeColor } from '@proj-airi/stage-layouts/composables/theme-color'
import { useBackgroundStore } from '@proj-airi/stage-layouts/stores/background'
import { encodeWav, normalizeSpeechSamples } from '@proj-airi/stage-ui/composables/audio/wav-encoder'
import { useLocalConversation } from '@proj-airi/stage-ui/composables/local-conversation'
import { useVAD } from '@proj-airi/stage-ui/stores/ai/models/vad'
import { useEmotionStore } from '@proj-airi/stage-ui/stores/modules/emotion'
import { useSettings, useSettingsAudioDevice } from '@proj-airi/stage-ui/stores/settings'
import { useTheme } from '@proj-airi/ui'
import { breakpointsTailwind, useBreakpoints, useFullscreen, useMouse } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import { computed, onMounted, onUnmounted, ref, useTemplateRef, watch } from 'vue'
import { useRouter } from 'vue-router'

import ChatPanel from '../components/Mitsuka/chat/ChatPanel.vue'
import HistoryPanel from '../components/Mitsuka/chat/HistoryPanel.vue'
import SettingsPanel from '../components/Mitsuka/settings/SettingsPanel.vue'
import IconButton from '../components/Mitsuka/shared/IconButton.vue'
import Sidebar from '../components/Mitsuka/sidebar/Sidebar.vue'
import Live2DStage from '../components/Mitsuka/stage/Live2DStage.vue'

import '../components/Mitsuka/theme.css'

// 127.0.0.1 rather than `localhost`: uvicorn binds `0.0.0.0`, which is IPv4
// only, while Windows resolves `localhost` to `::1` first — anything else
// holding the IPv6 loopback on this port answers instead of local-api.
const LOCAL_API_URL = 'http://127.0.0.1:8010'
const GREETING = 'Xin chào! Mình là Mitsuka ✨\nMình có thể giúp gì cho bạn hôm nay?'

const router = useRouter()
const { isDark, toggleDark } = useTheme()

// ── Conversation pipeline ──────────────────────────────────────────────────
const emotionStore = useEmotionStore()
const { emotion } = storeToRefs(emotionStore)
const localConv = useLocalConversation({
  baseUrl: LOCAL_API_URL,
  language: 'vi',
  onEmotion: (value) => { emotionStore.emotion = value },
})
const { state: localState, turn: localTurn, transcript: localTranscript, reply: localReply, error: localError } = localConv

const busy = computed(() => localState.value === 'transcribing' || localState.value === 'thinking')

// ── Chat transcript ────────────────────────────────────────────────────────
// `useLocalConversation` only tracks the *current* turn (one transcript, one
// reply), so the rendered thread is assembled here: a user bubble per new
// transcript, and one assistant bubble per turn whose text grows in place as
// the reply streams.
let messageSeq = 0
function nextMessageId() {
  messageSeq += 1
  return `mk-${messageSeq}`
}

const messages = ref<ChatMessage[]>([
  { id: nextMessageId(), role: 'assistant', content: GREETING, at: Date.now() },
])

// The assistant bubble of the turn currently being answered, if its first
// token has arrived. Kept for the whole turn — not cleared when the caret stops
// — because it is also what tells the typing indicator the reply has landed.
const replyId = ref<string | null>(null)
let userBubbleTurn = -1
// Set by `send()` right before starting a turn with an attached image, and
// consumed by the user-bubble watcher below — `processText` only carries the
// caption text through the turn, so the image rides alongside it here.
let pendingUserImageUrl: string | undefined
// Every object URL a message bubble might still be pointing at, so they can
// all be released together instead of leaking one per attached image.
const attachmentObjectUrls: string[] = []
function releaseAttachmentObjectUrls() {
  for (const url of attachmentObjectUrls.splice(0))
    URL.revokeObjectURL(url)
}
// A ref, not a plain local: `thinking` below is computed from it, so the
// typing indicator has to re-evaluate the moment the first token lands.
const replyTurn = ref(-1)
let streamSettleTimer: ReturnType<typeof setTimeout> | undefined

/** Stop the caret on the streaming bubble; the bubble itself stays this turn's. */
function stopCaret() {
  const target = messages.value.find(message => message.id === replyId.value)
  if (target)
    target.streaming = false
}

watch([localTurn, localTranscript], ([turn, text]) => {
  if (!text || turn === userBubbleTurn)
    return
  userBubbleTurn = turn
  const imageUrl = pendingUserImageUrl
  pendingUserImageUrl = undefined
  messages.value.push({ id: nextMessageId(), role: 'user', content: text, at: Date.now(), imageUrl })
})

watch([localTurn, localReply], ([turn, text]) => {
  if (!text)
    return

  if (turn !== replyTurn.value) {
    replyTurn.value = turn
    const id = nextMessageId()
    replyId.value = id
    messages.value.push({ id, role: 'assistant', content: text, at: Date.now(), streaming: true })
  }
  else {
    const target = messages.value.find(message => message.id === replyId.value)
    if (target)
      target.content = text
  }

  // The stream carries no per-turn "text complete" event — speech runs on well
  // past the last token — so the caret is dropped once tokens stop arriving.
  clearTimeout(streamSettleTimer)
  streamSettleTimer = setTimeout(stopCaret, 700)
})

watch(localState, (state) => {
  if (state === 'idle') {
    clearTimeout(streamSettleTimer)
    stopCaret()
  }
})

// Only until this turn's first token lands — once the bubble exists, the caret
// carries on showing that the reply is still arriving.
const thinking = computed(() => busy.value && replyTurn.value !== localTurn.value)

// `processText` only ever carries the typed caption to local-api — its chat
// model is text-only, so an attached image never reaches Mitsuka. It still
// rides along as a client-side attachment on the user's own bubble (see the
// `localTranscript` watcher above), and an image with no caption is shown
// immediately rather than round-tripped through a model that cannot see it.
async function send(text: string, image?: File) {
  if (!text && !image)
    return

  if (!text && image) {
    const url = URL.createObjectURL(image)
    attachmentObjectUrls.push(url)
    messages.value.push({ id: nextMessageId(), role: 'user', content: '', at: Date.now(), imageUrl: url })
    return
  }

  if (image) {
    const url = URL.createObjectURL(image)
    attachmentObjectUrls.push(url)
    pendingUserImageUrl = url
  }

  await localConv.processText(text)
}

function newConversation() {
  localConv.reset()
  clearTimeout(streamSettleTimer)
  replyId.value = null
  pendingUserImageUrl = undefined
  releaseAttachmentObjectUrls()
  messages.value = [{ id: nextMessageId(), role: 'assistant', content: GREETING, at: Date.now() }]
}

// ── Navigation ─────────────────────────────────────────────────────────────
const NAV_ITEMS: NavItem[] = [
  { id: 'chat', label: 'Trò chuyện', icon: 'i-solar:chat-round-line-outline' },
  { id: 'history', label: 'Lịch sử', icon: 'i-solar:history-outline' },
  { id: 'memory', label: 'Bộ nhớ', icon: 'i-solar:notebook-minimalistic-outline' },
  { id: 'tools', label: 'Công cụ', icon: 'i-solar:widget-4-outline', badge: 'MỚI' },
  { id: 'settings', label: 'Cài đặt', icon: 'i-solar:settings-outline' },
  { id: 'profile', label: 'Hồ sơ', icon: 'i-solar:user-circle-outline' },
]

// Sections airi already ships a full page for navigate there rather than being
// re-implemented inside this shell.
const NAV_ROUTES: Record<string, string> = {
  memory: '/settings/memory',
  tools: '/settings/modules',
  profile: '/settings/account',
}

const panel = ref<'chat' | 'history'>('chat')
const settingsOpen = ref(false)
const backgroundPickerOpen = ref(false)
const sidebarOpen = ref(false)

const activeNav = computed({
  get: () => panel.value,
  set: (id: string) => {
    sidebarOpen.value = false

    if (id === 'chat' || id === 'history') {
      panel.value = id
      return
    }
    if (id === 'settings') {
      settingsOpen.value = true
      return
    }

    const route = NAV_ROUTES[id]
    if (route)
      void router.push(route)
  },
})

const QUICK_ACTIONS: QuickAction[] = [
  { id: 'about', emoji: '👋', label: 'Giới thiệu Mitsuka', prompt: 'Bạn là ai vậy? Giới thiệu về bản thân bạn đi.' },
  { id: 'guide', emoji: '📖', label: 'Hướng dẫn sử dụng', prompt: 'Mình có thể trò chuyện với bạn như thế nào? Hướng dẫn mình vài cách nhé.' },
  { id: 'features', emoji: '✨', label: 'Các tính năng', prompt: 'Bạn có thể làm được những gì?' },
  { id: 'privacy', emoji: '🛡️', label: 'Bảo mật', prompt: 'Dữ liệu trò chuyện của mình được xử lý và lưu ở đâu?' },
]

const draft = ref('')

async function runQuickAction(action: QuickAction) {
  panel.value = 'chat'
  await send(action.prompt)
}

// ── Stage ──────────────────────────────────────────────────────────────────
const paused = ref(false)
const breakpoints = useBreakpoints(breakpointsTailwind)
const isMobile = breakpoints.smaller('md')
const { x: mouseX, y: mouseY } = useMouse()
const cursorPosition = computed(() => ({ x: mouseX.value, y: mouseY.value }))
const { isFullscreen, toggle: toggleFullscreen } = useFullscreen()

const backgroundStore = useBackgroundStore()
const { selectedOption, sampledColor } = storeToRefs(backgroundStore)
const { stageModelRenderer, stageViewControlsEnabled } = storeToRefs(useSettings())

// `Live2DStage` re-exposes the background provider's `surfaceEl` under the same
// name, which is the whole of what the sampler reads off this ref.
const backgroundSurface = useTemplateRef<{ surfaceEl?: HTMLElement | null }>('stage')
const { syncBackgroundTheme } = useBackgroundThemeColor({
  backgroundSurface: backgroundSurface as never,
  selectedOption,
  sampledColor,
})

// ── Microphone + VAD ───────────────────────────────────────────────────────
const audioDeviceStore = useSettingsAudioDevice()
const { stream, enabled: micEnabled, streamError: micStreamError } = storeToRefs(audioDeviceStore)

// `echoCancellation: true` (audio-device.ts) asks the browser to cancel out
// Mitsuka's own voice from the mic, but that only works reliably close to
// perfectly on a headset — through open speakers the reflected/reverberated
// copy the mic picks back up is often quieter than direct speech but not gone,
// so the plain 0.45 threshold below was tripping VAD on Mitsuka's own reply
// and self-triggering a barge-in mid-sentence. Raising the bar only while she
// is actually speaking keeps barge-in working (a real interruption is loud,
// close-mic speech) while filtering out the fainter self-echo.
const VAD_THRESHOLD_IDLE = 0.45
const VAD_THRESHOLD_WHILE_SPEAKING = 0.85
const vadThreshold = computed(() => localState.value === 'speaking' ? VAD_THRESHOLD_WHILE_SPEAKING : VAD_THRESHOLD_IDLE)

// Every step from here to `/emotion-vad` is logged under `[mic]`. Without it a
// broken pipeline is indistinguishable from a quiet room: the whole chain is
// callback-driven, so a step that never runs produces no console line, no
// network request and no error — nothing to tell "VAD never started" apart from
// "you did not speak".
const micError = ref<string>()

const {
  init: initVAD,
  dispose: disposeVAD,
  start: startVAD,
  loaded: vadLoaded,
  inferenceError: vadInferenceError,
} = useVAD(workletUrl, {
  threshold: vadThreshold,
  minSilenceDurationMs: ref(900),
  onSpeechStart: () => {
    console.info('[mic] speech-start')
    localConv.onSpeechStart()
  },
  onSpeechEnd: () => console.info('[mic] speech-end'),
  onSpeechReady: (buffer, duration) => {
    console.info('[mic] speech-ready | samples:', buffer.length, '| duration:', duration)
    const normalized = normalizeSpeechSamples(buffer)
    void localConv.process(encodeWav(normalized, 16000))
  },
})

// `initVAD` fetches the Silero VAD model over the network on first use
// (workers/vad/vad.ts pulls `onnx-community/silero-vad` through transformers.js,
// cached in the browser afterwards), so it is both the slowest step here and
// the one most likely to fail.
async function startAudioInteraction() {
  micError.value = undefined
  try {
    console.info('[mic] loading VAD model…')
    await initVAD()
    console.info('[mic] VAD model ready | mic stream:', stream.value ? 'present' : 'not acquired yet')
    if (stream.value) {
      await startVAD(stream.value)
      console.info('[mic] VAD attached to the mic stream')
    }
  }
  catch (error) {
    micError.value = `Không khởi động được micro: ${errorMessageFrom(error) ?? String(error)}`
    console.error('[mic] initialization failed', error)
  }
}

function stopAudioInteraction() {
  // Deliberately does not clear `micError`. A failed start turns the toggle
  // back off, which lands here immediately — clearing the message at this
  // point wiped the very explanation the user needs. `toggleMicrophone` clears
  // it when the user tries again.
  disposeVAD()
}

// The VAD emits its own `status: error` events after a successful load, which
// only ever landed in a ref nobody read.
watch(vadInferenceError, (message) => {
  if (!message)
    return
  console.error('[mic] VAD inference error:', message)
  micError.value = `Lỗi VAD: ${message}`
})

// getUserMedia fails one layer below the VAD — a denied permission or a
// device that no longer exists — and the store turns the toggle back off when
// it does. Say why, or the button just flicks off for no visible reason.
watch(micStreamError, (message) => {
  if (message)
    micError.value = `Không mở được micro: ${message}`
})

// A dead microphone is the more urgent of the two: a conversation error is
// about one turn, this is about the whole input path being unavailable. Each
// carries its own next step — telling someone to check local-api when the
// browser denied the microphone sends them to the wrong place.
const surfacedError = computed(() => {
  if (micError.value)
    return `${micError.value} — kiểm tra quyền micro của trình duyệt rồi thử lại nhé.`
  if (localError.value)
    return `${localError.value} — kiểm tra local-api rồi thử lại nhé.`
  return undefined
})

// Upstream airi asked for the microphone from its own hearing popover, so by
// the time anything flipped `enabled` the permission was already granted and
// enumerateDevices() was returning real device ids. This shell has no such
// step, and without one the very first click ran getUserMedia while every
// deviceId was still hidden behind the ungranted permission.
async function toggleMicrophone() {
  if (micEnabled.value) {
    micEnabled.value = false
    return
  }

  micError.value = undefined
  try {
    await audioDeviceStore.askPermission()
  }
  catch (error) {
    micError.value = `Không truy cập được micro: ${errorMessageFrom(error) ?? String(error)}`
    console.error('[mic] permission request failed', error)
    return
  }

  micEnabled.value = true
}

watch(micEnabled, async (value) => {
  if (value)
    await startAudioInteraction()
  else
    stopAudioInteraction()
}, { immediate: true })

watch([stream, () => vadLoaded.value], async ([activeStream, loaded]) => {
  if (micEnabled.value && loaded && activeStream) {
    try {
      await startVAD(activeStream)
      console.info('[mic] VAD attached to the mic stream')
    }
    catch (error) {
      micError.value = `Không gắn được micro vào VAD: ${errorMessageFrom(error) ?? String(error)}`
      console.error('[mic] VAD could not attach to the microphone', error)
    }
  }
})

// ── local-api health ───────────────────────────────────────────────────────
const serverOnline = ref<boolean | null>(null)
let healthTimer: ReturnType<typeof setInterval> | undefined

async function probeServer() {
  try {
    const response = await fetch(`${LOCAL_API_URL}/health`, { signal: AbortSignal.timeout(2500) })
    serverOnline.value = response.ok
  }
  catch {
    serverOnline.value = false
  }
}

onMounted(() => {
  syncBackgroundTheme()
  void probeServer()
  healthTimer = setInterval(() => void probeServer(), 15000)
})

onUnmounted(() => {
  stopAudioInteraction()
  clearTimeout(streamSettleTimer)
  releaseAttachmentObjectUrls()
  if (healthTimer)
    clearInterval(healthTimer)
})
</script>

<template>
  <div class="mitsuka-app" :class="{ 'mitsuka-app--light': !isDark }">
    <div class="mk-aurora mk-aurora--one" />
    <div class="mk-aurora mk-aurora--two" />

    <div v-if="sidebarOpen" class="mk-drawer-scrim" @click="sidebarOpen = false" />

    <div class="mk-shell-sidebar" :class="{ 'mk-shell-sidebar--open': sidebarOpen }">
      <Sidebar
        v-model="activeNav"
        :items="NAV_ITEMS"
        user-name="Bạn"
        :online="serverOnline"
        @learn-more="runQuickAction(QUICK_ACTIONS[0])"
        @open-settings="settingsOpen = true"
      />
    </div>

    <div class="mk-shell-stage">
      <IconButton
        class="mk-drawer-toggle"
        icon="i-solar:hamburger-menu-outline"
        label="Mở menu"
        @click="sidebarOpen = true"
      />
      <Live2DStage
        ref="stage"
        :background="selectedOption"
        :top-color="sampledColor"
        :cursor-position="cursorPosition"
        :enable-orbit-controls="!isMobile"
        :paused="paused"
        :listening="micEnabled"
        :view-controls="stageViewControlsEnabled"
        :fullscreen="isFullscreen"
        :live2d="stageModelRenderer === 'live2d'"
        :state="localState"
        :emotion="emotion"
        @toggle-listening="toggleMicrophone"
        @toggle-paused="paused = !paused"
        @toggle-view-controls="stageViewControlsEnabled = !stageViewControlsEnabled"
        @toggle-fullscreen="toggleFullscreen"
        @pick-background="backgroundPickerOpen = true"
        @open-settings="settingsOpen = true"
      />
    </div>

    <div class="mk-shell-chat">
      <ChatPanel
        v-if="panel === 'chat'"
        v-model:draft="draft"
        :messages="messages"
        :quick-actions="QUICK_ACTIONS"
        :thinking="thinking"
        :busy="busy"
        :listening="micEnabled"
        :dark="isDark"
        :error="surfacedError"
        @send="send"
        @pick-quick-action="runQuickAction"
        @toggle-listening="toggleMicrophone"
        @toggle-theme="toggleDark()"
        @new-conversation="newConversation"
        @open-settings="settingsOpen = true"
      />
      <HistoryPanel
        v-else
        :messages="messages"
        @open="panel = 'chat'"
        @clear="newConversation"
      />
    </div>

    <SettingsPanel v-model="settingsOpen" @pick-background="backgroundPickerOpen = true" />
    <BackgroundDialogPicker v-model="backgroundPickerOpen" />
  </div>
</template>

<style scoped>
.mitsuka-app {
  position: relative;
  display: grid;
  overflow: hidden;
  height: 100dvh;
  width: 100vw;
  background: radial-gradient(120% 120% at 50% 0%, var(--mk-bg-2) 0%, var(--mk-bg) 62%);
  padding: 0.85rem;
  gap: 0.85rem;
  grid-template-columns: 16.5rem minmax(0, 1fr) minmax(21rem, 28rem);
}

.mk-aurora {
  position: absolute;
  z-index: 0;
  width: 34rem;
  height: 34rem;
  border-radius: 999px;
  filter: blur(120px);
  opacity: 0.16;
  pointer-events: none;
}

.mk-aurora--one { top: -16rem; left: 18%; background: #8b5cf6; }
.mk-aurora--two { right: -12rem; bottom: -18rem; background: #ec4899; }

.mk-shell-sidebar,
.mk-shell-stage,
.mk-shell-chat {
  position: relative;
  z-index: 1;
  min-width: 0;
  min-height: 0;
}

.mk-shell-stage { display: flex; flex-direction: column; }
.mk-shell-stage > :last-child { flex: 1; min-height: 0; }

.mk-drawer-toggle { display: none; }

.mk-drawer-scrim {
  position: fixed;
  z-index: 60;
  inset: 0;
  background: rgb(6 2 12 / 0.55);
  backdrop-filter: blur(3px);
}

@media (max-width: 1180px) {
  .mitsuka-app { grid-template-columns: minmax(0, 1fr) minmax(20rem, 25rem); }

  .mk-shell-sidebar {
    position: fixed;
    z-index: 70;
    top: 0;
    bottom: 0;
    left: 0;
    width: 17rem;
    padding: 0.85rem;
    transform: translateX(-102%);
    transition: transform 220ms ease;
  }

  .mk-shell-sidebar--open { transform: translateX(0); }

  .mk-drawer-toggle {
    position: absolute;
    z-index: 30;
    top: 0.85rem;
    left: 0.85rem;
    display: grid;
    background: rgb(14 8 24 / 0.7);
    backdrop-filter: blur(18px);
  }

  /* The stage HUD would sit under the drawer button otherwise. */
  .mk-shell-stage :deep(.stage-hud--top) { left: 3.6rem; }
}

@media (max-width: 860px) {
  .mitsuka-app {
    padding: 0.6rem;
    gap: 0.6rem;
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(11rem, 30dvh) minmax(0, 1fr);
  }

  .mk-shell-stage :deep(.character-controller) { display: none; }
}

@media (prefers-reduced-motion: reduce) {
  .mk-shell-sidebar { transition: none; }
}
</style>

<route lang="yaml">
name: IndexScenePage
meta:
  layout: stage
  stageTransition:
    name: bubble-wave-out
</route>
