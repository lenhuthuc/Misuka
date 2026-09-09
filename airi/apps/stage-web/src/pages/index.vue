<script setup lang="ts">
import type { MitsukaTouchArea } from '@proj-airi/stage-ui-live2d'

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
import { breakpointsTailwind, useBreakpoints, useFullscreen, useMouse, useTitle } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import { computed, onMounted, onUnmounted, ref, useTemplateRef, watch } from 'vue'
import { useRouter } from 'vue-router'

import ChatPanel from '../components/Mitsuka/chat/ChatPanel.vue'
import HistoryPanel from '../components/Mitsuka/chat/HistoryPanel.vue'
import SettingsPanel from '../components/Mitsuka/settings/SettingsPanel.vue'
import IconButton from '../components/Mitsuka/shared/IconButton.vue'
import Sidebar from '../components/Mitsuka/sidebar/Sidebar.vue'
import Live2DStage from '../components/Mitsuka/stage/Live2DStage.vue'
import SceneBackground from '../components/Mitsuka/stage/SceneBackground.vue'

import '../components/Mitsuka/theme.css'

useTitle('Mitsuka')

// In the browser, use relative path so Vite / ngrok reverse proxy handles it seamlessly.
// Fallback to 127.0.0.1:8010 for SSR or node environments.
const LOCAL_API_URL = typeof window !== 'undefined' ? '' : 'http://127.0.0.1:8010'
const GREETING = 'Xin chào! Mình là Mitsuka ✨\nMình có thể giúp gì cho bạn hôm nay?'

const TOUCH_SCRIPTS: Record<MitsukaTouchArea, readonly string[]> = {
  head: [
    'Hì hì, xoa đầu nhẹ thôi nhé. Dễ chịu thật đó!',
    'Ưm… tay bạn ấm thật đấy.',
    'Ngoan nào… ơ, đáng lẽ mình phải nói câu đó với bạn chứ!',
    'Xoa thêm chút nữa cũng được… chỉ một chút thôi nhé.',
  ],
  chest: [
    'Ngốc !....',
    'Đ-đồ ngốc! Đừng chạm bất ngờ như vậy chứ…',
    'Này… chỗ đó không được tùy tiện đâu, ngốc!',
    'Baka… mình đỏ mặt mất rồi đó!',
  ],
  leftArm: [
    'Bạn chạm vào tay mình à? Muốn mình đi cùng không?',
    'Ơ… bạn muốn nắm tay sao?',
    'Tay mình ở đây nè. Đừng buông vội nhé.',
    'Có chuyện gì à? Mình đang nghe đây.',
  ],
  rightArm: [
    'Ơ, nắm tay thì phải báo trước chứ… nhưng cũng được.',
    'Muốn kéo mình đi đâu vậy?',
    'Nắm nhẹ thôi nhé… mình không chạy mất đâu.',
    'Bạn đang làm mình hơi ngại đó.',
  ],
  torso: [
    'Nhột quá! Bạn đúng là thích trêu mình nhỉ.',
    'Á… đừng cù mình mà!',
    'Hửm? Bạn đang gọi mình sao?',
    'Cứ chạm bất ngờ thế này, tim mình loạn mất thôi.',
  ],
  legs: [
    'Á, bất ngờ quá! Mình suýt mất thăng bằng rồi nè.',
    'Này, đừng nghịch chân mình chứ!',
    'Ơ kìa… chạm ở đó làm gì vậy?',
    'Cẩn thận nhé, mình mà ngã thì bạn phải đỡ đó.',
  ],
}

const IDLE_PROMPTS = [
  'Đây là một lời chủ động sau thời gian im lặng. Hãy kể một mẩu chuyện ngắn, ấm áp hoặc vui, bằng 2-4 câu. Không nhắc rằng đây là yêu cầu hệ thống và không đặt câu hỏi bắt buộc người dùng trả lời.',
  'Đây là một lời chủ động sau thời gian im lặng. Hãy chia sẻ một câu trích dẫn hay, ghi đúng tác giả nếu chắc chắn, rồi nói một suy nghĩ ngắn của Mitsuka. Nếu không chắc nguồn thì dùng một câu do chính Mitsuka viết và nói rõ điều đó.',
  'Đây là một lời chủ động sau thời gian im lặng. Hãy chọn một chủ đề văn hóa, công nghệ hoặc đời sống đang được quan tâm gần đây và kể ngắn gọn, thân thiện. Nếu có tìm kiếm web thì dùng thông tin mới; nếu không xác minh được thời gian thực, hãy gọi đó là một xu hướng chung, không khẳng định là tin nóng.',
  'Đây là một lời chủ động sau thời gian im lặng. Hãy nói một fun fact thú vị và một liên tưởng đáng yêu của Mitsuka, tổng cộng không quá 4 câu.',
]

const IDLE_EMOTIONS = [
  { v: 0.72, a: 0.28, d: 0.18 }, // gently delighted
  { v: 0.34, a: 0.7, d: 0.42 }, // excited to share
  { v: 0.18, a: -0.32, d: 0.08 }, // reflective
  { v: 0.52, a: 0.48, d: -0.16 }, // playful/shy
] as const

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
const serverOnline = ref<boolean | null>(null)

const preparedTouchAudio = new Map<string, ArrayBuffer>()
const lastTouchScript = new Map<MitsukaTouchArea, string>()
let idleTimer: ReturnType<typeof setTimeout> | undefined
const FIRST_IDLE_DELAY_MS = 90_000
const NEXT_IDLE_MIN_MS = 180_000
const NEXT_IDLE_JITTER_MS = 180_000

function randomItem<T>(items: readonly T[]): T {
  return items[Math.floor(Math.random() * items.length)]!
}

function scheduleIdleMoment(delay = NEXT_IDLE_MIN_MS + Math.random() * NEXT_IDLE_JITTER_MS) {
  clearTimeout(idleTimer)
  idleTimer = setTimeout(async () => {
    if (document.hidden || localState.value !== 'idle' || serverOnline.value === false) {
      scheduleIdleMoment(30_000)
      return
    }
    emotionStore.emotion = { ...randomItem(IDLE_EMOTIONS) }
    await localConv.processProactive(randomItem(IDLE_PROMPTS))
    scheduleIdleMoment()
  }, delay)
}

function noteUserActivity() {
  scheduleIdleMoment()
}

async function prepareTouchAudio() {
  // Sequential preparation avoids asking Piper to render every clip on the
  // same CPU at once. Once cached, pointer-to-sound latency is only decoding.
  for (const scripts of Object.values(TOUCH_SCRIPTS)) {
    for (const script of scripts) {
      const bytes = await localConv.prepareSpeech(script)
      if (bytes)
        preparedTouchAudio.set(script, bytes)
    }
  }
}

function handleCharacterTouch(area: MitsukaTouchArea) {
  noteUserActivity()
  const scripts = TOUCH_SCRIPTS[area]
  const previous = lastTouchScript.get(area)
  const candidates = scripts.length > 1 ? scripts.filter(script => script !== previous) : scripts
  const script = randomItem(candidates)
  lastTouchScript.set(area, script)
  void localConv.playScript(script, preparedTouchAudio.get(script))
}

// True while an attached image is being captioned (see `send()`) — happens
// before `processText` starts the turn, so `localState` alone would leave
// the composer looking idle for however long the VLM call takes.
const captioningImage = ref(false)
const busy = computed(() => localState.value === 'transcribing' || localState.value === 'thinking' || captioningImage.value)

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
// consumed by the user-bubble watcher below. `processText`'s query carries a
// VLM-generated description of the image folded in as extra context for the
// text-only chat model — `displayContent`, when set, overrides the bubble
// back to just what the user actually typed, so that description never
// leaks into their own chat history.
let pendingUserBubble: { imageUrl?: string, displayContent?: string } | undefined
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
  const bubble = pendingUserBubble
  pendingUserBubble = undefined
  messages.value.push({
    id: nextMessageId(),
    role: 'user',
    content: bubble?.displayContent ?? text,
    at: Date.now(),
    imageUrl: bubble?.imageUrl,
  })
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

// Chat's own model is text-only, so an attached image only ever reaches it
// as words: /v1/vision/caption runs a local VLM over the image and hands
// back a short description, which is folded into the query sent to
// `processText`. Best-effort like RAG/web search elsewhere in this pipeline —
// a failed or disabled captioning call degrades to "", never blocks sending.
async function captionImage(image: File): Promise<string> {
  captioningImage.value = true
  try {
    const form = new FormData()
    form.append('image', image)
    const response = await fetch(`${LOCAL_API_URL}/v1/vision/caption`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(35000),
    })
    if (!response.ok)
      return ''
    const body = await response.json() as { caption?: string }
    return body.caption ?? ''
  }
  catch (error) {
    console.warn('[vision] captioning failed, sending without it', error)
    return ''
  }
  finally {
    captioningImage.value = false
  }
}

async function send(text: string, image?: File) {
  if (!text && !image)
    return

  if (!image) {
    await localConv.processText(text)
    return
  }

  const url = URL.createObjectURL(image)
  attachmentObjectUrls.push(url)
  const caption = await captionImage(image)

  if (!caption) {
    // Nothing came back to add — either captioning is disabled/unavailable,
    // or it just failed. With no typed caption either, there is nothing left
    // for the text-only model to answer, so this stays a local-only bubble.
    if (!text) {
      messages.value.push({ id: nextMessageId(), role: 'user', content: '', at: Date.now(), imageUrl: url })
      return
    }
    pendingUserBubble = { imageUrl: url }
    await localConv.processText(text)
    return
  }

  // Phrased as the user telling Mitsuka what she is looking at, because
  // that is what she answers. Naming the machinery instead — "hệ thống
  // nhận diện nội dung ảnh là …" — makes a 1.7B model reply about the
  // recognition ("nội dung được nhận diện chính xác") rather than about
  // the picture.
  const query = text
    ? `${text}\n(Ảnh mình gửi kèm: ${caption})`
    : `Mình vừa gửi cho bạn một bức ảnh. ${caption} Bạn thấy sao?`

  // `displayContent: text` (not the combined `query`) keeps the description's
  // wording out of the user's chat history — they see what they typed (or
  // nothing) plus their image, never the auto-generated description.
  pendingUserBubble = { imageUrl: url, displayContent: text }
  await localConv.processText(query)
}

function newConversation() {
  localConv.reset()
  clearTimeout(streamSettleTimer)
  replyId.value = null
  pendingUserBubble = undefined
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
const mobileView = ref<'split' | 'chat' | 'stage'>('split')
const { x: mouseX, y: mouseY } = useMouse()
const cursorPosition = computed(() => ({ x: mouseX.value, y: mouseY.value }))
const { isFullscreen, toggle: toggleFullscreen } = useFullscreen()

const backgroundStore = useBackgroundStore()
const { selectedOption, sampledColor } = storeToRefs(backgroundStore)
const { stageModelRenderer, stageViewControlsEnabled } = storeToRefs(useSettings())

// `SceneBackground` exposes `surfaceEl` which is used to derive theme colors.
const sceneBackgroundRef = useTemplateRef<{ surfaceEl?: HTMLElement | null }>('sceneBackgroundRef')
const { syncBackgroundTheme } = useBackgroundThemeColor({
  backgroundSurface: sceneBackgroundRef as never,
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
  window.addEventListener('pointerdown', noteUserActivity, { passive: true })
  window.addEventListener('keydown', noteUserActivity)
  document.addEventListener('visibilitychange', noteUserActivity)
  scheduleIdleMoment(FIRST_IDLE_DELAY_MS)
  void prepareTouchAudio()
})

onUnmounted(() => {
  stopAudioInteraction()
  clearTimeout(streamSettleTimer)
  releaseAttachmentObjectUrls()
  clearTimeout(idleTimer)
  window.removeEventListener('pointerdown', noteUserActivity)
  window.removeEventListener('keydown', noteUserActivity)
  document.removeEventListener('visibilitychange', noteUserActivity)
  if (healthTimer)
    clearInterval(healthTimer)
})
</script>

<template>
  <div class="mitsuka-app" :class="[{ 'mitsuka-app--light': !isDark }, isMobile ? `mitsuka-app--m-${mobileView}` : '']">
    <!-- Full-screen Scene Background behind everything -->
    <SceneBackground
      ref="sceneBackgroundRef"
      :background="selectedOption"
      :top-color="sampledColor"
      class="mk-app-background"
    />

    <IconButton
      v-if="!isMobile"
      class="mk-drawer-toggle"
      icon="i-solar:hamburger-menu-outline"
      label="Mở menu"
      @click="sidebarOpen = true"
    />

    <!-- Mobile Top Navigation Bar -->
    <header v-if="isMobile" class="mk-mobile-topbar">
      <IconButton
        icon="i-solar:hamburger-menu-outline"
        label="Mở menu"
        size="sm"
        @click="sidebarOpen = true"
      />

      <div class="mk-mobile-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          class="mk-mobile-tab"
          :class="{ 'mk-mobile-tab--active': mobileView === 'chat' }"
          aria-label="Chế độ chỉ trò chuyện"
          @click="mobileView = 'chat'"
        >
          <span class="i-solar:chat-round-line-bold" />
          <span>Chat</span>
        </button>
        <button
          type="button"
          role="tab"
          class="mk-mobile-tab"
          :class="{ 'mk-mobile-tab--active': mobileView === 'split' }"
          aria-label="Chế độ chia đôi màn hình"
          @click="mobileView = 'split'"
        >
          <span class="i-solar:minimize-square-3-outline" />
          <span>Chia đôi</span>
        </button>
        <button
          type="button"
          role="tab"
          class="mk-mobile-tab"
          :class="{ 'mk-mobile-tab--active': mobileView === 'stage' }"
          aria-label="Chế độ xem Mitsuka"
          @click="mobileView = 'stage'"
        >
          <span class="i-solar:smile-circle-bold" />
          <span>Mitsuka</span>
        </button>
      </div>

      <IconButton
        :icon="isDark ? 'i-solar:sun-outline' : 'i-solar:moon-outline'"
        label="Đổi giao diện"
        size="sm"
        @click="toggleDark()"
      />
    </header>

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

    <div
      class="mk-shell-stage"
      :class="{ 'mk-shell--hidden-mobile': isMobile && mobileView === 'chat' }"
    >
      <Live2DStage
        :cursor-position="cursorPosition"
        :enable-orbit-controls="!isMobile"
        :paused="paused"
        :listening="micEnabled"
        :view-controls="stageViewControlsEnabled"
        :fullscreen="isFullscreen"
        :live2d="stageModelRenderer === 'live2d'"
        :state="localState"
        :emotion="emotion"
        @character-touch="handleCharacterTouch"
        @toggle-listening="toggleMicrophone"
        @toggle-paused="paused = !paused"
        @toggle-view-controls="stageViewControlsEnabled = !stageViewControlsEnabled"
        @toggle-fullscreen="toggleFullscreen"
        @pick-background="backgroundPickerOpen = true"
        @open-settings="settingsOpen = true"
      />
    </div>

    <div
      class="mk-shell-chat"
      :class="{ 'mk-shell--hidden-mobile': isMobile && mobileView === 'stage' }"
    >
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
  overflow: hidden;
  height: 100dvh;
  width: 100vw;
  background: #0b0710;
}

.mk-app-background {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
}

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

/* Desktop: Live2D stage on the left/center, transparent chat on the right */
.mk-shell-stage {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  right: 28rem;
  z-index: 5;
}

.mk-shell-chat {
  position: absolute;
  top: 1.2rem;
  right: 1.2rem;
  bottom: 1.2rem;
  z-index: 10;
  width: 27rem;
  max-width: calc(100vw - 2.4rem);
}

.mk-drawer-toggle {
  position: absolute;
  z-index: 30;
  top: 1.2rem;
  left: 1.2rem;
  display: grid;
  background: rgb(14 8 24 / 0.65);
  backdrop-filter: blur(18px);
}

.mk-drawer-scrim {
  position: fixed;
  z-index: 60;
  inset: 0;
  background: rgb(6 2 12 / 0.55);
  backdrop-filter: blur(3px);
}

.mk-mobile-topbar { display: none; }

@media (max-width: 860px) {
  .mk-mobile-topbar {
    position: absolute;
    top: 0.5rem;
    left: 0.5rem;
    right: 0.5rem;
    z-index: 25;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.35rem 0.6rem;
    border-radius: var(--mk-radius);
    background: rgba(18, 10, 28, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(18px);
    gap: 0.4rem;
  }

  .mk-mobile-tabs {
    display: flex;
    background: var(--mk-inset);
    border-radius: 999px;
    padding: 0.2rem;
    gap: 0.2rem;
    border: 1px solid var(--mk-border);
  }

  .mk-mobile-tab {
    display: flex;
    align-items: center;
    gap: 0.28rem;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    border: 0;
    background: transparent;
    color: var(--mk-ink-dim);
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 180ms ease;
  }

  .mk-mobile-tab--active {
    background: var(--mk-raise);
    color: var(--mk-ink);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }

  .mk-shell--hidden-mobile {
    display: none !important;
  }

  /* Mobile Split Mode: Stage in upper area, Chat in lower area */
  .mitsuka-app--m-split .mk-shell-stage {
    position: absolute;
    top: 3.2rem;
    left: 0;
    right: 0;
    height: 42dvh;
    z-index: 5;
  }

  .mitsuka-app--m-split .mk-shell-chat {
    position: absolute;
    top: calc(42dvh + 3.4rem);
    bottom: max(0.4rem, env(safe-area-inset-bottom));
    left: 0.5rem;
    right: 0.5rem;
    width: auto;
    max-width: none;
    height: auto;
    z-index: 10;
  }

  /* Mobile Full Chat Mode */
  .mitsuka-app--m-chat .mk-shell-stage {
    display: none !important;
  }
  .mitsuka-app--m-chat .mk-shell-chat {
    position: absolute;
    top: 3.4rem;
    bottom: max(0.4rem, env(safe-area-inset-bottom));
    left: 0.5rem;
    right: 0.5rem;
    width: auto;
    max-width: none;
    height: auto;
    z-index: 10;
  }

  /* Mobile Stage Mode: Character fills screen */
  .mitsuka-app--m-stage .mk-shell-stage {
    position: absolute;
    top: 3.2rem;
    bottom: 0;
    left: 0;
    right: 0;
    height: auto;
    z-index: 5;
  }
  .mitsuka-app--m-stage .mk-shell-chat {
    display: none !important;
  }
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
