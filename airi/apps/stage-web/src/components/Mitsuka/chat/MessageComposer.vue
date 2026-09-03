<script setup lang="ts">
import { onClickOutside, useObjectUrl } from '@vueuse/core'
import { nextTick, ref, shallowRef, watch } from 'vue'

import IconButton from '../shared/IconButton.vue'

const props = defineProps<{
  listening: boolean
  busy: boolean
  placeholder?: string
}>()

const emit = defineEmits<{
  (e: 'send', text: string, image?: File): void
  (e: 'toggleListening'): void
}>()

const model = defineModel<string>({ default: '' })

const EMOJIS = ['✨', '💜', '🌙', '🌸', '😊', '😂', '🥺', '😴', '👍', '🙏', '🎵', '☕', '🐱', '🍰', '🔥', '💡']

const textarea = ref<HTMLTextAreaElement | null>(null)
const emojiOpen = ref(false)
const emojiPanel = ref<HTMLElement | null>(null)
const composing = ref(false)

onClickOutside(emojiPanel, () => {
  emojiOpen.value = false
})

const fileInput = ref<HTMLInputElement | null>(null)
const attachedImage = shallowRef<File>()
const attachedImageUrl = useObjectUrl(attachedImage)

// A plain input driven by `fileInput.click()`, not `BasicInputFile`: that
// component wraps its slot in a <label>, and a <label> deliberately does not
// forward clicks that land on interactive content — our attach button — to
// its control, so the file dialog never opened.
function pickImage(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file)
    attachedImage.value = file
  // Let the same file be re-picked after it was removed.
  input.value = ''
}

function removeAttachedImage() {
  attachedImage.value = undefined
}

function resize() {
  const el = textarea.value
  if (!el)
    return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 140)}px`
}

watch(model, () => void nextTick(resize))

/** Insert at the caret rather than appending, so emoji land where the user is typing. */
function insert(text: string) {
  const el = textarea.value
  const start = el?.selectionStart ?? model.value.length
  const end = el?.selectionEnd ?? model.value.length
  model.value = model.value.slice(0, start) + text + model.value.slice(end)

  void nextTick(() => {
    resize()
    el?.focus()
    const caret = start + text.length
    el?.setSelectionRange(caret, caret)
  })
}

function pickEmoji(emoji: string) {
  insert(emoji)
  emojiOpen.value = false
}

function submit() {
  const text = model.value.trim()
  const image = attachedImage.value
  if ((!text && !image) || composing.value || props.busy)
    return
  model.value = ''
  attachedImage.value = undefined
  void nextTick(resize)
  emit('send', text, image)
}
</script>

<template>
  <form class="composer" @submit.prevent="submit">
    <div v-if="attachedImageUrl" class="composer-attachment">
      <img :src="attachedImageUrl" alt="Ảnh sẽ gửi kèm">
      <button
        type="button"
        class="composer-attachment-remove"
        aria-label="Bỏ ảnh đính kèm"
        title="Bỏ ảnh đính kèm"
        @click="removeAttachedImage"
      >
        <span class="i-solar:close-circle-bold" />
      </button>
    </div>

    <textarea
      ref="textarea"
      v-model="model"
      rows="1"
      :placeholder="placeholder ?? 'Nhập tin nhắn của bạn…'"
      aria-label="Tin nhắn gửi Mitsuka"
      @input="resize"
      @compositionstart="composing = true"
      @compositionend="composing = false"
      @keydown.enter.exact.prevent="submit"
    />

    <div class="composer-bar">
      <div class="composer-tools">
        <input
          ref="fileInput"
          class="composer-file"
          type="file"
          accept="image/*"
          tabindex="-1"
          aria-hidden="true"
          @change="pickImage"
        >
        <IconButton
          icon="i-solar:gallery-add-outline"
          label="Đính kèm ảnh"
          size="sm"
          @click="fileInput?.click()"
        />

        <div ref="emojiPanel" class="composer-emoji">
          <IconButton
            icon="i-solar:smile-circle-outline"
            label="Chèn biểu tượng cảm xúc"
            size="sm"
            :active="emojiOpen"
            @click="emojiOpen = !emojiOpen"
          />
          <div v-if="emojiOpen" class="emoji-pop">
            <button
              v-for="emoji in EMOJIS"
              :key="emoji"
              type="button"
              :aria-label="`Chèn ${emoji}`"
              @click="pickEmoji(emoji)"
            >
              {{ emoji }}
            </button>
          </div>
        </div>

        <IconButton
          :icon="listening ? 'i-solar:microphone-3-bold' : 'i-solar:microphone-3-outline'"
          :label="listening ? 'Tắt lắng nghe' : 'Bật lắng nghe'"
          :active="listening"
          size="sm"
          @click="emit('toggleListening')"
        />
      </div>

      <button class="composer-send" type="submit" :disabled="(!model.trim() && !attachedImage) || busy" :aria-label="busy ? 'Đang xử lý' : 'Gửi tin nhắn'">
        <span v-if="busy" class="i-svg-spinners:90-ring-with-bg" />
        <span v-else class="i-solar:plain-2-outline" />
      </button>
    </div>
  </form>
</template>

<style scoped>
.composer {
  position: relative;
  flex-shrink: 0;
  margin: 0 0.95rem 0.95rem;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius);
  background: var(--mk-inset);
  padding: 0.7rem 0.75rem 0.55rem;
  transition: border-color 160ms ease;
}

.composer:focus-within { border-color: rgb(171 125 255 / 0.4); }

.composer-attachment {
  position: relative;
  display: inline-block;
  margin-bottom: 0.55rem;
}

.composer-attachment img {
  display: block;
  max-width: 9rem;
  max-height: 6.5rem;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius-sm);
  object-fit: cover;
}

.composer-attachment-remove {
  position: absolute;
  top: -0.4rem;
  right: -0.4rem;
  display: grid;
  width: 1.2rem;
  height: 1.2rem;
  place-items: center;
  border: 0;
  border-radius: 50%;
  background: var(--mk-panel-solid);
  color: var(--mk-danger);
  cursor: pointer;
  font-size: 1.05rem;
  line-height: 1;
}

.composer textarea {
  display: block;
  width: 100%;
  min-height: 1.6rem;
  max-height: 8.75rem;
  border: 0;
  outline: none;
  background: transparent;
  color: var(--mk-ink);
  resize: none;
  font: inherit;
  font-size: 0.88rem;
  line-height: 1.5;
}

@media (max-width: 860px) {
  .composer {
    margin: 0 0.5rem 0.5rem;
    padding: 0.5rem 0.6rem 0.4rem;
  }

  .composer textarea {
    font-size: 16px; /* Prevents auto-zoom in iOS Safari */
    min-height: 1.4rem;
  }

  .composer-send {
    width: 2.15rem;
    height: 2.15rem;
  }
}

.composer textarea::placeholder { color: var(--mk-muted); }

.composer-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 0.45rem;
  gap: 0.5rem;
}

.composer-tools { display: flex; align-items: center; gap: 0.3rem; }

/* Hidden without `display: none`, which some browsers refuse to click(). */
.composer-file {
  position: absolute;
  width: 0;
  height: 0;
  opacity: 0;
  pointer-events: none;
}

.composer-emoji { position: relative; }

.emoji-pop {
  position: absolute;
  bottom: calc(100% + 0.45rem);
  left: 0;
  z-index: 40;
  display: grid;
  width: 11.5rem;
  border: 1px solid var(--mk-border-strong);
  border-radius: var(--mk-radius-sm);
  background: var(--mk-float);
  box-shadow: var(--mk-shadow);
  padding: 0.4rem;
  gap: 0.15rem;
  grid-template-columns: repeat(8, 1fr);
}

.emoji-pop button {
  border: 0;
  border-radius: 0.4rem;
  background: transparent;
  cursor: pointer;
  padding: 0.15rem;
  font-size: 0.95rem;
  line-height: 1.3;
}

.emoji-pop button:hover { background: var(--mk-hover); }

.composer-send {
  display: grid;
  width: 2.35rem;
  height: 2.35rem;
  flex: 0 0 auto;
  place-items: center;
  border: 0;
  border-radius: 50%;
  background: var(--mk-accent);
  color: #fff;
  cursor: pointer;
  font-size: 1rem;
  box-shadow: 0 10px 24px rgb(146 68 200 / 0.32);
  transition: transform 150ms ease, opacity 150ms ease;
}

.composer-send:hover:not(:disabled) { transform: translateY(-1px); }
.composer-send:disabled { cursor: default; opacity: 0.35; }

@media (prefers-reduced-motion: reduce) {
  .composer, .composer-send { transition: none; }
  .composer-send:hover:not(:disabled) { transform: none; }
}
</style>
