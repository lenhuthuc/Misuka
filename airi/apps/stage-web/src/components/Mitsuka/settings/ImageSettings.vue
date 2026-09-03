<script setup lang="ts">
import { BackgroundKind, useBackgroundStore } from '@proj-airi/stage-layouts/stores/background'
import { BasicInputFile } from '@proj-airi/ui'
import { storeToRefs } from 'pinia'
import { computed, ref, watch } from 'vue'

import Button from '../shared/Button.vue'

const emit = defineEmits<{ (e: 'pickBackground'): void }>()

const backgroundStore = useBackgroundStore()
const { options, selectedId } = storeToRefs(backgroundStore)

const uploadingFiles = ref<File[]>([])
const uploading = ref(false)
const uploadError = ref<string>()

/** Presets (wave/transparent) stay in the picker dialog — this gallery is only the user's own uploads. */
const images = computed(() => options.value.filter(option => option.kind === BackgroundKind.Image))

async function uploadFiles(files: File[]) {
  if (!files.length)
    return

  uploading.value = true
  uploadError.value = undefined
  try {
    for (const file of files) {
      const saved = await backgroundStore.addOption({
        id: `custom-${file.name}-${Date.now()}`,
        label: file.name || 'Ảnh tuỳ chỉnh',
        file,
        kind: BackgroundKind.Image,
      })
      backgroundStore.setSelection(saved)
    }
  }
  catch (error) {
    uploadError.value = 'Không thể tải ảnh lên, thử lại nhé.'
    console.error('[settings] background image upload failed', error)
  }
  finally {
    uploading.value = false
  }
}

// `BasicInputFile` always assigns a fresh array on pick (even re-picking the
// same file), so this fires correctly without needing to reset the ref back
// afterward — doing that would mutate the very source this watch depends on.
watch(uploadingFiles, files => void uploadFiles(files ?? []))

function select(id: string) {
  selectedId.value = id
}

function remove(id: string) {
  void backgroundStore.removeOption(id)
}
</script>

<template>
  <div class="settings-section">
    <div class="settings-row settings-row--stack">
      <span class="settings-label">
        Ảnh nền sân khấu
        <small>Ảnh của bạn, hiển thị phía sau Mitsuka trong lúc trò chuyện</small>
      </span>

      <div class="image-gallery">
        <button
          v-for="image in images"
          :key="image.id"
          type="button"
          class="image-tile"
          :class="{ 'image-tile--active': image.id === selectedId }"
          :aria-pressed="image.id === selectedId"
          @click="select(image.id)"
        >
          <img :src="image.src" :alt="image.label" loading="lazy">
          <div class="image-tile-remove" title="Xoá ảnh" @click.stop="remove(image.id)">
            <span class="i-solar:trash-bin-trash-bold" />
          </div>
        </button>

        <BasicInputFile v-model="uploadingFiles" accept="image/*" multiple class="image-upload">
          <span v-if="uploading" class="i-svg-spinners:90-ring-with-bg" />
          <template v-else>
            <span class="i-solar:gallery-add-outline" />
            <small>Tải ảnh lên</small>
          </template>
        </BasicInputFile>
      </div>

      <p v-if="uploadError" class="image-error">
        {{ uploadError }}
      </p>
    </div>

    <div class="settings-row">
      <span class="settings-label">
        Thư viện ảnh nền
        <small>Chọn từ hiệu ứng động hoặc ảnh nền có sẵn</small>
      </span>
      <Button size="sm" variant="soft" @click="emit('pickBackground')">
        Mở thư viện
      </Button>
    </div>
  </div>
</template>

<style scoped>
.settings-section {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.settings-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-radius: var(--mk-radius-sm);
  padding: 0.65rem 0.6rem;
  gap: 1rem;
}

.settings-row:hover { background: var(--mk-raise); }

.settings-row--stack {
  flex-direction: column;
  align-items: stretch;
  gap: 0.6rem;
}

.settings-row--stack:hover { background: transparent; }

.settings-label {
  display: flex;
  flex-direction: column;
  color: var(--mk-ink-dim);
  font-size: 0.82rem;
  gap: 0.12rem;
}

.settings-label small { color: var(--mk-muted); font-size: 0.68rem; }

.image-gallery {
  display: grid;
  gap: 0.5rem;
  grid-template-columns: repeat(auto-fill, minmax(5.25rem, 1fr));
}

.image-tile,
.image-upload {
  position: relative;
  display: flex;
  aspect-ratio: 4 / 3;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius-sm);
  background: var(--mk-inset);
  padding: 0;
  overflow: hidden;
  cursor: pointer;
}

.image-tile img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-tile--active {
  border-color: var(--mk-purple);
  box-shadow: 0 0 0 2px rgb(171 125 255 / 0.35);
}

.image-tile-remove {
  position: absolute;
  top: 0.25rem;
  right: 0.25rem;
  z-index: 1;
  display: grid;
  width: 1.35rem;
  height: 1.35rem;
  place-items: center;
  border-radius: 50%;
  background: rgb(6 2 12 / 0.55);
  color: #fff;
  opacity: 0;
  transition: opacity 120ms ease, background 120ms ease;
}

.image-tile:hover .image-tile-remove,
.image-tile:focus-visible .image-tile-remove { opacity: 1; }

.image-tile-remove:hover { background: var(--mk-danger); }

.image-upload {
  flex-direction: column;
  gap: 0.2rem;
  border-style: dashed;
  color: var(--mk-muted);
  font-size: 0.66rem;
  text-align: center;
}

.image-upload:hover {
  border-color: var(--mk-purple-soft);
  color: var(--mk-ink-dim);
}

.image-upload span[class*='i-'] { font-size: 1.15rem; }

.image-error {
  margin: 0;
  color: var(--mk-danger);
  font-size: 0.7rem;
}

@media (prefers-reduced-motion: reduce) {
  .image-tile-remove { transition: none; }
}
</style>
