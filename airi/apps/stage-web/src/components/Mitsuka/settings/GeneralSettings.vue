<script setup lang="ts">
import { useSettings, useSettingsAudioDevice } from '@proj-airi/stage-ui/stores/settings'
import { useTheme } from '@proj-airi/ui'
import { storeToRefs } from 'pinia'
import { computed } from 'vue'

import Toggle from '../shared/Toggle.vue'

const settings = useSettings()
const { language, disableTransitions, stageViewControlsEnabled } = storeToRefs(settings)
const { enabled: micEnabled } = storeToRefs(useSettingsAudioDevice())
const { isDark, toggleDark } = useTheme()

// `settings/language` is stored empty until the user picks one — that empty
// value is what "follow the browser" means to `modules/i18n.ts`, so it needs
// its own option or the select renders blank.
const LANGUAGES = [
  { value: '', label: 'Theo hệ thống' },
  { value: 'vi', label: 'Tiếng Việt' },
  { value: 'en', label: 'English' },
  { value: 'ja', label: '日本語' },
  { value: 'zh-Hans', label: '简体中文' },
  { value: 'ko', label: '한국어' },
]

// `disableTransitions` is stored inverted from how the row reads.
const motionEnabled = computed({
  get: () => !disableTransitions.value,
  set: (value: boolean) => { disableTransitions.value = !value },
})

const theme = computed({
  get: () => isDark.value ? 'dark' : 'light',
  set: (value: string) => {
    if ((value === 'dark') !== isDark.value)
      toggleDark()
  },
})
</script>

<template>
  <div class="settings-section">
    <label class="settings-row">
      <span class="settings-label">Ngôn ngữ</span>
      <select v-model="language" class="settings-select">
        <option v-for="item in LANGUAGES" :key="item.value" :value="item.value">
          {{ item.label }}
        </option>
      </select>
    </label>

    <label class="settings-row">
      <span class="settings-label">Chủ đề</span>
      <select v-model="theme" class="settings-select">
        <option value="dark">
          Tối
        </option>
        <option value="light">
          Sáng
        </option>
      </select>
    </label>

    <div class="settings-row">
      <span class="settings-label">
        Hiệu ứng chuyển động
        <small>Hoạt ảnh khi chuyển trang và mở hộp thoại</small>
      </span>
      <Toggle v-model="motionEnabled" label="Hiệu ứng chuyển động" />
    </div>

    <div class="settings-row">
      <span class="settings-label">
        Tự động lắng nghe
        <small>Bật micro để Mitsuka nghe khi bạn nói</small>
      </span>
      <Toggle v-model="micEnabled" label="Tự động lắng nghe" />
    </div>

    <div class="settings-row">
      <span class="settings-label">
        Điều khiển góc nhìn
        <small>Hiện thanh trượt phóng to / thu nhỏ trên sân khấu</small>
      </span>
      <Toggle v-model="stageViewControlsEnabled" label="Điều khiển góc nhìn" />
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

.settings-label {
  display: flex;
  flex-direction: column;
  color: var(--mk-ink-dim);
  font-size: 0.82rem;
  gap: 0.12rem;
}

.settings-label small {
  color: var(--mk-muted);
  font-size: 0.68rem;
}

.settings-select {
  min-width: 9rem;
  border: 1px solid var(--mk-border);
  border-radius: 0.6rem;
  outline: none;
  background: var(--mk-inset);
  color: var(--mk-ink);
  padding: 0.4rem 0.55rem;
  font: inherit;
  font-size: 0.78rem;
}

.settings-select:focus { border-color: rgb(171 125 255 / 0.5); }
</style>
