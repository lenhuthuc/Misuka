<script setup lang="ts">
import { useSettings } from '@proj-airi/stage-ui/stores/settings'
import { storeToRefs } from 'pinia'
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'

import Modal from '../shared/Modal.vue'
import Slider from '../shared/Slider.vue'
import Toggle from '../shared/Toggle.vue'
import GeneralSettings from './GeneralSettings.vue'
import ImageSettings from './ImageSettings.vue'

const emit = defineEmits<{ (e: 'pickBackground'): void }>()

const open = defineModel<boolean>({ default: false })

const router = useRouter()
const settings = useSettings()
const { themeColorsHue, themeColorsHueDynamic } = storeToRefs(settings)

interface Tab {
  id: string
  label: string
  icon: string
  /** Sections airi already has full pages for link out instead of duplicating them. */
  links?: { label: string, description: string, to: string }[]
}

const TABS: Tab[] = [
  { id: 'general', label: 'Chung', icon: 'i-solar:settings-outline' },
  { id: 'appearance', label: 'Giao diện', icon: 'i-solar:palette-outline' },
  { id: 'images', label: 'Hình ảnh', icon: 'i-solar:gallery-wide-outline' },
  {
    id: 'character',
    label: 'Nhân vật',
    icon: 'i-solar:user-circle-outline',
    links: [
      { label: 'Thẻ nhân vật', description: 'Tính cách, lời chào và giọng nói của Mitsuka', to: '/settings/airi-card' },
      { label: 'Sân khấu & mô hình', description: 'Chọn mô hình Live2D / VRM và cách hiển thị', to: '/settings/scene' },
    ],
  },
  {
    id: 'ai',
    label: 'AI',
    icon: 'i-solar:magic-stick-3-outline',
    links: [
      { label: 'Nhà cung cấp', description: 'Kết nối tới các dịch vụ mô hình ngôn ngữ', to: '/settings/providers' },
      { label: 'Mô hình', description: 'Chọn mô hình dùng cho hội thoại', to: '/settings/models' },
    ],
  },
  {
    id: 'voice',
    label: 'Giọng nói',
    icon: 'i-solar:soundwave-outline',
    links: [
      { label: 'Tổng hợp & nhận dạng giọng nói', description: 'Cấu hình TTS và STT trong phần nhà cung cấp', to: '/settings/providers' },
    ],
  },
  {
    id: 'memory',
    label: 'Bộ nhớ',
    icon: 'i-solar:notebook-minimalistic-outline',
    links: [
      { label: 'Bộ nhớ dài hạn', description: 'Những gì Mitsuka ghi nhớ giữa các cuộc trò chuyện', to: '/settings/memory' },
    ],
  },
  {
    id: 'tools',
    label: 'Công cụ',
    icon: 'i-solar:widget-4-outline',
    links: [
      { label: 'Mô-đun', description: 'Bật / tắt các mô-đun mở rộng của airi', to: '/settings/modules' },
    ],
  },
  {
    id: 'advanced',
    label: 'Nâng cao',
    icon: 'i-solar:tuning-outline',
    links: [
      { label: 'Hệ thống', description: 'Tùy chọn hệ thống và nhà phát triển', to: '/settings/system' },
      { label: 'Dữ liệu', description: 'Sao lưu, nhập và xóa dữ liệu cục bộ', to: '/settings/data' },
    ],
  },
]

const activeId = ref('general')
const activeTab = computed(() => TABS.find(tab => tab.id === activeId.value) ?? TABS[0])

function go(to: string) {
  open.value = false
  void router.push(to)
}
</script>

<template>
  <Modal v-model="open" title="Cài đặt" wide>
    <div class="settings-panel">
      <nav class="settings-tabs" aria-label="Nhóm cài đặt">
        <button
          v-for="tab in TABS"
          :key="tab.id"
          class="settings-tab"
          :class="{ 'settings-tab--active': tab.id === activeId }"
          type="button"
          @click="activeId = tab.id"
        >
          <span :class="tab.icon" />
          {{ tab.label }}
        </button>
      </nav>

      <div class="settings-content mk-scroll">
        <GeneralSettings v-if="activeId === 'general'" />

        <div v-else-if="activeId === 'appearance'" class="settings-block">
          <Slider
            v-model="themeColorsHue"
            label="Tông màu chủ đạo"
            :min="0"
            :max="360"
            :step="1"
            :display="`${Math.round(themeColorsHue)}°`"
          />
          <div class="settings-inline">
            <span class="settings-inline-label">
              Tông màu chuyển động
              <small>Xoay dần sắc độ theo thời gian</small>
            </span>
            <Toggle v-model="themeColorsHueDynamic" label="Tông màu chuyển động" />
          </div>
        </div>

        <ImageSettings
          v-else-if="activeId === 'images'"
          @pick-background="emit('pickBackground'); open = false"
        />

        <div v-else class="settings-block">
          <button
            v-for="link in activeTab.links"
            :key="link.to"
            class="settings-link"
            type="button"
            @click="go(link.to)"
          >
            <span class="settings-link-text">
              <strong>{{ link.label }}</strong>
              <small>{{ link.description }}</small>
            </span>
            <span class="i-solar:arrow-right-linear" />
          </button>
        </div>
      </div>
    </div>
  </Modal>
</template>

<style scoped>
.settings-panel {
  display: grid;
  min-height: 22rem;
  grid-template-columns: 11rem minmax(0, 1fr);
}

.settings-tabs {
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--mk-border);
  padding: 0.6rem 0.5rem;
  gap: 0.1rem;
}

.settings-tab {
  display: flex;
  align-items: center;
  border: 0;
  border-radius: var(--mk-radius-sm);
  background: transparent;
  color: var(--mk-muted);
  cursor: pointer;
  padding: 0.5rem 0.6rem;
  font: inherit;
  font-size: 0.78rem;
  font-weight: 600;
  gap: 0.5rem;
  text-align: left;
}

.settings-tab:hover { background: var(--mk-raise); color: var(--mk-ink-dim); }

.settings-tab--active {
  background: linear-gradient(100deg, rgb(171 125 255 / 0.26), rgb(255 111 181 / 0.16));
  color: var(--mk-ink);
}

.settings-content { padding: 0.9rem 1rem; }

.settings-block {
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.settings-inline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.settings-inline-label {
  display: flex;
  flex-direction: column;
  color: var(--mk-ink-dim);
  font-size: 0.82rem;
  gap: 0.12rem;
}

.settings-inline-label small { color: var(--mk-muted); font-size: 0.68rem; }

.settings-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius-sm);
  background: var(--mk-raise);
  color: var(--mk-ink-dim);
  cursor: pointer;
  padding: 0.7rem 0.75rem;
  font: inherit;
  gap: 1rem;
  text-align: left;
}

.settings-link:hover { border-color: var(--mk-border-strong); background: var(--mk-hover); }
.settings-link-text { display: flex; flex-direction: column; gap: 0.15rem; }
.settings-link-text strong { font-size: 0.82rem; }
.settings-link-text small { color: var(--mk-muted); font-size: 0.7rem; }

@media (max-width: 720px) {
  .settings-panel { grid-template-columns: 1fr; }
  .settings-tabs {
    flex-direction: row;
    overflow-x: auto;
    border-right: 0;
    border-bottom: 1px solid var(--mk-border);
  }
  .settings-tab { white-space: nowrap; }
}
</style>
