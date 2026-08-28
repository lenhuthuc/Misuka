<script setup lang="ts">
import { useTheme } from '@proj-airi/ui'

import Button from '../shared/Button.vue'

import '../theme.css'

const emit = defineEmits<{ (e: 'start'): void }>()

const open = defineModel<boolean>({ default: false })

// Teleported out of `.mitsuka-app` like the other Mitsuka overlays, so it has to
// re-declare both the token scope and the light variant itself.
const { isDark } = useTheme()

// What Mitsuka actually does today, in the order a first-time reader cares
// about: how you talk to her, what she keeps, what she can look up, and where
// all of it lives.
const FEATURES = [
  {
    icon: 'i-solar:microphone-3-bold-duotone',
    title: 'Trò chuyện bằng giọng nói',
    body: 'Cứ nói tự nhiên — Mitsuka nghe, hiểu và trả lời lại bằng giọng nói.',
  },
  {
    icon: 'i-solar:notebook-bookmark-bold-duotone',
    title: 'Nhớ những gì bạn kể',
    body: 'Chuyện cũ được ghi lại để lần sau bạn không phải nhắc lại từ đầu.',
  },
  {
    icon: 'i-solar:magnifer-bold-duotone',
    title: 'Tự tra cứu khi cần',
    body: 'Gặp câu hỏi cần thông tin mới, Mitsuka sẽ tìm trên web rồi mới trả lời.',
  },
  {
    icon: 'i-solar:shield-check-bold-duotone',
    title: 'Ở lại trên máy bạn',
    body: 'Mô hình chạy cục bộ — không cần đăng nhập, không cần API key.',
  },
]
</script>

<template>
  <Teleport to="body">
    <Transition name="mk-welcome">
      <div
        v-if="open"
        class="mk-welcome mitsuka-app"
        :class="{ 'mitsuka-app--light': !isDark }"
        role="dialog"
        aria-modal="true"
        aria-label="Chào mừng đến với Mitsuka"
      >
        <div class="mk-welcome-aurora mk-welcome-aurora--one" />
        <div class="mk-welcome-aurora mk-welcome-aurora--two" />

        <section class="mk-welcome-card">
          <div class="mk-welcome-badge">
            <span class="mk-welcome-badge-spark mk-welcome-badge-spark--left">✦</span>
            <span class="mk-welcome-badge-face">◕‿◕</span>
            <span class="mk-welcome-badge-spark mk-welcome-badge-spark--right">✧</span>
          </div>

          <h1 class="mk-welcome-title">
            Mitsuka
          </h1>
          <p class="mk-welcome-tagline">
            Trợ lý AI của riêng bạn <span class="mk-welcome-heart">♥</span>
          </p>

          <p class="mk-welcome-lede">
            Mở lên là trò chuyện được ngay — không có bước đăng nhập, không có
            biểu mẫu nào phải điền trước.
          </p>

          <ul class="mk-welcome-features">
            <li v-for="feature in FEATURES" :key="feature.title" class="mk-welcome-feature">
              <span class="mk-welcome-feature-icon" :class="feature.icon" />
              <div class="mk-welcome-feature-text">
                <strong>{{ feature.title }}</strong>
                <span>{{ feature.body }}</span>
              </div>
            </li>
          </ul>

          <Button class="mk-welcome-cta" @click="emit('start')">
            Bắt đầu trò chuyện <span class="i-solar:arrow-right-linear" />
          </Button>

          <p class="mk-welcome-note">
            Mitsuka trả lời qua local-api ở <code>127.0.0.1:8010</code> — nhớ bật
            máy chủ đó trước khi bắt đầu.
          </p>
        </section>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.mk-welcome {
  position: fixed;
  z-index: 9999;
  display: grid;
  overflow: hidden auto;
  inset: 0;
  align-content: center;
  justify-items: center;
  background: radial-gradient(120% 120% at 50% 0%, var(--mk-bg-2) 0%, var(--mk-bg) 62%);
  padding: 1.5rem 1rem;
}

.mk-welcome-aurora {
  position: absolute;
  z-index: 0;
  width: 34rem;
  height: 34rem;
  border-radius: 999px;
  filter: blur(120px);
  opacity: 0.16;
  pointer-events: none;
}

.mk-welcome-aurora--one { top: -16rem; left: 12%; background: #8b5cf6; }
.mk-welcome-aurora--two { right: -10rem; bottom: -18rem; background: #ec4899; }

.mk-welcome-card {
  position: relative;
  z-index: 1;
  display: flex;
  width: min(32rem, 100%);
  flex-direction: column;
  align-items: center;
  border: 1px solid var(--mk-border-strong);
  border-radius: var(--mk-radius-lg);
  background: var(--mk-modal);
  padding: 2rem 1.75rem 1.5rem;
  box-shadow: var(--mk-shadow);
  text-align: center;
}

.mk-welcome-badge {
  display: flex;
  width: 4.5rem;
  height: 4.5rem;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--mk-border-strong);
  border-radius: 999px;
  background: var(--mk-accent);
  box-shadow: var(--mk-glow-pink);
  gap: 0.15rem;
}

.mk-welcome-badge-face {
  color: #fff;
  font-size: 0.95rem;
  font-weight: 800;
  letter-spacing: -0.04em;
}

.mk-welcome-badge-spark { color: rgb(255 255 255 / 0.85); font-size: 0.6rem; }
.mk-welcome-badge-spark--left { align-self: flex-start; margin-top: 1.15rem; }
.mk-welcome-badge-spark--right { align-self: flex-end; margin-bottom: 1.15rem; }

.mk-welcome-title {
  margin: 0.9rem 0 0;
  background: var(--mk-logo-gradient);
  background-clip: text;
  color: transparent;
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: 0.01em;
}

.mk-welcome-tagline {
  margin: 0.15rem 0 0;
  color: var(--mk-muted);
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.mk-welcome-heart { color: var(--mk-pink); }

.mk-welcome-lede {
  max-width: 24rem;
  margin: 0.85rem 0 0;
  color: var(--mk-ink-dim);
  font-size: 0.86rem;
  line-height: 1.6;
  opacity: 0.88;
}

.mk-welcome-features {
  display: grid;
  width: 100%;
  margin: 1.35rem 0 0;
  padding: 0;
  gap: 0.55rem;
  list-style: none;
}

.mk-welcome-feature {
  display: flex;
  align-items: flex-start;
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius);
  background: var(--mk-raise);
  padding: 0.7rem 0.85rem;
  gap: 0.7rem;
  text-align: left;
}

.mk-welcome-feature-icon {
  flex: 0 0 auto;
  margin-top: 0.1rem;
  color: var(--mk-pink-soft);
  font-size: 1.15rem;
}

.mk-welcome-feature-text { display: flex; min-width: 0; flex-direction: column; gap: 0.15rem; }

.mk-welcome-feature-text strong {
  color: var(--mk-ink);
  font-size: 0.82rem;
  font-weight: 800;
}

.mk-welcome-feature-text span {
  color: var(--mk-muted);
  font-size: 0.76rem;
  line-height: 1.5;
}

.mk-welcome-cta {
  margin-top: 1.4rem;
  padding: 0.65rem 1.6rem;
  font-size: 0.9rem;
}

.mk-welcome-note {
  max-width: 22rem;
  margin: 0.9rem 0 0;
  color: var(--mk-muted);
  font-size: 0.7rem;
  line-height: 1.5;
  opacity: 0.8;
}

.mk-welcome-note code {
  border-radius: 5px;
  background: var(--mk-inset);
  padding: 0.05rem 0.3rem;
  font-family: ui-monospace, SFMono-Regular, monospace;
}

.mk-welcome-enter-active, .mk-welcome-leave-active { transition: opacity 220ms ease; }
.mk-welcome-enter-from, .mk-welcome-leave-to { opacity: 0; }
.mk-welcome-enter-active .mk-welcome-card, .mk-welcome-leave-active .mk-welcome-card { transition: transform 220ms ease; }
.mk-welcome-enter-from .mk-welcome-card, .mk-welcome-leave-to .mk-welcome-card { transform: translateY(12px) scale(0.98); }

@media (max-width: 480px) {
  .mk-welcome-card { padding: 1.5rem 1.15rem 1.15rem; }
  .mk-welcome-title { font-size: 1.75rem; }
}

@media (prefers-reduced-motion: reduce) {
  .mk-welcome-enter-active, .mk-welcome-leave-active,
  .mk-welcome-enter-active .mk-welcome-card, .mk-welcome-leave-active .mk-welcome-card { transition: none; }
}
</style>
