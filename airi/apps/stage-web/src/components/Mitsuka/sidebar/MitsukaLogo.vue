<template>
  <div class="mk-logo">
    <div class="mk-logo-row">
      <span class="mk-logo-spark mk-logo-spark--left">✦</span>
      <h1>Mitsuka</h1>
      <span class="mk-logo-spark mk-logo-spark--right">✧</span>
    </div>
    <p>AI Assistant <span class="mk-logo-heart">♥</span></p>
  </div>
</template>

<style scoped>
.mk-logo { padding: 0.35rem 0.15rem 0.15rem; }

.mk-logo-row {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.mk-logo-row h1 {
  margin: 0;
  background: var(--mk-logo-gradient);
  background-clip: text;
  color: transparent;
  font-size: 1.45rem;
  font-weight: 800;
  letter-spacing: 0.01em;
}

.mk-logo-spark {
  color: var(--mk-pink-soft);
  font-size: 0.8rem;
  text-shadow: 0 0 10px rgb(255 121 193 / 0.55);
}

.mk-logo-spark--left { font-size: 0.7rem; opacity: 0.75; }

.mk-logo p {
  margin: 0.1rem 0 0;
  color: var(--mk-muted);
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.mk-logo-heart { color: var(--mk-pink); }
</style>
