<script setup lang="ts">
import { SpeechProviderSettings } from '@proj-airi/stage-ui/components'

const providerId = 'piper-tts'
const defaultModel = 'piper'
</script>

<template>
  <SpeechProviderSettings :provider-id="providerId" :default-model="defaultModel" />
</template>

<route lang="yaml">
meta:
  layout: settings
  stageTransition:
    name: slide
</route>
