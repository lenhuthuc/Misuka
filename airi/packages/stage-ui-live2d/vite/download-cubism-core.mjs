// Downloads the Live2D Cubism Core runtime into an app's `public/` directory.
//
// Replaces `@proj-airi/unplugin-live2d-sdk`'s `DownloadLive2DSDK()`, which
// hardcodes SDK 5-r.3 in its output paths (its `from` option only changes the
// URL, not the folder it copies from, so it cannot be pointed at a newer
// release). That matters here because the Core caps which `moc3` file version
// it will load, and rejecting a too-new model surfaces only as
// `CubismMoc.create` throwing "Unknown error":
//
//   Core 5.0.0 (SDK 5-r.3) → moc3 v5 → Cubism Editor 5.0
//   Core 6.0.1 (SDK 5-r.5) → moc3 v6 → Cubism Editor 5.1
//
// `assets/models/models/TiredGirl_V1.moc3` is moc3 v6, so the app needs 5-r.5.
// Bump CUBISM_SDK_VERSION (and the matching <script src> in each index.html)
// when a model exported from a newer editor stops loading.
//
// Only the Core file is extracted; the rest of the SDK zip (~20 MiB of samples
// and framework sources) is never written to disk.

import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'

import JSZip from 'jszip'

export const CUBISM_SDK_VERSION = '5-r.5'

const CORE_ENTRY = 'Core/live2dcubismcore.min.js'

async function exists(path) {
  try {
    await stat(path)
    return true
  }
  catch {
    return false
  }
}

/**
 * @param {{ version?: string, from?: string, cacheDir?: string }} [options]
 */
export function DownloadCubismCore(options = {}) {
  const version = options.version ?? CUBISM_SDK_VERSION
  const from = options.from ?? `https://cubism.live2d.com/sdk-web/bin/CubismSdkForWeb-${version}.zip`

  return {
    name: 'download-live2d-cubism-core',
    async configResolved(config) {
      const relativePath = join('assets', 'js', `CubismSdkForWeb-${version}`, CORE_ENTRY)
      const publicPath = resolve(join(config.root, 'public', relativePath))
      if (await exists(publicPath))
        return

      const cachePath = resolve(join(options.cacheDir ?? join(config.root, '.cache'), relativePath))

      let core
      if (await exists(cachePath)) {
        core = await readFile(cachePath)
      }
      else {
        config.logger.info(`Downloading Live2D Cubism Core ${version}...`)
        const response = await fetch(from)
        if (!response.ok)
          throw new Error(`Failed to download Cubism SDK ${version}: ${response.status} ${response.statusText}`)

        const zip = await JSZip.loadAsync(await response.arrayBuffer())
        const entry = Object.keys(zip.files).find(name => name.endsWith(CORE_ENTRY))
        if (!entry)
          throw new Error(`Cubism SDK ${version} contains no ${CORE_ENTRY}`)

        core = await zip.file(entry).async('nodebuffer')

        await mkdir(dirname(cachePath), { recursive: true })
        await writeFile(cachePath, core)
      }

      await mkdir(dirname(publicPath), { recursive: true })
      await writeFile(publicPath, core)
      config.logger.info(`Live2D Cubism Core ${version} ready.`)
    },
  }
}
