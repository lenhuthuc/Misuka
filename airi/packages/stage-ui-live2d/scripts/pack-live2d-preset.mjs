// Packs a Live2D Cubism 4 model directory into the single-file `.zip` preset
// format that `display-models.ts` ships and `live2d-zip-loader.ts` reads.
//
// The preset zip is committed (unlike `assets/live2d/models/*`, which is
// gitignored and populated by the upstream Download plugin), so a fresh clone
// renders the stage without a network fetch. Re-run this whenever the source
// model under `assets/models/` changes:
//
//   node packages/stage-ui-live2d/scripts/pack-live2d-preset.mjs
//   node packages/stage-ui-live2d/scripts/pack-live2d-preset.mjs <srcDir> <outZip>
//
// Entry names are always written with forward slashes — `Cubism4ModelSettings`
// resolves texture/physics paths relative to the `.model3.json` entry, and
// Windows-style separators would break that lookup.

import process from 'node:process'

import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import JSZip from 'jszip'

const scriptDir = dirname(fileURLToPath(import.meta.url))
// packages/stage-ui-live2d/scripts -> repository root (Mitsuka/)
const repoRoot = resolve(scriptDir, '..', '..', '..', '..')

const defaultSrcDir = join(repoRoot, 'assets', 'models', 'models')
const defaultOutZip = join(
  repoRoot,
  'airi',
  'packages',
  'stage-ui',
  'src',
  'assets',
  'live2d',
  'preset',
  'tiredgirl.zip',
)

async function collectFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const full = join(dir, entry.name)
    if (entry.isDirectory())
      files.push(...await collectFiles(full))
    else if (entry.isFile())
      files.push(full)
  }

  return files
}

async function main() {
  const srcDir = resolve(process.argv[2] ?? defaultSrcDir)
  const outZip = resolve(process.argv[3] ?? defaultOutZip)

  const files = await collectFiles(srcDir)
  if (!files.some(file => file.endsWith('.model3.json')))
    throw new Error(`No *.model3.json found under ${srcDir}`)

  const zip = new JSZip()
  for (const file of files) {
    const entryName = relative(srcDir, file).split(/[\\/]/).join('/')
    zip.file(entryName, await readFile(file))
  }

  const buffer = await zip.generateAsync({
    type: 'nodebuffer',
    compression: 'DEFLATE',
    compressionOptions: { level: 9 },
    // Deterministic timestamps keep the committed zip from churning on every run.
    date: new Date(0),
  })

  await mkdir(dirname(outZip), { recursive: true })
  await writeFile(outZip, buffer)

  console.info(`[pack-live2d-preset] ${files.length} files -> ${outZip} (${(buffer.length / 1024 / 1024).toFixed(2)} MiB)`)
}

await main()
