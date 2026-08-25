import { describe, expect, it } from 'vitest'

import { extractSpeakableChunks, MIN_CHUNK_LEN } from './local-conversation-tts'

/**
 * ROOT CAUSE these tests pin down:
 *
 * The first sentence-by-sentence implementation closed a sentence only on a
 * terminator *followed by whitespace*, and reported "nothing was ever queued"
 * through a boolean the caller checked after the stream ended. A reply ending
 * in "." therefore left its entire text in the tail buffer, and the caller
 * spoke that tail and then the full reply again — every single-sentence answer
 * was heard twice, the second time from the top. Speaking the whole reply in
 * one request made that unrepresentable, at the cost of the user hearing
 * nothing for the whole of generation.
 *
 * The property that makes streaming safe again is that every character of the
 * buffer comes back in exactly one of the two return values, so "spoken twice"
 * cannot be expressed. Most of what follows checks that partition directly.
 */
describe('extractSpeakableChunks', () => {
  it('emits a sentence once its boundary has arrived', () => {
    expect(extractSpeakableChunks('Xin chào bạn. Mình khoẻ')).toEqual([['Xin chào bạn.'], ' Mình khoẻ'])
  })

  it('emits several sentences in speaking order', () => {
    const buffer = 'Đây là câu thứ nhất. Đây là câu thứ hai! Đây là câu thứ ba? Còn lại'
    const [chunks, remaining] = extractSpeakableChunks(buffer)
    expect(chunks).toEqual(['Đây là câu thứ nhất.', 'Đây là câu thứ hai!', 'Đây là câu thứ ba?'])
    expect(remaining).toBe(' Còn lại')
  })

  it('holds back a sentence whose terminator has no whitespace after it yet', () => {
    // Mid-stream this is not a boundary: the next token may be "14".
    expect(extractSpeakableChunks('Kết quả là 3.')).toEqual([[], 'Kết quả là 3.'])
  })

  it('does not split a decimal number', () => {
    expect(extractSpeakableChunks('Giá trị 3.14 là hằng số')).toEqual([[], 'Giá trị 3.14 là hằng số'])
  })

  it('keeps the whole reply as the remainder when no boundary closed', () => {
    // This is the exact shape that used to be spoken twice.
    const reply = 'Mình vẫn ổn nhé.'
    expect(extractSpeakableChunks(reply)).toEqual([[], reply])
  })

  it('returns every character exactly once, split between chunks and remainder', () => {
    const buffer = 'Một câu đủ dài. Câu thứ hai cũng vậy! Phần đuôi'
    const [chunks, remaining] = extractSpeakableChunks(buffer)
    // Whitespace is trimmed off each chunk, so compare with it collapsed away.
    expect((chunks.join('') + remaining).replace(/\s+/g, '')).toBe(buffer.replace(/\s+/g, ''))
  })

  it('folds a fragment shorter than the minimum into the next chunk', () => {
    // The old splitter *filtered* short fragments out, so "Vâng." was dropped
    // from the spoken reply entirely rather than being spoken with what follows.
    const [chunks] = extractSpeakableChunks('Vâng. Mình hiểu ý bạn rồi. ')
    expect(chunks).toEqual(['Vâng. Mình hiểu ý bạn rồi.'])
  })

  it('treats a line break as a boundary', () => {
    const [chunks, remaining] = extractSpeakableChunks('Danh sách như sau\nMục đầu tiên')
    expect(chunks).toEqual(['Danh sách như sau'])
    expect(remaining).toBe('Mục đầu tiên')
  })

  it('closes after a closing quote so the punctuation is not spoken alone', () => {
    const [chunks] = extractSpeakableChunks('Cô ấy nói "xong rồi." Rồi đi ra')
    expect(chunks).toEqual(['Cô ấy nói "xong rồi."'])
  })

  it('collapses an ellipsis into one boundary rather than three', () => {
    const [chunks] = extractSpeakableChunks('Ừ thì cũng được... Để mình xem')
    expect(chunks).toEqual(['Ừ thì cũng được...'])
  })

  it('is stable when fed its own remainder, which is how the stream drives it', () => {
    let pending = ''
    const spoken: string[] = []
    for (const token of ['Xin ', 'chào ', 'bạn. ', 'Mình ', 'là ', 'Mitsuka.']) {
      pending += token
      const [chunks, remaining] = extractSpeakableChunks(pending)
      pending = remaining
      spoken.push(...chunks)
    }
    // The last sentence has no trailing whitespace, so it is still pending —
    // the caller flushes it once the stream ends, and only from here.
    expect(spoken).toEqual(['Xin chào bạn.'])
    expect(pending.trim()).toBe('Mình là Mitsuka.')
  })

  it('has a minimum short enough not to swallow an ordinary sentence', () => {
    expect(MIN_CHUNK_LEN).toBeLessThan('Mình khoẻ, cảm ơn bạn.'.length)
  })
})
