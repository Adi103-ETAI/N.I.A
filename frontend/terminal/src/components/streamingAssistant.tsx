/**
 * NIA TUI Streaming Assistant — renders streaming token deltas with markdown.
 *
 * Ported from Hermes Agent's ui-tui/src/components/streamingAssistant.tsx (118 LOC).
 *
 * As message.delta events arrive, this component accumulates the text and
 * re-renders the markdown. To avoid re-parsing the entire markdown on every
 * token (expensive), it batches updates with a short debounce (~16ms = 60fps).
 *
 * When the message.complete event arrives, the final text is rendered once
 * with full markdown.
 */

import { Box, Text } from 'ink'
import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { Markdown } from './markdown.js'
import { Thinking } from './thinking.js'
import type { Theme } from '../theme.js'
import type { ThinkingMode } from '../domain/types.js'

interface StreamingAssistantProps {
  buffer: string
  thinking?: string
  thinkingMode?: ThinkingMode
  isStreaming: boolean
  t: Theme
}

const STREAM_BATCH_MS = 16 // ~60fps

export const StreamingAssistant = memo(function StreamingAssistant({
  buffer,
  thinking,
  thinkingMode = 'collapsed',
  isStreaming,
  t,
}: StreamingAssistantProps) {
  const [displayBuffer, setDisplayBuffer] = useState(buffer)
  const lastUpdate = useRef(0)

  useEffect(() => {
    const now = Date.now()
    if (now - lastUpdate.current >= STREAM_BATCH_MS) {
      setDisplayBuffer(buffer)
      lastUpdate.current = now
    } else {
      const timer = setTimeout(() => {
        setDisplayBuffer(buffer)
        lastUpdate.current = Date.now()
      }, STREAM_BATCH_MS)
      return () => clearTimeout(timer)
    }
  }, [buffer])

  // Show thinking block if present.
  const showThinking = thinking && thinking.trim().length > 0

  return (
    <Box flexDirection="column">
      {showThinking && (
        <Thinking
          text={thinking}
          mode={thinkingMode}
          isStreaming={isStreaming}
          t={t}
        />
      )}
      {displayBuffer ? (
        <Markdown t={t}>{displayBuffer}</Markdown>
      ) : isStreaming ? (
        <Text color={t.color.muted}>…</Text>
      ) : null}
    </Box>
  )
})
