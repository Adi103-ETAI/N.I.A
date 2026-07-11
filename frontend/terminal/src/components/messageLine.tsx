/**
 * NIA TUI Message Line — renders a single message in the transcript.
 *
 * Ported from Hermes Agent's ui-tui/src/components/messageLine.tsx (259 LOC).
 *
 * Renders different message roles with distinct visual treatment:
 *   - user:    prompt symbol + text
 *   - assistant: markdown rendered + optional thinking block
 *   - system:   dimmed text
 *   - tool:     tool name + truncated result
 */

import { Box, Text } from 'ink'
import { memo } from 'react'
import { Markdown } from './markdown.js'
import { Thinking } from './thinking.js'
import type { Theme } from '../theme.js'
import type { Msg, ThinkingMode } from '../domain/types.js'

interface MessageLineProps {
  msg: Msg
  thinkingMode?: ThinkingMode
  t: Theme
}

export const MessageLine = memo(function MessageLine({ msg, thinkingMode = 'collapsed', t }: MessageLineProps) {
  switch (msg.role) {
    case 'user':
      return (
        <Box flexDirection="column" marginY={0}>
          <Box>
            <Text color={t.color.prompt} bold>{t.brand.prompt} </Text>
            <Text>{msg.text}</Text>
          </Box>
        </Box>
      )

    case 'assistant':
      return (
        <Box flexDirection="column" marginY={0}>
          {msg.thinking && (
            <Thinking
              text={msg.thinking}
              mode={thinkingMode}
              tokens={msg.thinkingTokens}
              tools={msg.tools}
              isStreaming={false}
              t={t}
            />
          )}
          {msg.text && <Markdown t={t}>{msg.text}</Markdown>}
          {msg.todos && msg.todos.length > 0 && (
            <Box flexDirection="column" marginLeft={2}>
              {msg.todos.map(todo => (
                <Box key={todo.id}>
                  <Text>
                    {todo.status === 'completed' ? '☑' : todo.status === 'in_progress' ? '◐' : todo.status === 'cancelled' ? '⊘' : '☐'}{' '}
                  </Text>
                  <Text color={todo.status === 'completed' ? t.color.muted : t.color.text} dimColor={todo.status === 'completed'}>
                    {todo.content}
                  </Text>
                </Box>
              ))}
            </Box>
          )}
        </Box>
      )

    case 'system':
      return (
        <Box>
          <Text color={t.color.muted} dimColor>{msg.text}</Text>
        </Box>
      )

    case 'tool':
      return (
        <Box flexDirection="column" marginY={0}>
          {msg.tools && msg.tools.length > 0 && (
            <Box>
              <Text color={t.color.muted}> {t.brand.tool} </Text>
              <Text color={t.color.label}>{msg.tools.join(' → ')}</Text>
            </Box>
          )}
          {msg.text && (
            <Box marginLeft={2}>
              <Text color={t.color.muted} dimColor>
                {msg.text.length > 500 ? msg.text.slice(0, 500) + '…' : msg.text}
              </Text>
            </Box>
          )}
        </Box>
      )

    default:
      return (
        <Box>
          <Text>{msg.text}</Text>
        </Box>
      )
  }
})
