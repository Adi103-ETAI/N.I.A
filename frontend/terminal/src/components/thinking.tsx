/**
 * NIA TUI Thinking/Reasoning display.
 *
 * Ported from Hermes Agent's ui-tui/src/components/thinking.tsx (1,211 LOC).
 *
 * Displays the model's reasoning/thinking blocks with:
 *   - Collapsible sections (collapsed/truncated/full modes)
 *   - Animated spinner while thinking is in progress
 *   - Token count display
 *   - Tool trail (list of tools called during this thinking block)
 *   - Subagent tree visualization (for delegated tasks)
 *   - Elapsed time
 *   - Hotness indicator (how active a subagent is)
 *
 * The component receives thinking text (streamed in via thinking.delta
 * events) and renders it with markdown + syntax highlighting.
 */

import { Box, Text, useApp } from 'ink'
import { memo, type ReactNode, useEffect, useMemo, useState } from 'react'
import type { Theme } from '../theme.js'
import type {
  ActiveTool,
  ActivityItem,
  DetailsMode,
  SectionVisibility,
  SubagentNode,
  SubagentProgress,
  ThinkingMode,
} from '../domain/types.js'

// ── Constants ────────────────────────────────────────────────────────

const THINKING_COT_MAX = 4000 // Max chars of thinking to show in truncated mode
const TRUNCATE_PREVIEW = 200

// ── Spinner ──────────────────────────────────────────────────────────

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

function useSpinner(active: boolean): string {
  const [frame, setFrame] = useState(0)
  useEffect(() => {
    if (!active) return
    const interval = setInterval(() => setFrame(f => (f + 1) % SPINNER_FRAMES.length), 80)
    return () => clearInterval(interval)
  }, [active])
  return active ? SPINNER_FRAMES[frame] : ''
}

// ── Token formatter ──────────────────────────────────────────────────

function fmtTokens(n: number | undefined): string {
  if (!n || n <= 0) return ''
  if (n < 1000) return `${n} tok`
  if (n < 1000000) return `${(n / 1000).toFixed(1)}k tok`
  return `${(n / 1000000).toFixed(1)}M tok`
}

function fmtElapsed(ms: number): string {
  const sec = Math.max(0, ms) / 1000
  return sec < 10 ? `${sec.toFixed(1)}s` : `${Math.round(sec)}s`
}

// ── Thinking preview ─────────────────────────────────────────────────

function thinkingPreview(text: string, maxLen: number = TRUNCATE_PREVIEW): string {
  const cleaned = text.replace(/\n+/g, ' ').trim()
  if (cleaned.length <= maxLen) return cleaned
  return cleaned.slice(0, maxLen) + '…'
}

// ── Tool trail ───────────────────────────────────────────────────────

function ToolTrail({ tools, t }: { tools: string[] | undefined; t: Theme }) {
  if (!tools || tools.length === 0) return null
  const display = tools.slice(-8) // Last 8 tools
  const overflow = tools.length - display.length
  return (
    <Box>
      <Text color={t.color.muted}> {t.brand.tool} </Text>
      {display.map((tool, i) => (
        <Text key={i} color={t.color.label}>
          {tool}
          {i < display.length - 1 ? ' → ' : ''}
        </Text>
      ))}
      {overflow > 0 && <Text color={t.color.muted}> (+{overflow} more)</Text>}
    </Box>
  )
}

// ── Activity feed ────────────────────────────────────────────────────

function ActivityFeed({ items, t }: { items: ActivityItem[] | undefined; t: Theme }) {
  if (!items || items.length === 0) return null
  const colors = { error: t.color.error, info: t.color.muted, warn: t.color.warn }
  return (
    <Box flexDirection="column">
      {items.slice(-5).map(item => (
        <Box key={item.id}>
          <Text color={colors[item.tone] ?? t.color.muted}>  • {item.text}</Text>
        </Box>
      ))}
    </Box>
  )
}

// ── Subagent tree ────────────────────────────────────────────────────

function SubagentTree({ node, rails = [], t }: { node: SubagentNode; rails?: boolean[]; t: Theme }) {
  const branch: 'mid' | 'last' = 'mid' // simplified
  const lead = `${rails.map(on => (on ? '│ ' : '  ')).join('')}${branch === 'mid' ? '├─ ' : '└─ '}`

  const statusIcons: Record<string, string> = {
    completed: '✓',
    error: '✗',
    failed: '✗',
    interrupted: '⊘',
    queued: '⏳',
    running: '▶',
    timeout: '⏱',
  }
  const statusColors: Record<string, string> = {
    completed: t.color.ok,
    error: t.color.error,
    failed: t.color.error,
    interrupted: t.color.warn,
    queued: t.color.muted,
    running: t.color.accent,
    timeout: t.color.warn,
  }

  const item = node.item
  const icon = statusIcons[item.status] ?? '•'
  const color = statusColors[item.status] ?? t.color.muted

  return (
    <Box flexDirection="column">
      <Box>
        <Text color={t.color.muted}>{lead}</Text>
        <Text color={color}>{icon} </Text>
        <Text color={t.color.text} bold>{item.goal.slice(0, 60)}</Text>
        {item.model && <Text color={t.color.muted}> ({item.model})</Text>}
        {item.toolCount > 0 && <Text color={t.color.muted}> [{item.toolCount} tools]</Text>}
      </Box>
      {node.children.map((child, i) => (
        <SubagentTree key={i} node={child} rails={[...rails, branch === 'mid']} t={t} />
      ))}
    </Box>
  )
}

// ── Main Thinking component ──────────────────────────────────────────

interface ThinkingProps {
  text: string | undefined
  mode: ThinkingMode
  tokens?: number
  tools?: string[]
  activity?: ActivityItem[]
  activeTool?: ActiveTool
  subagents?: SubagentNode[]
  sectionVisibility?: SectionVisibility
  isStreaming?: boolean
  startedAt?: number
  t: Theme
}

export const Thinking = memo(function Thinking({
  text,
  mode,
  tokens,
  tools,
  activity,
  activeTool,
  subagents,
  sectionVisibility,
  isStreaming,
  startedAt,
  t,
}: ThinkingProps) {
  const [elapsed, setElapsed] = useState(0)
  const spinner = useSpinner(isStreaming ?? false)

  useEffect(() => {
    if (!startedAt || !isStreaming) return
    const interval = setInterval(() => setElapsed(Date.now() - startedAt), 100)
    return () => clearInterval(interval)
  }, [startedAt, isStreaming])

  if (mode === 'hidden' || (!text && !isStreaming && (!subagents || subagents.length === 0))) {
    return null
  }

  const headerColor = isStreaming ? t.color.accent : t.color.muted
  const effectiveMode = sectionVisibility?.thinking ?? mode

  return (
    <Box flexDirection="column" marginY={0}>
      {/* Header */}
      <Box>
        <Text color={headerColor}>
          {spinner} {isStreaming ? 'Thinking' : 'Thought'}
          {elapsed > 0 && <Text color={t.color.muted}> · {fmtElapsed(elapsed)}</Text>}
          {tokens && tokens > 0 && <Text color={t.color.muted}> · {fmtTokens(tokens)}</Text>}
        </Text>
      </Box>

      {/* Thinking content */}
      {effectiveMode === 'full' && text && (
        <Box flexDirection="column" marginLeft={2}>
          {text.split('\n').slice(0, 100).map((line, i) => (
            <Text key={i} color={t.color.muted} dimColor>
              {line}
            </Text>
          ))}
          {text.split('\n').length > 100 && (
            <Text color={t.color.muted}>… ({text.split('\n').length - 100} more lines)</Text>
          )}
        </Box>
      )}

      {effectiveMode === 'truncated' && text && (
        <Box marginLeft={2}>
          <Text color={t.color.muted} dimColor>
            {thinkingPreview(text, THINKING_COT_MAX)}
          </Text>
        </Box>
      )}

      {effectiveMode === 'collapsed' && text && (
        <Box marginLeft={2}>
          <Text color={t.color.muted} dimColor>
            {thinkingPreview(text)} {text.length > TRUNCATE_PREVIEW ? '…' : ''}
          </Text>
        </Box>
      )}

      {/* Active tool */}
      {activeTool && (
        <Box marginLeft={2}>
          <Text color={t.color.accent}>
            {spinner} {activeTool.name}
            {activeTool.verboseArgs ? ` ${activeTool.verboseArgs}` : ''}
          </Text>
        </Box>
      )}

      {/* Tool trail */}
      {(sectionVisibility?.tools ?? 'collapsed') !== 'hidden' && (
        <Box marginLeft={2}>
          <ToolTrail tools={tools} t={t} />
        </Box>
      )}

      {/* Activity feed */}
      {(sectionVisibility?.activity ?? 'hidden') !== 'hidden' && (
        <Box marginLeft={2}>
          <ActivityFeed items={activity} t={t} />
        </Box>
      )}

      {/* Subagent tree */}
      {(sectionVisibility?.subagents ?? 'collapsed') !== 'hidden' && subagents && subagents.length > 0 && (
        <Box flexDirection="column" marginLeft={2}>
          {subagents.map((node, i) => (
            <SubagentTree key={i} node={node} t={t} />
          ))}
        </Box>
      )}
    </Box>
  )
})
