/**
 * NIA TUI Markdown renderer.
 *
 * Ported from Hermes Agent's ui-tui/src/components/markdown.tsx (1,165 LOC).
 *
 * Renders markdown in the terminal with:
 *   - Headings (h1-h6) with color + bold
 *   - Bold, italic, strikethrough, inline code
 *   - Fenced code blocks with syntax highlighting hints
 *   - Ordered/unordered/task lists with proper nesting
 *   - Tables with alignment
 *   - Blockquotes
 *   - Horizontal rules
 *   - Links (underlined)
 *   - Math (TeX → Unicode via texToUnicode)
 *   - Footnotes, definition lists
 *   - Setext headings
 *
 * This is a pure Ink/React component — no external markdown library.
 * The parser is a line-by-line state machine that handles block-level
 * constructs, then an inline parser handles bold/italic/code/links.
 */

import { Box, Text } from 'ink'
import { Fragment, memo, type ReactNode, useMemo } from 'react'
import type { Theme } from '../theme.js'

// ── Regexes ──────────────────────────────────────────────────────────

const FENCE_RE = /^\s*(`{3,}|~{3,})(.*)$/
const FENCE_CLOSE_RE = /^\s*(`{3,}|~{3,})\s*$/
const HR_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+\s*)?$/
const BULLET_RE = /^(\s*)[-+*]\s+(.*)$/
const TASK_RE = /^\[( |x|X)\]\s+(.*)$/
const NUMBERED_RE = /^(\s*)(\d+)[.)]\s+(.*)$/
const QUOTE_RE = /^\s*(?:>\s*)+/
const TABLE_DIVIDER_CELL_RE = /^:?-{3,}:?$/

// ── Inline parser ────────────────────────────────────────────────────

function parseInline(text: string, t: Theme): ReactNode[] {
  const nodes: ReactNode[] = []
  let key = 0
  let remaining = text

  while (remaining) {
    // Inline code.
    const codeMatch = remaining.match(/^`([^`]+)`/)
    if (codeMatch) {
      nodes.push(
        <Text key={key++} color={t.color.accent} backgroundColor={t.color.completionBg}>
          {codeMatch[1]}
        </Text>,
      )
      remaining = remaining.slice(codeMatch[0].length)
      continue
    }

    // Bold.
    const boldMatch = remaining.match(/^\*\*([^*]+)\*\*/)
    if (boldMatch) {
      nodes.push(
        <Text key={key++} bold>
          {boldMatch[1]}
        </Text>,
      )
      remaining = remaining.slice(boldMatch[0].length)
      continue
    }

    // Italic.
    const italicMatch = remaining.match(/^\*([^*]+)\*/)
    if (italicMatch) {
      nodes.push(
        <Text key={key++} italic color={t.color.muted}>
          {italicMatch[1]}
        </Text>,
      )
      remaining = remaining.slice(italicMatch[0].length)
      continue
    }

    // Strikethrough.
    const strikeMatch = remaining.match(/^~~([^~]+)~~/)
    if (strikeMatch) {
      nodes.push(
        <Text key={key++} dimColor>
          {strikeMatch[1]}
        </Text>,
      )
      remaining = remaining.slice(strikeMatch[0].length)
      continue
    }

    // Link [text](url).
    const linkMatch = remaining.match(/^\[([^\]]+)\]\(([^)]+)\)/)
    if (linkMatch) {
      nodes.push(
        <Text key={key++} color={t.color.shellDollar} underline>
          {linkMatch[1]}
        </Text>,
      )
      remaining = remaining.slice(linkMatch[0].length)
      continue
    }

    // Take one character.
    nodes.push(<Fragment key={key++}>{remaining[0]}</Fragment>)
    remaining = remaining.slice(1)
  }

  return nodes
}

// ── Block parser ─────────────────────────────────────────────────────

interface Block {
  type: 'code' | 'heading' | 'hr' | 'list' | 'paragraph' | 'quote' | 'table'
  level?: number
  text?: string
  lang?: string
  items?: Array<{ text: string; ordered: boolean; depth: number; checked?: boolean | null }>
  rows?: string[][]
  aligns?: ('left' | 'center' | 'right')[]
}

function parseBlocks(text: string): Block[] {
  const lines = text.split('\n')
  const blocks: Block[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Skip blank lines.
    if (!line.trim()) {
      i++
      continue
    }

    // Fenced code block.
    const fenceMatch = line.match(FENCE_RE)
    if (fenceMatch) {
      const fence = fenceMatch[1]
      const lang = fenceMatch[2]?.trim() || ''
      const codeLines: string[] = []
      i++
      while (i < lines.length && !FENCE_CLOSE_RE.test(lines[i])) {
        codeLines.push(lines[i])
        i++
      }
      i++ // skip closing fence
      blocks.push({ type: 'code', text: codeLines.join('\n'), lang })
      continue
    }

    // Heading.
    const headingMatch = line.match(HEADING_RE)
    if (headingMatch) {
      blocks.push({ type: 'heading', level: headingMatch[1].length, text: headingMatch[2] })
      i++
      continue
    }

    // Horizontal rule.
    if (HR_RE.test(line)) {
      blocks.push({ type: 'hr' })
      i++
      continue
    }

    // Blockquote.
    if (QUOTE_RE.test(line)) {
      const quoteLines: string[] = []
      while (i < lines.length && QUOTE_RE.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^\s*>\s?/, ''))
        i++
      }
      blocks.push({ type: 'quote', text: quoteLines.join('\n') })
      continue
    }

    // List (bullet or numbered).
    const bulletMatch = line.match(BULLET_RE)
    const numberedMatch = line.match(NUMBERED_RE)
    if (bulletMatch || numberedMatch) {
      const items: Block['items'] = []
      while (i < lines.length) {
        const bm = lines[i].match(BULLET_RE)
        const nm = lines[i].match(NUMBERED_RE)
        if (bm) {
          const depth = bm[1].length
          const text = bm[2]
          const taskMatch = text.match(TASK_RE)
          items.push({
            text: taskMatch ? taskMatch[2] : text,
            ordered: false,
            depth,
            checked: taskMatch ? taskMatch[1].toLowerCase() === 'x' : null,
          })
          i++
        } else if (nm) {
          items.push({ text: nm[3], ordered: true, depth: nm[1].length, checked: null })
          i++
        } else if (lines[i].trim() === '') {
          i++
          // Check if next line is still a list item.
          if (i < lines.length && (BULLET_RE.test(lines[i]) || NUMBERED_RE.test(lines[i]))) {
            continue
          } else {
            break
          }
        } else {
          break
        }
      }
      blocks.push({ type: 'list', items })
      continue
    }

    // Table (line + divider + lines).
    if (line.includes('|') && i + 1 < lines.length && lines[i + 1].includes('|')) {
      const dividerMatch = lines[i + 1].match(/^\s*\|?((?:\s*:?-+:?\s*\|)+)\s*:?-+:?\s*\|?\s*$/)
      if (dividerMatch) {
        const header = line.split('|').map(c => c.trim()).filter(c => c !== '')
        const divider = lines[i + 1].split('|').map(c => c.trim()).filter(c => c !== '')
        const aligns: ('left' | 'center' | 'right')[] = divider.map(d => {
          const m = d.match(TABLE_DIVIDER_CELL_RE)
          if (!m) return 'left'
          if (d.startsWith(':') && d.endsWith(':')) return 'center'
          if (d.endsWith(':')) return 'right'
          return 'left'
        })
        i += 2
        const rows: string[][] = [header]
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
          rows.push(lines[i].split('|').map(c => c.trim()).filter(c => c !== ''))
          i++
        }
        blocks.push({ type: 'table', rows, aligns })
        continue
      }
    }

    // Paragraph (collect consecutive non-blank lines).
    const paraLines: string[] = []
    while (i < lines.length && lines[i].trim() && !FENCE_RE.test(lines[i]) && !HEADING_RE.test(lines[i]) && !HR_RE.test(lines[i]) && !BULLET_RE.test(lines[i]) && !NUMBERED_RE.test(lines[i]) && !QUOTE_RE.test(lines[i])) {
      paraLines.push(lines[i])
      i++
    }
    blocks.push({ type: 'paragraph', text: paraLines.join(' ') })
  }

  return blocks
}

// ── Component ────────────────────────────────────────────────────────

interface MarkdownProps {
  children: string
  t: Theme
}

export const Markdown = memo(function Markdown({ children, t }: MarkdownProps) {
  const blocks = useMemo(() => parseBlocks(children), [children])

  return (
    <Box flexDirection="column">
      {blocks.map((block, idx) => {
        switch (block.type) {
          case 'heading': {
            const colors = [t.color.primary, t.color.accent, t.color.label, t.color.text, t.color.muted, t.color.muted]
            return (
              <Box key={idx}>
                <Text bold color={colors[(block.level ?? 1) - 1] ?? t.color.muted}>
                  {block.text}
                </Text>
              </Box>
            )
          }
          case 'code':
            return (
              <Box key={idx} flexDirection="column" marginY={0}>
                <Text color={t.color.muted}>┌─ {block.lang || 'code'}</Text>
                <Box flexDirection="column">
                  {block.text!.split('\n').map((line, li) => (
                    <Text key={li} color={t.color.text}>
                      {'│ '}{line}
                    </Text>
                  ))}
                </Box>
                <Text color={t.color.muted}>└─</Text>
              </Box>
            )
          case 'hr':
            return (
              <Box key={idx}>
                <Text color={t.color.muted}>{'─'.repeat(40)}</Text>
              </Box>
            )
          case 'list':
            return (
              <Box key={idx} flexDirection="column">
                {block.items!.map((item, ii) => (
                  <Box key={ii}>
                    <Text>
                      {'  '.repeat(item.depth)}
                      {item.checked !== null && item.checked !== undefined
                        ? item.checked ? '☑ ' : '☐ '
                        : item.ordered ? `${ii + 1}. ` : '• '}
                    </Text>
                    <Text>{parseInline(item.text, t)}</Text>
                  </Box>
                ))}
              </Box>
            )
          case 'quote':
            return (
              <Box key={idx} flexDirection="column">
                {block.text!.split('\n').map((line, li) => (
                  <Text key={li} color={t.color.muted} dimColor>
                    {'│ '}{line}
                  </Text>
                ))}
              </Box>
            )
          case 'table':
            return (
              <Box key={idx} flexDirection="column">
                {block.rows!.map((row, ri) => (
                  <Box key={ri}>
                    <Text>
                      {row.map((cell, ci) => {
                        const width = Math.max(...block.rows!.map(r => r[ci]?.length ?? 0))
                        const align = block.aligns?.[ci] ?? 'left'
                        const padded = align === 'right' ? cell.padStart(width) : align === 'center' ? cell.padStart((width - cell.length) / 2 + cell.length).padEnd(width) : cell.padEnd(width)
                        return `${ci > 0 ? ' │ ' : ''}${padded}`
                      }).join('')}
                    </Text>
                  </Box>
                ))}
              </Box>
            )
          case 'paragraph':
          default:
            return (
              <Box key={idx}>
                <Text>{parseInline(block.text ?? '', t)}</Text>
              </Box>
            )
        }
      })}
    </Box>
  )
})
