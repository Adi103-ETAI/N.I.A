/**
 * NIA TUI App Layout — the main layout container.
 *
 * Ported from Hermes Agent's ui-tui/src/components/appLayout.tsx (564 LOC).
 *
 * Lays out the TUI into regions:
 *   - Header (banner/logo/session info)
 *   - Conversation area (transcript + streaming assistant) — grows
 *   - Side panel (todos, active session info) — optional
 *   - Status bar (model, cwd, permissions, cost)
 *   - Input area (prompt input)
 *   - Overlays (approval, clarify, command picker, model picker)
 *   - Footer (keyboard hints)
 */

import { Box, Text } from 'ink'
import type { ReactNode } from 'react'
import type { Theme } from '../theme.js'

interface AppLayoutProps {
  header?: ReactNode
  children: ReactNode
  sidePanel?: ReactNode
  statusBar?: ReactNode
  input?: ReactNode
  overlay?: ReactNode
  footer?: ReactNode
  t: Theme
}

export function AppLayout({ header, children, sidePanel, statusBar, input, overlay, footer, t }: AppLayoutProps) {
  return (
    <Box flexDirection="column" height="100%">
      {/* Header */}
      {header && (
        <Box flexShrink={0}>
          {header}
        </Box>
      )}

      {/* Main content area: conversation + side panel */}
      <Box flexDirection="row" flexGrow={1}>
        {/* Conversation */}
        <Box flexDirection="column" flexGrow={1} overflow="hidden">
          {children}
        </Box>

        {/* Side panel */}
        {sidePanel && (
          <Box flexShrink={0} flexDirection="column" borderStyle="single" borderColor={t.color.border} minWidth={30} maxWidth={50}>
            {sidePanel}
          </Box>
        )}
      </Box>

      {/* Overlay (approval, clarify, etc.) */}
      {overlay && (
        <Box flexShrink={0}>
          {overlay}
        </Box>
      )}

      {/* Status bar */}
      {statusBar && (
        <Box flexShrink={0}>
          {statusBar}
        </Box>
      )}

      {/* Input */}
      {input && (
        <Box flexShrink={0}>
          {input}
        </Box>
      )}

      {/* Footer */}
      {footer && (
        <Box flexShrink={0}>
          {footer}
        </Box>
      )}
    </Box>
  )
}
