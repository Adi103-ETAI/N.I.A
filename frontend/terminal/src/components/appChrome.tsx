/**
 * NIA TUI App Chrome — header + branding + session info display.
 *
 * Ported from Hermes Agent's ui-tui/src/components/appChrome.tsx (804 LOC).
 *
 * Renders the TUI chrome:
 *   - Brand icon + name
 *   - Session model + cwd + profile
 *   - Usage stats (tokens, cost)
 *   - Update indicator
 *   - Connection status
 */

import { Box, Text } from 'ink'
import { memo } from 'react'
import type { Theme } from '../theme.js'
import type { SessionInfo, Usage } from '../domain/types.js'

interface AppChromeProps {
  sessionInfo: SessionInfo | null
  sid: string
  t: Theme
}

export const AppChrome = memo(function AppChrome({ sessionInfo, sid, t }: AppChromeProps) {
  const model = sessionInfo?.model ?? '…'
  const cwd = sessionInfo?.cwd ?? '…'
  const profile = sessionInfo?.profile_name
  const usage = sessionInfo?.usage
  const version = sessionInfo?.version

  return (
    <Box flexDirection="column" flexShrink={0}>
      <Box>
        <Text color={t.color.primary} bold>
          {t.brand.icon} {t.brand.name}
        </Text>
        {version && (
          <Text color={t.color.muted}> v{version}</Text>
        )}
        {profile && (
          <Text color={t.color.muted}> · {profile}</Text>
        )}
        <Text color={t.color.muted}> · {model}</Text>
      </Box>
      <Box>
        <Text color={t.color.muted} dimColor>
          {cwd}
        </Text>
        {sid && (
          <Text color={t.color.muted} dimColor> · {sid.slice(0, 8)}</Text>
        )}
      </Box>
      {usage && (
        <Box>
          <Text color={t.color.muted} dimColor>
            {usage.calls} calls · {(usage.input + usage.output).toLocaleString()} tokens
            {usage.cost_usd ? ` · $${usage.cost_usd.toFixed(4)}` : ''}
            {usage.context_percent ? ` · ${usage.context_percent}% ctx` : ''}
          </Text>
        </Box>
      )}
    </Box>
  )
})
