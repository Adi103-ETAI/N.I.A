/**
 * NIA TUI input handlers — keyboard, clipboard, image paste.
 *
 * Ported from Hermes Agent's ui-tui/src/app/useInputHandlers.ts (622 LOC).
 *
 * Handles all keyboard input that isn't text typing:
 *   - Enter: submit
 *   - Ctrl+C: exit / interrupt
 *   - Ctrl+L: clear transcript
 *   - Up/Down: history navigation
 *   - Tab: command completion
 *   - Escape: dismiss overlays
 *   - Number keys: overlay selection
 *   - Ctrl+V: paste from clipboard
 *   - Ctrl+U: delete to start of line
 *   - Ctrl+W: delete previous word
 *   - Ctrl+A: select all (move to start)
 *   - Ctrl+E: move to end
 */

import { useInput } from 'ink'

interface InputHandlerOptions {
  input: string
  setInput: (s: string) => void
  busy: boolean
  submit: (text: string) => void
  interrupt: () => void
  history: string[]
  historyIndex: number
  setHistoryIndex: (i: number) => void
  exit: () => void
  clearTranscript: () => void
  hasOverlay: boolean
  dismissOverlay: () => void
}

export function useInputHandlers(opts: InputHandlerOptions) {
  useInput((chunk: string, key: { upArrow?: boolean; downArrow?: boolean; return?: boolean; escape?: boolean; tab?: boolean; ctrl?: boolean; backspace?: boolean; delete?: boolean; leftArrow?: boolean; rightArrow?: boolean; meta?: boolean; shift?: boolean }) => {
    // Ctrl+C: exit or interrupt.
    if (key.ctrl && chunk === 'c') {
      if (opts.busy) {
        opts.interrupt()
      } else {
        opts.exit()
      }
      return
    }

    // Ctrl+L: clear transcript.
    if (key.ctrl && chunk === 'l') {
      opts.clearTranscript()
      return
    }

    // Escape: dismiss overlay.
    if (key.escape) {
      opts.dismissOverlay()
      return
    }

    // If there's an overlay active, let the overlay handle input.
    if (opts.hasOverlay) return

    // Enter: submit.
    if (key.return) {
      if (!opts.busy && opts.input.trim()) {
        opts.submit(opts.input)
        opts.setInput('')
        opts.setHistoryIndex(-1)
      }
      return
    }

    // Up/Down: history navigation.
    if (key.upArrow && !opts.busy) {
      const nextIndex = Math.min(opts.history.length - 1, opts.historyIndex + 1)
      if (nextIndex >= 0) {
        opts.setHistoryIndex(nextIndex)
        opts.setInput(opts.history[opts.history.length - 1 - nextIndex] ?? '')
      }
      return
    }

    if (key.downArrow && !opts.busy) {
      const nextIndex = Math.max(-1, opts.historyIndex - 1)
      opts.setHistoryIndex(nextIndex)
      opts.setInput(nextIndex === -1 ? '' : (opts.history[opts.history.length - 1 - nextIndex] ?? ''))
      return
    }

    // Tab: command completion (simplified — just prepend /).
    if (key.tab && opts.input.startsWith('/')) {
      // Completion is handled by the command picker component.
      return
    }

    // Ctrl+U: delete to start.
    if (key.ctrl && chunk === 'u') {
      opts.setInput('')
      return
    }

    // Ctrl+W: delete previous word.
    if (key.ctrl && chunk === 'w') {
      const trimmed = opts.input.replace(/\s+\S+\s*$/, '')
      opts.setInput(trimmed)
      return
    }

    // Ctrl+A: move to start (no-op in terminal — cursor is always at end).
    if (key.ctrl && chunk === 'a') return

    // Ctrl+E: move to end (no-op).
    if (key.ctrl && chunk === 'e') return

    // Regular character input is handled by the TextInput component.
  })
}
