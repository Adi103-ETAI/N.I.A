/**
 * NIA TUI Turn controller — manages the agent turn lifecycle.
 *
 * Ported from Hermes Agent's ui-tui/src/app/turnController.ts (1,050 LOC).
 *
 * The turn controller tracks the state of the current agent turn:
 *   - idle: no turn in progress
 *   - submitting: user submitted, waiting for gateway to acknowledge
 *   - streaming: gateway acknowledged, streaming tokens
 *   - tool_running: a tool is executing
 *   - interrupting: user requested interrupt, waiting for gateway
 *   - done: turn completed (transitions back to idle)
 *
 * It also manages:
 *   - Steer queue (user can queue a steer message mid-turn)
 *   - Interrupt state (user can interrupt mid-turn)
 *   - Queued messages (messages submitted while busy are queued)
 *   - Tool call tracking (count, active tool name)
 *   - Turn duration timing
 */

import type { ActiveTool } from '../domain/types.js'

export type TurnState = 'idle' | 'submitting' | 'streaming' | 'tool_running' | 'interrupting' | 'done'

interface QueuedMessage {
  text: string
}

interface TurnControllerState {
  state: TurnState
  toolCallCount: number
  activeTool: ActiveTool | undefined
  startedAt: number | undefined
  steerQueue: string[]
  queuedMessages: QueuedMessage[]
  interrupted: boolean
}

let state: TurnControllerState = {
  state: 'idle',
  toolCallCount: 0,
  activeTool: undefined,
  startedAt: undefined,
  steerQueue: [],
  queuedMessages: [],
  interrupted: false,
}

const listeners = new Set<() => void>()

function notify() {
  for (const fn of listeners) fn()
}

export function getTurnState(): TurnControllerState {
  return state
}

export function patchTurnState(patch: Partial<TurnControllerState>) {
  state = { ...state, ...patch }
  notify()
}

export function subscribeTurn(fn: () => void): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

// ── Turn lifecycle ───────────────────────────────────────────────────

export function startTurn() {
  patchTurnState({
    state: 'submitting',
    toolCallCount: 0,
    activeTool: undefined,
    startedAt: Date.now(),
    steerQueue: [],
    interrupted: false,
  })
}

export function onStreamingStart() {
  patchTurnState({ state: 'streaming' })
}

export function onToolStart(tool: ActiveTool) {
  patchTurnState({
    state: 'tool_running',
    activeTool: tool,
    toolCallCount: state.toolCallCount + 1,
  })
}

export function onToolComplete() {
  patchTurnState({
    state: 'streaming',
    activeTool: undefined,
  })
}

export function endTurn() {
  patchTurnState({
    state: 'idle',
    activeTool: undefined,
    startedAt: undefined,
  })

  // Submit any queued messages.
  if (state.queuedMessages.length > 0) {
    const next = state.queuedMessages.shift()!
    // The caller (useMainApp) picks this up.
    patchTurnState({ queuedMessages: state.queuedMessages })
    // Return the queued text for the caller to submit.
    return next.text
  }
  return null
}

// ── Interrupt ────────────────────────────────────────────────────────

export function requestInterrupt() {
  if (state.state === 'idle' || state.state === 'done') return false
  patchTurnState({ state: 'interrupting', interrupted: true })
  return true
}

export function clearInterrupt() {
  patchTurnState({ interrupted: false })
}

// ── Steer ────────────────────────────────────────────────────────────

export function queueSteer(text: string) {
  if (state.state === 'idle' || state.state === 'done') return false
  patchTurnState({ steerQueue: [...state.steerQueue, text] })
  return true
}

export function drainSteer(): string[] {
  const queue = state.steerQueue
  patchTurnState({ steerQueue: [] })
  return queue
}

// ── Queue ────────────────────────────────────────────────────────────

export function queueMessage(text: string) {
  patchTurnState({ queuedMessages: [...state.queuedMessages, { text }] })
}

export function getQueuedCount(): number {
  return state.queuedMessages.length
}
