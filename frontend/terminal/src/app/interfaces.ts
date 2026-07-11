/**
 * NIA TUI App interfaces — shared contracts between app hooks.
 *
 * Ported from Hermes Agent's ui-tui/src/app/interfaces.ts (462 LOC).
 */

import type { GatewayClient } from '../gatewayClient.js'
import type { Msg, SessionInfo, TodoItem, SubagentProgress } from '../domain/types.js'
import type { Theme } from '../theme.js'

export interface SessionState {
  sid: string
  setSid: (id: string) => void
  newSession: () => void
  resumeById: (id: string) => void
  sessionInfo: SessionInfo | null
  setSessionInfo: (info: SessionInfo | null) => void
  recoverSidRef: { current: string | null }
  STARTUP_RESUME_ID: string | null
  setCatalog: (catalog: unknown) => void
}

export interface TranscriptState {
  messages: Msg[]
  appendMessage: (msg: Msg) => void
  clearMessages: () => void
  setHistoryItems: (items: unknown[]) => void
  panel: { current: unknown }
  assistantBuffer: string
  setAssistantBuffer: (s: string) => void
}

export interface ComposerState {
  input: string
  setInput: (s: string) => void
}

export interface SubmissionState {
  submitRef: { current: ((text: string) => void) | null }
}

export interface SystemState {
  bellOnComplete: () => void
  stdout: { write: (s: string) => void }
  sys: { exit: (code?: number) => void }
}

export interface VoiceState {
  voiceEnabled: boolean
  setVoiceEnabled: (b: boolean) => void
  voiceProcessing: boolean
  setVoiceProcessing: (b: boolean) => void
  voiceRecording: boolean
  setVoiceRecording: (b: boolean) => void
}

export interface GatewayState {
  gateway: { rpc: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T> }
}

export interface GatewayEventHandlerContext {
  gateway: GatewayState['gateway']
  session: SessionState
  system: SystemState
  transcript: TranscriptState
  composer: ComposerState
  submission: SubmissionState
  voice: VoiceState
}

export interface OverlayState {
  approval: { command: string; description: string; allowPermanent?: boolean } | null
  clarify: { question: string; choices: string[] | null; requestId: string } | null
  sudo: { requestId: string } | null
  secret: { envVar: string; prompt: string; requestId: string } | null
  confirm: { title: string; detail?: string; danger?: boolean; onConfirm: () => void } | null
}

export interface UiState {
  busy: boolean
  theme: Theme
  bgTasks: Set<string>
  todos: TodoItem[]
  subagents: Map<string, SubagentProgress>
}
