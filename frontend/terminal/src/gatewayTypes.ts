/**
 * NIA TUI gateway types — RPC response interfaces + the GatewayEvent union.
 *
 * Ported from Hermes Agent's ui-tui/src/gatewayTypes.ts (714 LOC).
 * Every event type the gateway can emit + every RPC response shape.
 */

import type { SessionInfo, SubagentStatus, Usage } from './domain/types.js'

export interface GatewaySkin {
  banner_hero?: string
  banner_logo?: string
  branding?: Record<string, string>
  colors?: Record<string, string>
  help_header?: string
  tool_prefix?: string
}

export interface GatewayCompletionItem {
  display: string
  meta?: string
  text: string
}

export interface GatewayTranscriptMessage {
  context?: string
  name?: string
  role: 'assistant' | 'system' | 'tool' | 'user'
  text?: string
}

export interface CommandsCatalogResponse {
  canon?: Record<string, string>
  categories?: SlashCategory[]
  pairs?: [string, string][]
  skill_count?: number
  sub?: Record<string, string[]>
  warning?: string
}

export interface CompletionResponse {
  items?: GatewayCompletionItem[]
  replace_from?: number
}

export interface SlashExecResponse {
  output?: string
  warning?: string
}

export interface ConfigDisplayConfig {
  bell_on_complete?: boolean
  busy_input_mode?: string
  details_mode?: string
  inline_diffs?: boolean
  mouse_tracking?: boolean | null | number | string
  sections?: Record<string, string>
  show_cost?: boolean
  show_reasoning?: boolean
  streaming?: boolean
  thinking_mode?: string
  tui_statusbar?: 'bottom' | 'off' | 'on' | 'top' | boolean
}

export interface ConfigFullResponse {
  config?: {
    display?: ConfigDisplayConfig
    voice?: { record_key?: unknown }
    paste_collapse_threshold?: number
  }
}

export interface ConfigSetResponse {
  confirm_message?: string
  confirm_required?: boolean
  info?: SessionInfo
  value?: string
  warning?: string
}

export interface SessionCreateResponse {
  info?: SessionInfo
  session_id: string
}

export interface SessionResumeResponse {
  info?: SessionInfo
  message_count?: number
  messages: GatewayTranscriptMessage[]
  session_id: string
  started_at?: number
}

export type LiveSessionStatus = 'idle' | 'starting' | 'waiting' | 'working'

export interface SessionActiveItem {
  current?: boolean
  id: string
  last_active?: number
  message_count?: number
  model?: string
  preview?: string
  started_at?: number
  status?: LiveSessionStatus
  title?: string
}

export interface SessionListResponse {
  sessions?: Array<{
    id: string
    message_count: number
    preview: string
    started_at: number
    title: string
  }>
}

export interface SessionUsageResponse {
  cache_read?: number
  cache_write?: number
  calls?: number
  compressions?: number
  context_max?: number
  context_percent?: number
  context_used?: number
  cost_usd?: number
  input?: number
  model?: string
  output?: number
  total?: number
}

export interface SessionStatusResponse {
  output?: string
}

export interface PromptSubmitResponse {
  ok?: boolean
}

export interface ModelOptionProvider {
  auth_type?: string
  authenticated?: boolean
  is_current?: boolean
  key_env?: string
  models?: string[]
  name: string
  slug: string
}

export interface ModelOptionsResponse {
  model?: string
  provider?: string
  providers?: ModelOptionProvider[]
}

export interface ShellExecResponse {
  code: number
  stderr?: string
  stdout?: string
}

export interface ImageAttachResponse {
  height?: number
  name?: string
  token_estimate?: number
  width?: number
}

export interface VoiceToggleResponse {
  enabled?: boolean
}

export interface ToolsListResponse {
  tools?: Array<{ name: string; description: string }>
}

export interface SubagentEventPayload {
  depth?: number
  goal?: string
  model?: string
  status?: SubagentStatus
  subagent_id?: string
  summary?: string
  task_index?: number
  text?: string
  tool_count?: number
  tool_name?: string
  parent_id?: null | string
}

export interface DelegationStatusResponse {
  active?: Array<{
    depth?: number
    goal?: string
    model?: null | string
    parent_id?: null | string
    started_at?: number
    status?: string
    subagent_id?: string
    tool_count?: number
  }>
  paused?: boolean
}

export interface SlashCategory {
  name: string
  pairs: [string, string][]
}

// ── The GatewayEvent union — every event the gateway can emit ────────

export type GatewayEvent =
  | { payload?: { skin?: GatewaySkin }; session_id?: string; type: 'gateway.ready' }
  | { payload?: GatewaySkin; session_id?: string; type: 'skin.changed' }
  | { payload: SessionInfo; session_id?: string; type: 'session.info' }
  | { payload?: { text?: string }; session_id?: string; type: 'thinking.delta' }
  | { payload?: undefined; session_id?: string; type: 'message.start' }
  | { payload?: { kind?: string; text?: string }; session_id?: string; type: 'status.update' }
  | { payload?: { line: string }; session_id?: string; type: 'gateway.stderr' }
  | {
      payload?: { cwd?: string; python?: string; stderr_tail?: string }
      session_id?: string
      type: 'gateway.start_timeout'
    }
  | { payload?: { preview?: string }; session_id?: string; type: 'gateway.protocol_error' }
  | {
      payload?: { text?: string; verbose?: boolean }
      session_id?: string
      type: 'reasoning.delta' | 'reasoning.available'
    }
  | {
      payload: { args_text?: string; name?: string; tool_id: string; todos?: unknown[] }
      session_id?: string
      type: 'tool.start'
    }
  | {
      payload: {
        duration_s?: number
        error?: string
        name?: string
        result_text?: string
        tool_id: string
        todos?: unknown[]
      }
      session_id?: string
      type: 'tool.complete'
    }
  | {
      payload: { choices: string[] | null; question: string; request_id: string }
      session_id?: string
      type: 'clarify.request'
    }
  | {
      payload: { allow_permanent?: boolean; command: string; description: string }
      session_id?: string
      type: 'approval.request'
    }
  | { payload: { request_id: string }; session_id?: string; type: 'sudo.request' }
  | { payload: { env_var: string; prompt: string; request_id: string }; session_id?: string; type: 'secret.request' }
  | { payload: { task_id: string; text: string }; session_id?: string; type: 'background.complete' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.spawn_requested' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.start' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.thinking' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.tool' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.progress' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.complete' }
  | { payload?: { rendered?: string; text?: string }; session_id?: string; type: 'message.delta' }
  | {
      payload?: { reasoning?: string; rendered?: string; text?: string; usage?: Usage }
      session_id?: string
      type: 'message.complete'
    }
  | { payload?: { message?: string }; session_id?: string; type: 'error' }
  | { payload?: { state?: 'idle' | 'listening' | 'transcribing' }; session_id?: string; type: 'voice.status' }
  | { payload?: { text?: string }; session_id?: string; type: 'voice.transcript' }
  | { payload?: { name?: string; preview?: string }; session_id?: string; type: 'tool.progress' }
  | { payload?: { name?: string }; session_id?: string; type: 'tool.generating' }
