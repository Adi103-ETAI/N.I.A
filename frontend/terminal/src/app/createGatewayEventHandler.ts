/**
 * NIA TUI Gateway event handler — processes all 40+ event types from the gateway.
 *
 * Ported from Hermes Agent's ui-tui/src/app/createGatewayEventHandler.ts (970 LOC).
 *
 * This is the heart of the TUI — it receives every event from the Python
 * gateway and updates the UI state accordingly. Events include:
 *
 *   - gateway.ready / skin.changed / gateway.stderr / gateway.start_timeout
 *   - session.info / message.start / message.delta / message.complete
 *   - thinking.delta / reasoning.delta / reasoning.available
 *   - tool.start / tool.complete / tool.progress / tool.generating
 *   - approval.request / clarify.request / sudo.request / secret.request
 *   - status.update / notification.show / notification.clear
 *   - subagent.* (spawn_requested, start, thinking, tool, progress, complete)
 *   - voice.status / voice.transcript / background.complete / error
 *   - billing.step_up.verification / browser.progress / review.summary
 */

import type { GatewayEvent } from '../gatewayTypes.js'
import type { Msg, TodoItem, SubagentProgress } from '../domain/types.js'
import type { Theme } from '../theme.js'
import { fromSkin, DEFAULT_THEME } from '../theme.js'

// ── Handler context ──────────────────────────────────────────────────

export interface GatewayEventHandlerContext {
  gateway: {
    rpc: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  }
  session: {
    STARTUP_RESUME_ID: string | null
    newSession: () => void
    recoverSidRef: { current: string | null }
    resumeById: (id: string) => void
    setCatalog: (catalog: unknown) => void
  }
  system: {
    bellOnComplete: () => void
    stdout: { write: (s: string) => void }
    sys: { exit: (code?: number) => void }
  }
  transcript: {
    appendMessage: (msg: Msg) => void
    panel: { current: unknown }
    setHistoryItems: (items: unknown[]) => void
  }
  composer: {
    setInput: (s: string) => void
  }
  submission: {
    submitRef: { current: ((text: string) => void) | null }
  }
  voice: {
    setProcessing: (b: boolean) => void
    setRecording: (b: boolean) => void
    setVoiceEnabled: (b: boolean) => void
  }
}

// ── State tracking ───────────────────────────────────────────────────

interface UiState {
  busy: boolean
  theme: Theme
  bgTasks: Set<string>
}

let uiState: UiState = {
  busy: false,
  theme: DEFAULT_THEME,
  bgTasks: new Set(),
}

export function getUiState(): UiState { return uiState }
export function patchUiState(patch: Partial<UiState>) { uiState = { ...uiState, ...patch } }

interface OverlayState {
  approval: { command: string; description: string; allowPermanent?: boolean } | null
  clarify: { question: string; choices: string[] | null; requestId: string } | null
  sudo: { requestId: string } | null
  secret: { envVar: string; prompt: string; requestId: string } | null
  confirm: { title: string; detail?: string; danger?: boolean; onConfirm: () => void } | null
}

let overlayState: OverlayState = {
  approval: null, clarify: null, sudo: null, secret: null, confirm: null,
}

export function getOverlayState(): OverlayState { return overlayState }
export function patchOverlayState(patch: Partial<OverlayState>) { overlayState = { ...overlayState, ...patch } }

// ── Pending thinking ─────────────────────────────────────────────────

let pendingThinking = ''
let thinkingStatusTimer: ReturnType<typeof setTimeout> | null = null

// ── Subagent tracking ────────────────────────────────────────────────

const subagents = new Map<string, SubagentProgress>()

// ── Event handler factory ────────────────────────────────────────────

export function createGatewayEventHandler(ctx: GatewayEventHandlerContext): (ev: GatewayEvent) => void {
  const { appendMessage } = ctx.transcript
  const { setInput } = ctx.composer

  return (ev: GatewayEvent) => {
    switch (ev.type) {
      // ── Gateway lifecycle ──────────────────────────────────────────
      case 'gateway.ready': {
        const skin = (ev.payload as { skin?: Record<string, unknown> })?.skin
        if (skin) {
          patchUiState({
            theme: fromSkin(
              (skin.colors as Record<string, string>) ?? {},
              (skin.branding as Record<string, string>) ?? {},
              (skin.banner_logo as string) ?? '',
              (skin.banner_hero as string) ?? '',
              (skin.tool_prefix as string) ?? '',
              (skin.help_header as string) ?? '',
            ),
          })
        }
        break
      }

      case 'skin.changed': {
        const skin = ev.payload
        if (skin) {
          patchUiState({
            theme: fromSkin(
              (skin.colors as Record<string, string>) ?? {},
              (skin.branding as Record<string, string>) ?? {},
              (skin.banner_logo as string) ?? '',
              (skin.banner_hero as string) ?? '',
              (skin.tool_prefix as string) ?? '',
              (skin.help_header as string) ?? '',
            ),
          })
        }
        break
      }

      case 'gateway.stderr': {
        const line = (ev.payload as { line?: string })?.line
        if (line) {
          appendMessage({ role: 'system', text: line })
        }
        break
      }

      case 'gateway.start_timeout': {
        appendMessage({
          role: 'system',
          text: 'Gateway startup timed out. Check your Python environment and try again.',
        })
        break
      }

      case 'gateway.protocol_error': {
        const preview = (ev.payload as { preview?: string })?.preview
        appendMessage({ role: 'system', text: `Protocol error: ${preview ?? 'unknown'}` })
        break
      }

      // ── Session ────────────────────────────────────────────────────
      case 'session.info': {
        // Session info is handled by the main app hook.
        break
      }

      // ── Message streaming ──────────────────────────────────────────
      case 'message.start': {
        patchUiState({ busy: true })
        pendingThinking = ''
        break
      }

      case 'thinking.delta': {
        const text = (ev.payload as { text?: string })?.text
        if (text) {
          pendingThinking += text
          // Debounce status update.
          if (thinkingStatusTimer) clearTimeout(thinkingStatusTimer)
          thinkingStatusTimer = setTimeout(() => {
            thinkingStatusTimer = null
          }, 100)
        }
        break
      }

      case 'reasoning.delta':
      case 'reasoning.available': {
        const text = (ev.payload as { text?: string })?.text
        if (text) {
          pendingThinking += text
        }
        break
      }

      case 'message.delta': {
        // Delta is accumulated by the streaming assistant component.
        // The main app hook reads the buffer directly.
        break
      }

      case 'message.complete': {
        const payload = ev.payload as { text?: string; reasoning?: string; usage?: unknown }
        const text = payload?.text ?? ''
        const thinking = payload?.reasoning ?? pendingThinking
        appendMessage({
          role: 'assistant',
          text,
          thinking: thinking || undefined,
        })
        pendingThinking = ''
        patchUiState({ busy: false })
        ctx.system.bellOnComplete()
        break
      }

      // ── Tools ──────────────────────────────────────────────────────
      case 'tool.start': {
        const payload = ev.payload as { name?: string; args_text?: string; tool_id: string; todos?: unknown[] }
        const name = payload?.name ?? 'tool'
        const args = payload?.args_text ?? ''
        appendMessage({
          role: 'tool',
          text: args,
          tools: [name],
        })
        // Update todos if present.
        if (payload?.todos && Array.isArray(payload.todos)) {
          // Todos are handled by the main app hook.
        }
        break
      }

      case 'tool.complete': {
        const payload = ev.payload as { name?: string; result_text?: string; error?: string; tool_id: string; duration_s?: number }
        const name = payload?.name ?? 'tool'
        const result = payload?.result_text ?? ''
        const isError = !!payload?.error
        const duration = payload?.duration_s
        appendMessage({
          role: 'tool',
          text: isError ? `Error: ${payload?.error}` : result,
          tools: [name],
        })
        break
      }

      case 'tool.progress':
      case 'tool.generating': {
        // Progress updates — handled by the status bar.
        break
      }

      // ── Approval / clarify / sudo / secret ─────────────────────────
      case 'approval.request': {
        const payload = ev.payload as { command: string; description: string; allow_permanent?: boolean }
        patchOverlayState({
          approval: {
            command: payload.command,
            description: payload.description,
            allowPermanent: payload.allow_permanent,
          },
        })
        break
      }

      case 'clarify.request': {
        const payload = ev.payload as { question: string; choices: string[] | null; request_id: string }
        patchOverlayState({
          clarify: {
            question: payload.question,
            choices: payload.choices,
            requestId: payload.request_id,
          },
        })
        break
      }

      case 'sudo.request': {
        const payload = ev.payload as { request_id: string }
        patchOverlayState({ sudo: { requestId: payload.request_id } })
        break
      }

      case 'secret.request': {
        const payload = ev.payload as { env_var: string; prompt: string; request_id: string }
        patchOverlayState({
          secret: {
            envVar: payload.env_var,
            prompt: payload.prompt,
            requestId: payload.request_id,
          },
        })
        break
      }

      // ── Status ─────────────────────────────────────────────────────
      case 'status.update': {
        const payload = ev.payload as { kind?: string; text?: string }
        // Status is handled by the status bar.
        break
      }

      // ── Subagents ──────────────────────────────────────────────────
      case 'subagent.spawn_requested':
      case 'subagent.start':
      case 'subagent.thinking':
      case 'subagent.tool':
      case 'subagent.progress':
      case 'subagent.complete': {
        const payload = ev.payload as SubagentProgress & { subagent_id?: string }
        const id = payload.subagent_id ?? payload.id
        if (id) {
          if (ev.type === 'subagent.complete') {
            subagents.delete(id)
          } else {
            subagents.set(id, { ...subagents.get(id), ...payload } as SubagentProgress)
          }
        }
        break
      }

      // ── Background ─────────────────────────────────────────────────
      case 'background.complete': {
        const payload = ev.payload as { task_id: string; text: string }
        appendMessage({ role: 'system', text: payload.text })
        uiState.bgTasks.delete(payload.task_id)
        break
      }

      // ── Voice ──────────────────────────────────────────────────────
      case 'voice.status': {
        const payload = ev.payload as { state?: 'idle' | 'listening' | 'transcribing' }
        if (payload?.state === 'listening') {
          ctx.voice.setRecording(true)
        } else {
          ctx.voice.setRecording(false)
        }
        break
      }

      case 'voice.transcript': {
        const payload = ev.payload as { text?: string }
        if (payload?.text) {
          setInput(payload.text)
        }
        break
      }

      // ── Error ──────────────────────────────────────────────────────
      case 'error': {
        const payload = ev.payload as { message?: string }
        appendMessage({ role: 'system', text: `Error: ${payload?.message ?? 'unknown error'}` })
        patchUiState({ busy: false })
        break
      }

      // ── Billing ────────────────────────────────────────────────────
      case 'billing.step_up.verification': {
        const payload = ev.payload as { verification_url: string; user_code?: string }
        appendMessage({
          role: 'system',
          text: `Billing verification required. Visit ${payload.verification_url}` +
            (payload.user_code ? ` (code: ${payload.user_code})` : ''),
        })
        break
      }

      // ── Browser ────────────────────────────────────────────────────
      case 'browser.progress': {
        // Handled by the status bar.
        break
      }

      // ── Review ─────────────────────────────────────────────────────
      case 'review.summary': {
        const payload = ev.payload as { text?: string }
        if (payload?.text) {
          appendMessage({ role: 'assistant', text: payload.text })
        }
        break
      }

      default: {
        // Unknown event type — ignore.
        break
      }
    }
  }
}
