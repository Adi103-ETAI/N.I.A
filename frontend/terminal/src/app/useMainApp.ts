/**
 * NIA TUI main app hook — orchestrates the entire TUI lifecycle.
 *
 * Ported from Hermes Agent's ui-tui/src/app/useMainApp.ts (1,145 LOC).
 *
 * This hook ties together:
 *   - GatewayClient (spawn/attach, RPC, events)
 *   - createGatewayEventHandler (event → state updates)
 *   - turnController (turn lifecycle)
 *   - Session management (create, resume, list)
 *   - Transcript (messages, streaming buffer)
 *   - Composer (input state)
 *   - Overlays (approval, clarify, sudo, secret, confirm)
 *   - Voice mode
 *   - Slash command catalog
 *   - Theme/skin
 *   - Startup flow (auto-resume, initial prompt)
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { GatewayClient } from '../gatewayClient.js'
import { createGatewayEventHandler, getUiState, patchUiState, getOverlayState, patchOverlayState } from './createGatewayEventHandler.js'
import { startTurn, endTurn, onStreamingStart, onToolStart, onToolComplete, requestInterrupt, queueSteer, getTurnState, subscribeTurn } from './turnController.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg, SessionInfo } from '../domain/types.js'
import type { GatewayEvent } from '../gatewayTypes.js'

export function useMainApp() {
  const [messages, setMessages] = useState<Msg[]>([])
  const [assistantBuffer, setAssistantBuffer] = useState('')
  const [sid, setSid] = useState('')
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [theme, setTheme] = useState(DEFAULT_THEME)
  const [commands, setCommands] = useState<string[]>([])
  const [overlay, setOverlay] = useState(getOverlayState())
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [voiceRecording, setVoiceRecording] = useState(false)
  const [voiceProcessing, setVoiceProcessing] = useState(false)

  const gatewayRef = useRef<GatewayClient | null>(null)
  const submitRef = useRef<((text: string) => void) | null>(null)
  const recoverSidRef = useRef<string | null>(null)
  const STARTUP_RESUME_ID = useRef<string | null>(null)

  // ── Gateway client setup ───────────────────────────────────────────

  const gateway = useMemo(() => {
    const gw = new GatewayClient()
    gatewayRef.current = gw
    return gw
  }, [])

  // ── Transcript helpers ─────────────────────────────────────────────

  const appendMessage = useCallback((msg: Msg) => {
    setMessages(prev => [...prev, msg])
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    setAssistantBuffer('')
  }, [])

  // ── Event handler ──────────────────────────────────────────────────

  const eventHandler = useMemo(() => {
    return createGatewayEventHandler({
      gateway: { rpc: (method, params) => gateway.request(method, params) },
      session: {
        STARTUP_RESUME_ID: STARTUP_RESUME_ID.current,
        newSession: () => {
          clearMessages()
          gateway.request('session.create', {}).then((resp: any) => {
            if (resp?.session_id) setSid(resp.session_id)
          }).catch(() => {})
        },
        recoverSidRef,
        resumeById: (id: string) => {
          gateway.request('session.resume', { session_id: id }).then((resp: any) => {
            if (resp?.session_id) {
              setSid(resp.session_id)
              if (resp.messages) {
                setMessages(resp.messages.map((m: any) => ({
                  role: m.role ?? 'system',
                  text: m.text ?? '',
                })))
              }
            }
          }).catch(() => {})
        },
        setCatalog: (catalog: any) => {
          if (catalog?.pairs) {
            setCommands(catalog.pairs.map((p: [string, string]) => p[0]))
          }
        },
      },
      system: {
        bellOnComplete: () => { process.stdout.write('\x07') },
        stdout: process.stdout,
        sys: { exit: (code?: number) => process.exit(code) },
      },
      transcript: {
        appendMessage,
        panel: { current: null },
        setHistoryItems: () => {},
      },
      composer: { setInput },
      submission: { submitRef },
      voice: {
        setProcessing,
        setRecording: setVoiceRecording,
        setVoiceEnabled,
      },
    })
  }, [gateway, appendMessage, clearMessages])

  // ── Gateway lifecycle ──────────────────────────────────────────────

  useEffect(() => {
    gateway.start()
    gateway.on('event', (ev: GatewayEvent) => {
      // Handle streaming deltas directly (they're too high-frequency for
      // the event handler's appendMessage pattern).
      if (ev.type === 'message.delta') {
        const text = (ev.payload as any)?.text ?? ''
        setAssistantBuffer(prev => prev + text)
        onStreamingStart()
        return
      }

      // Handle session.info to update session info + commands.
      if (ev.type === 'session.info') {
        setSessionInfo(ev.payload as SessionInfo)
        return
      }

      // Handle tool.start/complete for turn tracking.
      if (ev.type === 'tool.start') {
        const payload = ev.payload as any
        onToolStart({
          id: payload.tool_id,
          name: payload.name ?? 'tool',
          startedAt: Date.now(),
        })
        return
      }

      if (ev.type === 'tool.complete') {
        onToolComplete()
        return
      }

      // Handle message.complete to end the turn.
      if (ev.type === 'message.complete') {
        const payload = ev.payload as any
        const text = payload?.text ?? ''
        const thinking = payload?.reasoning
        setMessages(prev => [...prev, {
          role: 'assistant',
          text,
          thinking: thinking || undefined,
        }])
        setAssistantBuffer('')
        endTurn()
        return
      }

      // Delegate everything else to the event handler.
      eventHandler(ev)

      // Sync UI state from the event handler.
      const ui = getUiState()
      setBusy(ui.busy)
      setTheme(ui.theme)
      setOverlay(getOverlayState())
    })

    gateway.on('exit', (code: number | null) => {
      setMessages(prev => [...prev, {
        role: 'system',
        text: `Gateway exited (code ${code ?? 0})`,
      }])
    })

    gateway.drain()

    // Fetch command catalog.
    gateway.request('commands.catalog', {}).then((catalog: any) => {
      if (catalog?.pairs) {
        setCommands(catalog.pairs.map((p: [string, string]) => p[0]))
      }
    }).catch(() => {})

    return () => {
      gateway.kill('cleanup')
    }
  }, [gateway, eventHandler])

  // ── Submit ─────────────────────────────────────────────────────────

  const submit = useCallback((text: string) => {
    if (!text.trim() || busy) return

    // Add user message to transcript.
    setMessages(prev => [...prev, { role: 'user', text }])

    // Start the turn.
    startTurn()
    setBusy(true)
    setAssistantBuffer('')

    // Submit to gateway.
    gateway.request('prompt.submit', { session_id: sid, text }).catch((err: Error) => {
      setMessages(prev => [...prev, { role: 'system', text: `Error: ${err.message}` }])
      endTurn()
      setBusy(false)
    })
  }, [busy, sid, gateway])

  submitRef.current = submit

  // ── Interrupt ──────────────────────────────────────────────────────

  const interrupt = useCallback(() => {
    if (!requestInterrupt()) return
    gateway.request('session.interrupt', { session_id: sid }).catch(() => {})
  }, [sid, gateway])

  // ── Return ─────────────────────────────────────────────────────────

  return {
    messages,
    assistantBuffer,
    sid,
    sessionInfo,
    input,
    setInput,
    busy,
    theme,
    commands,
    overlay,
    voiceEnabled,
    voiceRecording,
    voiceProcessing,
    submit,
    interrupt,
    gateway,
    newSession: () => {
      clearMessages()
      gateway.request('session.create', {}).then((resp: any) => {
        if (resp?.session_id) setSid(resp.session_id)
      }).catch(() => {})
    },
  }
}
