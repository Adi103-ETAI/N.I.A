/**
 * NIA TUI gateway client — manages the Python gateway subprocess (or WebSocket
 * attach), sends JSON-RPC requests, and emits gateway events.
 *
 * Ported from Hermes Agent's ui-tui/src/gatewayClient.ts (794 LOC).
 *
 * Two transport modes:
 *   1. Spawned: spawns `python -m niaharness.tui_gateway.entry` as a child
 *      process, communicates via stdin/stdout JSON-lines.
 *   2. Attached: connects to a running gateway via WebSocket (NIA_TUI_GATEWAY_URL).
 *
 * Events are buffered until the consumer subscribes (via drain()), so events
 * that arrive during startup (gateway.ready, session.info) are not lost.
 *
 * RPC requests are tracked by ID with per-request timeouts. Pending requests
 * are rejected on transport exit/restart.
 */

import { type ChildProcess, spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { existsSync } from 'node:fs'
import { delimiter, resolve } from 'node:path'
import { createInterface } from 'node:readline'

import type { GatewayEvent } from './gatewayTypes.js'

const MAX_GATEWAY_LOG_LINES = 200
const MAX_LOG_LINE_BYTES = 4096
const MAX_BUFFERED_EVENTS = 2000
const MAX_LOG_PREVIEW = 240
const STARTUP_TIMEOUT_MS = Math.max(5000, parseInt(process.env.NIA_TUI_STARTUP_TIMEOUT_MS ?? '15000', 10) || 15000)
const REQUEST_TIMEOUT_MS = Math.max(30000, parseInt(process.env.NIA_TUI_RPC_TIMEOUT_MS ?? '120000', 10) || 120000)
const WS_CONNECTING = 0
const WS_OPEN = 1
const WS_CLOSING = 2
const WS_CLOSED = 3

const getWebSocketCtor = (): typeof WebSocket | undefined => {
  try {
    return typeof WebSocket !== 'undefined' ? WebSocket : undefined
  } catch {
    return undefined
  }
}

const truncateLine = (line: string) =>
  line.length > MAX_LOG_LINE_BYTES ? `${line.slice(0, MAX_LOG_LINE_BYTES)}… [truncated ${line.length} bytes]` : line

const describeChild = (proc: ChildProcess | null) => {
  if (!proc) return 'pid=none'
  return `pid=${proc.pid ?? 'unknown'} killed=${proc.killed} exitCode=${proc.exitCode ?? 'null'} signal=${proc.signalCode ?? 'null'}`
}

const resolveGatewayAttachUrl = () => {
  const raw = process.env.NIA_TUI_GATEWAY_URL?.trim()
  return raw ? raw : null
}

const resolvePython = (root: string) => {
  const configured = process.env.NIA_PYTHON?.trim() || process.env.PYTHON?.trim()
  if (configured) return configured
  const venv = process.env.VIRTUAL_ENV?.trim()
  const hit = [
    venv && resolve(venv, 'bin/python'),
    venv && resolve(venv, 'Scripts/python.exe'),
    resolve(root, '.venv/bin/python'),
    resolve(root, '.venv/bin/python3'),
    resolve(root, 'venv/bin/python'),
    resolve(root, 'venv/bin/python3'),
  ].find(p => p && existsSync(p))
  return hit || (process.platform === 'win32' ? 'python' : 'python3')
}

const asGatewayEvent = (value: unknown): GatewayEvent | null =>
  value && typeof value === 'object' && !Array.isArray(value) && typeof (value as { type?: unknown }).type === 'string'
    ? (value as GatewayEvent)
    : null

const redactUrl = (raw: string): string => {
  if (!raw) return raw
  try {
    const url = new URL(raw)
    const userInfo = url.username || url.password ? '***@' : ''
    const query = url.search ? '?***' : ''
    return `${url.protocol}//${userInfo}${url.host}${url.pathname}${query}`
  } catch {
    const queryIdx = raw.indexOf('?')
    return queryIdx >= 0 ? `${raw.slice(0, queryIdx)}?***` : raw
  }
}

// ── Circular buffer for logs + events ────────────────────────────────

class CircularBuffer<T> {
  private items: T[] = []
  constructor(private capacity: number) {}
  push(item: T) {
    this.items.push(item)
    if (this.items.length > this.capacity) this.items.shift()
  }
  drain(): T[] {
    const out = this.items
    this.items = []
    return out
  }
  clear() { this.items = [] }
  tail(n: number): T[] { return this.items.slice(-n) }
}

// ── Pending request tracking ─────────────────────────────────────────

interface Pending {
  id: string
  method: string
  reject: (e: Error) => void
  resolve: (v: unknown) => void
  timeout: ReturnType<typeof setTimeout>
}

// ── GatewayClient ────────────────────────────────────────────────────

export class GatewayClient extends EventEmitter {
  private proc: ChildProcess | null = null
  private ws: WebSocket | null = null
  private wsConnectPromise: Promise<void> | null = null
  private attachUrl: null | string = null
  private reqId = 0
  private logs = new CircularBuffer<string>(MAX_GATEWAY_LOG_LINES)
  private pending = new Map<string, Pending>()
  private bufferedEvents = new CircularBuffer<GatewayEvent>(MAX_BUFFERED_EVENTS)
  private pendingExit: number | null | undefined
  private ready = false
  private readyTimer: ReturnType<typeof setTimeout> | null = null
  private subscribed = false
  private drainGeneration = 0
  private stdoutRl: ReturnType<typeof createInterface> | null = null
  private stderrRl: ReturnType<typeof createInterface> | null = null

  constructor() {
    super()
    this.setMaxListeners(0)
  }

  // ── Event publishing ───────────────────────────────────────────────

  private publish(ev: GatewayEvent) {
    if (ev.type === 'gateway.ready') {
      this.ready = true
      if (this.readyTimer) { clearTimeout(this.readyTimer); this.readyTimer = null }
    }
    if (this.subscribed) {
      this.emit('event', ev)
    } else {
      this.bufferedEvents.push(ev)
    }
  }

  // ── Transport lifecycle ────────────────────────────────────────────

  private resetStartupState() {
    this.rejectPending(new Error('gateway restarting'))
    this.ready = false
    this.subscribed = false
    this.drainGeneration += 1
    this.bufferedEvents.clear()
    this.pendingExit = undefined
    this.stdoutRl?.close()
    this.stderrRl?.close()
    this.stdoutRl = null
    this.stderrRl = null
    if (this.readyTimer) { clearTimeout(this.readyTimer); this.readyTimer = null }
  }

  private startReadyTimer(python: string, cwd: string) {
    this.readyTimer = setTimeout(() => {
      if (this.ready) return
      const stderrTail = this.getLogTail(20)
      this.pushLog(`[startup] timed out waiting for gateway.ready (python=${python}, cwd=${cwd})`)
      this.publish({ type: 'gateway.start_timeout', payload: { cwd, python, stderr_tail: stderrTail } })
    }, STARTUP_TIMEOUT_MS)
  }

  private handleTransportExit(code: null | number, reason?: string) {
    if (this.readyTimer) { clearTimeout(this.readyTimer); this.readyTimer = null }
    this.pushLog(`[lifecycle] transport exit code=${code ?? 'null'} reason=${reason ?? 'none'}`)
    this.rejectPending(new Error(reason || `gateway exited${code === null ? '' : ` (${code})`}`))
    if (this.subscribed) {
      this.emit('exit', code)
    } else {
      this.pendingExit = code
    }
  }

  // ── Spawned gateway (stdio) ────────────────────────────────────────

  private startSpawnedGateway(root: string) {
    const python = resolvePython(root)
    const cwd = process.env.NIA_CWD || root
    const env = { ...process.env }
    const pyPath = env.PYTHONPATH?.trim()
    env.PYTHONPATH = pyPath ? `${root}${delimiter}${pyPath}` : root
    env.NIA_PYTHON_SRC_ROOT = root

    this.startReadyTimer(python, cwd)
    this.proc = spawn(python, ['-m', 'niaharness.tui_gateway.entry'], { cwd, env, stdio: ['pipe', 'pipe', 'pipe'] })
    this.pushLog(`[lifecycle] spawned gateway child ${describeChild(this.proc)} python=${python} cwd=${cwd}`)

    this.stdoutRl = createInterface({ input: this.proc.stdout! })
    this.stdoutRl.on('line', raw => {
      try {
        this.dispatch(JSON.parse(raw))
      } catch {
        const preview = raw.trim().slice(0, MAX_LOG_PREVIEW) || '(empty line)'
        this.pushLog(`[protocol] malformed stdout: ${preview}`)
        this.publish({ type: 'gateway.protocol_error' as any, payload: { preview } })
      }
    })

    this.stderrRl = createInterface({ input: this.proc.stderr! })
    this.stderrRl.on('line', raw => {
      const line = truncateLine(raw.trim())
      if (!line) return
      this.pushLog(line)
      this.publish({ type: 'gateway.stderr' as any, payload: { line } })
    })

    const ownedProc = this.proc
    this.proc.on('error', err => {
      if (this.proc !== ownedProc) return
      this.proc = null
      this.handleTransportExit(1, `gateway error: ${err.message}`)
    })
    this.proc.on('exit', (code, signal) => {
      if (this.proc !== ownedProc) return
      this.handleTransportExit(code)
    })
  }

  // ── Attached gateway (WebSocket) ───────────────────────────────────

  private startAttachedGateway(attachUrl: string) {
    const safeUrl = redactUrl(attachUrl)
    this.startReadyTimer('websocket', safeUrl)
    const WebSocketCtor = getWebSocketCtor()
    if (!WebSocketCtor) {
      this.handleTransportExit(1, 'gateway websocket unavailable')
      return
    }
    try {
      const ws = new WebSocketCtor(attachUrl)
      this.ws = ws

      const connectPromise = new Promise<void>((resolve, reject) => {
        ws.addEventListener('open', () => resolve(), { once: true })
        ws.addEventListener('error', () => reject(new Error('gateway websocket connection failed')), { once: true })
        ws.addEventListener('close', ev => reject(new Error(`gateway websocket closed (${ev.code}) during connect`)), { once: true })
      })
      connectPromise.catch(() => {})
      this.wsConnectPromise = connectPromise

      ws.addEventListener('message', ev => {
        const text = typeof ev.data === 'string' ? ev.data : null
        if (!text) return
        for (const line of text.splitlines()) {
          try { this.dispatch(JSON.parse(line)) } catch { /* malformed */ }
        }
      })
      ws.addEventListener('close', ev => {
        if (this.ws !== ws) return
        this.ws = null
        this.wsConnectPromise = null
        this.handleTransportExit(ev.code, `gateway websocket closed${ev.code ? ` (${ev.code})` : ''}`)
      })
      ws.addEventListener('error', () => {
        this.pushLog('[gateway] websocket transport error')
      })
    } catch {
      this.handleTransportExit(1, 'gateway websocket startup failed')
    }
  }

  // ── Public API ─────────────────────────────────────────────────────

  start() {
    const root = process.env.NIA_PYTHON_SRC_ROOT ?? resolve(import.meta.dirname, '../../')
    const attachUrl = resolveGatewayAttachUrl()
    this.attachUrl = attachUrl
    this.resetStartupState()

    if (this.proc && !this.proc.killed && this.proc.exitCode === null) {
      this.proc.kill()
    }
    this.proc = null
    if (this.ws) { try { this.ws.close() } catch {} this.ws = null }

    if (attachUrl) {
      this.startAttachedGateway(attachUrl)
    } else {
      this.startSpawnedGateway(root)
    }
  }

  drain() {
    const generation = this.drainGeneration
    queueMicrotask(() => {
      if (this.drainGeneration !== generation) return
      this.subscribed = true
      for (const ev of this.bufferedEvents.drain()) {
        this.emit('event', ev)
      }
      if (this.pendingExit !== undefined) {
        const code = this.pendingExit
        this.pendingExit = undefined
        this.emit('exit', code)
      }
    })
  }

  getLogTail(limit = 20): string {
    return this.logs.tail(Math.max(1, limit)).join('\n')
  }

  private pushLog(line: string) {
    this.logs.push(truncateLine(line))
  }

  private dispatch(msg: Record<string, unknown>) {
    const id = msg.id as string | undefined
    const p = id ? this.pending.get(id) : undefined
    if (p) {
      clearTimeout(p.timeout)
      this.pending.delete(p.id)
      if (msg.error) {
        const err = msg.error as { message?: unknown }
        p.reject(new Error(typeof err?.message === 'string' ? err.message : 'request failed'))
      } else {
        p.resolve(msg.result)
      }
      return
    }
    if (msg.method === 'event') {
      const ev = asGatewayEvent(msg.params)
      if (ev) this.publish(ev)
    }
  }

  private rejectPending(err: Error) {
    for (const p of this.pending.values()) {
      clearTimeout(p.timeout)
      p.reject(err)
    }
    this.pending.clear()
  }

  private onTimeout = (id: string) => {
    const p = this.pending.get(id)
    if (p) {
      this.pending.delete(id)
      p.reject(new Error(`timeout: ${p.method}`))
    }
  }

  request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    const attachUrl = resolveGatewayAttachUrl()

    if (attachUrl) {
      if (this.attachUrl !== attachUrl) {
        this.rejectPending(new Error('gateway attach url changed'))
        this.start()
      }
      // WebSocket path
      if (!this.ws || this.ws.readyState === WS_CLOSED || this.ws.readyState === WS_CLOSING) {
        this.start()
      }
      if (this.ws?.readyState === WS_CONNECTING) {
        return this.wsConnectPromise!.then(() => this._wsRequest<T>(method, params))
      }
      return this._wsRequest<T>(method, params)
    }

    // stdio path
    if (!this.proc?.stdin || this.proc.killed || this.proc.exitCode !== null) {
      this.start()
    }
    if (!this.proc?.stdin) {
      return Promise.reject(new Error('gateway not running'))
    }

    const id = `r${++this.reqId}`
    return new Promise<T>((resolveP, rejectP) => {
      const timeout = setTimeout(this.onTimeout, REQUEST_TIMEOUT_MS, id)
      timeout.unref?.()
      this.pending.set(id, { id, method, reject: rejectP, resolve: v => resolveP(v as T), timeout })
      try {
        this.proc!.stdin!.write(JSON.stringify({ id, jsonrpc: '2.0', method, params }) + '\n')
      } catch (e) {
        const p = this.pending.get(id)
        if (p) { clearTimeout(p.timeout); this.pending.delete(id) }
        rejectP(e instanceof Error ? e : new Error(String(e)))
      }
    })
  }

  private _wsRequest<T>(method: string, params: Record<string, unknown>): Promise<T> {
    if (!this.ws || this.ws.readyState !== WS_OPEN) {
      return Promise.reject(new Error(`gateway not connected: ${method}`))
    }
    const id = `r${++this.reqId}`
    return new Promise<T>((resolveP, rejectP) => {
      const timeout = setTimeout(this.onTimeout, REQUEST_TIMEOUT_MS, id)
      timeout.unref?.()
      this.pending.set(id, { id, method, reject: rejectP, resolve: v => resolveP(v as T), timeout })
      try {
        this.ws!.send(JSON.stringify({ id, jsonrpc: '2.0', method, params }))
      } catch (e) {
        const p = this.pending.get(id)
        if (p) { clearTimeout(p.timeout); this.pending.delete(id) }
        rejectP(e instanceof Error ? e : new Error(String(e)))
      }
    })
  }

  kill(reason = 'requested') {
    const proc = this.proc
    proc?.kill()
    this.pushLog(`[lifecycle] GatewayClient.kill reason=${reason} ${describeChild(proc)}`)
    if (this.ws) { try { this.ws.close() } catch {} this.ws = null }
    if (this.readyTimer) { clearTimeout(this.readyTimer); this.readyTimer = null }
    this.rejectPending(new Error('gateway closed'))
  }
}
