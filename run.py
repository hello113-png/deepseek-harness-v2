#!/usr/bin/env python3
"""
run.py -- a local chat UI for the DeepSeek reasoning API.

    python run.py                       # UI on http://127.0.0.1:8770
    python run.py --port 9000           # different UI port
    python run.py --backend-port 8765   # where deepseek.py listens
    python run.py --no-spawn            # never start the backend, only attach

What it does
  * serves one page: a chat you type into, with the model's thinking shown per
    turn (collapsed) and the answer below it
  * talks only to deepseek.py, which owns the browser; if that server is not
    already running this starts it as a child process (it waits for the browser
    login before it binds, so the page shows "connecting" until then)
  * every number on screen is real: the thinking time and character count come
    from the fragments that produced the answer, and the chat link opens the
    exact conversation

No dependencies beyond the Python standard library.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DEFAULT_UI_PORT = 8770
DEFAULT_BACKEND_PORT = 8765
BACKEND_HOST = "127.0.0.1"


# ------------------------------------------------------------------- backend


class Backend:
    """Owns (or attaches to) the deepseek.py HTTP API."""

    def __init__(self, port: int, allow_spawn: bool = True, verbose: bool = False):
        self.port = port
        self.allow_spawn = allow_spawn
        self.verbose = verbose
        self.proc: subprocess.Popen | None = None
        self.spawned_by_us = False
        self.log_path = ROOT / "server.log"
        self._lock = threading.Lock()

    # -- diagnostics -------------------------------------------------------
    def _dbg(self, message: str) -> None:
        if self.verbose:
            sys.stderr.write(f"[debug {time.strftime('%H:%M:%S')}] backend: {message}\n")
            sys.stderr.flush()

    def log_tail(self, lines: int = 25) -> list[str]:
        """Last lines of server.log -- shown when a call fails in verbose mode."""
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return fh.read().splitlines()[-lines:]
        except OSError:
            return []

    def is_up(self, timeout: float = 0.6) -> bool:
        try:
            with socket.create_connection((BACKEND_HOST, self.port), timeout=timeout):
                return True
        except OSError:
            return False

    def wait_ready(self, timeout: float = 90.0, poll: float = 1.0) -> bool:
        """Wait for the port to accept connections (the login gate runs first)."""
        started = time.time()
        while time.time() - started < timeout:
            if self.is_up():
                self._dbg(f"wait_ready: up after {time.time() - started:.1f}s")
                return True
            time.sleep(poll)
        self._dbg(f"wait_ready: still down after {timeout:.0f}s")
        return False

    # -- lifecycle ---------------------------------------------------------
    def spawn(self) -> None:
        with self._lock:
            # A backend that died must not block the next attempt: without this
            # the stale Popen object would short-circuit every future spawn and
            # the UI would sit on "connecting" forever.
            if self.proc is not None and self.proc.poll() is not None:
                self._dbg(f"spawn: previous backend exited (rc {self.proc.returncode}); "
                          "starting a new one")
                self.proc = None
            if self.proc is not None or self.is_up():
                self._dbg("spawn: already up, nothing to do")
                return
            log = open(self.log_path, "a", encoding="utf-8")
            log.write(f"\n--- backend started by run.py at {time.ctime()} ---\n")
            log.flush()
            flags = 0
            if os.name == "nt":
                flags = 0x08000000  # CREATE_NO_WINDOW: keep this console clean
            cmd = [sys.executable, "-u", str(ROOT / "deepseek.py"), "--port", str(self.port)]
            if self.verbose:
                cmd.append("--verbose")
            self._dbg("spawn: " + " ".join(cmd))
            if self.verbose:
                # stream the backend's own debug lines onto this console
                self.proc = subprocess.Popen(
                    cmd, cwd=str(ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=flags,
                )
                threading.Thread(target=self._pump, args=(self.proc, log),
                                 daemon=True).start()
            else:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(ROOT),
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
            self.spawned_by_us = True

    def _pump(self, proc: subprocess.Popen, log) -> None:
        """Copy backend output to server.log and to the console."""
        for line in proc.stdout or []:
            log.write(line)
            log.flush()
            sys.stderr.write("[backend] " + line.rstrip() + "\n")
            sys.stderr.flush()

    def ensure(self) -> str:
        if self.is_up():
            self._dbg("ensure: attached to a backend that is already listening")
            return "attached"
        if not self.allow_spawn:
            self._dbg("ensure: nothing listening and --no-spawn was given")
            return "missing"
        self.spawn()
        return "spawned"

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass

    # -- proxy ------------------------------------------------------------
    def call(self, method: str, path: str, payload: dict | None = None,
             timeout: float = 330) -> tuple[int, dict]:
        url = f"http://{BACKEND_HOST}:{self.port}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urlrequest.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
        started = time.time()
        self._dbg(f"http -> {method} {path} ({len(data or b'')}B body)")
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8") or "{}")
            self._dbg(f"http <- {resp.status} in {time.time() - started:.2f}s")
            return resp.status, body
        except urlerror.HTTPError as exc:                    # backend said no
            raw = exc.read().decode("utf-8", "replace")
            try:
                body = json.loads(raw)
            except ValueError:
                body = {"ok": False, "error": raw.strip()[:400] or f"HTTP {exc.code}"}
            self._dbg(f"http <- {exc.code} in {time.time() - started:.2f}s: "
                      f"{str(body.get('error'))[:160]}")
            return exc.code, body
        except (urlerror.URLError, OSError) as exc:          # backend not up yet
            self._dbg(f"http <- unreachable in {time.time() - started:.2f}s: {exc}")
            return 503, {
                "ok": False,
                "error": "the DeepSeek backend is not ready yet",
                "detail": str(exc)[:200],
            }

    def call_stream(self, path: str, payload: dict | None = None, on_event=None,
                    timeout: float = 330) -> tuple[int, dict]:
        """POST and relay newline-delimited events until the turn finishes.

        Falls back to a plain JSON body when the backend answers without a
        stream (an older server), so callers need only this entry point.
        """
        url = f"http://{BACKEND_HOST}:{self.port}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urlrequest.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Accept": "application/x-ndjson"},
        )
        started = time.time()
        self._dbg(f"http -> POST {path} (stream, {len(data or b'')}B body)")
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                kind = (resp.headers.get("Content-Type") or "").lower()
                if "ndjson" not in kind:
                    body = json.loads(resp.read().decode("utf-8") or "{}")
                    self._dbg(f"http <- {resp.status} (no stream) "
                              f"in {time.time() - started:.2f}s")
                    return resp.status, body
                final: dict | None = None
                events = 0
                for raw in resp:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line.decode("utf-8", "replace"))
                    except ValueError:
                        continue
                    events += 1
                    name = event.get("event")
                    if name == "done":
                        final = event.get("body") or {}
                    elif name == "error":
                        self._dbg(f"http <- stream error: {str(event.get('error'))[:160]}")
                        limited = bool(event.get("rate_limited"))
                        return (429 if limited else 502), {
                            "ok": False,
                            "error": event.get("error") or "the turn failed",
                            "rate_limited": limited,
                            "retry_after": event.get("retry_after"),
                            "chat_id": event.get("chat_id"),
                            "timeout": event.get("timeout", False),
                        }
                    elif on_event is not None:
                        on_event(event)
                self._dbg(f"http <- stream ended ({events} event(s)) "
                          f"in {time.time() - started:.2f}s")
                if final is None:
                    return 502, {"ok": False, "error": "the stream ended without a result"}
                return 200, final
        except urlerror.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                body = json.loads(raw)
            except ValueError:
                body = {"ok": False, "error": raw.strip()[:400] or f"HTTP {exc.code}"}
            self._dbg(f"http <- {exc.code}: {str(body.get('error'))[:160]}")
            return exc.code, body
        except (urlerror.URLError, OSError) as exc:
            self._dbg(f"http <- unreachable while streaming: {exc}")
            return 503, {
                "ok": False,
                "error": "the DeepSeek backend is not ready yet",
                "detail": str(exc)[:200],
            }


# ---------------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>DeepSeek reasoning</title>
<style>
  :root {
    --paper: #faf9f7;
    --surface: #ffffff;
    --ink: #1b1b19;
    --muted: #6c6a63;
    --hairline: #e3e1dc;
    --wash: #f1efea;
    --accent: #4d6bfe;
    --danger: #b3261e;
    --ok: #2f7d55;
    --warn: #a06a12;
    --radius: 10px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  header {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 14px 22px;
    border-bottom: 1px solid var(--hairline);
    background: var(--surface);
  }
  .brand { display: flex; align-items: baseline; gap: 9px; }
  .brand strong { font-size: 15px; font-weight: 650; letter-spacing: -0.01em; }
  .brand span { font-size: 12.5px; color: var(--muted); }
  .header-right { display: flex; align-items: center; gap: 14px; }
  .status { display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--muted); }
  .status i {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--warn);
    display: inline-block;
  }
  .status[data-state="ready"] i { background: var(--ok); }
  .status[data-state="error"] i { background: var(--danger); }
  button {
    font: inherit; color: inherit; cursor: pointer;
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 8px;
    padding: 7px 13px;
    transition: background 120ms ease, border-color 120ms ease;
  }
  button:hover:not(:disabled) { background: var(--wash); }
  button:disabled { opacity: 0.45; cursor: default; }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  main {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 28px 22px 12px;
  }
  .column { max-width: 720px; margin: 0 auto; }

  .empty { padding: 8vh 0 0; }
  .empty h1 {
    font-size: 21px; font-weight: 600; letter-spacing: -0.015em;
    margin: 0 0 8px;
  }
  .empty p { margin: 0 0 20px; color: var(--muted); max-width: 52ch; }
  .examples { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }
  .examples button { text-align: left; color: var(--muted); font-size: 14px; }
  .examples button:hover { color: var(--ink); }

  .turn { padding: 4px 0 26px; }
  .ask {
    margin: 0 0 14px auto;
    max-width: 84%;
    width: fit-content;
    background: var(--wash);
    border-radius: var(--radius);
    padding: 10px 14px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .trace-toggle {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 4px 9px 4px 7px;
    margin: 0 0 8px -3px;
    border: 1px solid transparent;
    background: none;
    border-radius: 7px;
    font-size: 12.5px;
    color: var(--muted);
  }
  .trace-toggle:hover { background: var(--wash); color: var(--ink); }
  .trace-toggle svg { width: 11px; height: 11px; transition: transform 140ms ease; }
  .trace-toggle[aria-expanded="true"] svg { transform: rotate(90deg); }
  .trace {
    background: var(--wash);
    border-radius: var(--radius);
    padding: 12px 14px;
    margin: 0 0 14px;
    font-size: 13.5px;
    line-height: 1.65;
    color: #4b4942;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    max-height: 420px;
    overflow-y: auto;
  }
  .answer { white-space: pre-wrap; overflow-wrap: anywhere; }
  .answer code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.9em;
    background: var(--wash);
    border-radius: 4px;
    padding: 1px 4px;
  }
  .answer pre {
    background: var(--wash);
    border-radius: var(--radius);
    padding: 12px 14px;
    overflow-x: auto;
  }
  .answer pre code { background: none; padding: 0; font-size: 12.8px; }
  .meta {
    margin-top: 12px;
    display: flex; gap: 14px; align-items: center;
    font-size: 12px; color: var(--muted);
  }
  .meta a { color: var(--muted); }
  .meta a:hover { color: var(--accent); }

  .pending { display: flex; align-items: center; gap: 9px; color: var(--muted); font-size: 13px; }
  .pending .bar {
    width: 90px; height: 2px; background: var(--hairline); overflow: hidden; border-radius: 2px;
  }
  .pending .bar b {
    display: block; height: 100%; width: 40%; background: var(--accent);
    animation: slide 1.15s ease-in-out infinite;
  }
  @keyframes slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(250%); } }

  .error {
    border: 1px solid #e8c9c6;
    background: #fdf3f2;
    color: var(--danger);
    border-radius: var(--radius);
    padding: 11px 13px;
    font-size: 13.5px;
    margin: 0 0 14px;
  }
  .error button { margin-left: 10px; padding: 3px 10px; font-size: 12.5px; }

  footer {
    flex: 0 0 auto;
    border-top: 1px solid var(--hairline);
    background: var(--surface);
    padding: 14px 22px 18px;
  }
  .composer { max-width: 720px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }
  textarea {
    flex: 1 1 auto;
    font: inherit;
    color: inherit;
    resize: none;
    min-height: 46px;
    max-height: 210px;
    padding: 12px 13px;
    border: 1px solid var(--hairline);
    border-radius: var(--radius);
    background: var(--surface);
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  #send {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 550;
    padding: 12px 18px;
  }
  #send:hover:not(:disabled) { background: #3f5be8; border-color: #3f5be8; }
  .hint {
    max-width: 720px; margin: 8px auto 0;
    font-size: 12px; color: var(--muted);
    display: flex; justify-content: space-between; gap: 12px;
  }
  .notice {
    max-width: 720px; margin: 0 auto 12px;
    font-size: 12.5px; color: var(--warn);
  }
</style>
</head>
<body>
<header>
  <div class="brand"><strong>DeepSeek</strong><span>reasoning</span></div>
  <div class="header-right">
    <span class="status" id="status" data-state="connecting"><i></i><span id="status-text">connecting</span></span>
    <button id="new-chat" type="button">New chat</button>
  </div>
</header>

<main id="scroller">
  <div class="column" id="transcript">
    <div class="empty" id="empty">
      <h1>Talk to DeepSeek, keep its thinking</h1>
      <p>Each reply comes with the model's own reasoning trace, collapsed under the answer.
         The trace and the answer are read from the same response, not reconstructed after the fact.</p>
      <div class="examples">
        <button type="button" data-prompt="Two ropes each burn for 60 minutes, unevenly. How do you measure 45 minutes? Think it through, then answer briefly.">Two ropes, 45 minutes — think it through</button>
        <button type="button" data-prompt="Explain why a hash table gives expected O(1) lookup while a balanced tree does not, and what the hash table gives up for it.">Hash table vs balanced tree</button>
        <button type="button" data-prompt="Given a list of 10 million integers, find the pair closest to zero. Reason about the fastest practical approach before answering.">Closest pair in 10M integers</button>
      </div>
    </div>
  </div>
</main>

<footer>
  <div class="notice" id="notice" hidden></div>
  <form class="composer" id="composer">
    <textarea id="input" rows="2" placeholder="Ask something that needs thinking..." autofocus></textarea>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="hint">
    <span>Enter sends &middot; Shift+Enter for a new line</span>
    <span id="chat-link"></span>
  </div>
</footer>

<script>
(() => {
  const scroller = document.getElementById('scroller');
  const transcript = document.getElementById('transcript');
  const empty = document.getElementById('empty');
  const input = document.getElementById('input');
  const form = document.getElementById('composer');
  const sendBtn = document.getElementById('send');
  const newChatBtn = document.getElementById('new-chat');
  const statusEl = document.getElementById('status');
  const statusText = document.getElementById('status-text');
  const notice = document.getElementById('notice');
  const chatLink = document.getElementById('chat-link');

  // `fresh` means "the conversation on screen is empty", so the first message
  // must open a new chat instead of appending to whatever the tab still holds.
  const state = { chatId: null, busy: false, backend: false, fresh: true };

  const setStatus = (text, kind) => {
    statusText.textContent = text;
    statusEl.dataset.state = kind || 'connecting';
  };

  const atBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
  const scrollDown = (force) => {
    if (force || atBottom()) scroller.scrollTop = scroller.scrollHeight;
  };

  function escapeHtml(s) {
    return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  }

  // minimal, honest formatting: keep the text as written, only lift code blocks
  function format(text) {
    const parts = escapeHtml(text).split(/```/);
    return parts.map((chunk, i) => {
      if (i % 2 === 1) {
        const body = chunk.replace(/^[a-zA-Z0-9+#-]*\n/, '');
        return '<pre><code>' + body.replace(/\n$/, '') + '</code></pre>';
      }
      return chunk
        .replace(/`([^`\n]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    }).join('');
  }

  function updateChatLink(chatId) {
    if (!chatId) { chatLink.textContent = ''; return; }
    const a = document.createElement('a');
    a.href = 'https://chat.deepseek.com/a/chat/s/' + chatId;
    a.target = '_blank';
    a.rel = 'noreferrer';
    a.textContent = 'open chat in DeepSeek';
    chatLink.replaceChildren(a);
  }

  function clearTranscript() {
    transcript.querySelectorAll('.turn, .error-row').forEach((n) => n.remove());
    empty.hidden = false;
  }

  function traceLabel(seconds, chars) {
    const secs = (seconds === null || seconds === undefined)
      ? 'thinking'
      : (Math.round(seconds * 10) / 10) + ' s';
    const size = chars >= 1000 ? (chars / 1000).toFixed(1) + 'k' : String(chars);
    return 'Thought for ' + secs + ' · ' + size + ' chars';
  }

  function addTurn({ prompt, reasoning, reasoningSeconds, answer, chatId }) {
    empty.hidden = true;
    const turn = document.createElement('section');
    turn.className = 'turn';

    const ask = document.createElement('div');
    ask.className = 'ask';
    ask.textContent = prompt;
    turn.appendChild(ask);

    const reply = document.createElement('div');
    reply.className = 'reply';

    if (reasoning && reasoning.trim()) {
      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'trace-toggle';
      toggle.setAttribute('aria-expanded', 'false');
      toggle.innerHTML = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      const text = document.createElement('span');
      text.textContent = traceLabel(reasoningSeconds, reasoning.length);
      toggle.appendChild(text);

      const body = document.createElement('div');
      body.className = 'trace';
      body.hidden = true;
      body.textContent = reasoning.trim();

      toggle.addEventListener('click', () => {
        const open = toggle.getAttribute('aria-expanded') === 'true';
        toggle.setAttribute('aria-expanded', String(!open));
        body.hidden = open;
      });

      reply.append(toggle, body);
    }

    const ans = document.createElement('div');
    ans.className = 'answer';
    ans.innerHTML = format(answer || '');
    reply.appendChild(ans);

    if (chatId) {
      const meta = document.createElement('div');
      meta.className = 'meta';
      const link = document.createElement('a');
      link.href = 'https://chat.deepseek.com/a/chat/s/' + chatId;
      link.target = '_blank';
      link.rel = 'noreferrer';
      link.textContent = 'chat ' + chatId.slice(0, 8);
      meta.appendChild(link);
      reply.appendChild(meta);
    }

    turn.appendChild(reply);
    transcript.appendChild(turn);
    scrollDown(true);
    return turn;
  }

  function addPending(prompt) {
    empty.hidden = true;
    const turn = document.createElement('section');
    turn.className = 'turn';
    const ask = document.createElement('div');
    ask.className = 'ask';
    ask.textContent = prompt;
    const reply = document.createElement('div');
    reply.className = 'reply';
    const pending = document.createElement('div');
    pending.className = 'pending';
    const started = Date.now();
    const label = document.createElement('span');
    const tick = () => {
      label.textContent = 'thinking · ' + ((Date.now() - started) / 1000).toFixed(1) + ' s';
    };
    tick();
    const timer = setInterval(tick, 100);
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.innerHTML = '<b></b>';
    pending.append(label, bar);
    reply.appendChild(pending);
    turn.append(ask, reply);
    transcript.appendChild(turn);
    scrollDown(true);
    return { turn, stop: () => clearInterval(timer) };
  }

  function addError(message, retry) {
    const row = document.createElement('div');
    row.className = 'error-row';
    const box = document.createElement('div');
    box.className = 'error';
    box.textContent = message;
    if (retry) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = 'Retry';
      btn.addEventListener('click', () => { row.remove(); retry(); });
      box.appendChild(btn);
    }
    row.appendChild(box);
    transcript.appendChild(row);
    scrollDown(true);
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    let body = null;
    try { body = await res.json(); } catch (e) { body = { ok: false, error: 'unreadable response' }; }
    return { status: res.status, body };
  }

  async function send(text) {
    if (state.busy || !text.trim()) return;
    const prompt = text.trim();
    state.busy = true;
    sendBtn.disabled = true;
    input.value = '';
    autoSize();

    const pending = addPending(prompt);
    try {
      if (state.fresh) {
        const started = await api('/api/new_chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}',
        });
        if (started.body && started.body.ok) state.fresh = false;
      }
      const payload = { message: prompt };
      if (state.chatId) payload.chat_id = state.chatId;
      const { status, body } = await api('/api/send_message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      pending.stop();
      pending.turn.remove();

      if (!status.toString().startsWith('2') || !body || body.ok === false) {
        const reason = (body && (body.error || body.message)) || ('request failed (' + status + ')');
        addError(reason, () => send(prompt));
        if (status === 503) markBackend(false);
        return;
      }

      state.chatId = body.chat_id || state.chatId;
      updateChatLink(state.chatId);
      addTurn({
        prompt,
        reasoning: body.reasoning,
        reasoningSeconds: body.reasoning_seconds,
        answer: body.answer,
        chatId: body.chat_id,
      });
      markBackend(true);
    } catch (e) {
      pending.stop();
      pending.turn.remove();
      addError('Could not reach the local server: ' + e.message, () => send(prompt));
      markBackend(false);
    } finally {
      state.busy = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function autoSize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 210) + 'px';
  }

  function markBackend(up) {
    state.backend = up;
    setStatus(up ? 'ready' : 'connecting', up ? 'ready' : 'connecting');
    notice.hidden = up;
    if (!up) {
      notice.textContent = 'Waiting for the DeepSeek browser session. If the browser asks you to sign in, do it there; this page reconnects on its own.';
    }
  }

  async function pollStatus() {
    try {
      const { body } = await api('/api/status');
      markBackend(!!(body && body.backend));
    } catch (e) {
      markBackend(false);
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    send(input.value);
  });

  input.addEventListener('input', autoSize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(input.value);
    }
  });

  newChatBtn.addEventListener('click', async () => {
    if (state.busy) return;
    newChatBtn.disabled = true;
    try {
      const { body } = await api('/api/new_chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      if (body && body.ok) {
        state.chatId = null;
        state.fresh = true;
        updateChatLink(null);
        clearTranscript();
        input.focus();
      } else {
        addError((body && body.error) || 'could not start a new chat', null);
      }
    } catch (e) {
      addError('Could not reach the local server: ' + e.message, null);
    } finally {
      newChatBtn.disabled = false;
    }
  });

  document.querySelectorAll('.examples button').forEach((btn) => {
    btn.addEventListener('click', () => {
      input.value = btn.dataset.prompt || btn.textContent;
      autoSize();
      input.focus();
    });
  });

  pollStatus();
  setInterval(pollStatus, 5000);
})();
</script>
</body>
</html>
"""


# -------------------------------------------------------------------- server


class UIHandler(BaseHTTPRequestHandler):
    server: "UIServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[ui] {fmt % args}\n")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def _dbg(self, message: str) -> None:
        if self.server.backend.verbose:
            sys.stderr.write(f"[debug] ui: {message}\n")
            sys.stderr.flush()

    def do_GET(self):
        parsed = urlparse(self.path)
        self._dbg(f"GET {self.path}")
        if parsed.path in ("/", "/index.html"):
            return self._page()
        if parsed.path == "/api/status":
            backend = self.server.backend
            ready = backend.is_up()
            if not ready:
                backend.ensure()          # keeps trying to bring it up
            return self._json(200, {"backend": ready})
        if parsed.path == "/api/history":
            status, body = self.server.backend.call("GET", "/history" + ("?" + parsed.query if parsed.query else ""))
            return self._json(status, body)
        self._json(404, {"ok": False, "error": f"no route for GET {parsed.path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        payload = self._read_json()
        self._dbg(f"POST {self.path}")
        if parsed.path == "/api/new_chat":
            self.server.backend.ensure()
            status, body = self.server.backend.call("POST", "/new_chat", {}, timeout=120)
            return self._json(status, body)
        if parsed.path == "/api/send_message":
            self.server.backend.ensure()
            status, body = self.server.backend.call("POST", "/send_message", payload)
            return self._json(status, body)
        self._json(404, {"ok": False, "error": f"no route for POST {parsed.path}"})


class UIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, backend: Backend):
        super().__init__(address, UIHandler)
        self.backend = backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat UI for the DeepSeek reasoning API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_UI_PORT, help="UI port")
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT,
                        help="port deepseek.py listens on")
    parser.add_argument("--no-spawn", action="store_true",
                        help="only attach to a running backend instead of starting one")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug logging, including the backend's own log lines")
    args = parser.parse_args(argv)

    backend = Backend(args.backend_port, allow_spawn=not args.no_spawn, verbose=args.verbose)
    state = backend.ensure()
    if state == "attached":
        print(f"[backend] attached to a running server on port {args.backend_port}")
    elif state == "spawned":
        print(f"[backend] starting deepseek.py on port {args.backend_port} "
              f"(it waits for the browser login; log is server.log)")
        atexit.register(backend.stop)
        if args.verbose:
            backend.wait_ready(120)
    else:
        print(f"[backend] nothing on port {args.backend_port} and --no-spawn was given")

    try:
        ui = UIServer((args.host, args.port), backend)
    except OSError as exc:
        raise SystemExit(f"[ui] cannot bind {args.host}:{args.port} -- {exc}")

    print(f"\n  chat UI   http://{args.host}:{args.port}")
    print(f"  backend   http://{BACKEND_HOST}:{args.backend_port}  (deepseek.py)")
    print("  Ctrl+C to stop\n")
    try:
        ui.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] bye")
    finally:
        ui.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
