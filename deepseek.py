#!/usr/bin/env python3
"""
deepseek.py -- a localhost HTTP API around chat.deepseek.com, driven through
Playwriter (your already-signed-in browser), exposing DeepSeek's *reasoning*.

Why Playwriter and not a bundled browser: Selenium/Chromium cannot attach to a
tab owned by the Playwriter extension, and a fresh automation profile is signed
out (DeepSeek binds the session to a bearer token in localStorage, not just a
cookie). So the browser is *yours*, the login is done once by you, and Python
only orchestrates.

---------------------------------------------------------------------------
TRANSPORT: WHY DATA NEVER GOES THROUGH THE COMMAND LINE

Windows caps a whole command line at 32767 characters, and CreateProcess
reports the overflow as WinError 206 ("The filename or extension is too long"),
which CPython raises as FileNotFoundError -- indistinguishable, if you only look
at the exception type, from "node.exe does not exist". A 27 KB file read sent
back to the model is enough to cross that line, so:

  * the JavaScript is passed with -e (fixed size, ~3 KB),
  * the *payload* (message, chat id, budget) is written to a temp JSON file whose
    path is injected into the script (@@PAYLOAD_FILE@@),
  * the *result* comes back in a temp JSON file written by the page (@@OUT@@).

The command line therefore stays constant no matter how large a file the model
asks for. --verbose prints every exec with its argv size and exit code.

---------------------------------------------------------------------------
WHAT ACTUALLY CARRIES THE REASONING (reverse-engineered)

DeepSeek's "Deep thinking" output is not a separate model and not a page-only
artefact: a single completion response is split into typed *fragments*.

  POST https://chat.deepseek.com/api/v0/chat/completion        (SSE)
      event: ready
      data: {"request_message_id":3,"response_message_id":4,"model_type":"default"}
      data: {"v":{"response":{"message_id":4,"thinking_enabled":true,"status":"WIP",
               "fragments":[{"id":2,"type":"THINK","content":"We", ...}]}}}
                                                   ^^^^^ reasoning
      data: {"p":"response/fragments/-1/content","v":" need"}       # token append
      data: {"p":"response/fragments/-1/elapsed_secs","o":"SET","v":6.62}
      data: {"p":"response/fragments","o":"APPEND",
             "v":[{"id":3,"type":"RESPONSE","content":"With", ...}]}  # answer starts
      data: {"p":"response","o":"BATCH","v":[{"p":"quasi_status","v":"FINISHED"}]}
      data: {"p":"response/status","o":"SET","v":"FINISHED"}
      event: close

  Fragment types seen: THINK (reasoning), RESPONSE (answer), REQUEST (the user
  prompt), SEARCH (web references). Concatenate THINK fragments -> reasoning.

  The same fragments are served, already assembled, by

  GET https://chat.deepseek.com/api/v0/chat/history_messages?chat_session_id=<id>
      Authorization: Bearer <localStorage.userToken .value>

  which is what /history uses and what /send_message polls -- no scrolling, no
  virtualized list, and the assistant message's `status` turns into "FINISHED"
  exactly when the turn is done.

  DOM fallback (semantic class names, present even while the reasoning is
  collapsed):  .ds-message > .ds-think-content            reasoning
               .ds-message > .ds-markdown.ds-assistant-message-main-content
               label: "Thinking..." -> "Thought for N seconds"

LOGIN
  localStorage "userToken" = {"value": "<bearer>"}  + httpOnly cookie ds_session_id.
  There is no credential login here on purpose: at startup the server waits for
  you to sign in inside the browser, then it never needs credentials again.

API
  POST /send_message   {"message": "...", "chat_id": "<optional>"}
                       -> {"reasoning", "reasoning_seconds", "answer", ...}
  POST /new_chat       -> starts a fresh conversation
  GET  /history?q=<chat_id>  -> all turns with their reasoning

  --verbose (or --debug) prints resolution, every exec, payload sizes and timings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 8765
DEFAULT_WAIT = 240
WINDOWS_ARGV_LIMIT = 32767      # CreateProcess lpCommandLine cap, in characters


# ----------------------------------------------------------------------- debug

VERBOSE = False


def dbg(message: str) -> None:
    """One-line diagnostic, only with --verbose. Goes to stderr, never stdout."""
    if VERBOSE:
        sys.stderr.write(f"[debug {time.strftime('%H:%M:%S')}] {message}\n")
        sys.stderr.flush()


def short(text: str | None, limit: int = 200) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[:limit] + "…"


# --------------------------------------------------------------------- browser


class PlaywriterError(RuntimeError):
    """The browser side failed (tab gone, JS threw, CLI timed out...)."""


def node_search_paths() -> list[str]:
    """Directories checked for a Node binary, used in error messages."""
    paths: list[Path] = []
    if os.name == "nt":
        for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA", "ProgramData"):
            base = os.environ.get(var)
            if base:
                paths += [Path(base) / "nodejs", Path(base) / "Programs" / "nodejs"]
        paths += [Path("C:/Program Files/nodejs"), Path("C:/Program Files (x86)/nodejs")]
    else:
        paths += [Path("/usr/local/bin"), Path("/usr/bin"), Path("/opt/homebrew/bin")]
    return [str(p) for p in paths]


def find_node() -> str | None:
    """Absolute path to Node, without trusting PATH to contain it."""
    exe = shutil.which("node")
    dbg(f"find_node: shutil.which('node') -> {exe!r}")
    if exe:
        return exe
    names = ("node.exe",) if os.name == "nt" else ("node",)
    for directory in node_search_paths():
        for name in names:
            candidate = Path(directory) / name
            exists = candidate.is_file()
            dbg(f"find_node: {candidate} -> {'yes' if exists else 'no'}")
            if exists:
                return str(candidate)
    return None


def find_playwriter_cli() -> str | None:
    """Locate playwriter's cli.js in an npx cache or a global npm install."""
    roots: list[Path] = []
    if os.environ.get("LOCALAPPDATA"):
        roots.append(Path(os.environ["LOCALAPPDATA"]) / "npm-cache" / "_npx")
    roots += [
        Path.home() / ".npm" / "_npx",
        Path.home() / "AppData" / "Roaming" / "npm" / "node_modules",
        Path.home() / ".npm-global" / "lib" / "node_modules",
        Path("/usr/local/lib/node_modules"),
        Path("/usr/lib/node_modules"),
        Path(__file__).resolve().parent / "node_modules",
    ]
    for root in roots:
        if not root.is_dir():
            dbg(f"find_cli: {root} -> not a directory")
            continue
        direct = root / "playwriter" / "dist" / "cli.js"
        if direct.is_file():
            dbg(f"find_cli: {direct} -> yes (direct)")
            return str(direct)
        for hit in root.glob("*/node_modules/playwriter/dist/cli.js"):
            if hit.is_file():
                dbg(f"find_cli: {hit} -> yes")
                return str(hit)
        dbg(f"find_cli: {root} -> no playwriter/dist/cli.js")
    return None


def js_path(path: str) -> str:
    """Path form that is safe inside a JS string on both platforms."""
    return path.replace("\\", "/")


class Playwriter:
    """Thin, synchronous bridge: one `playwriter -e` call per operation."""

    def __init__(self, session: str | None = None, tab_group: str = "deepseek"):
        self.session = session
        self.tab_group = tab_group
        self._base_cmd: list[str] | None = None

    # -- CLI discovery -----------------------------------------------------
    #
    # Node is deliberately located on disk rather than through PATH: this server
    # may be started from a shell that never saw Node's installer (or by a
    # launcher with a minimal environment), and on Windows a bare "npx" cannot
    # be executed at all -- CreateProcess does not apply PATHEXT, so the real
    # target is npx.cmd and the call dies with FileNotFoundError.
    def _resolve(self) -> list[str]:
        if self._base_cmd:
            return self._base_cmd

        override = os.environ.get("PLAYWRITER_BIN")
        if override:
            path = Path(override).expanduser()
            dbg(f"resolve: PLAYWRITER_BIN={override!r} exists={path.exists()}")
            if not path.exists():
                raise PlaywriterError(f"PLAYWRITER_BIN points at a missing file: {override}")
            # a .js path needs Node in front of it; an executable is used as-is
            self._base_cmd = ([find_node() or "node", str(path)]
                              if path.suffix.lower() == ".js" else [str(path)])
            dbg(f"resolve: {self._base_cmd}")
            return self._base_cmd

        node, cli = find_node(), find_playwriter_cli()
        if node and cli:
            self._base_cmd = [node, cli]
            dbg(f"resolve: node + cli.js -> {node}")
            return self._base_cmd

        exe = shutil.which("playwriter")
        dbg(f"resolve: shutil.which('playwriter') -> {exe!r}")
        if exe:
            self._base_cmd = [exe]
            return self._base_cmd

        npx = shutil.which("npx")
        dbg(f"resolve: shutil.which('npx') -> {npx!r}")
        if npx:
            self._base_cmd = [npx, "--yes", "playwriter@latest"]
            return self._base_cmd

        raise PlaywriterError(
            "cannot run Playwriter: Node.js was not found.\n"
            f"  node on PATH: no\n"
            f"  checked: {', '.join(node_search_paths()) or '(nothing)'}\n"
            "  fix: install Node.js, add it to PATH, or set PLAYWRITER_BIN to \n"
            "       ...\\playwriter\\dist\\cli.js"
        )

    def describe(self) -> str:
        """The concrete command used, for startup logs and bug reports."""
        cmd = self._resolve()
        return " ".join(cmd[:2]) + (f" {' '.join(cmd[2:])}" if len(cmd) > 2 else "")

    # -- execution ---------------------------------------------------------
    @staticmethod
    def argv_size(cmd: list[str]) -> int:
        """Characters the OS will see, including separators."""
        return sum(len(part) + 1 for part in cmd)

    def _run(self, argv: list[str], env: dict | None = None, timeout: float = 120):
        cmd = [*self._resolve(), *argv]
        total = self.argv_size(cmd)
        dbg(f"exec: {Path(cmd[0]).name} {len(cmd) - 1} arg(s), argv {total} chars, "
            f"timeout {timeout:.0f}s")

        if os.name == "nt" and total > WINDOWS_ARGV_LIMIT:
            raise PlaywriterError(
                f"refusing to launch: the command line is {total} chars, over the "
                f"Windows limit of {WINDOWS_ARGV_LIMIT}.\n"
                "  Large data must travel through a file, not an argument "
                "(see the TRANSPORT note in this file)."
            )

        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            dbg(f"exec: timed out after {timeout:.0f}s")
            raise PlaywriterError(f"playwriter timed out after {timeout:.0f}s") from exc
        except FileNotFoundError as exc:
            # FileNotFoundError is ambiguous on Windows: it covers "no such
            # executable", "no such working directory" AND WinError 206
            # (command line too long). Say which one it was.
            winerror = getattr(exc, "winerror", None)
            if winerror == 206:
                detail = (f"the command line is too long ({total} chars; "
                          f"Windows allows {WINDOWS_ARGV_LIMIT})")
            elif winerror in (2, 3):
                detail = ("the executable or the working directory was not found "
                          f"(cwd={os.getcwd()!r})")
            else:
                detail = f"{exc.strerror or exc} (winerror={winerror})"
            dbg(f"exec: FAILED {type(exc).__name__} winerror={winerror} errno={exc.errno}")
            raise PlaywriterError(
                f"cannot execute {cmd[0]!r}: {detail}\n"
                f"  resolved command: {self.describe()}\n"
                f"  argv length: {total}"
            ) from exc
        except OSError as exc:
            dbg(f"exec: OSError {exc} winerror={getattr(exc, 'winerror', None)}")
            raise PlaywriterError(f"cannot execute {cmd[0]!r}: {exc}") from exc

        elapsed = time.time() - started
        dbg(f"exec: rc={proc.returncode} stdout={len(proc.stdout or '')}B "
            f"stderr={len(proc.stderr or '')}B in {elapsed:.2f}s")
        if proc.returncode != 0:
            dbg(f"exec: stderr tail: {short(proc.stderr, 400)}")
        return proc

    # -- session -----------------------------------------------------------
    def ensure_session(self) -> str:
        if self.session:
            return self.session
        dbg(f"session: creating a new one (tab group {self.tab_group!r})")
        proc = self._run(["session", "new", "--tab-group", self.tab_group], timeout=180)
        blob = (proc.stdout or "") + (proc.stderr or "")
        match = re.search(r"Session (\S+) created", blob)
        if not match:
            dbg(f"session: could not parse: {short(blob, 400)}")
            raise PlaywriterError(
                "could not create a Playwriter session (is the browser extension "
                f"connected?)\n{blob.strip()[:600]}"
            )
        self.session = match.group(1)
        dbg(f"session: {self.session}")
        return self.session

    # -- payload materialisation ------------------------------------------
    def prepare(self, code: str, payload: dict | None = None) -> tuple[str, list[str]]:
        """Substitute temp paths into `code`; return (code, files to delete).

        Split out from eval_js so --test can measure the argv a script produces
        without executing anything.
        """
        files: list[str] = []
        fd, out_path = tempfile.mkstemp(prefix="ds_out_", suffix=".json")
        os.close(fd)
        files.append(out_path)
        code = code.replace("@@OUT@@", json.dumps(js_path(out_path)))

        if payload is not None:
            fd, payload_path = tempfile.mkstemp(prefix="ds_payload_", suffix=".json")
            os.close(fd)
            files.append(payload_path)
            with open(payload_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            code = code.replace("@@PAYLOAD_FILE@@", json.dumps(js_path(payload_path)))
            dbg(f"prepare: payload {os.path.getsize(payload_path)}B -> {payload_path}")
        elif "@@PAYLOAD_FILE@@" in code:
            raise PlaywriterError("script needs a payload but none was given")

        return code, files

    def argv_for(self, code: str, timeout: float = 60) -> list[str]:
        """The exact argv eval_js would run (used by --test)."""
        return [*self._resolve(), "-s", self.session or "<session>",
                "--timeout", str(int(timeout * 1000)), "-e", code]

    # -- the one primitive -------------------------------------------------
    def eval_js(self, code: str, payload: dict | None = None, timeout: float = 60) -> dict:
        """Run `code` in the sandbox; it must call __done(obj).

        Nothing large travels through argv: the page writes its result to
        @@OUT@@ and reads its input from @@PAYLOAD_FILE@@.
        """
        self.ensure_session()
        code, files = self.prepare(code, payload)
        dbg(f"eval_js: code {len(code)} chars, session {self.session}, timeout {timeout:.0f}s")
        try:
            proc = self._run(
                ["-s", self.session, "--timeout", str(int(timeout * 1000)), "-e", code],
                timeout=timeout + 45,
            )
            out_path = files[0]
            if os.path.getsize(out_path) > 0:
                with open(out_path, encoding="utf-8") as fh:
                    result = json.load(fh)
                dbg(f"eval_js: result keys {sorted(result)[:8]} ({len(result)} keys)")
                return result
            blob = ((proc.stdout or "") + (proc.stderr or "")).strip()
            dbg(f"eval_js: no result file; output: {short(blob, 400)}")
            raise PlaywriterError(f"no result from the browser: {blob[-800:]}")
        finally:
            for path in files:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def eval_js_stream(self, code: str, payload: dict | None = None, timeout: float = 60,
                       on_snapshot=None) -> dict:
        """eval_js, but forward snapshots the page writes before it finishes.

        JS_SEND rewrites @@OUT@@ after every poll with {"__stream": true, ...}
        and finally with its result object, whose write is how we know it is
        done. Reading that file here lets the HTTP layer relay progress while
        the browser call is still running.
        """
        self.ensure_session()
        code, files = self.prepare(code, payload)
        out_path = files[0]
        cmd = [*self._resolve(), "-s", self.session, "--timeout",
               str(int(timeout * 1000)), "-e", code]
        total = self.argv_size(cmd)
        dbg(f"eval_js_stream: code {len(code)} chars, argv {total} chars, "
            f"timeout {timeout:.0f}s")
        if os.name == "nt" and total > WINDOWS_ARGV_LIMIT:
            raise PlaywriterError(
                f"refusing to launch: the command line is {total} chars, over the "
                f"Windows limit of {WINDOWS_ARGV_LIMIT}."
            )
        started = time.time()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
        except OSError as exc:
            raise PlaywriterError(f"cannot execute {cmd[0]!r}: {exc}") from exc
        tail: list[str] = []

        def drain() -> None:
            for line in proc.stderr or []:
                tail.append(line.rstrip())
                del tail[:-30]

        Thread(target=drain, daemon=True).start()
        result = None
        seen = None
        try:
            while True:
                if os.path.exists(out_path):
                    try:
                        info = os.stat(out_path)
                        signature = (info.st_size, info.st_mtime_ns)
                    except OSError:
                        signature = None
                    if signature and signature != seen:
                        seen = signature
                        try:
                            with open(out_path, encoding="utf-8") as fh:
                                data = json.load(fh)
                        except (OSError, ValueError):
                            data = None           # caught a half-written file
                        if isinstance(data, dict):
                            if data.get("__stream"):
                                dbg(f"eval_js_stream: snapshot reasoning "
                                    f"{len(data.get('reasoning') or '')} chars, answer "
                                    f"{len(data.get('answer') or '')} chars")
                                if on_snapshot is not None:
                                    try:
                                        on_snapshot(data)
                                    except Exception as exc:   # display must not kill the call
                                        dbg(f"eval_js_stream: snapshot handler failed: {exc}")
                            else:
                                result = data
                if proc.poll() is not None:
                    break
                if time.time() - started > timeout + 40:
                    proc.kill()
                    raise PlaywriterError(f"playwriter timed out after {timeout:.0f}s")
                time.sleep(0.2)
            # the final write can land between the last check and process exit
            if result is None and os.path.exists(out_path):
                try:
                    with open(out_path, encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict) and not data.get("__stream"):
                        result = data
                except (OSError, ValueError):
                    pass
        finally:
            for path in files:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if result is None:
            blob = "\n".join(tail)
            dbg(f"eval_js_stream: no result; stderr: {short(blob, 400)}")
            raise PlaywriterError(f"no result from the browser: {blob[-800:] or 'no output'}")
        dbg(f"eval_js_stream: done in {time.time() - started:.2f}s")
        return result


# The shared prologue: rescue a stale/closed tab, expose the result channel.
PRELUDE = r"""
const __fs = require("node:fs");
const __OUT = @@OUT@@;
const __done = (obj) => { __fs.writeFileSync(__OUT, JSON.stringify(obj)); console.log("__DS_OK__"); };
if (!state.page || state.page.isClosed()) {
  const __pages = context.pages();
  state.page = __pages.find((p) => p.url().includes("chat.deepseek.com")) || await context.newPage();
}
const page = state.page;
const sleep = (ms) => page.waitForTimeout(ms);
const chatIdOf = (u) => (String(u).match(/\/s\/([0-9a-fA-F-]{20,})/) || [])[1] || null;
const openChat = async (id) => {
  if (!id) { if (!page.url().includes("deepseek.com")) await page.goto("https://chat.deepseek.com/", { waitUntil: "commit" }); return; }
  if (!page.url().includes(id)) await page.goto("https://chat.deepseek.com/a/chat/s/" + id, { waitUntil: "commit" });
};
const apiGet = async (path) => page.evaluate(async ({ p }) => {
  const raw = localStorage.getItem("userToken");
  let token = null;
  try { token = JSON.parse(raw).value; } catch (e) { token = raw; }
  if (!token) return { __noToken: true };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 15000);
  try {
    const r = await fetch(p, { headers: { Authorization: "Bearer " + token }, signal: ctrl.signal });
    return await r.json();
  } catch (e) {
    return { __fetchError: String(e) };
  } finally {
    clearTimeout(timer);
  }
}, { p: path });
const messagesOf = (j) => ((((j || {}).data || {}).biz_data || {}).chat_messages) || [];
const ensureDeepThinking = async () => page.evaluate(() => {
  const t = Array.from(document.querySelectorAll(".ds-toggle-button"))
    .find((e) => /deep thinking/i.test(e.textContent || ""));
  if (!t) return "toggle-missing";
  if ((t.getAttribute("aria-pressed") || "") === "true") return "already-on";
  t.click();
  return "enabled";
});
const scrapeDom = async () => page.evaluate(() => {
  // A code block renders as .md-code-block with a banner (language + Copy /
  // Download buttons) around a <pre>. Taking innerText of the whole answer
  // would mix that chrome into the text, so blocks are rebuilt as fences.
  const renderAnswer = (root) => {
    if (!root) return "";
    const parts = [];
    for (const child of Array.from(root.children)) {
      if (child.classList && child.classList.contains("md-code-block")) {
        const label = child.querySelector(".md-code-block-banner-wrap span");
        const pre = child.querySelector("pre");
        parts.push("```" + (label ? (label.innerText || "").trim() : "") + "\n" +
                   (pre ? (pre.innerText || "").replace(/\s+$/, "") : "") + "\n```");
      } else {
        parts.push((child.innerText || "").trim());
      }
    }
    return parts.join("\n\n").trim();
  };
  const msgs = Array.from(document.querySelectorAll(".ds-message"));
  const last = msgs.reverse().find((m) => m.querySelector(".ds-think-content, .ds-markdown.ds-assistant-message-main-content"));
  if (!last) return null;
  const think = last.querySelector(".ds-think-content");
  const answer = last.querySelector(".ds-markdown.ds-assistant-message-main-content");
  const text = last.innerText || "";
  return {
    reasoning: think ? (think.innerText || "").trim() : "",
    answer: renderAnswer(answer),
    label: (text.match(/(Thought for [\d.]+ seconds?|Thinking[^\n]*)/) || [])[1] || null
  };
});

// DeepSeek renders a "Continue" button under an answer it stopped early. The
// page only has it inside the newest assistant message, so the search is scoped
// there: an older message's button must never be clicked by mistake.
const clickContinue = async () => {
  try {
    return await page.evaluate(() => {
  const label = /^continue(\s+generating)?$/i;
  const visible = (el) => {
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 1 && rect.height > 1;
  };
  const msgs = Array.from(document.querySelectorAll(".ds-message"))
    .filter((m) => m.querySelector(".ds-markdown.ds-assistant-message-main-content, .ds-think-content"));
  const scope = msgs[msgs.length - 1] || null;
  if (!scope) return null;
  const nodes = Array.from(scope.querySelectorAll(
    ".ds-button__content, .ds-button, button, [role='button']"));
  let found = null;
  for (const node of nodes) {
    const text = (node.innerText || "").replace(/\s+/g, " ").trim();
    if (!label.test(text)) continue;
    const button = node.closest("button, [role='button'], .ds-button") || node;
    if (visible(button)) found = button;      // last match: the one furthest down
  }
  if (!found) return null;
  found.click();
  return (found.innerText || "").trim() || "Continue";
    });
  } catch (err) {
    console.log("continue: probe failed: " + err);
    return null;                 // never let the probe break a turn
  }
};
"""

JS_CHECK_LOGIN = PRELUDE + r"""
if (!page.url().includes("deepseek.com")) await page.goto("https://chat.deepseek.com/", { waitUntil: "commit" });
await sleep(1500);
const info = await page.evaluate(() => {
  const raw = localStorage.getItem("userToken");
  let token = null;
  try { token = JSON.parse(raw).value; } catch (e) { token = raw; }
  return {
    has_token: !!token,
    has_composer: !!document.querySelector("textarea[name='search']"),
    url: location.href,
    on_signin: location.pathname.includes("sign_in")
  };
});
__done({ ...info, logged_in: info.has_token && info.has_composer });
"""

JS_NEW_CHAT = PRELUDE + r"""
await page.goto("https://chat.deepseek.com/", { waitUntil: "commit" });
await page.locator("textarea[name='search']").first().waitFor({ state: "visible", timeout: 25000 });
const thinking = await ensureDeepThinking();
await sleep(400);
__done({ ok: true, url: page.url(), chat_id: null, thinking });
"""

JS_SEND = PRELUDE + r"""
const __payload = JSON.parse(__fs.readFileSync(@@PAYLOAD_FILE@@, "utf8"));
const MESSAGE = __payload.message;
const CHAT_ID = __payload.chat_id || null;
const BUDGET = __payload.budget;

if (CHAT_ID) await openChat(CHAT_ID);
if (!page.url().includes("deepseek.com")) await page.goto("https://chat.deepseek.com/", { waitUntil: "commit" });
await page.locator("textarea[name='search']").first().waitFor({ state: "visible", timeout: 25000 });

let chatId = CHAT_ID || chatIdOf(page.url());
let baseline = 0;
if (chatId) {
  const pre = await apiGet("/api/v0/chat/history_messages?chat_session_id=" + chatId);
  if (pre && pre.__noToken) __done({ ok: false, error: "not_logged_in" });
  baseline = messagesOf(pre).reduce((mx, m) => Math.max(mx, m.message_id || 0), 0);
}

const thinking = await ensureDeepThinking();

// The page renders tokens as they arrive, so the DOM is the only live view:
// history_messages stays empty until the turn is FINISHED. Remember what the
// previous answer looked like so its text can never be mistaken for this one.
const baseDom = await scrapeDom();
const baseReason = (baseDom && baseDom.reasoning) || "";
const baseAnswer = (baseDom && baseDom.answer) || "";

// A send can be refused before any turn starts: rate limit, "too many
// requests", a lost login. That must not look like a slow model -- and because
// nothing reaches the chat, the client may safely send again later.
const bannerTexts = async () => page.evaluate(() => Array.from(document.querySelectorAll(
  "[role='alert'], [class*='toast'], [class*='Toast'], [class*='notice'], [class*='error']"))
  .map((n) => (n.innerText || "").trim()));
const baseBanners = (await bannerTexts()).join("\u0000");
const bannerText = async () => {
  for (const text of await bannerTexts()) {
    if (!text || text.length > 300 || baseBanners.includes(text)) continue;
    if (/rate.?limit|too many|frequen|try again|busy|limit reached|请求过于频繁|稍后再试|服务器繁忙/i.test(text)) return text;
  }
  return null;
};
let blocked = null;

const box = page.locator("textarea[name='search']").first();
await box.click();
await box.fill(MESSAGE, { timeout: 20000 });
await page.keyboard.press("Enter");

const started = Date.now();
while (!chatId && Date.now() - started < 25000) {
  blocked = await bannerText();
  if (blocked) break;
  chatId = chatIdOf(page.url());
  if (!chatId) await sleep(600);
}
blocked = blocked || (await bannerText());

let assistant = null;
let lastError = null;
let polls = 0;
let streamed = "";
const collect = (msg, type) => (msg.fragments || [])
  .filter((f) => String(f.type || "").toUpperCase() === type)
  .map((f) => f.content || "")
  .join("");
const writeSnapshot = (reasoning, answer, seconds) => {
  const stamp = reasoning.length + ":" + answer.length + ":" +
                reasoning.slice(-24) + answer.slice(-24);
  if (stamp === streamed) return;
  streamed = stamp;
  __fs.writeFileSync(__OUT, JSON.stringify({
    __stream: true, chat_id: chatId, reasoning: reasoning, answer: answer, seconds: seconds
  }));
};
const emitSnapshot = async (candidate) => {
  const dom = await scrapeDom();
  let reasoning = (dom && dom.reasoning) || "";
  let answer = (dom && dom.answer) || "";
  if (reasoning === baseReason && answer === baseAnswer) return;   // still the old turn
  if (!reasoning && !answer && candidate) {
    reasoning = collect(candidate, "THINK");
    answer = collect(candidate, "RESPONSE");
  }
  const think = ((candidate || {}).fragments || [])
    .find((f) => String(f.type || "").toUpperCase() === "THINK") || {};
  const fromLabel = /([\d.]+)\s+second/.exec((dom && dom.label) || "");
  const seconds = think.elapsed_secs === undefined
    ? (fromLabel ? parseFloat(fromLabel[1]) : null)
    : think.elapsed_secs;
  writeSnapshot(reasoning, answer, seconds);
};
let sawAny = false;
const MAX_CONTINUE_CLICKS = 100;
const answerLength = (msg) => ((msg.fragments || [])
  .map((f) => f.content || "").join("")).length;
let continues = 0;
let longest = 0;
let clicksWithoutGrowth = 0;
let lastClick = 0;
let lastGrowth = started;
let candidate = null;
while (Date.now() - started < BUDGET) {
  candidate = null;
  if (chatId) {
    const res = await apiGet("/api/v0/chat/history_messages?chat_session_id=" + chatId);
    polls += 1;
    if (res && (res.__noToken || res.__fetchError)) lastError = res.__noToken ? "not_logged_in" : "fetch_failed";
    const messages = messagesOf(res);
    // The user's own message lands first. If not even that appears, the send
    // was refused and polling for an answer would burn the whole budget.
    if (messages.some((m) => (m.message_id || 0) > baseline)) sawAny = true;
    const fresh = messages.filter((m) => (m.message_id || 0) > baseline && m.role === "ASSISTANT");
    candidate = fresh[fresh.length - 1] || null;
    if (candidate || chatId) await emitSnapshot(candidate);
    if (candidate) {
      const length = answerLength(candidate);
      if (length > longest) {
        longest = length;
        clicksWithoutGrowth = 0;
        lastGrowth = Date.now();
      }
    }
    if (candidate && String(candidate.status || "").toUpperCase() === "FINISHED" &&
        (candidate.fragments || []).some((f) => (f.content || "").length > 0)) {
      // A visible "Continue" means the answer stopped early. Click it and keep
      // polling instead of accepting a truncated reply. Three clicks in a row
      // with no new text mean the button is stale, so the answer is final.
      if (continues < MAX_CONTINUE_CLICKS && clicksWithoutGrowth < 3 &&
          Date.now() - lastClick > 1200) {
        const clicked = await clickContinue();
        if (clicked) {
          continues += 1;
          clicksWithoutGrowth += 1;
          lastClick = Date.now();
          console.log("continue: clicked " + clicked + " (#" + continues + ")");
          await sleep(600);
          continue;
        }
      }
      assistant = candidate;
      break;
    }
  }
  if (!sawAny && !blocked) {
    blocked = await bannerText();
    if (!blocked && !lastError && Date.now() - started > 20000) blocked = "send_blocked";
    if (blocked) break;
  }
  // Stalled mid-turn: the text has not grown for a few seconds and the newest
  // assistant message offers Continue. Only a real message is in scope, so an
  // older conversation's button can never be clicked here.
  if (!blocked && candidate && continues < MAX_CONTINUE_CLICKS &&
      Date.now() - lastClick > 1200 && Date.now() - lastGrowth > 4000) {
    const clicked = await clickContinue();
    if (clicked) {
      continues += 1;
      lastClick = Date.now();
      lastGrowth = Date.now();
      console.log("continue: clicked " + clicked + " mid-turn (#" + continues + ")");
    }
  }
  await sleep(700);
}

const dom = await scrapeDom();
// The last poll can land just before the page renders its final tokens, so ship
// one last snapshot from the finished DOM. The short sleep lets the reading
// side pick it up before __done replaces the file.
if (dom && ((dom.reasoning || "") !== baseReason || (dom.answer || "") !== baseAnswer)) {
  const fromLabel = /([\d.]+)\s+second/.exec(dom.label || "");
  writeSnapshot(dom.reasoning || "", dom.answer || "",
                fromLabel ? parseFloat(fromLabel[1]) : null);
  await sleep(250);
}
const fragments = ((assistant || {}).fragments || []).map((f) => ({
  type: f.type,
  content: f.content || "",
  seconds: f.elapsed_secs === undefined ? null : f.elapsed_secs,
  references: f.references || []
}));

const blockedTurn = !assistant && !!blocked;

__done({
  ok: !blockedTurn && !!(assistant || dom),
  timeout: !assistant,
  chat_id: chatId,
  chat_url: page.url(),
  message_id: assistant ? assistant.message_id : null,
  thinking_enabled: assistant ? assistant.thinking_enabled : null,
  status: assistant ? assistant.status : null,
  fragments,
  dom,
  thinking: thinking,
  polls,
  continues,
  error: blocked || lastError || (assistant ? null : "timed_out_waiting_for_answer"),
  blocked: blockedTurn,
  detail: blocked || null
});
"""

JS_HISTORY = PRELUDE + r"""
const __payload = JSON.parse(__fs.readFileSync(@@PAYLOAD_FILE@@, "utf8"));
const CHAT_ID = __payload.chat_id;
const valid = /^[0-9a-fA-F-]{20,}$/.test(String(CHAT_ID));

if (!valid) {
  __done({ ok: false, error: "bad_chat_id" });
} else {
  if (!page.url().includes("deepseek.com")) await page.goto("https://chat.deepseek.com/", { waitUntil: "commit" });
  const res = await apiGet("/api/v0/chat/history_messages?chat_session_id=" + encodeURIComponent(CHAT_ID));
  if (!res || res.__noToken) {
    __done({ ok: false, error: "not_logged_in" });
  } else if (res.__fetchError) {
    __done({ ok: false, error: "fetch_failed", detail: res.__fetchError });
  } else {
    const messages = messagesOf(res).map((m) => ({
      id: m.message_id,
      role: m.role,
      status: m.status,
      thinking_enabled: m.thinking_enabled,
      inserted_at: m.inserted_at,
      fragments: (m.fragments || []).map((f) => ({
        type: f.type,
        content: f.content || "",
        seconds: f.elapsed_secs === undefined ? null : f.elapsed_secs,
        references: f.references || []
      }))
    }));
    const session = ((res.data || {}).chat_session) || {};
    __done({ ok: res.code === 0, code: res.code, msg: res.msg, chat_id: CHAT_ID,
             title: session.title || null, messages });
  }
}
"""


# --------------------------------------------------------------- normalisation

RATE_HINTS = ("rate limit", "rate-limit", "ratelimit", "too many", "frequen",
              "try again later", "busy", "limit reached", "请求过于频繁", "稍后再试", "繁忙")


def looks_rate_limited(*parts: object) -> bool:
    """True when the text smells like a throttle, not a real failure."""
    text = " ".join(str(part or "") for part in parts).lower()
    return any(hint in text for hint in RATE_HINTS)


def fragments_to_text(fragments: list[dict], *types: str) -> str:
    wanted = {t.upper() for t in types}
    return "".join(f.get("content") or "" for f in fragments if (f.get("type") or "").upper() in wanted)


def thinking_seconds(fragments: list[dict]) -> float | None:
    for f in fragments:
        if (f.get("type") or "").upper() == "THINK":
            return f.get("seconds")
    return None


def references_of(fragments: list[dict]) -> list:
    refs = []
    for f in fragments:
        refs.extend(f.get("references") or [])
    return refs


def pair_turns(messages: list[dict]) -> list[dict]:
    """Walk the message list, pairing each USER prompt with the reply after it."""
    turns: list[dict] = []
    pending: dict | None = None
    for msg in messages:
        role = (msg.get("role") or "").upper()
        frags = msg.get("fragments") or []
        if role == "USER":
            pending = {
                "message_id": msg.get("id"),
                "user": fragments_to_text(frags, "REQUEST") or fragments_to_text(frags, "USER"),
                "reasoning": "",
                "reasoning_seconds": None,
                "answer": "",
            }
            turns.append(pending)
        elif role == "ASSISTANT":
            turn = pending if pending is not None else {
                "message_id": None, "user": "", "reasoning": "", "reasoning_seconds": None, "answer": ""
            }
            turn["reasoning"] = fragments_to_text(frags, "THINK")
            turn["answer"] = fragments_to_text(frags, "RESPONSE")
            turn["reasoning_seconds"] = thinking_seconds(frags)
            turn["answer_message_id"] = msg.get("id")
            if msg.get("status"):
                turn["status"] = msg.get("status")
            pending = None
    return turns


# ------------------------------------------------------------------- HTTP server


class DeepSeekAPI(ThreadingHTTPServer):
    daemon_threads = True
    # Windows happily double-binds a busy port when SO_REUSEADDR is on, which
    # silently routes requests to a stale server. Fail instead.
    allow_reuse_address = False

    def __init__(self, address, browser: Playwriter, wait_timeout: float):
        super().__init__(address, Handler)
        self.browser = browser
        self.wait_timeout = wait_timeout
        self.lock = Lock()


class Handler(BaseHTTPRequestHandler):
    server: DeepSeekAPI
    protocol_version = "HTTP/1.1"

    # -- helpers -----------------------------------------------------------
    def log_message(self, fmt, *args):  # quiet, timestamped one-liner
        sys.stderr.write(f"[http] {self.address_string()} {fmt % args}\n")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        dbg(f"response {status}: {len(body)}B -> {short(json.dumps(payload), 160)}")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        dbg(f"body: {length}B declared, {len(raw)}B read")
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("body must be JSON")
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    def _browser_call(self, fn, *args, **kwargs) -> dict:
        started = time.time()
        try:
            with self.server.lock:                   # one tab, one operation
                return fn(*args, **kwargs)
        except PlaywriterError:
            dbg(f"browser call failed after {time.time() - started:.2f}s")
            if VERBOSE:
                traceback.print_exc(file=sys.stderr)
            raise
        finally:
            dbg(f"browser call took {time.time() - started:.2f}s")

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        dbg(f"GET {self.path}")
        if parsed.path == "/history":
            params = parse_qs(parsed.query)
            chat_id = (params.get("q") or params.get("chat_id") or [""])[0].strip()
            if not chat_id:
                return self._json(400, {"ok": False, "error": "missing ?q=<chat_id>"})
            return self._history(chat_id)
        self._json(404, {"ok": False, "error": f"no route for GET {parsed.path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        dbg(f"POST {self.path}")
        if parsed.path == "/send_message":
            return self._send_message()
        if parsed.path == "/new_chat":
            return self._new_chat()
        self._json(404, {"ok": False, "error": f"no route for POST {parsed.path}"})

    # -- handlers ----------------------------------------------------------
    def _send_message(self):
        try:
            body = self._body()
        except ValueError as exc:
            return self._json(400, {"ok": False, "error": str(exc)})
        message = (body.get("message") or body.get("prompt") or "").strip()
        chat_id = (body.get("chat_id") or "").strip() or None
        if not message:
            return self._json(400, {"ok": False, "error": "missing 'message'"})
        dbg(f"send: message {len(message)} chars, chat_id {chat_id}, stream {bool(body.get('stream'))}")
        if body.get("stream"):
            return self._send_message_stream(message, chat_id)

        payload = {"message": message, "chat_id": chat_id,
                   "budget": int(self.server.wait_timeout * 1000)}
        try:
            raw = self._browser_call(self.server.browser.eval_js, JS_SEND, payload,
                                     self.server.wait_timeout + 60)
        except PlaywriterError as exc:
            return self._json(502, {"ok": False, "error": str(exc), "stage": "send"})

        if not raw.get("ok"):
            error = raw.get("error") or "no answer captured"
            limited = looks_rate_limited(error, raw.get("detail"))
            return self._json(
                429 if limited else 502,
                {
                    "ok": False,
                    "error": error,
                    "detail": raw.get("detail"),
                    "stage": "send",
                    "rate_limited": limited,
                    "retry_after": raw.get("retry_after") or (8 if limited else None),
                    "chat_id": raw.get("chat_id"),
                    "timeout": raw.get("timeout", False),
                },
            )

        return self._json(200, self._send_result(raw))

    @staticmethod
    def _send_result(raw: dict) -> dict:
        """One shape for a finished turn, however it was delivered."""
        fragments = raw.get("fragments") or []
        dom = raw.get("dom") or {}
        reasoning = fragments_to_text(fragments, "THINK") or dom.get("reasoning", "")
        answer = fragments_to_text(fragments, "RESPONSE") or dom.get("answer", "")
        dbg(f"send: {len(fragments)} fragment(s), reasoning {len(reasoning)} chars, "
            f"answer {len(answer)} chars, polls {raw.get('polls')}"
            f"{' continues ' + str(raw.get('continues')) if raw.get('continues') else ''}")
        return {
            "ok": True,
            "chat_id": raw.get("chat_id"),
            "chat_url": raw.get("chat_url"),
            "message_id": raw.get("message_id"),
            "reasoning": reasoning,
            "reasoning_seconds": thinking_seconds(fragments),
            "thinking_enabled": raw.get("thinking_enabled"),
            "thinking_toggle": raw.get("thinking"),
            "answer": answer,
            "sources": references_of(fragments),
            "source": "api" if fragments else "dom",
            "fragment_types": [f.get("type") for f in fragments],
            "dom_label": dom.get("label"),
        }

    def _send_message_stream(self, message: str, chat_id: str | None) -> None:
        """Answer with newline-delimited JSON events while the turn runs.

        Each line is {event: reasoning|answer|done|error, ...}. The client reads
        them as they arrive; the browser call underneath writes snapshots every
        ~0.7 s and blocks this thread for the whole turn.
        """
        payload = {"message": message, "chat_id": chat_id,
                   "budget": int(self.server.wait_timeout * 1000)}
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        sent = {"reasoning": "", "answer": ""}
        state = {"closed": False}

        def emit(event: dict) -> None:
            if state["closed"]:
                return
            try:
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                state["closed"] = True
                dbg("stream: client went away")

        def on_snapshot(snapshot: dict) -> None:
            for key in ("reasoning", "answer"):
                text = snapshot.get(key) or ""
                if text != sent[key]:
                    sent[key] = text
                    emit({"event": key, "text": text, "seconds": snapshot.get("seconds"),
                          "chat_id": snapshot.get("chat_id")})

        try:
            raw = self._browser_call(self.server.browser.eval_js_stream, JS_SEND, payload,
                                     self.server.wait_timeout + 60, on_snapshot)
        except PlaywriterError as exc:
            emit({"event": "error", "error": str(exc), "stage": "send"})
            return
        if not raw.get("ok"):
            error = raw.get("error") or "no answer captured"
            limited = looks_rate_limited(error, raw.get("detail"))
            emit({"event": "error", "error": error, "chat_id": raw.get("chat_id"),
                  "timeout": raw.get("timeout", False), "rate_limited": limited,
                  "retry_after": raw.get("retry_after") or (8 if limited else None)})
            return
        emit({"event": "done", "body": self._send_result(raw)})

    def _new_chat(self):
        try:
            raw = self._browser_call(self.server.browser.eval_js, JS_NEW_CHAT, None, 90)
        except PlaywriterError as exc:
            return self._json(502, {"ok": False, "error": str(exc), "stage": "new_chat"})
        dbg(f"new_chat: url={raw.get('url')} thinking={raw.get('thinking')}")
        return self._json(200, {"ok": True, "chat_id": None, "url": raw.get("url"),
                                "thinking": raw.get("thinking")})

    def _history(self, chat_id: str):
        try:
            raw = self._browser_call(self.server.browser.eval_js, JS_HISTORY,
                                     {"chat_id": chat_id}, 90)
        except PlaywriterError as exc:
            return self._json(502, {"ok": False, "error": str(exc), "stage": "history"})
        if not raw.get("ok"):
            status = 404 if raw.get("code") not in (None, 0) else 502
            return self._json(status, {"ok": False, "error": raw.get("error") or raw.get("msg"),
                                       "chat_id": chat_id})
        turns = pair_turns(raw.get("messages") or [])
        dbg(f"history: {len(raw.get('messages') or [])} message(s) -> {len(turns)} turn(s)")
        return self._json(
            200,
            {
                "ok": True,
                "chat_id": chat_id,
                "title": raw.get("title"),
                "turns": turns,
                "message_count": len(raw.get("messages") or []),
            },
        )


# ------------------------------------------------------------------------ startup


def wait_for_login(browser: Playwriter, timeout: float | None) -> None:
    print("=" * 78)
    print("DEEPSEEK REASONING API")
    print("  Your browser is driven through the Playwriter extension.")
    print("  If chat.deepseek.com asks you to sign in, do it in that browser now.")
    print("  The server starts as soon as a signed-in composer is detected.")
    print("=" * 78)
    started = time.time()
    last_note = 0.0
    while True:
        try:
            info = browser.eval_js(JS_CHECK_LOGIN, None, 60)
        except PlaywriterError as exc:
            print(f"[login] browser not ready yet: {str(exc).splitlines()[0][:120]}")
            info = {}
        dbg(f"login: probe -> {info}")
        if info.get("logged_in"):
            print(f"[login] signed in -- tab {info.get('url')}")
            return
        now = time.time()
        if now - last_note > 8:
            state = "sign-in page" if info.get("on_signin") else "not signed in"
            print(f"[login] waiting for you to log in ({state}) ... {now - started:5.0f}s")
            last_note = now
        if timeout is not None and now - started > timeout:
            raise SystemExit("[login] timed out waiting for a signed-in session")
        time.sleep(3)


def serve(host: str, port: int, session: str | None, tab_group: str,
          wait_timeout: float, login_timeout: float | None, check_login: bool) -> None:
    browser = Playwriter(session=session, tab_group=tab_group)
    dbg(f"serve: host={host} port={port} session={session} wait={wait_timeout}")
    browser.ensure_session()
    print(f"[playwriter] session {browser.session} via {browser.describe()}")
    if check_login:
        wait_for_login(browser, login_timeout)

    try:
        api = DeepSeekAPI((host, port), browser, wait_timeout)
    except OSError as exc:
        raise SystemExit(f"[server] cannot bind {host}:{port} -- {exc} (another instance running?)")
    print(f"\nREADY  http://{host}:{port}")
    print(f"  POST /send_message  {{\"message\": \"...\"}}           (optionally \"chat_id\")")
    print(f"  POST /new_chat")
    print(f"  GET  /history?q=<chat_id>")
    print("  Ctrl+C to stop\n")
    try:
        api.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] bye")
    finally:
        api.server_close()


def main(argv: list[str] | None = None) -> int:
    global VERBOSE
    parser = argparse.ArgumentParser(
        description="DeepSeek reasoning HTTP API over Playwriter.",
        epilog="--verbose logs resolution, every exec (with argv size), payload size and timing.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--session", help="reuse an existing Playwriter session id")
    parser.add_argument("--tab-group", default="deepseek")
    parser.add_argument("--wait", type=float, default=DEFAULT_WAIT,
                        help=f"seconds to wait for each answer (default {DEFAULT_WAIT})")
    parser.add_argument("--login-timeout", type=float, default=None,
                        help="give up after N seconds of waiting for login (default: wait forever)")
    parser.add_argument("--skip-login-check", action="store_true",
                        help="start serving immediately, without the login gate")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug logging on stderr (resolution, execs, payloads, timings)")
    args = parser.parse_args(argv)
    VERBOSE = args.verbose
    dbg("verbose logging enabled")
    serve(args.host, args.port, args.session, args.tab_group, args.wait,
          args.login_timeout, not args.skip_login_check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
