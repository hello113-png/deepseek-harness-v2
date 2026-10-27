#!/usr/bin/env python3
"""
cli.py -- terminal chat with DeepSeek that can act on your files.

    python cli.py                          # asks for a working directory
    python cli.py --dir ~/project          # start in that directory
    python cli.py --dir . --show-thinking   # print the reasoning trace too
    python cli.py --dir . --mcp-config mcp.json   # add tools from MCP servers
    python cli.py --no-stream              # wait for whole answers instead
    python cli.py --verbose                # debug logs, including the backend's
    python cli.py --test                   # test every moving part, print every log
    python cli.py --test --offline         # same, but skip the live browser checks

  Answers stream: the page renders tokens as they arrive, so the client picks
  them up every ~0.7 s and prints them live. Tool requests are collapsed to a
  marker instead of dumping their JSON onto the terminal.

  MCP servers (stdio only) are loaded from <working dir>/mcp.json or the file
  named by --mcp-config, and each of their tools becomes an action next to the
  built-in ones. They start in parallel in the background while the browser
  connects, and a server that dies is restarted on its next call. See
  mcp_client.py for the config shape and the per-server "timeout" key.

Skills
  Every .txt (or .md) file in <working dir>/skills — or in the skills folder
  next to cli.py — is advertised to the model by name and first line. It loads
  one on demand with {"action": "skill", "name": "..."} and follows it, so a
  big folder costs nothing until a skill is actually used.

How the agent gets hands
  DeepSeek's web chat has no function calling, so the client defines a text
  protocol and teaches it to the model in the first message of every
  conversation (see TOOL_GUIDE below). The model asks for work by emitting a
  fenced block tagged `tool` holding one JSON object (a `<tool>` tag is
  accepted too, because models drift):

      ```tool
      {"action": "read_file", "path": "app.py"}
      ```

  This client runs every block in order, confined to the chosen directory, and
  sends the results back as the next message. A turn only ends when the model
  writes {END_MESSAGE}; a reply without it (and without a tool block) is taken
  as an unfinished message and answered with "continue…". Supported actions:
  list_dir, stat, read_file, write_file, write_lines (replace or insert a line
  range), edit_file, find_files, search_files, mkdir, move, delete, run_command,
  web_search, fetch_url.

Rate limits
  A send the page refuses before any turn starts (429, a rate-limit banner, or
  a message that never lands in the chat) is retried with exponential backoff
  and jitter, after a short gap between turns. A turn that may already exist in
  the chat is never resent, so the question cannot be duplicated.

Commands
  /new [dir]   start a fresh conversation (re-sends the tool guide)
  /dir [path]  show or change the working directory
  /think       toggle the reasoning trace
  /tools       list what the agent can do
  /skills      list the skill files the model can load
  /mcp         list the MCP servers and their tools
  /mcp-add-http add a Streamable HTTP MCP server by URL
  /help        this list
  /quit        leave

No dependencies: standard library plus the deepseek.py backend (started here
if it is not already running).
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepseek import (  # noqa: E402
    JS_SEND,
    Playwriter,
    PlaywriterError,
    find_node,
    find_playwriter_cli,
    looks_rate_limited,
)
from mcp_client import (  # noqa: E402
    START_TIMEOUT,
    McpError,
    McpHttpServer,
    McpRegistry,
    McpServer,
)
from run import Backend  # noqa: E402  (reuses the tested spawn/attach + proxy)

MAX_READ = 2_000_000        # character backstop for pathological (minified, CJK) files
MAX_WORDS = 100_000         # words per read; a larger read is refused, not trimmed
MAX_WRITE = 200_000         # characters accepted for a single write
MAX_TOOL_ROUNDS = 100        # tool round-trips per user message
MAX_CONTINUES = 100         # extra "continue" prompts while the end token is missing
MAX_SEND_RETRIES = 3        # transient send failures (rate limit, blocked send)
RETRY_BASE = 4.0            # seconds; doubles per attempt, plus jitter
RETRY_CAP = 60.0            # never wait longer than this between attempts
MIN_SEND_GAP = 1.0          # seconds between turns, to stay under DeepSeek's limit
MAX_RESULT_CHARS = 2_100_000  # total text sent back in one results message
WINDOWS_ARGV_LIMIT = 32_767  # CreateProcess command-line cap (see deepseek.py)

MAX_COMMAND_OUTPUT = 30_000   # characters of stdout/stderr per command
DEFAULT_COMMAND_TIMEOUT = 60  # seconds a command may run before it is killed
MAX_COMMAND_TIMEOUT = 300     # hard ceiling, whatever the model asks for
MAX_SCAN_BYTES = 5_000_000    # largest file searched by search_files
MAX_FIND_RESULTS = 400        # ceiling for find_files / search_files output
MAX_FETCH_BYTES = 2_000_000   # bytes read from a web page
MAX_SKILL_CHARS = 60_000      # one skill's instructions, sent whole or truncated
MAX_FETCH_TEXT = 40_000       # characters of page text handed to the model
DEFAULT_FETCH_TEXT = 12_000   # default page text size
HTTP_TIMEOUT = 30             # seconds per web request
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
SKIPPED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
                ".mypy_cache", ".pytest_cache", ".ruff_cache"}


# -------------------------------------------------------------------- console

def _init_console() -> bool:
    """Make stdout UTF-8 safe and report whether the fancy glyphs are usable."""
    original = sys.stdout.encoding or "ascii"
    try:                                   # never crash on an unencodable char
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if os.name == "nt" and sys.stdout.isatty():
        try:                               # UTF-8 code page for the console window
            os.system("chcp 65001 > nul")
            return True
        except OSError:
            pass
    try:
        "─·⚙❯┌│└—✓✗•".encode(original)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


UNICODE = _init_console()
RULE_CH = "─" if UNICODE else "-"
SEP = " · " if UNICODE else "  |  "
GLYPH_TOOL = "⚙" if UNICODE else "*"
GLYPH_PROMPT = "❯" if UNICODE else ">"
GLYPH_TL, GLYPH_V, GLYPH_BL = ("┌", "│", "└") if UNICODE else ("+", "|", "+")
GLYPH_OK, GLYPH_BAD, GLYPH_SKIP = ("✓", "✗", "•") if UNICODE else ("+", "x", "-")


# ---------------------------------------------------------------------- debug

class Log:
    """Debug channel. Off by default, on with --verbose (stderr, never stdout)."""

    enabled = False

    @classmethod
    def enable(cls) -> None:
        cls.enabled = True

    @classmethod
    def dbg(cls, message: str) -> None:
        if cls.enabled:
            sys.stderr.write(f"[debug {time.strftime('%H:%M:%S')}] {message}\n")
            sys.stderr.flush()

    @classmethod
    def warn(cls, message: str) -> None:
        sys.stderr.write(f"[warn] {message}\n")
        sys.stderr.flush()


def short(text, limit: int = 200) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[:limit] + "…"


# --------------------------------------------------------------------- colours

class Ink:
    """ANSI styling that disappears when stdout is not a terminal."""

    on = sys.stdout.isatty()

    @classmethod
    def _wrap(cls, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if cls.on else text

    @classmethod
    def dim(cls, t): return cls._wrap("2", t)

    @classmethod
    def bold(cls, t): return cls._wrap("1", t)

    @classmethod
    def accent(cls, t): return cls._wrap("36", t)

    @classmethod
    def ok(cls, t): return cls._wrap("32", t)

    @classmethod
    def warn(cls, t): return cls._wrap("33", t)

    @classmethod
    def bad(cls, t): return cls._wrap("31", t)


def rule(width: int = 74, char: str | None = None) -> str:
    return Ink.dim((char or RULE_CH) * width)


# ----------------------------------------------------------------------- tools

class ToolError(Exception):
    pass


class Skills:
    """Instructions the model loads by name, from <working dir>/skills/*.txt.

    Only the name and the first line go into the tool guide, so a big folder
    costs almost nothing until the model actually asks for a skill. The
    working directory wins over the folder next to cli.py, where the built-in
    examples live. `.md` files are accepted as well as `.txt`.
    """

    SUFFIXES = (".txt", ".md")

    def __init__(self, root: Path, extra: Path | None = None):
        self.roots = [root / "skills"]
        if extra is not None and extra.resolve() != root.resolve():
            self.roots.append(extra / "skills")
        self.items: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        self.items = {}
        for folder in self.roots:
            if not folder.is_dir():
                continue
            try:
                entries = sorted(folder.iterdir())
            except OSError as exc:
                Log.dbg(f"skills: cannot list {folder}: {exc}")
                continue
            for path in entries:
                if not path.is_file() or path.suffix.lower() not in self.SUFFIXES:
                    continue
                name = path.stem.strip()
                if not name or name in self.items:
                    continue
                try:
                    body = path.read_text(encoding="utf-8", errors="replace").strip()
                except OSError as exc:
                    Log.dbg(f"skills: cannot read {path}: {exc}")
                    continue
                if not body:
                    continue
                first = next((line.strip(" \t#") for line in body.splitlines()
                              if line.strip()), "")
                self.items[name] = {"name": name, "path": path, "body": body,
                                    "description": short(first, 140)}
        Log.dbg(f"skills: {len(self.items)} found in {[str(r) for r in self.roots]}")

    def find(self, name: str) -> dict | None:
        item = self.items.get(name)
        if item is None:
            self.load()                     # a file may have been added mid-run
            item = self.items.get(name) or next(
                (value for key, value in self.items.items() if key.lower() == name.lower()),
                None)
        return item

    def names(self) -> list[str]:
        return list(self.items)


class Tools:
    """The filesystem and web operations, confined to `root`."""

    ACTIONS = (
        "list_dir", "stat", "read_file", "write_file", "write_lines", "edit_file",
        "find_files", "search_files", "mkdir", "move", "delete",
        "run_command", "web_search", "fetch_url", "skill",
    )

    def __init__(self, root: Path, mcp: McpRegistry | None = None):
        self.root = root.resolve()
        self.mcp = mcp
        self.skills = Skills(self.root, extra=Path(__file__).resolve().parent)
        self.actions = tuple(self.ACTIONS)     # MCP tools extend this per instance
        if mcp is not None:
            self.actions += tuple(mcp.actions())
        Log.dbg(f"tools: root {self.root}, {len(self.actions)} action(s)")
        if mcp is not None and mcp.tools:
            Log.dbg(f"tools: MCP actions {', '.join(mcp.actions())}")

    # -- path handling ----------------------------------------------------
    def resolve(self, rel: str) -> Path:
        raw = (rel or ".").strip()
        if raw in ("", "."):
            return self.root
        candidate = Path(raw)
        if candidate.is_absolute():
            Log.dbg(f"tools: refused absolute path {raw!r}")
            raise ToolError("absolute paths are not allowed; use a path inside the working directory")
        target = (self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            Log.dbg(f"tools: refused escape {raw!r} -> {target}")
            raise ToolError(f"'{raw}' is outside the working directory")
        return target

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)) or "."
        except ValueError:
            return str(path)

    # -- actions ----------------------------------------------------------
    def list_dir(self, args: dict) -> tuple[str, str]:
        target = self.resolve(args.get("path") or ".")
        if not target.is_dir():
            raise ToolError(f"not a directory: {self._rel(target)}")
        entries = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if child.is_dir():
                entries.append(f"{child.name}/")
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entries.append(f"{child.name}  ({size} B)")
        listing = "\n".join(entries) if entries else "(empty)"
        summary = f"{self._rel(target)} -> {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"
        Log.dbg(f"tools: list_dir {summary}")
        return summary, f"{self._rel(target)}:\n{listing}"

    def read_file(self, args: dict) -> tuple[str, str]:
        target = self.resolve(args.get("path"))
        if not target.is_file():
            Log.dbg(f"tools: read_file {self._rel(target)} -> not a file")
            raise ToolError(f"not a file: {self._rel(target)}")
        data = target.read_bytes()
        if b"\x00" in data[:4096]:
            Log.dbg(f"tools: read_file {self._rel(target)} -> binary, refused")
            raise ToolError(f"{self._rel(target)} looks binary; not sending it")
        text = data.decode("utf-8", "replace")
        rel = self._rel(target)
        lines = text.splitlines()
        first = args.get("start", args.get("from"))
        last = args.get("end", args.get("to"))
        if first is not None or last is not None:
            if not lines:
                raise ToolError(f"read_file: {rel} is empty")
            try:
                start = max(int(first or 1), 1)
                end = min(int(last if last is not None else len(lines)), len(lines))
            except (TypeError, ValueError):
                raise ToolError("read_file: 'start'/'end' (or 'from'/'to') must be integers")
            if start > end:
                raise ToolError(f"read_file: start {start} is past end {end} ({len(lines)} lines)")
            # Only the requested slice leaves the machine: no bytes before start,
            # none after end, and an oversized slice is refused rather than cut.
            window = "\n".join(lines[start - 1:end])
            self._read_budget(window, rel, start, end, len(lines))
            span = f"lines {start}-{end}/{len(lines)}"
            summary = f"read_file {rel} {span}"
            Log.dbg(f"tools: {summary}, window {len(window)} chars, {len(window.split())} words")
            return summary, f"--- start of {rel} ({span}) ---\n{window}\n--- end of {rel} ---"
        total_lines = len(lines) or 1
        self._read_budget(text, rel, 1, total_lines, total_lines)
        summary = f"read_file {rel} -> {len(data)} B, {total_lines} lines"
        Log.dbg(f"tools: {summary}, text {len(text)} chars, {len(text.split())} words")
        return summary, f"--- start of {rel} ---\n{text}\n--- end of {rel} ---"

    def _read_budget(self, text: str, rel: str, start: int, end: int,
                     total_lines: int) -> None:
        """Block, do not truncate: one read has to fit the per-message budget.

        The model gets told exactly how many lines it may ask for next, so a
        refusal is actionable instead of a dead end.
        """
        words = len(text.split())
        chars = len(text)
        if words <= MAX_WORDS and chars <= MAX_READ:
            return
        over = (f"about {words:,} words" if words > MAX_WORDS
                else f"{chars:,} characters")
        span = f"lines {start}-{end}" if end > start else f"line {start}"
        count = max(end - start + 1, 1)
        by_words = max(1, int(MAX_WORDS * count / max(words, 1)))
        by_chars = max(1, int(MAX_READ * count / max(chars, 1)))
        step = max(1, min(by_words, by_chars, count))
        hint = (f"read it in ranges of about {step} lines, starting with "
                f"start={start}, end={min(start + step - 1, total_lines)}")
        Log.dbg(f"tools: read_file blocked, {rel} {span}: {words} words, {chars} chars")
        raise ToolError(
            f"read_file blocked: {rel} {span} is {over}; one message may carry at most "
            f"{MAX_WORDS:,} words / {MAX_READ:,} characters — {hint}"
        )

    def write_file(self, args: dict) -> tuple[str, str]:
        target = self.resolve(args.get("path"))
        content = args.get("content")
        if content is None:
            raise ToolError("write_file needs a 'content' field")
        content = str(content)
        if len(content) > MAX_WRITE:
            raise ToolError(f"refusing to write {len(content)} characters (limit {MAX_WRITE})")
        if target.is_dir():
            raise ToolError(f"{self._rel(target)} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        summary = f"wrote {self._rel(target)} ({len(content.encode('utf-8'))} B)"
        Log.dbg(f"tools: {summary}")
        return summary, f"wrote {self._rel(target)}: {len(content)} characters"

    def write_lines(self, args: dict) -> tuple[str, str]:
        """Replace or insert whole lines, leaving the rest of the file untouched.

        `start` and `end` are 1-based and inclusive; without `end` the content is
        inserted before `start`. Line endings and the trailing-newline style are
        preserved, so a CRLF file stays CRLF.
        """
        target = self.resolve(args.get("path"))
        content = args.get("content")
        if content is None:
            raise ToolError("write_lines needs a 'content' field")
        if isinstance(content, (list, tuple)):
            content = "\n".join(str(item) for item in content)
        content = str(content)
        if len(content) > MAX_WRITE:
            raise ToolError(f"refusing to write {len(content)} characters (limit {MAX_WRITE})")
        if not target.is_file():
            raise ToolError(f"not a file: {self._rel(target)}; write_file creates new files")
        raw = target.read_bytes()
        if b"\x00" in raw[:4096]:
            Log.dbg(f"tools: write_lines {self._rel(target)} -> binary, refused")
            raise ToolError(f"{self._rel(target)} looks binary; not writing it")
        rel = self._rel(target)
        text = raw.decode("utf-8", "replace")
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.splitlines()
        first = args.get("start", args.get("from"))
        last = args.get("end", args.get("to"))
        if first is None:
            raise ToolError("write_lines needs 'start' (the 1-based line number to write at)")
        try:
            start = int(first)
            end = int(last) if last is not None else None
        except (TypeError, ValueError):
            raise ToolError("write_lines: 'start'/'end' (or 'from'/'to') must be integers")
        if start < 1:
            raise ToolError("write_lines: 'start' is 1-based, so it must be 1 or more")
        if end is not None and end < start:
            raise ToolError(f"write_lines: end {end} is before start {start} ({len(lines)} lines)")
        fresh = content.splitlines()
        if end is None:
            if start > len(lines) + 1:
                raise ToolError(f"write_lines: start {start} is past the last line "
                                f"({len(lines)} lines); append with start={len(lines) + 1}")
            updated = lines[:start - 1] + fresh + lines[start - 1:]
            what = f"inserted {len(fresh)} line(s) before line {start}"
        else:
            if start > len(lines):
                raise ToolError(f"write_lines: start {start} is past the last line ({len(lines)} lines)")
            stop = min(end, len(lines))
            updated = lines[:start - 1] + fresh + lines[stop:]
            what = f"replaced lines {start}-{stop} with {len(fresh)} line(s)"
        body = newline.join(updated) + (newline if raw.endswith(b"\n") and updated else "")
        if len(body) > MAX_WRITE:
            raise ToolError(f"the file would be {len(body)} characters (limit {MAX_WRITE})")
        joined = "\n".join(updated)
        if b"\x00" in joined.encode("utf-8", "replace")[:4096]:
            raise ToolError("refusing to write NUL bytes into a text file")
        target.write_text(body, encoding="utf-8", newline="")
        summary = f"write_lines {rel} {what} -> {len(updated)} lines"
        Log.dbg(f"tools: {summary}, {len(body)} chars")
        return summary, (f"{rel}: {what}; now {len(updated)} lines, {len(body)} characters")

    def stat(self, args: dict) -> tuple[str, str]:
        target = self.resolve(args.get("path"))
        if not target.exists():
            raise ToolError(f"does not exist: {self._rel(target)}")
        info = target.stat()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime))
        if target.is_dir():
            entries = sum(1 for _ in target.iterdir())
            kind, detail = "directory", f"entries: {entries}"
        else:
            data = target.read_bytes()
            if b"\x00" in data[:4096]:
                kind, detail = "binary file", "not text; read_file will refuse it"
            else:
                text = data.decode("utf-8", "replace")
                lines = text.count("\n") + (0 if text.endswith("\n") else 1) if text else 0
                kind, detail = "text file", f"lines: {lines}, characters: {len(text)}"
        summary = f"stat {self._rel(target)} -> {kind}, {info.st_size} B"
        Log.dbg(f"tools: {summary}")
        payload = (f"{self._rel(target)}\n  type: {kind}\n  size: {info.st_size} B\n"
                   f"  modified: {stamp}\n  {detail}")
        return summary, payload

    def edit_file(self, args: dict) -> tuple[str, str]:
        target = self.resolve(args.get("path"))
        find = args.get("find")
        replace = args.get("replace")
        if find is None or replace is None:
            raise ToolError("edit_file needs both 'find' and 'replace'")
        find, replace = str(find), str(replace)
        if not find:
            raise ToolError("edit_file: 'find' is empty")
        if not target.is_file():
            raise ToolError(f"not a file: {self._rel(target)}")
        text = target.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(find)
        if occurrences == 0:
            Log.dbg(f"tools: edit_file {self._rel(target)} -> no match for {short(find, 60)!r}")
            raise ToolError(f"{short(find, 80)!r} is not in {self._rel(target)}; read the file and copy the exact text")
        replace_all = bool(args.get("all"))
        if occurrences > 1 and not replace_all:
            raise ToolError(f"{short(find, 80)!r} appears {occurrences} times in {self._rel(target)}; "
                            f"use a longer snippet or pass \"all\": true")
        updated = text.replace(find, replace, -1 if replace_all else 1)
        if len(updated) > MAX_WRITE:
            raise ToolError(f"the edited file would be {len(updated)} characters (limit {MAX_WRITE})")
        target.write_text(updated, encoding="utf-8")
        changed = occurrences if replace_all else 1
        lines = updated.count("\n") + 1
        summary = f"edit_file {self._rel(target)} -> {changed} replacement(s)"
        Log.dbg(f"tools: {summary}")
        return summary, (f"{self._rel(target)}: replaced {changed} occurrence(s); "
                         f"now {len(updated)} characters, {lines} lines")

    def find_files(self, args: dict) -> tuple[str, str]:
        pattern = str(args.get("pattern") or "*").strip() or "*"
        base = self.resolve(args.get("path") or ".")
        limit = max(1, min(int(args.get("limit") or 200), MAX_FIND_RESULTS))
        if not base.is_dir():
            raise ToolError(f"not a directory: {self._rel(base)}")
        matches: list[str] = []
        try:
            for path in base.glob(pattern):
                if _skipped(path, self.root):
                    continue
                matches.append(self._rel(path) + ("/" if path.is_dir() else ""))
                if len(matches) >= limit:
                    break
        except (ValueError, OSError) as exc:
            raise ToolError(f"bad glob {pattern!r}: {exc}")
        matches.sort()
        summary = f"find_files {pattern!r} -> {len(matches)} match(es)"
        Log.dbg(f"tools: {summary}")
        if not matches:
            return summary, f"no file matches {pattern!r} in {self._rel(base)} (use **/ for every level)"
        body = "\n".join(matches)
        if len(matches) >= limit:
            body += f"\n… stopped at the {limit}-match limit …"
        return summary, f"{self._rel(base)}: {pattern}\n{body}"

    def search_files(self, args: dict) -> tuple[str, str]:
        pattern = str(args.get("pattern") or "")
        if not pattern:
            raise ToolError("search_files needs a 'pattern' (a regular expression)")
        base = self.resolve(args.get("path") or ".")
        name_glob = str(args.get("glob") or "*") or "*"
        limit = max(1, min(int(args.get("limit") or 80), MAX_FIND_RESULTS))
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        try:
            needle = re.compile(pattern, flags)
        except re.error as exc:
            Log.dbg(f"tools: search_files: {pattern!r} is not a regex ({exc}); using it literally")
            needle = re.compile(re.escape(pattern), flags)
        if not base.is_dir():
            raise ToolError(f"not a directory: {self._rel(base)}")
        hits: list[str] = []
        files_hit = 0
        scanned = 0
        for path in sorted(base.rglob(name_glob)):
            if not path.is_file() or _skipped(path, self.root):
                continue
            try:
                if path.stat().st_size > MAX_SCAN_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                continue
            scanned += 1
            found_here = 0
            for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if not needle.search(line):
                    continue
                hits.append(f"{self._rel(path)}:{number}: {line.strip()[:300]}")
                found_here += 1
                if len(hits) >= limit:
                    break
            if found_here:
                files_hit += 1
            if len(hits) >= limit:
                break
        summary = (f"search_files {short(pattern, 40)!r} -> {len(hits)} hit(s) "
                   f"in {files_hit} of {scanned} file(s)")
        Log.dbg(f"tools: {summary}")
        if not hits:
            return summary, (f"no match for {pattern!r} in {scanned} file(s) under "
                             f"{self._rel(base)}" + (f" matching {name_glob}" if name_glob != "*" else ""))
        body = "\n".join(hits)
        if len(hits) >= limit:
            body += f"\n… stopped at the {limit}-hit limit …"
        return summary, body

    def mkdir(self, args: dict) -> tuple[str, str]:
        target = self.resolve(args.get("path"))
        existed = target.exists()
        if existed and not target.is_dir():
            raise ToolError(f"{self._rel(target)} is a file")
        target.mkdir(parents=True, exist_ok=True)
        summary = f"mkdir {self._rel(target)}" + (" (already there)" if existed else "")
        Log.dbg(f"tools: {summary}")
        return summary, f"{self._rel(target)} is a directory{' (it already existed)' if existed else ''}"

    def move(self, args: dict) -> tuple[str, str]:
        origin = args.get("path") or args.get("from")
        if not origin or not args.get("to"):
            raise ToolError("move needs 'path' (or 'from') and 'to'")
        source = self.resolve(origin)
        if not source.exists():
            raise ToolError(f"does not exist: {self._rel(source)}")
        destination = self.resolve(args.get("to"))
        if destination.exists() and not args.get("overwrite"):
            raise ToolError(f"{self._rel(destination)} already exists; pass \"overwrite\": true to replace it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == self.root:
            raise ToolError("the working directory itself cannot be moved")
        shutil.move(str(source), str(destination))
        summary = f"move {self._rel(source)} -> {self._rel(destination)}"
        Log.dbg(f"tools: {summary}")
        return summary, f"moved {self._rel(source)} to {self._rel(destination)}"

    def delete(self, args: dict) -> tuple[str, str]:
        target = self.resolve(args.get("path"))
        if target == self.root:
            raise ToolError("refusing to delete the working directory itself")
        if not target.exists():
            raise ToolError(f"does not exist: {self._rel(target)}")
        if target.is_dir():
            count = sum(1 for _ in target.rglob("*"))
            shutil.rmtree(target)
            summary = f"delete {self._rel(target)} -> directory, {count} entr{'y' if count == 1 else 'ies'}"
        else:
            size = target.stat().st_size
            target.unlink()
            summary = f"delete {self._rel(target)} -> {size} B file"
        Log.dbg(f"tools: {summary}")
        return summary, f"deleted {self._rel(target)}"

    def run_command(self, args: dict) -> tuple[str, str]:
        command = str(args.get("command") or "").strip()
        if not command:
            raise ToolError("run_command needs a 'command'")
        try:
            timeout = int(args.get("timeout") or DEFAULT_COMMAND_TIMEOUT)
        except (TypeError, ValueError):
            raise ToolError("run_command: 'timeout' must be a number of seconds")
        timeout = max(1, min(timeout, MAX_COMMAND_TIMEOUT))
        cwd = self.resolve(args.get("cwd") or ".")
        if not cwd.is_dir():
            raise ToolError(f"cwd is not a directory: {self._rel(cwd)}")
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        Log.dbg(f"tools: run_command in {self._rel(cwd)}: {short(command, 120)!r}")
        started = time.time()
        try:
            proc = subprocess.run(command, shell=True, cwd=str(cwd), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            partial = "\n".join(part for part in (
                (exc.stdout or "").strip(), (exc.stderr or "").strip()) if part)
            partial = partial[:MAX_COMMAND_OUTPUT]
            summary = f"run_command -> timed out after {timeout}s"
            Log.dbg(f"tools: {summary}")
            return summary, (f"$ {command}\ntimed out after {timeout} seconds"
                             + (f"; output so far:\n{partial}" if partial else "; no output"))
        except OSError as exc:
            raise ToolError(f"could not start the command: {exc}")
        elapsed = time.time() - started
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        summary = f"run_command -> exit {proc.returncode} in {elapsed:.2f}s"
        Log.dbg(f"tools: {summary}, stdout {len(stdout)} chars, stderr {len(stderr)} chars")
        blocks = [f"$ {command}", f"exit code: {proc.returncode} ({elapsed:.2f}s, cwd {self._rel(cwd)})"]
        if stdout:
            blocks.append("stdout:\n" + stdout[:MAX_COMMAND_OUTPUT]
                          + ("\n… stdout truncated …" if len(stdout) > MAX_COMMAND_OUTPUT else ""))
        if stderr:
            blocks.append("stderr:\n" + stderr[:MAX_COMMAND_OUTPUT]
                          + ("\n… stderr truncated …" if len(stderr) > MAX_COMMAND_OUTPUT else ""))
        if not stdout and not stderr:
            blocks.append("(no output)")
        return summary, "\n\n".join(blocks)

    def web_search(self, args: dict) -> tuple[str, str]:
        query = " ".join(str(args.get("query") or "").split())
        if not query:
            raise ToolError("web_search needs a 'query'")
        try:
            count = max(1, min(int(args.get("count") or 8), 10))
        except (TypeError, ValueError):
            count = 8
        provider, results = search_web(query, count)
        lines = [f"{index}. {item['title']}\n   {item['url']}\n   {item['snippet']}"
                 for index, item in enumerate(results, 1)]
        summary = f"web_search {short(query, 40)!r} -> {len(results)} result(s) via {provider}"
        Log.dbg(f"tools: {summary}")
        return summary, f"query: {query}\nprovider: {provider}\n\n" + "\n\n".join(lines)

    def fetch_url(self, args: dict) -> tuple[str, str]:
        url = str(args.get("url") or "").strip()
        if not url:
            raise ToolError("fetch_url needs a 'url'")
        try:
            budget = int(args.get("max_chars") or DEFAULT_FETCH_TEXT)
        except (TypeError, ValueError):
            budget = DEFAULT_FETCH_TEXT
        budget = max(500, min(budget, MAX_FETCH_TEXT))
        content_type, text, final_url = fetch_page(url, budget)
        summary = (f"fetch_url {urllib.parse.urlparse(final_url).netloc} -> "
                   f"{len(text)} chars of text")
        Log.dbg(f"tools: {summary} ({content_type or 'no content type'})")
        header = f"url: {final_url}\ncontent-type: {content_type or 'unknown'}"
        return summary, header + "\n\n" + text

    def skill(self, args: dict) -> tuple[str, str]:
        """Return one skill's instructions, or list what is available."""
        name = str(args.get("name") or args.get("skill") or "").strip()
        if not name:
            if not self.skills.items:
                where = self.root / "skills"
                return ("no skills", f"no skill files in {where} "
                        f"(add a .txt file with the instructions)")
            listing = "\n".join(f"  {item['name']}  {item['description']}"
                                for item in self.skills.items.values())
            return f"{len(self.skills.items)} skill(s)", "available skills:\n" + listing
        item = self.skills.find(name)
        if item is None:
            available = ", ".join(self.skills.names()) or "none"
            raise ToolError(f"no skill named {name!r}; available: {available}")
        body = item["body"]
        if len(body) > MAX_SKILL_CHARS:
            body = body[:MAX_SKILL_CHARS] + f"\n… truncated at {MAX_SKILL_CHARS} characters …"
        summary = f"skill {item['name']} -> {len(body)} chars"
        Log.dbg(f"tools: {summary} from {item['path']}")
        return summary, f"--- skill: {item['name']} ---\n{body}\n--- end of skill ---"

    def execute(self, action: str, args: dict) -> tuple[str, str]:
        """Dispatch one tool call. Named execute, not run, because 'run_command' exists."""
        if action not in self.actions:
            raise ToolError(f"unknown action '{action}' (have: {', '.join(self.actions)})")
        started = time.time()
        try:
            if action in self.ACTIONS:
                return getattr(self, action)(args)
            return self._mcp_call(action, args)
        finally:
            Log.dbg(f"tools: {action} took {time.time() - started:.3f}s")

    def _mcp_call(self, action: str, args: dict) -> tuple[str, str]:
        if self.mcp is None:
            raise ToolError(f"{action} belongs to an MCP server, but none is loaded")
        arguments = args.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {key: value for key, value in args.items() if key != "action"}
        summary, text, is_error = self.mcp.call(action, arguments)
        Log.dbg(f"tools: MCP {summary}")
        if is_error:
            raise ToolError(text)
        return summary, text


# ----------------------------------------------------------------- web helpers

def _skipped(path: Path, root: Path) -> bool:
    """True for paths inside a directory nobody wants searched (.git, venv...)."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in SKIPPED_DIRS for part in parts[:-1])


def _plain(markup: str) -> str:
    text = re.sub(r"<[^>]+>", " ", markup or "")
    return html_lib.unescape(re.sub(r"\s+", " ", text)).strip()


class _TextExtractor(HTMLParser):
    """HTML -> readable text: keep block structure, drop scripts and styles."""

    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "header", "footer", "pre", "table", "ul", "ol",
             "blockquote", "hr", "form", "nav", "main", "aside", "figure", "dt", "dd"}
    SKIP = {"script", "style", "noscript", "svg", "iframe", "template", "canvas"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._pre = 0

    def handle_starttag(self, tag, attrs):  # noqa: D401 - parser hook
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "pre":
            self._pre += 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):  # noqa: D401 - parser hook
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "pre" and self._pre:
            self._pre -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):  # noqa: D401 - parser hook
        if self._skip or not data:
            return
        if self._pre:
            self.parts.append(data)
            return
        collapsed = re.sub(r"[ \t\r\f\v]+", " ", data)
        if collapsed.strip():
            self.parts.append(collapsed)
        elif self.parts and not self.parts[-1].endswith((" ", "\n")):
            self.parts.append(" ")

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
        return parser.text()
    except Exception:                                  # malformed HTML, never fatal
        Log.dbg("web: HTML parser failed; falling back to tag stripping")
        return _plain(markup)


def _http(url: str, data: bytes | None = None, timeout: float = HTTP_TIMEOUT,
          headers: dict | None = None) -> tuple[str, bytes, str]:
    merged = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    }
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, data=data, headers=merged)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_FETCH_BYTES)
        return response.headers.get("Content-Type", ""), raw, response.geturl()


def _ddg_target(href: str) -> str:
    """DuckDuckGo wraps result links; unwrap the real URL."""
    href = html_lib.unescape(href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    query = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
    if "uddg" in query:
        return query["uddg"][0]
    return href


def _ddg_search(query: str, count: int) -> list[dict]:
    data = urllib.parse.urlencode({"q": query, "b": "", "kl": "wt-wt"}).encode()
    _, raw, _ = _http("https://html.duckduckgo.com/html/", data=data, headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://html.duckduckgo.com/",
    })
    page = raw.decode("utf-8", "replace")
    if "result__a" not in page and "anomaly" in page.lower():
        raise RuntimeError("duckduckgo asked for a captcha")
    snippets = [_plain(m) for m in
                re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)]
    results = []
    for index, match in enumerate(re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)):
        results.append({
            "title": _plain(match.group(2)),
            "url": _ddg_target(match.group(1)),
            "snippet": snippets[index] if index < len(snippets) else "",
        })
        if len(results) >= count:
            break
    if not results:
        raise RuntimeError("duckduckgo returned no results")
    return results


MARGINALIA_BASE = "https://old-search.marginalia.nu"


def _marginalia_search(query: str, count: int) -> list[dict]:
    """Small independent index; used when DuckDuckGo refuses to answer."""
    url = f"{MARGINALIA_BASE}/search?" + urllib.parse.urlencode({"query": query})
    page = ""
    for attempt in range(3):
        _, raw, _ = _http(url, timeout=45)
        page = raw.decode("utf-8", "replace")
        token = re.search(r'href="(/search\?query=[^"]+&sst=[^"]+)"', page)
        if not token:
            break
        Log.dbg(f"web: marginalia wants a pause (attempt {attempt + 1})")
        time.sleep(1.5)
        url = MARGINALIA_BASE + html_lib.unescape(token.group(1))
    results = []
    for block in re.split(r'<section[^>]+class="card search-result"', page)[1:]:
        link = (re.search(r'class="title"[^>]+href="([^"]+)"', block)
                or re.search(r'<div class="url"><a[^>]+href="([^"]+)"', block))
        if not link:
            continue
        title = re.search(r'class="title"[^>]*>(.*?)</a>', block, re.S)
        snippet = re.search(r'<p class="description">(.*?)</p>', block, re.S)
        results.append({
            "title": _plain(title.group(1)) if title else link.group(1),
            "url": html_lib.unescape(link.group(1)),
            "snippet": _plain(snippet.group(1)) if snippet else "",
        })
        if len(results) >= count:
            break
    if not results:
        raise RuntimeError("marginalia returned no results")
    return results


def _searxng_search(query: str, count: int) -> list[dict]:
    base = (os.environ.get("SEARXNG_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("SEARXNG_URL is not set")
    url = f"{base}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    _, raw, _ = _http(url)
    payload = json.loads(raw.decode("utf-8", "replace"))
    results = [{"title": _plain(r.get("title", "")), "url": r.get("url", ""),
                "snippet": _plain(r.get("content", ""))}
               for r in payload.get("results", [])[:count]]
    if not results:
        raise RuntimeError("searxng returned no results")
    return results


def search_web(query: str, count: int = 8) -> tuple[str, list[dict]]:
    """Try each provider in turn and report which one answered."""
    providers = []
    if os.environ.get("SEARXNG_URL"):
        providers.append(("searxng", _searxng_search))
    providers += [("duckduckgo", _ddg_search), ("marginalia", _marginalia_search)]
    problems: list[str] = []
    for name, finder in providers:
        attempts = 2 if name == "duckduckgo" else 1
        for attempt in range(1, attempts + 1):
            try:
                Log.dbg(f"web: {name} attempt {attempt} for {query!r}")
                results = finder(query, count)
                if results:
                    return name, results
                problems.append(f"{name}: no results")
            except Exception as exc:                   # a search must never kill a turn
                problems.append(f"{name}: {short(exc, 90)}")
                Log.dbg(f"web: {name} attempt {attempt} failed: {exc}")
            if attempt < attempts:
                time.sleep(3)
    raise ToolError("search failed — " + "; ".join(problems[-4:])
                    + " (set SEARXNG_URL to a SearXNG instance for reliable search)")


def fetch_page(url: str, budget: int) -> tuple[str, str, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ToolError("fetch_url only accepts absolute http:// or https:// URLs")
    try:
        content_type, raw, final_url = _http(url)
    except urllib.error.HTTPError as exc:
        raise ToolError(f"{url} answered HTTP {exc.code} {exc.reason}")
    except urllib.error.URLError as exc:
        raise ToolError(f"could not reach {url}: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        raise ToolError(f"{url} did not answer: {exc}")
    charset = "utf-8"
    match = re.search(r"charset=([\w.:-]+)", content_type or "", re.I)
    if match:
        charset = match.group(1)
    try:
        body = raw.decode(charset, "replace")
    except LookupError:
        body = raw.decode("utf-8", "replace")
    kind = (content_type or "").lower()
    if "json" in kind:
        try:
            text = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
        except ValueError:
            text = body
    elif "html" in kind or (not kind and body.lstrip().startswith("<")):
        text = html_to_text(body)
    elif kind.startswith("text/") or not kind:
        text = body
    else:
        raise ToolError(f"{url} returned {content_type}; only text, HTML and JSON are read")
    text = text.strip()
    total = len(text)
    if total > budget:
        text = text[:budget] + f"\n… truncated at {budget} of {total} characters …"
    return content_type, text, final_url


# -------------------------------------------------------------------- protocol

# A reply is only finished when it carries this token. Anything else means the
# model was cut off (long answers arrive in pieces), so the client asks again.
END_TOKEN = "{END_MESSAGE}"
CONTINUE_PROMPT = 'continue, but if your message is already over, say "{END_MESSAGE}"'

TOOL_GUIDE = """You are connected to a local terminal client that can act on the user's computer. Work only through the protocol below.

WORKING DIRECTORY
  {root}
  Relative paths resolve against it. File actions cannot leave it.

HOW TO ASK FOR WORK
  Reply with one or more fenced blocks tagged `tool`, each holding exactly one JSON object:

  ```tool
  {{"action": "read_file", "path": "app.py"}}
  ```

  The client runs every block in the order written and sends you the results as the next message. Never describe or quote a file you have not read.

ACTIONS
  list_dir     {{"path": "."}}                                  entries with sizes
  stat         {{"path": "app.py"}}                             size, type, mtime, line count
  read_file    {{"path": "app.py", "start": 40, "end": 80}}     only lines 40-80; omit both for the whole file
                                                              (max {words:,} words per read, or it is refused)
  write_file   {{"path": "app.py", "content": "full text"}}      replaces the whole file
  write_lines  {{"path": "app.py", "start": 40, "end": 42, "content": "new lines"}}
                                                              replaces lines 40-42; omit "end" to insert before line 40
  edit_file    {{"path": "app.py", "find": "old", "replace": "new", "all": false}}
  find_files   {{"pattern": "**/*.py"}}                          glob over file names
  search_files {{"pattern": "regex", "glob": "*.py"}}            contents; optional "ignore_case": true
  mkdir        {{"path": "src/new"}}
  move         {{"path": "a.txt", "to": "b.txt"}}                optional "overwrite": true
  delete       {{"path": "old.txt"}}                             a file or a whole directory
  run_command  {{"command": "pytest -q"}}                        optional "timeout": 60 (max 300)
  web_search   {{"query": "python subprocess timeout"}}          optional "count": 8 (max 10)
  fetch_url    {{"url": "https://docs.python.org/3/"}}           optional "max_chars": 12000
  skill        {{"name": "deploy"}}                             that skill's instructions; no name lists them
{skills}{mcp_tools}
RULES
  1. When the user names a file, act on exactly that path. If it is missing, say so and list the directory; never substitute another file.
  2. Read before you change. Use write_lines to replace or insert a line range, edit_file for an exact text swap, write_file only for new files or a full rewrite. On a long file ask for a line range (start/end): only that slice is sent back, so read it in pieces instead of asking for everything at once.
  3. Paths are relative to the working directory. No absolute paths, no "..".
  4. search_files reports `path:line: text`; find_files matches names, so use **/ to reach every level.
  5. run_command is not sandboxed. Keep it inside the working directory, never start long-lived servers, and do not run commands that change things the user did not ask about.
  6. web_search returns titles, URLs and snippets. Use fetch_url on the pages you actually need, and cite the URLs.
  7. move and delete change the user's disk: only when the task requires it, and say what you renamed or removed.
  8. One action per block, prose outside blocks.
  9. A read over {words:,} words ({chars:,} characters) is refused outright, not trimmed: ask for a smaller range. Binary files are refused too. Say so instead of guessing.
  10. After working, state which files changed and what you did.
  11. Your turn is over only when you end it with {{END_MESSAGE}}, alone on its last line, and that reply carries no `tool` block. If a reply has neither, the client assumes you were cut off and sends you this next: continue, but if your message is already over, say "{{END_MESSAGE}}"
  12. If a skill below covers the task, load it with the `skill` action BEFORE you start and follow it; skills are the user's own instructions and outrank your defaults.

Reply with one short sentence confirming you understand, ending with {{END_MESSAGE}}, and nothing else."""

TOOL_BLOCK = re.compile(r"```[ \t]*(?:tool|tool_call|tool_code)[ \t]*\r?\n(.*?)```", re.S | re.I)
TAG_BLOCK = re.compile(r"<(tool|tool_call|tool_code)>\s*(.*?)\s*</\1>", re.S | re.I)
ANY_BLOCK = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n(.*?)```", re.S | re.I)


def extract_tool_calls(answer: str, problems: list[str] | None = None,
                       valid_actions=None) -> list[dict]:
    """Pull tool requests out of a reply.

    Primary form: ```tool {json} ```. Two tolerances exist because models drift:
    `<tool>{json}</tool>` is accepted, and any fenced JSON block naming a known
    action works too. A block that clearly means to be a call but cannot be run
    is appended to `problems` so the caller can ask the model to correct itself.
    """
    actions = tuple(valid_actions or Tools.ACTIONS)
    calls: list[dict] = []
    seen: set[str] = set()
    for pattern, strict in ((TOOL_BLOCK, True), (TAG_BLOCK, True), (ANY_BLOCK, False)):
        for match in pattern.finditer(answer):
            body = (match.groups()[-1] or "").strip()
            if not body or body in seen:
                continue
            try:
                obj = json.loads(body)
            except ValueError:
                seen.add(body)
                if strict:
                    note = f"a tool block was not valid JSON: {short(body, 120)!r}"
                    Log.dbg("parse: " + note)
                    if problems is not None:
                        problems.append(note)
                else:
                    Log.dbg(f"parse: fenced block is not JSON: {short(body, 80)}")
                continue
            if not isinstance(obj, dict):
                seen.add(body)
                continue
            # MCP clients often write {"name": ...}; accept that spelling too.
            action = obj.get("action") or obj.get("tool") or obj.get("name")
            if action not in actions:
                seen.add(body)
                if strict:
                    note = (f"action {action!r} is not supported; valid actions are "
                            + ", ".join(actions))
                    Log.dbg("parse: " + note)
                    if problems is not None:
                        problems.append(note)
                else:
                    Log.dbg(f"parse: ignoring unknown action {action!r}")
                continue
            args = obj.get("args") if isinstance(obj.get("args"), dict) else obj
            calls.append({"action": action, "args": args})
            seen.add(body)
    Log.dbg(f"parse: {len(calls)} tool call(s) out of {len(answer)} chars"
            f"{', ' + str(len(problems)) + ' problem(s)' if problems else ''}")
    return calls


def strip_tool_blocks(answer: str, valid_actions=None) -> str:
    """Remove the blocks that asked for work, keep ordinary code blocks."""
    actions = tuple(valid_actions or Tools.ACTIONS)

    def drop_if_call(match: re.Match) -> str:
        try:
            obj = json.loads(match.group(1).strip())
        except ValueError:
            return match.group(0)
        if isinstance(obj, dict) and obj.get("action") in actions:
            return ""
        return match.group(0)

    cleaned = TOOL_BLOCK.sub("", answer)      # tool-tagged blocks are always control
    cleaned = TAG_BLOCK.sub("", cleaned)      # <tool>…</tool> is control too
    cleaned = ANY_BLOCK.sub(drop_if_call, cleaned)
    return cleaned.strip()


def strip_end_token(answer: str) -> str:
    """Drop the protocol token: it is a signal, not something to print."""
    if END_TOKEN not in answer:
        return answer
    return answer.replace(END_TOKEN, "").rstrip()


def _token_prefix_len(text: str) -> int:
    """How many trailing characters could still turn into the end token."""
    limit = min(len(END_TOKEN), len(text))
    for size in range(limit, 0, -1):
        if END_TOKEN.startswith(text[-size:]):
            return size
    return 0


# ------------------------------------------------------------------- streaming

def meta_text(seconds, chars: int) -> str:
    bits = []
    if seconds is not None:
        try:
            bits.append(f"thought {float(seconds):.1f}s")
        except (TypeError, ValueError):
            pass
    if chars:
        bits.append(f"{chars / 1000:.1f}k chars" if chars >= 1000 else f"{chars} chars")
    return SEP.join(bits)


class AnswerStream:
    """Print answer text as it arrives while holding tool requests back.

    The page renders a tool call as a bare JSON object (no fence, no language),
    and the API form fences it, so both shapes are recognised here. Otherwise a
    raw printer would splash a 200 KB write_file payload onto the terminal.
    """

    HIDDEN = ("tool", "tool_call", "tool_code")
    JSON_START = '{"action"'

    def __init__(self, write, valid_actions):
        self.write = write
        self.valid_actions = set(valid_actions)
        self.pending = ""
        self.in_block = False
        self.in_json = False
        self.fed = ""                    # everything received, printed or held

    def feed(self, text: str) -> None:
        self.fed += text
        self.pending += text
        self._drain()

    def merge(self, final_answer: str) -> None:
        """Append the tail the last snapshot missed, when the texts line up."""
        answer = (final_answer or "").strip()
        if not answer:
            return
        # Diff against everything fed, not against what is on screen: text
        # waiting behind a possible fence marker is already in the buffer.
        seen = self.fed.rstrip()
        if not seen:
            self.feed(answer)
        elif answer.startswith(seen):
            self.feed(answer[len(seen):])
        elif not seen.startswith(answer):
            Log.dbg("stream: DOM text and API text diverge; keeping what was printed")

    def _emit(self, text: str) -> None:
        if END_TOKEN in text:
            text = strip_end_token(text)
            if not text:
                return
        self.write(text)

    def finish(self) -> None:
        if not self.pending:
            return
        text, self.pending = self.pending, ""
        if text.strip() and END_TOKEN.startswith(text.strip()):
            return                       # the token was cut off; show nothing
        if (self.in_block or self.in_json) and text.lstrip().startswith(("```tool", "```tool_call", "```tool_code", self.JSON_START)):
            self.write(Ink.dim("  [tool request]") + "\n")
        else:
            self._emit(text)
        self.in_block = False
        self.in_json = False

    # -- internals --------------------------------------------------------
    def _drain(self) -> None:
        while True:
            if END_TOKEN in self.pending:
                # The token is a signal, never text: drop it, print what led to
                # it, and keep scanning what follows. Doing it here (not in the
                # emitter) means it cannot leak when chunks split it.
                before, _, after = self.pending.partition(END_TOKEN)
                self.pending = after
                self.in_block = self.in_json = False
                if before:
                    self._emit(before)
                continue

            if self.in_block:
                index = self.pending.find("```", 3)
                if index < 0:
                    return
                block = self.pending[:index + 3]
                self.pending = self.pending[index + 3:]
                self.in_block = False
                self._render(block)
                continue

            if self.in_json:
                end = self._json_end(self.pending)
                if end < 0:
                    return
                text = self.pending[:end]
                self.pending = self.pending[end:]
                self.in_json = False
                self._render_json(text)
                continue

            stripped = self.pending.lstrip()
            offset = len(self.pending) - len(stripped)
            fence = stripped.find("```")
            json_start = stripped.startswith(self.JSON_START)
            if fence >= 0 and not json_start:
                if offset + fence:
                    self._emit(self.pending[:offset + fence])
                self.pending = stripped[fence:]
                self.in_block = True
                continue
            if json_start:
                if offset:
                    self._emit(self.pending[:offset])
                self.pending = stripped
                self.in_json = True
                continue
            # no marker: hold back enough to catch a marker split across chunks
            keep = max(len(self.JSON_START) - 1, 2, _token_prefix_len(self.pending))
            if len(self.pending) > keep:
                self._emit(self.pending[:-keep])
                self.pending = self.pending[-keep:]
            return

    @staticmethod
    def _json_end(text: str) -> int:
        """Index just past the closing brace, or -1 while it is incomplete."""
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index + 1
        return -1

    def _render_json(self, text: str) -> None:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj.get("action") in self.valid_actions:
            self.write(Ink.dim("  [tool request]") + "\n")
        else:
            self._emit(text)

    def _render(self, block: str) -> None:
        match = re.match(r"```[ \t]*([\w-]*)[ \t]*\r?\n?(.*?)```\s*$", block, re.S)
        hidden = False
        if match:
            language = (match.group(1) or "").lower()
            body = (match.group(2) or "").strip()
            if language in self.HIDDEN:
                hidden = True
            elif language in ("", "json") and body:
                try:
                    obj = json.loads(body)
                except ValueError:
                    obj = None
                if isinstance(obj, dict) and obj.get("action") in self.valid_actions:
                    hidden = True
        if hidden:
            self.write(Ink.dim("  [tool request]") + "\n")
        else:
            self._emit(block)


class StreamView:
    """Render one streamed turn: reasoning dimmed, answer live."""

    def __init__(self, write, say, show_thinking: bool, valid_actions,
                 quiet: bool = False):
        self.write = write
        self.say = say
        self.show_thinking = show_thinking
        self.quiet = quiet
        self.seen_reasoning = 0
        self.seen_answer = 0
        self.shown_reasoning = False
        self.answer_open = False
        self.tail = ""
        self.filter = None if quiet else AnswerStream(write, valid_actions)

    def event(self, event: dict) -> None:
        if self.quiet:
            return
        kind, text = event.get("event"), (event.get("text") or "")
        if kind == "reasoning":
            if not self.show_thinking:
                self.seen_reasoning = len(text)
                return
            if not self.shown_reasoning:
                self.say(Ink.dim("thinking…"))
                self.shown_reasoning = True
            delta = text[self.seen_reasoning:]
            self.seen_reasoning = len(text)
            if delta:
                self.write(Ink.dim(delta))
                self.tail = delta[-1]
        elif kind == "answer":
            if not self.answer_open:
                self.answer_open = True
                if self.shown_reasoning:
                    self.write("\n")
                    meta = meta_text(event.get("seconds"), self.seen_reasoning)
                    self.say(Ink.dim("deepseek" + ("  " + meta if meta else "")))
                else:
                    self.say(Ink.dim("deepseek"))
            delta = text[self.seen_answer:]
            self.seen_answer = len(text)
            if delta and self.filter is not None:
                self.filter.feed(delta)
                self.tail = delta[-1]

    def finish(self, body: dict | None = None) -> None:
        if self.quiet:
            return
        if self.filter is not None:
            self.filter.merge((body or {}).get("answer") or "")
            self.filter.finish()
        if (self.answer_open or self.shown_reasoning) and self.tail != "\n":
            self.write("\n")


# ----------------------------------------------------------------- the session

class Agent:
    def __init__(self, api: Backend, root: Path, show_thinking: bool = False,
                 max_rounds: int = MAX_TOOL_ROUNDS, quiet: bool = False,
                 stream: bool = True, mcp: McpRegistry | None = None,
                 require_end: bool = True):
        self.api = api
        self.root = root.resolve()
        self.mcp = mcp
        self.tools = Tools(self.root, mcp)
        self.show_thinking = show_thinking
        self.require_end = require_end         # --test turns the token rule off
        self.max_rounds = max_rounds
        self.quiet = quiet                     # --test drives this quietly
        self.stream = stream                   # --no-stream disables live text
        self._streamed = False
        self.chat_id: str | None = None
        self.turns = 0
        self.last_send = 0.0                   # pacing against DeepSeek's rate limit

    def _write(self, text: str) -> None:
        """Stream text without a newline; silenced while --test runs."""
        if self.quiet:
            return
        sys.stdout.write(text)
        sys.stdout.flush()

    # -- output -----------------------------------------------------------
    def _say(self, text: str, kind: str = "plain") -> None:
        """Print to the transcript, or hand it to the debug log when quiet."""
        if self.quiet:
            Log.dbg("agent: " + short(text, 300))
            return
        print(text)

    # -- plumbing ---------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: float = 400) -> dict:
        size = len(json.dumps(payload, ensure_ascii=False))
        Log.dbg(f"agent: POST {path} ({size}B payload, timeout {timeout:.0f}s)")
        started = time.time()
        status, body = self.api.call("POST", path, payload, timeout=timeout)
        Log.dbg(f"agent: POST {path} -> {status} in {time.time() - started:.2f}s")
        if status >= 400 or not body.get("ok"):
            detail = body.get("error") or f"HTTP {status}"
            Log.dbg(f"agent: failure detail: {short(json.dumps(body), 300)}")
            raise RuntimeError(detail)
        return body

    def new_chat(self) -> None:
        self._post("/new_chat", {}, timeout=120)
        self.chat_id = None
        self.turns = 0
        Log.dbg("agent: new chat started")

    @staticmethod
    def _retry_delay(status: int, body: dict, attempt: int) -> float | None:
        """Seconds to wait before retrying a send, or None if retrying cannot help.

        Only refusals that happened before the turn started are retried: a rate
        limit, a send the page would not accept, or a backend that was not up.
        A turn that timed out may already exist in the chat, so resending it
        would duplicate the question.
        """
        error = str(body.get("error") or "")
        local = "backend is not ready yet" in error
        transient = (status == 429 or bool(body.get("rate_limited"))
                     or error == "send_blocked" or local)
        if not transient:
            return None
        try:
            suggested = float(body.get("retry_after") or 0)
        except (TypeError, ValueError):
            suggested = 0.0
        delay = suggested or RETRY_BASE * (2 ** attempt)
        return min(delay + random.uniform(0, 1.5), RETRY_CAP)

    def _send(self, message: str) -> dict:
        attempt = 0
        while True:
            gap = MIN_SEND_GAP - (time.time() - self.last_send)
            if self.last_send and gap > 0:
                time.sleep(gap)
            payload = {"message": message}
            if self.chat_id:
                payload["chat_id"] = self.chat_id
            self._streamed = False
            view = None
            if self.stream:
                payload["stream"] = True
                view = StreamView(self._write, self._say, self.show_thinking,
                                  self.tools.actions, self.quiet)
            Log.dbg(f"agent: sending {len(message)} chars"
                    f"{' into chat ' + self.chat_id if self.chat_id else ' (new chat)'}"
                    f"{' (streaming)' if self.stream else ''}"
                    f"{' (attempt ' + str(attempt + 1) + ')' if attempt else ''}")
            started = time.time()
            try:
                if self.stream:
                    def on_event(event: dict) -> None:
                        if event.get("event") in ("reasoning", "answer"):
                            self._streamed = True
                        if view is not None:
                            view.event(event)

                    status, body = self.api.call_stream("/send_message", payload,
                                                        on_event=on_event, timeout=400)
                else:
                    status, body = self.api.call("POST", "/send_message", payload,
                                                 timeout=400)
            finally:
                self.last_send = time.time()
            if view is not None:
                view.finish(body)
            Log.dbg(f"agent: POST /send_message -> {status} in {time.time() - started:.2f}s"
                    f"{' (streamed)' if self._streamed else ''}")
            if status < 400 and body.get("ok"):
                self.chat_id = body.get("chat_id") or self.chat_id
                self.turns += 1
                Log.dbg(f"agent: chat {self.chat_id}, message {body.get('message_id')}, "
                        f"fragments {body.get('fragment_types')}, source {body.get('source')}, "
                        f"reasoning {len(body.get('reasoning') or '')} chars, "
                        f"answer {len(body.get('answer') or '')} chars")
                return body

            detail = body.get("error") or f"HTTP {status}"
            wait = self._retry_delay(status, body, attempt)
            if wait is None or attempt >= MAX_SEND_RETRIES or self._streamed:
                Log.dbg(f"agent: failure detail: {short(json.dumps(body), 300)}")
                raise RuntimeError(detail)
            attempt += 1
            note = (f"  {short(detail, 120).lower()}; retrying in {wait:.0f}s "
                    f"(attempt {attempt}/{MAX_SEND_RETRIES})")
            self._say(Ink.warn(note))
            Log.dbg(f"agent: retrying after {wait:.1f}s: {detail}")
            time.sleep(wait)

    # -- display ----------------------------------------------------------
    def _show_meta(self, body: dict, prefix: str = "deepseek") -> None:
        secs = body.get("reasoning_seconds")
        chars = len(body.get("reasoning") or "")
        bits = []
        if secs is not None:
            bits.append(f"thought {round(secs, 1)}s")
        if chars:
            bits.append(f"{chars / 1000:.1f}k chars" if chars >= 1000 else f"{chars} chars")
        if body.get("thinking_toggle") and body["thinking_toggle"] != "already-on":
            bits.append(body["thinking_toggle"])
        meta = SEP.join(bits)
        self._say(Ink.dim(f"{prefix}  {meta}" if meta else prefix))

    def _show_reasoning(self, body: dict) -> None:
        if not self.show_thinking:
            return
        reasoning = (body.get("reasoning") or "").strip()
        if not reasoning:
            return
        self._say(Ink.dim(f"  {GLYPH_TL} reasoning"))
        for line in reasoning.splitlines():
            self._say(Ink.dim(f"  {GLYPH_V} " + line))
        self._say(Ink.dim(f"  {GLYPH_BL}"))

    @staticmethod
    def _label(call: dict) -> str:
        """What to show on the tool line: the interesting argument, not always a path."""
        args = call["args"]
        action = call["action"]
        if action not in Tools.ACTIONS:          # an MCP tool
            values = args.get("arguments") if isinstance(args.get("arguments"), dict) else {
                key: value for key, value in args.items() if key != "action"}
            inner = ", ".join(f"{key}={short(str(value), 30)}"
                              for key, value in list(values.items())[:2])
            return f"{action}({short(inner, 70)})"
        if action == "web_search":
            target = args.get("query")
        elif action == "fetch_url":
            target = args.get("url")
        elif action == "run_command":
            target = args.get("command")
        elif action == "move":
            target = f"{args.get('path') or args.get('from')} -> {args.get('to')}"
        elif action == "search_files":
            target = args.get("pattern")
        elif action == "find_files":
            target = args.get("pattern")
        elif action == "skill":
            target = args.get("name") or "(list)"
        elif action == "write_lines":
            target = (f"{args.get('path')} lines {args.get('start')}-"
                      f"{args.get('end') or args.get('start')}")
        elif action == "edit_file":
            target = args.get("path")
        else:
            target = args.get("path") or "."
        return f"{action}({short(str(target), 60)!r})"

    # -- the loop ---------------------------------------------------------
    def mcp_section(self) -> str:
        """The MCP part of the tool guide, empty when no server is loaded."""
        if self.mcp is None or not self.mcp.tools:
            return ""
        lines = "\n".join(self.mcp.guide_lines())
        return ("\nMCP TOOLS (from servers the user configured; arguments go at top level,\n"
                "the same way the actions above do)\n" + lines + "\n")

    def skills_section(self) -> str:
        """Names and first lines only: the full text is loaded on demand."""
        items = self.tools.skills.items
        if not items:
            return ""
        lines = "\n".join(f"  {item['name']}  {item['description']}"
                           for item in items.values())
        return ("\nSKILLS (the user's own instructions, from the skills folder; they\n"
                "outrank your defaults, so load the one that fits the task first)\n"
                "  skill  {{\"name\": \"<skill>\"}}   load its full instructions\n"
                + lines + "\n")

    def prime(self) -> str:
        guide = TOOL_GUIDE.format(root=self.root, words=MAX_WORDS, chars=MAX_READ,
                                  skills=self.skills_section(),
                                  mcp_tools=self.mcp_section())
        Log.dbg(f"agent: priming with the tool guide ({len(guide)} chars) for {self.root}")
        streaming, self.stream = self.stream, False     # the caller prints the reply
        try:
            body = self._send(guide)
        finally:
            self.stream = streaming
        text = strip_end_token(body.get("answer") or "").strip()
        if not text:
            Log.warn("the model did not confirm the tool guide; tools may misbehave")
        return text

    def ask(self, message: str) -> str:
        """Run rounds until the model writes the end token, then return its prose."""
        body = self._send(message)
        parts: list[str] = []
        continues = 0
        tool_rounds = 0
        rounds = 0
        while True:
            rounds += 1
            answer = body.get("answer") or ""
            problems: list[str] = []
            calls = extract_tool_calls(answer, problems, self.tools.actions)
            prose = strip_end_token(strip_tool_blocks(answer, self.tools.actions)).strip()
            Log.dbg(f"agent: round {rounds} (tools {tool_rounds}/{self.max_rounds}, "
                    f"continues {continues}/{MAX_CONTINUES}): {len(calls)} tool call(s), "
                    f"{len(problems)} problem(s), prose {len(prose)} chars, "
                    f"end token {'yes' if END_TOKEN in answer else 'no'}")

            if not self._streamed:                 # streamed turns already shown
                self._show_meta(body)
                self._show_reasoning(body)
                if prose:
                    self._say(prose)

            if not calls:
                if problems:
                    # The reply meant to call a tool but could not be run. Saying
                    # nothing here would end the turn as if the model were done.
                    note = "; ".join(problems)
                    self._say(Ink.warn("  " + short(note, 200)))
                    correction = (
                        "The client could not run that: " + note +
                        "\nReply with one fenced block, exactly like this:\n"
                        "```tool\n{\"action\": \"read_file\", \"path\": \"app.py\"}\n```"
                    )
                    continues += 1                 # a broken round is still a round
                    if continues > MAX_CONTINUES:
                        note = ("Stopped after {0} continuation(s): the model never said {1}."
                                .format(MAX_CONTINUES, END_TOKEN))
                        self._say(Ink.warn(note))
                        return "\n\n".join(parts) or note
                    Log.dbg("agent: sending a format correction back to the model")
                    body = self._send(correction)
                    continue
                if prose:
                    parts.append(prose)
                if END_TOKEN in answer or not self.require_end:
                    Log.dbg(f"agent: finished after {rounds} round(s)")
                    return "\n\n".join(parts)
                # No tool block and no token: the message was cut off or just
                # unfinished. Ask for the rest instead of ending the turn.
                continues += 1
                if continues > MAX_CONTINUES:
                    note = ("Stopped after {0} continuation(s): the model never said {1}."
                            .format(MAX_CONTINUES, END_TOKEN))
                    self._say(Ink.warn(note))
                    return "\n\n".join(parts) or note
                if not prose:
                    self._say(Ink.warn("  the model returned an empty answer"))
                self._say(Ink.dim("  " + GLYPH_V + " no " + END_TOKEN +
                                  "; asking it to continue"))
                Log.dbg("agent: no end token, sending the continuation prompt")
                body = self._send(CONTINUE_PROMPT)
                continue

            if tool_rounds >= self.max_rounds:
                note = ("Stopped after {0} tool rounds; ask again if you need more."
                        .format(self.max_rounds))
                self._say(Ink.warn(note))
                return "\n\n".join(parts) or note
            tool_rounds += 1

            results, summaries = [], []
            for index, call in enumerate(calls, 1):
                label = self._label(call)
                Log.dbg(f"agent: running [{index}] {call['action']} "
                        f"args={short(json.dumps(call['args'], ensure_ascii=False), 160)}")
                try:
                    summary, payload = self.tools.execute(call["action"], call["args"])
                    self._say(Ink.accent(f"  {GLYPH_TOOL} {label}") + Ink.dim(f"  {summary}"))
                except (ToolError, OSError) as exc:
                    summary, payload = f"{label} failed: {exc}", f"{label} failed: {exc}"
                    self._say(Ink.bad(f"  {GLYPH_TOOL} {label}") + Ink.dim(f"  {exc}"))
                    Log.dbg(f"agent: tool error: {exc}")
                summaries.append(f"[{index}] {summary}")
                results.append(f"[{index}] {summary}\n{payload}")

            joined = "\n\n".join(results)
            if len(joined) > MAX_RESULT_CHARS:
                Log.dbg(f"agent: results {len(joined)} chars, capping at {MAX_RESULT_CHARS}")
                joined = joined[:MAX_RESULT_CHARS] + "\n\n… more results omitted (size cap) …"
            header = ("Note: " + "; ".join(problems) + "\n\n") if problems else ""
            feedback = (
                header + "Tool results:\n\n" + joined +
                "\n\nContinue. Use another `tool` block only if you still need "
                "something, and end with " + END_TOKEN + " when you are finished."
            )
            Log.dbg(f"agent: returning {len(feedback)} chars of tool results to the model")
            body = self._send(feedback)


# ------------------------------------------------------------------ self test

DEMO_MCP_SERVER = r'''"""A tiny stdio MCP server, written by --test to prove the client works."""
import json
import sys

TOOLS = [{
    "name": "echo",
    "description": "Echo text back to the caller",
    "inputSchema": {"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
}, {
    "name": "shout",
    "description": "Echo text back in upper case",
    "inputSchema": {"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
}]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        message = json.loads(line)
    except ValueError:
        continue
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                  "serverInfo": {"name": "demo", "version": "1"}}
    elif method == "tools/list":
        # Two pages, so the client has to follow nextCursor to see `shout`.
        cursor = (message.get("params") or {}).get("cursor")
        if cursor == "page-2":
            result = {"tools": TOOLS[1:]}
        else:
            result = {"tools": TOOLS[:1], "nextCursor": "page-2"}
    elif method == "tools/call":
        params = message.get("params") or {}
        arguments = params.get("arguments") or {}
        text = str(arguments.get("text", ""))
        if params.get("name") == "shout":
            text = text.upper()
        result = {"content": [{"type": "text", "text": "echo:" + text}],
                  "isError": False}
    elif method == "shutdown":
        result = {}
    elif method == "exit":
        break
    else:
        result = None
    if request_id is not None and result is not None:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id,
                                     "result": result}) + "\n")
        sys.stdout.flush()
'''


class DemoHttpMcp:
    """A tiny Streamable HTTP MCP server, written by --test to prove the client.

    It insists on the SSE Accept header and on the session id after initialize,
    serves tools/list in two pages, and sends a progress notification before the
    tools/call result, so the client must pick the reply with its own id.
    """

    PROTOCOL = "2024-11-05"
    TOOLS = [
        {"name": "echo", "description": "Echo text back to the caller",
         "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
        {"name": "shout", "description": "Echo text back in upper case",
         "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    ]

    def __init__(self, port: int = 0):
        self.session_id = "demo-session"
        self.got_session = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):        # keep the test output clean
                pass

            def _json(self, code: int, body: dict, headers: dict | None = None) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            def _sse(self, messages: list[dict]) -> None:
                data = "".join("event: message\ndata: " + json.dumps(item) + "\n\n"
                               for item in messages).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_DELETE(self):
                outer.session_id = None
                self.send_response(204)
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    message = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    return self._json(400, {"error": "bad json"})
                if "text/event-stream" not in (self.headers.get("Accept") or ""):
                    return self._json(406, {"error": "Accept must allow text/event-stream"})
                method = message.get("method")
                request_id = message.get("id")
                if method != "initialize":
                    if self.headers.get("Mcp-Session-Id") != outer.session_id:
                        return self._json(400, {"error": "bad or missing session"})
                    outer.got_session += 1
                if method == "initialize":
                    outer.session_id = "demo-session"      # a fresh session per initialize
                    result = {"protocolVersion": outer.PROTOCOL,
                              "capabilities": {"tools": {}},
                              "serverInfo": {"name": "http-demo", "version": "1"}}
                    return self._json(200, {"jsonrpc": "2.0", "id": request_id,
                                            "result": result},
                                      {"Mcp-Session-Id": outer.session_id})
                if method == "notifications/initialized":
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if method == "tools/list":
                    cursor = (message.get("params") or {}).get("cursor")
                    if cursor == "page-2":
                        result = {"tools": outer.TOOLS[1:]}
                    else:
                        result = {"tools": outer.TOOLS[:1], "nextCursor": "page-2"}
                    return self._sse([{"jsonrpc": "2.0", "id": request_id,
                                       "result": result}])
                if method == "tools/call":
                    params = message.get("params") or {}
                    text = str((params.get("arguments") or {}).get("text", ""))
                    if params.get("name") == "shout":
                        text = text.upper()
                    return self._sse([
                        {"jsonrpc": "2.0", "method": "notifications/progress",
                         "params": {"progress": 1}},
                        {"jsonrpc": "2.0", "id": request_id,
                         "result": {"content": [{"type": "text",
                                                   "text": "echo:" + text}],
                                    "isError": False}},
                    ])
                return self._json(200, {"jsonrpc": "2.0", "id": request_id,
                                        "error": {"code": -32601,
                                                  "message": "unknown method"}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> int:
        self.thread.start()
        return self.port

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class _ScriptedApi:
    """A stand-in backend for the end-token check: it replays canned replies."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.sent: list[str] = []

    def call(self, method: str, path: str, payload: dict | None = None,
             timeout: float = 30, **_kwargs):
        if path == "/send_message":
            self.sent.append(str((payload or {}).get("message") or ""))
            answer = self.replies.pop(0) if self.replies else ""
            return 200, {"ok": True, "answer": answer, "chat_id": "test"}
        return 200, {"ok": True}

    def call_stream(self, path: str, payload: dict, on_event=None, timeout: float = 400):
        return self.call("POST", path, payload)


class SelfTest:
    """Exercises every moving part; each check prints its debug lines."""

    def __init__(self, api_port: int, offline: bool = False):
        self.api_port = api_port
        self.offline = offline
        self.api: Backend | None = None
        self._backend: tuple[bool | None, str] | None = None
        self.results: list[tuple[str, str, str]] = []
        self.dirs: list[Path] = []
        self.last_chat: str | None = None

    # -- helpers ----------------------------------------------------------
    def tempdir(self) -> Path:
        path = Path(tempfile.mkdtemp(prefix="ds_selftest_"))
        self.dirs.append(path)
        Log.dbg(f"test: temp dir {path}")
        return path

    def cleanup(self) -> None:
        for path in self.dirs:
            shutil.rmtree(path, ignore_errors=True)

    def ensure_backend(self) -> tuple[bool | None, str]:
        """Start or attach to the backend once; remember the outcome."""
        if self._backend is not None:
            return self._backend
        if self.offline:
            self._backend = (None, "--offline was given")
            return self._backend
        self.api = Backend(self.api_port, allow_spawn=True, verbose=Log.enabled)
        state = self.api.ensure()
        Log.dbg(f"test: backend state '{state}'")
        if not self.api.wait_ready(120):
            tail = "\n    ".join(self.api.log_tail(10))
            self._backend = (None, f"the backend never became ready; server.log tail:\n    {tail}")
        elif state == "attached":
            self._backend = (True, "attached to a backend that was already running")
        else:
            self._backend = (True, f"started by this run (pid {self.api.proc.pid if self.api.proc else '?'})")
        return self._backend

    # -- reporting --------------------------------------------------------
    def run(self) -> int:
        checks = [
            ("python", self.check_python, False),
            ("console", self.check_console, False),
            ("node", self.check_node, False),
            ("playwriter", self.check_playwriter, False),
            ("argv-limit", self.check_argv_limit, False),
            ("argv-overflow", self.check_argv_overflow, False),
            ("tools-list", self.check_tools_list, False),
            ("tools-read", self.check_tools_read, False),
            ("read-range", self.check_tools_read_range, False),
            ("tools-write", self.check_tools_write, False),
            ("write-lines", self.check_tools_write_lines, False),
            ("skills", self.check_skills, False),
            ("tools-escape", self.check_tools_escape, False),
            ("tools-binary", self.check_tools_binary, False),
            ("read-limit", self.check_read_limit, False),
            ("tools-stat", self.check_tools_stat, False),
            ("tools-edit", self.check_tools_edit, False),
            ("tools-find", self.check_tools_find, False),
            ("tools-search", self.check_tools_search, False),
            ("tools-move", self.check_tools_move, False),
            ("tools-delete", self.check_tools_delete, False),
            ("tools-run", self.check_tools_run, False),
            ("url-guard", self.check_url_guard, False),
            ("html-text", self.check_html_text, False),
            ("mcp", self.check_mcp, False),
            ("mcp-http", self.check_mcp_http, False),
            ("parse-blocks", self.check_parse_blocks, False),
            ("parse-problems", self.check_parse_problems, False),
            ("strip-blocks", self.check_strip_blocks, False),
            ("end-token", self.check_end_token, False),
            ("rate-limit", self.check_rate_limit, False),
            ("rate-retry", self.check_rate_retry, False),
            ("prompt-size", self.check_prompt_size, False),
            ("web-search", self.check_web_search, True),
            ("backend", self.check_backend, True),
            ("new-chat", self.check_new_chat, True),
            ("send", self.check_send, True),
            ("send-large", self.check_send_large, True),
            ("tool-loop", self.check_tool_loop, True),
            ("history", self.check_history, True),
        ]
        passed = failed = skipped = 0
        for index, (name, fn, live) in enumerate(checks, 1):
            tag = Ink.dim("live  ") if live else Ink.dim("offline")
            label = f"[{index:2d}/{len(checks)}] {name:<14}"
            started = time.time()
            try:
                ok, detail = fn()
            except Exception as exc:                       # a check must never kill the run
                ok, detail = False, f"{type(exc).__name__}: {exc}"
                Log.dbg("test: traceback\n" + traceback.format_exc())
            elapsed = time.time() - started
            if ok is True:
                mark, passed = Ink.ok(f"{GLYPH_OK} PASS"), passed + 1
            elif ok is None:
                mark, skipped = Ink.warn(f"{GLYPH_SKIP} SKIP"), skipped + 1
            else:
                mark, failed = Ink.bad(f"{GLYPH_BAD} FAIL"), failed + 1
            print(f"{label} {tag} {mark} {Ink.dim('%.2fs' % elapsed)}")
            print(f"{'':>22}{Ink.dim(short(detail, 220))}")
            self.results.append((name, "PASS" if ok is True else "SKIP" if ok is None else "FAIL", detail))
        return self.finish(passed, failed, skipped)

    def finish(self, passed: int, failed: int, skipped: int) -> int:
        self.cleanup()
        print(rule())
        summary = f"self test: {passed} passed, {failed} failed, {skipped} skipped"
        print(Ink.bad(summary) if failed else Ink.ok(summary))
        if skipped and not self.offline:
            print(Ink.dim("skipped checks needed the browser session or the network; run "
                          "`python deepseek.py --verbose` for the browser side"))
        if failed:
            print(Ink.dim("failed:"))
            for name, status, detail in self.results:
                if status == "FAIL":
                    print(Ink.bad(f"  {name}: ") + short(detail, 300))
            if self.api is not None:
                print(Ink.dim("server.log tail:"))
                for line in self.api.log_tail(15):
                    print(Ink.dim("  " + line))
        return 1 if failed else 0

    # -- offline checks ---------------------------------------------------
    def check_python(self):
        info = (f"python {sys.version.split()[0]} on {sys.platform}, "
                f"{sys.executable}, cwd {os.getcwd()}")
        Log.dbg("test: " + info)
        return True, info

    def check_console(self):
        detail = (f"encoding {sys.stdout.encoding}, glyphs "
                  f"{'unicode' if UNICODE else 'ascii fallback'}, colour "
                  f"{'on' if Ink.on else 'off (not a tty)'}")
        Log.dbg("test: " + detail)
        try:
            probe = "".join([RULE_CH, GLYPH_TOOL, GLYPH_PROMPT, GLYPH_OK])
            probe.encode(sys.stdout.encoding or "ascii")
        except (UnicodeEncodeError, LookupError) as exc:
            return False, f"cannot print the interface glyphs: {exc}"
        return True, detail

    def check_node(self):
        path = find_node()
        if not path:
            return None, "no Node.js found; browser checks will be skipped"
        proc = subprocess.run([path, "--version"], capture_output=True,
                              text=True, timeout=60, errors="replace")
        ok = proc.returncode == 0 and proc.stdout.strip().startswith("v")
        return ok, f"{path} -> {short((proc.stdout or proc.stderr).strip(), 60)} (rc {proc.returncode})"

    def check_playwriter(self):
        pw = Playwriter()
        cli = find_playwriter_cli()
        if not cli:
            return False, "playwriter's cli.js was not found in any npx cache or global install"
        proc = subprocess.run([*pw._resolve(), "--version"], capture_output=True,
                              text=True, timeout=90, errors="replace")
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and "playwriter" in out
        return ok, f"{pw.describe()} -> {short(out.strip(), 90)} (rc {proc.returncode})"

    def check_argv_limit(self):
        """The bug that started all of this: a big message must not use argv."""
        pw = Playwriter()
        try:
            pw._resolve()
        except PlaywriterError as exc:
            return None, f"no Playwriter command resolved: {short(exc, 120)}"
        big = ("line of a large file\n" * 14_000)[:300_000]
        code, files = pw.prepare(JS_SEND, {"message": big, "chat_id": None, "budget": 1000})
        try:
            argv = pw.argv_for(code)
            total = Playwriter.argv_size(argv)
            payload_size = os.path.getsize(files[1])
            problems = []
            if "@@PAYLOAD_FILE@@" in code or "@@OUT@@" in code:
                problems.append("a placeholder was left unsubstituted")
            if payload_size < len(big.encode("utf-8")):
                problems.append(f"payload file is short ({payload_size} < {len(big)})")
            if os.name == "nt" and total > WINDOWS_ARGV_LIMIT:
                problems.append(f"argv {total} chars exceeds the limit")
            if problems:
                return False, "; ".join(problems)
            return True, (f"{len(big)}-char message -> payload file {payload_size}B, "
                          f"argv {total} chars (limit {WINDOWS_ARGV_LIMIT})")
        finally:
            for path in files:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def check_argv_overflow(self):
        """An oversized argv must fail with a message that names the real cause."""
        if os.name != "nt":
            return None, "the 32767-character limit only applies on Windows"
        pw = Playwriter(session="selftest")
        try:
            pw._run(["-e", "x" * 40_000], timeout=10)
        except PlaywriterError as exc:
            text = str(exc)
            ok = ("command line" in text and str(WINDOWS_ARGV_LIMIT) in text
                  and "refusing to launch" in text)
            return ok, short(text.splitlines()[0], 160)
        return False, "a 40,000-character command line was not refused"

    def check_tools_list(self):
        root = self.tempdir()
        (root / "a.txt").write_text("a", "utf-8")
        (root / "sub").mkdir()
        summary, payload = Tools(root).list_dir({"path": "."})
        ok = "a.txt" in payload and "sub/" in payload and "2 entries" in summary
        return ok, f"{summary}; payload has both entries: {ok}"

    def check_tools_read(self):
        root = self.tempdir()
        (root / "note.md").write_text("hello from the test\n", "utf-8")
        summary, payload = Tools(root).read_file({"path": "note.md"})
        ok = "hello from the test" in payload and "end of note.md" in payload
        return ok, summary

    def check_tools_write(self):
        root = self.tempdir()
        summary, _ = Tools(root).write_file({"path": "deep/out.txt", "content": "written"})
        target = root / "deep" / "out.txt"
        ok = target.is_file() and target.read_text("utf-8") == "written"
        return ok, f"{summary}; on disk: {target.exists()}"

    def check_tools_write_lines(self):
        """Replace and insert line ranges, keeping the file's line endings."""
        root = self.tempdir()
        tools = Tools(root)
        path = root / "lines.txt"
        path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
        problems: list[str] = []
        tools.write_lines({"path": "lines.txt", "start": 2, "end": 3,
                           "content": "TWO\nTHREE"})
        after_replace = path.read_text(encoding="utf-8")
        if after_replace != "one\nTWO\nTHREE\nfour\nfive\n":
            problems.append(f"replace -> {after_replace!r}")
        tools.write_lines({"path": "lines.txt", "start": 5, "content": "inserted"})
        after_insert = path.read_text(encoding="utf-8")
        if after_insert != "one\nTWO\nTHREE\nfour\ninserted\nfive\n":
            problems.append(f"insert -> {after_insert!r}")
        crlf = root / "crlf.txt"
        crlf.write_bytes(b"a\r\nb\r\nc\r\n")
        tools.write_lines({"path": "crlf.txt", "start": 2, "end": 2, "content": "B"})
        if crlf.read_bytes() != b"a\r\nB\r\nc\r\n":
            problems.append(f"crlf -> {crlf.read_bytes()!r}")
        bad = [{"path": "lines.txt", "start": 0, "content": "x"},
               {"path": "lines.txt", "start": 99, "content": "x"},
               {"path": "lines.txt", "start": 3, "end": 2, "content": "x"},
               {"path": "missing.txt", "start": 1, "content": "x"}]
        for spec in bad:
            try:
                tools.write_lines(spec)
                problems.append(f"accepted bad spec {spec}")
            except ToolError:
                pass
        return (not problems), ("; ".join(problems) if problems else
                                f"replace, insert, CRLF preserved and {len(bad)} bad ranges refused")

    def check_rate_limit(self):
        """Only refusals that never reached the model are worth retrying."""
        cases = [
            (429, {"error": "rate limit reached"}, True),
            (502, {"error": "send_blocked"}, True),
            (503, {"error": "the DeepSeek backend is not ready yet"}, True),
            (502, {"error": "timed_out_waiting_for_answer", "timeout": True}, False),
            (502, {"error": "not_logged_in"}, False),
            (502, {"error": "the stream ended without a result"}, False),
        ]
        wrong = []
        for status, body, expected in cases:
            delay = Agent._retry_delay(status, body, 0)
            if (delay is not None) != expected:
                wrong.append(f"{body['error']}: retry={delay}")
        if not looks_rate_limited("Too many requests, try again later"):
            wrong.append("classifier missed a throttle")
        if looks_rate_limited("here is your answer"):
            wrong.append("classifier false positive")
        delay = Agent._retry_delay(429, {"error": "rate limit", "retry_after": 12}, 0)
        if delay is None or not (12 <= delay <= RETRY_CAP):
            wrong.append(f"retry_after ignored: {delay}")
        return (not wrong), ("; ".join(wrong) if wrong else
                             f"{len(cases)} failure kinds classified, retry_after honoured")

    def check_rate_retry(self):
        """A 429 that never reached the model is retried; a timeout is not."""
        class RateApi:
            def __init__(self, status: int, body: dict):
                self.status, self.body, self.calls = status, body, 0

            def call(self, method, path, payload=None, timeout=30, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return self.status, dict(self.body)
                return 200, {"ok": True, "answer": "done " + END_TOKEN, "chat_id": "t"}

            def call_stream(self, path, payload, on_event=None, timeout=400):
                return self.call("POST", path, payload)

        limited = RateApi(429, {"ok": False, "error": "rate limit reached",
                                "rate_limited": True, "retry_after": 0.2})
        first = Agent(limited, self.tempdir(), quiet=True, stream=False)
        answer = first.ask("hi")
        retried = limited.calls == 2 and "done" in answer

        timed_out = RateApi(502, {"ok": False, "error": "timed_out_waiting_for_answer",
                                  "timeout": True})
        second = Agent(timed_out, self.tempdir(), quiet=True, stream=False)
        try:
            second.ask("hi")
            gave_up = False
        except RuntimeError:
            gave_up = timed_out.calls == 1
        ok = retried and gave_up
        return ok, (f"429 retried ({limited.calls} calls, answer {short(answer, 20)!r}); "
                    f"timeout not retried ({timed_out.calls} call)")

    def check_tools_escape(self):
        root = self.tempdir()
        tools = Tools(root)
        attempts = ["../outside.txt", "sub/../../escape.txt",
                    str(Path.home() / "boom.txt"), "/etc/passwd",
                    "C:\\Windows\\win.ini"]
        blocked = []
        for attempt in attempts:
            try:
                tools.resolve(attempt)
            except ToolError:
                blocked.append(attempt)
        ok = len(blocked) == len(attempts)
        return ok, f"{len(blocked)}/{len(attempts)} refused: {', '.join(blocked)}"

    def check_tools_read_range(self):
        root = self.tempdir()
        body = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"
        (root / "ten.txt").write_text(body, "utf-8")
        tools = Tools(root)
        summary, payload = tools.read_file({"path": "ten.txt", "start": 3, "end": 5})
        sliced = ("(lines 3-5/10)" in payload and "line3" in payload and "line5" in payload
                  and "line2" not in payload and "line6" not in payload)
        tail_summary, tail = tools.read_file({"path": "ten.txt", "from": 8})
        tail_ok = "line8" in tail and "line10" in tail and "line7" not in tail
        (root / "wide.txt").write_text("w" * (MAX_READ + 5_000) + "\ntail\n", "utf-8")
        oversized = False
        try:
            tools.read_file({"path": "wide.txt", "start": 1, "end": 2})
        except ToolError as exc:
            oversized = "blocked" in str(exc)
        refused = False
        try:
            tools.read_file({"path": "ten.txt", "start": 7, "end": 3})
        except ToolError:
            refused = True
        ok = sliced and tail_ok and oversized and refused
        return ok, (f"{summary}; alias {tail_summary}; "
                    f"oversized slice blocked: {oversized}; backwards range refused: {refused}")

    def check_tools_binary(self):
        root = self.tempdir()
        (root / "blob.bin").write_bytes(b"\x00\x01\x02\x03" * 100)
        try:
            Tools(root).read_file({"path": "blob.bin"})
        except ToolError as exc:
            return True, f"refused: {exc}"
        return False, "a binary file was handed to the model"

    def check_read_limit(self):
        """Over budget is blocked with a way forward; exactly at the limit passes."""
        root = self.tempdir()
        tools = Tools(root)
        (root / "big.txt").write_text(
            " ".join(f"w{i}" for i in range(MAX_WORDS + 800)), "utf-8")
        blocked, message = False, ""
        try:
            tools.read_file({"path": "big.txt"})
        except ToolError as exc:
            blocked, message = True, str(exc)
        (root / "edge.txt").write_text(
            " ".join(f"w{i}" for i in range(MAX_WORDS)), "utf-8")
        edge_summary, _ = tools.read_file({"path": "edge.txt"})
        (root / "lines.txt").write_text(
            "\n".join("word " * 40 for _ in range(200)), "utf-8")
        slice_summary, _ = tools.read_file({"path": "lines.txt", "start": 1, "end": 20})
        ok = (blocked and "blocked" in message and "start=" in message
              and edge_summary.startswith("read_file edge.txt")
              and slice_summary.startswith("read_file lines.txt lines 1-20"))
        return ok, f"blocked: {short(message, 150)} | at limit: ok | slice: {slice_summary}"

    def check_tools_stat(self):
        root = self.tempdir()
        (root / "s.py").write_text("one\ntwo\n", "utf-8")
        summary, payload = Tools(root).stat({"path": "s.py"})
        ok = "text file" in payload and "lines: 2" in payload and "size:" in payload
        return ok, f"{summary}; {short(payload.replace(chr(10), ' / '), 110)}"

    def check_tools_edit(self):
        root = self.tempdir()
        (root / "e.py").write_text("value = 1\nprint(value)\n", "utf-8")
        summary, _ = Tools(root).edit_file(
            {"path": "e.py", "find": "value = 1", "replace": "value = 2"})
        text = (root / "e.py").read_text("utf-8")
        (root / "dup.py").write_text("a\na\n", "utf-8")
        ambiguous = False
        try:
            Tools(root).edit_file({"path": "dup.py", "find": "a", "replace": "b"})
        except ToolError:
            ambiguous = True
        ok = text == "value = 2\nprint(value)\n" and ambiguous
        return ok, f"{summary}; ambiguous edit refused: {ambiguous}"

    def check_tools_find(self):
        root = self.tempdir()
        (root / "pkg").mkdir()
        (root / "pkg" / "mod.py").write_text("x = 1\n", "utf-8")
        (root / "notes.txt").write_text("n\n", "utf-8")
        summary, payload = Tools(root).find_files({"pattern": "**/*.py"})
        expected = str(Path("pkg") / "mod.py")
        ok = expected in payload and "notes.txt" not in payload
        return ok, f"{summary}; {short(payload.replace(chr(10), ', '), 90)}"

    def check_tools_search(self):
        root = self.tempdir()
        (root / "a.py").write_text("alpha\nneedle here\nomega\n", "utf-8")
        (root / "b.bin").write_bytes(b"\x00nee\x00dle")
        summary, payload = Tools(root).search_files({"pattern": "needle"})
        ok = "a.py:2: needle here" in payload and "b.bin" not in payload
        return ok, f"{summary}; {short(payload, 90)}"

    def check_tools_move(self):
        root = self.tempdir()
        (root / "one.txt").write_text("1", "utf-8")
        summary, _ = Tools(root).move({"path": "one.txt", "to": "nested/two.txt"})
        moved = (root / "nested" / "two.txt").is_file() and not (root / "one.txt").exists()
        (root / "three.txt").write_text("3", "utf-8")
        guarded = False
        try:
            Tools(root).move({"path": "nested/two.txt", "to": "three.txt"})
        except ToolError:
            guarded = True
        ok = moved and guarded
        return ok, f"{summary}; overwrite guarded: {guarded}"

    def check_tools_delete(self):
        root = self.tempdir()
        (root / "dead.txt").write_text("x", "utf-8")
        (root / "gone").mkdir()
        (root / "gone" / "inner.txt").write_text("y", "utf-8")
        Tools(root).delete({"path": "dead.txt"})
        Tools(root).delete({"path": "gone"})
        refused = False
        try:
            Tools(root).delete({"path": "."})
        except ToolError:
            refused = True
        ok = (not (root / "dead.txt").exists() and not (root / "gone").exists()
              and refused)
        return ok, f"file and directory removed; deleting the root refused: {refused}"

    def check_tools_run(self):
        root = self.tempdir()
        summary, payload = Tools(root).run_command({"command": "echo tool-run-ok"})
        bad_summary, bad_payload = Tools(root).run_command(
            {"command": 'python -c "import sys; sys.exit(3)"'})
        ok = ("tool-run-ok" in payload and "exit code: 0" in payload
              and "exit code: 3" in bad_payload)
        return ok, f"{summary}; failing command reported as {short(bad_summary, 40)}"

    def check_url_guard(self):
        attempts = ["file:///etc/passwd", "javascript:alert(1)",
                    "ftp://example.com/x", "not-a-url"]
        blocked = 0
        for url in attempts:
            try:
                fetch_page(url, 500)
            except ToolError:
                blocked += 1
        return blocked == len(attempts), f"{blocked}/{len(attempts)} non-http URLs refused"

    def check_html_text(self):
        markup = ("<html><head><title>T</title><style>.x{color:red}</style></head><body>"
                  "<h1>Heading</h1><p>First para with <b>bold</b> text.</p>"
                  "<script>var leak = 'no';</script><ul><li>item</li></ul></body></html>")
        text = html_to_text(markup)
        ok = ("Heading" in text and "First para with bold text." in text
              and "item" in text and "leak" not in text and "color:red" not in text)
        return ok, short(text.replace(chr(10), " / "), 130)

    def check_web_search(self):
        if self.offline:
            return None, "--offline was given"
        try:
            provider, results = search_web("python subprocess TimeoutExpired", 5)
        except Exception as exc:                    # network, captcha, DNS... all skips
            return None, f"no search provider answered: {short(exc, 160)}"
        first = results[0]
        return True, (f"{len(results)} result(s) via {provider}; first "
                      f"{short(first.get('title'), 50)!r} <{short(first.get('url'), 60)}>")

    def check_mcp(self):
        """Start a real stdio server, list its tool, and call it."""
        root = self.tempdir()
        server = root / "demo_mcp.py"
        server.write_text(DEMO_MCP_SERVER, encoding="utf-8")
        config = root / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"demo": {
            "command": sys.executable, "args": [str(server)]}}}), encoding="utf-8")
        registry = McpRegistry.load(config, verbose=Log.enabled)
        if registry is None or not registry.servers:
            return False, f"the registry did not start: {registry.errors if registry else 'none'}"
        try:
            actions = registry.actions()
            # `shout` is on the second tools/list page, so seeing it proves the
            # client followed nextCursor instead of stopping at page one.
            if "demo__echo" not in actions or "demo__shout" not in actions:
                return False, f"expected both tools across two pages, got {actions}"
            tools = Tools(root, registry)
            if "demo__echo" not in tools.actions:
                return False, "the tool layer did not expose the MCP action"
            summary, payload = tools.execute("demo__echo", {"arguments": {"text": "hi"}})
            calls = extract_tool_calls('```tool\n{"action": "demo__echo", "text": "x"}\n```',
                                       None, tools.actions)
            # Kill the server mid-session: the next call must restart it once.
            server = registry.servers[0]
            started = time.time()
            server.proc.kill()
            server.proc.wait(5)
            summary2, payload2, failed2 = registry.call("demo__shout", {"text": "hi"})
            revived = (not failed2) and "echo:HI" in payload2
            ok = "echo:hi" in payload and len(calls) == 1 and revived
            return ok, (f"{summary}; two pages; {summary2} "
                        f"(restart in {time.time() - started:.1f}s): {revived}")
        finally:
            registry.close()

    def check_mcp_http(self):
        """Streamable HTTP: sessions, SSE replies, interleaved progress, pages."""
        demo = DemoHttpMcp()
        demo.start()
        root = self.tempdir()
        config = root / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"web": {
            "url": f"http://127.0.0.1:{demo.port}/mcp"}}}), encoding="utf-8")
        try:
            registry = McpRegistry.load(config, verbose=Log.enabled)
            if registry is None or not registry.servers:
                return False, f"no HTTP server started: {registry.errors if registry else 'none'}"
            try:
                actions = registry.actions()
                if "web__echo" not in actions or "web__shout" not in actions:
                    return False, f"expected both tools across two SSE pages, got {actions}"
                summary, text, is_error = registry.call("web__echo", {"text": "hi"})
                ok = (not is_error) and "echo:hi" in text
                return ok, (f"{summary}; session echoed on {demo.got_session} request(s), "
                            f"two SSE pages, progress notification skipped: {ok}")
            finally:
                registry.close()
        finally:
            demo.stop()

    def check_skills(self):
        """skills/*.txt is listed by name and loaded only when asked for."""
        root = self.tempdir()
        folder = root / "skills"
        folder.mkdir()
        (folder / "deploy.txt").write_text(
            "# Deploy\nRun the deploy script.\nStep 1: build.", encoding="utf-8")
        (folder / "review.md").write_text(
            "Review code for bugs.\nBe strict.", encoding="utf-8")
        (folder / "notes.json").write_text("{}", encoding="utf-8")
        tools = Tools(root)
        problems = []
        names = sorted(tools.skills.names())
        # Other folders may hold real skills, so check for these two, not equality.
        for expected in ("deploy", "review"):
            if expected not in names:
                problems.append(f"missing {expected} in {names}")
        if "notes" in names:
            problems.append("a non-text file was loaded as a skill")
        if tools.skills.items["deploy"]["description"] != "Deploy":
            problems.append("description")
        summary, body = tools.execute("skill", {"name": "deploy"})
        if "Step 1: build" not in body or "Run the deploy script" not in body:
            problems.append("skill body")
        _summary, listing = tools.execute("skill", {})
        if "review" not in listing or "Review code for bugs" not in listing:
            problems.append("listing")
        for name in ("nope", "../deploy", "skills/deploy"):
            try:
                tools.execute("skill", {"name": name})
                problems.append(f"accepted {name!r}")
            except ToolError:
                pass
        if not tools.execute("skill", {"name": "Deploy"})[1].startswith("--- skill: deploy"):
            problems.append("case-insensitive lookup")
        return (not problems), ("; ".join(problems) if problems else
                                f"{summary}; {len(names)} skills listed, bodies loaded on demand")

    def check_parse_blocks(self):
        cases = {
            "tool fence": ('```tool\n{"action": "read_file", "path": "a.py"}\n```', 1),
            "json fence": ('```json\n{"action": "list_dir", "path": "."}\n```', 1),
            "nested args": ('```tool\n{"action": "write_file", "args": {"path": "b", "content": "x"}}\n```', 1),
            "unknown action": ('```tool\n{"action": "delete_file", "path": "a"}\n```', 0),
            "plain code": ('```python\nprint("not a call")\n```', 0),
            "broken json": ('```tool\n{not json}\n```', 0),
            "two calls": ('```tool\n{"action": "list_dir", "path": "."}\n```\n'
                          '```tool\n{"action": "read_file", "path": "x"}\n```', 2),
            "tag block": ('<tool>{"action": "list_dir", "path": "."}</tool>', 1),
            "tag arg block": ('<tool_call>{"action": "search_files", "pattern": "x"}'
                              '</tool_call>', 1),
        }
        wrong = []
        for label, (text, expected) in cases.items():
            got = len(extract_tool_calls(text))
            Log.dbg(f"test: {label}: {got} call(s), expected {expected}")
            if got != expected:
                wrong.append(f"{label} ({got} != {expected})")
        return (not wrong), (f"all {len(cases)} parse cases correct" if not wrong
                             else "; ".join(wrong))

    def check_parse_problems(self):
        """A tool block that cannot run must be reported, not silently dropped."""
        cases = {
            "unknown action": '```tool\n{"action": "edit", "path": "a"}\n```',
            "broken json": '```tool\n{"action": read_file}\n```',
            "tag with garbage": '<tool>run the tests</tool>',
        }
        good = '```tool\n{"action": "list_dir"}\n```'
        wrong = []
        for label, text in cases.items():
            problems: list[str] = []
            extract_tool_calls(text, problems)
            Log.dbg(f"test: problem case {label}: {problems}")
            if not problems:
                wrong.append(label)
        problems = []
        extract_tool_calls(good, problems)
        if problems:
            wrong.append(f"false positive on a valid block: {problems}")
        return (not wrong), (f"{len(cases)} broken blocks reported" if not wrong
                             else "; ".join(wrong))

    def check_strip_blocks(self):
        text = ("Reading now.\n```tool\n{\"action\": \"read_file\", \"path\": \"a.py\"}\n```\n"
                "Here is code:\n```python\nprint(1)\n```\n")
        stripped = strip_tool_blocks(text)
        ok = ("read_file" not in stripped and "print(1)" in stripped
              and "Here is code" in stripped)
        return ok, f"kept the ordinary block, dropped the control block: {ok}"

    def check_end_token(self):
        """No token means another round; the token itself is never printed."""
        api = _ScriptedApi(["First half of the answer.",
                            "Second half, now finished. " + END_TOKEN])
        agent = Agent(api, self.tempdir(), quiet=True, stream=False)
        answer = agent.ask("hi")
        continued = len(api.sent) == 2 and api.sent[1] == CONTINUE_PROMPT
        Log.dbg(f"test: sent messages {[short(m, 40) for m in api.sent]}")

        shown: list[str] = []
        stream = AnswerStream(shown.append, Tools.ACTIONS)
        stream.feed("done " + END_TOKEN[:5])      # the token split across chunks
        stream.feed(END_TOKEN[5:])
        stream.finish()
        printed = "".join(shown)

        # Continues have their own budget (100), not the tool-round one: with
        # max_rounds=2 the old loop would have stopped after two replies.
        api2 = _ScriptedApi(["part one.", "part two.", "part three.",
                             "the end " + END_TOKEN])
        agent2 = Agent(api2, self.tempdir(), quiet=True, stream=False, max_rounds=2)
        continued_lots = agent2.ask("hi")
        beyond_tool_budget = (len(api2.sent) == 4 and "part three" in continued_lots)

        ok = (continued and beyond_tool_budget
              and "First half" in answer and "Second half" in answer
              and END_TOKEN not in answer and END_TOKEN not in printed
              and printed.strip() == "done")
        return ok, (f"no token -> continuation prompt sent: {continued}; "
                    f"3 continues past a 2-round budget: {beyond_tool_budget}; "
                    f"answer {short(answer, 50)!r}; printed {short(printed, 40)!r}")

    def check_prompt_size(self):
        guide = TOOL_GUIDE.format(root="C:\\some\\working\\directory", words=MAX_WORDS,
                                  chars=MAX_READ, skills="", mcp_tools="")
        loop = ("Tool results:\n\n[1] read_file x -> 10 B\n--- start of x ---\n0\n--- end of x ---"
                "\n\nContinue. Use another `tool` block only if you still need "
                "something, and end with " + END_TOKEN + " when you are finished.")
        ok = len(guide) < 4_500 and len(loop) < 500
        return ok, f"tool guide {len(guide)} chars, per-round wrapper {len(loop)} chars"

    # -- live checks ------------------------------------------------------
    def check_backend(self):
        ok, detail = self.ensure_backend()
        if ok is None:
            return None, detail
        Log.dbg(f"test: backend pid {self.api.proc.pid if self.api.proc else None}, "
                f"spawned_by_us={self.api.spawned_by_us}")
        return True, detail

    def _agent(self, root: Path | None = None) -> Agent:
        # These checks call ask() and expect it to return after one reply, so the
        # end-token rule is off here; check_end_token covers it on its own.
        return Agent(self.api, root or self.tempdir(), quiet=True, require_end=False)

    def check_new_chat(self):
        ok, detail = self.ensure_backend()
        if ok is None:
            return None, detail
        body = self.api.call("POST", "/new_chat", {})[1]
        if not body.get("ok"):
            return None, f"the backend answered: {short(body.get('error'), 160)}"
        return True, f"thinking toggle: {body.get('thinking')}, url {body.get('url')}"

    def check_send(self):
        ok, detail = self.ensure_backend()
        if ok is None:
            return None, detail
        agent = self._agent()
        agent.new_chat()
        agent.prime()
        prompt = ("A bat and a ball cost $1.10 together, and the bat costs $1.00 more "
                  "than the ball. Work it out step by step, then state the ball's price.")
        attempts = []
        for attempt in (1, 2):
            # DeepSeek decides per turn whether to show reasoning at all, so a
            # single answer without a THINK fragment is not a client failure.
            ask = prompt if attempt == 1 else (
                "Reason it through in your thinking before answering: " + prompt)
            body = agent._send(ask)
            types = [t.upper() for t in (body.get("fragment_types") or []) if t]
            reasoning = body.get("reasoning") or ""
            answer = body.get("answer") or ""
            self.last_chat = body.get("chat_id") or self.last_chat
            attempts.append(f"#{attempt}: fragments {types}, reasoning {len(reasoning)} chars, "
                            f"answer {short(answer, 40)!r}")
            Log.dbg(f"test: send attempt {attempt}: {attempts[-1]}")
            if answer.strip() and "THINK" in types and reasoning.strip():
                return True, (f"{len(reasoning)} chars of reasoning "
                              f"({body.get('reasoning_seconds')}s), answer {short(answer, 60)!r}, "
                              f"source {body.get('source')}; " + "; ".join(attempts))
        return False, "; ".join(attempts) or "no answer"

    def check_send_large(self):
        """The exact case that used to die with WinError 206."""
        ok, detail = self.ensure_backend()
        if ok is None:
            return None, detail
        agent = self._agent()
        dump = "\n".join(f"line {i}: a moderately long line of file content" for i in range(700))
        message = ("Tool results:\n\n[1] read_file big.py -> 40 KB\n"
                   f"--- start of big.py ---\n{dump}\n--- end of big.py ---\n\n"
                   "Reply with the single word: received")
        body = agent._send(message)
        answer = (body.get("answer") or "").strip()
        ok = bool(answer)
        return ok, (f"sent {len(message)} chars (the old failure was ~27 KB, limit "
                    f"{WINDOWS_ARGV_LIMIT}); answer {short(answer, 60)!r}")

    def check_tool_loop(self):
        """End to end: the model reads a file and writes another one."""
        ok, detail = self.ensure_backend()
        if ok is None:
            return None, detail
        root = self.tempdir()
        (root / "secret.txt").write_text("4271", "utf-8")
        agent = self._agent(root)
        agent.new_chat()
        agent.prime()
        agent.ask("Read secret.txt, then write its exact contents to answer.txt. "
                  "Answer with just the word done.")
        target = root / "answer.txt"
        if not target.is_file():
            return False, f"answer.txt was never written; dir contains {sorted(p.name for p in root.iterdir())}"
        content = target.read_text("utf-8").strip()
        return content == "4271", f"answer.txt contains {content!r} (expected '4271')"

    def check_history(self):
        ok, detail = self.ensure_backend()
        if ok is None:
            return None, detail
        if not self.last_chat:
            return None, "no chat id from the send check"
        status, body = self.api.call("GET", f"/history?q={self.last_chat}")
        turns = body.get("turns") or []
        with_reasoning = [t for t in turns if (t.get("reasoning") or "").strip()]
        ok = status == 200 and bool(body.get("ok")) and bool(turns) and bool(with_reasoning)
        return ok, (f"HTTP {status}, {len(turns)} turn(s), {len(with_reasoning)} with reasoning, "
                    f"title {short(body.get('title'), 30)!r}")


# ---------------------------------------------------------------------- banner

def banner(root: Path, ready: bool) -> None:
    print()
    print(Ink.bold("DeepSeek") + Ink.dim(f"  terminal{SEP}file access"))
    print(Ink.dim(f"directory  {root}"))
    print(Ink.dim("commands   /new  /dir  /think  /tools  /mcp  /mcp-add  /help  /quit"))
    print(rule())
    if not ready:
        print(Ink.warn("waiting for the DeepSeek backend (log in to chat.deepseek.com in your browser)"))


# ------------------------------------------------------------------------ main

def open_mcp(args, root: Path) -> McpRegistry | None:
    """Start the configured MCP servers, if any. A broken server never blocks."""
    if args.no_mcp:
        return None
    config = Path(args.mcp_config).expanduser() if args.mcp_config else root / "mcp.json"
    if not args.mcp_config and not config.is_file():
        Log.dbg(f"mcp: no config at {config}")
        return None
    print(Ink.dim(f"[mcp] loading {config}"))
    timeout = getattr(args, "mcp_timeout", None) or START_TIMEOUT
    registry = McpRegistry.load(config, verbose=Log.enabled, start_timeout=timeout)
    if registry is None:
        return None
    for error in registry.errors:
        print(Ink.warn(f"[mcp] {error}"))
    if registry.servers:
        print(Ink.dim(f"[mcp] {registry.summary()}"))
    if registry.tools:
        print(Ink.dim("[mcp] actions: " + ", ".join(registry.actions())))
    return registry if registry.servers else None


def start_mcp_loading(args, root: Path):
    """Load MCP servers while the browser connects: that wait is already spent.

    Returns a job for join_mcp_loading(), or None when no server is configured.
    """
    if args.no_mcp:
        return None
    config = Path(args.mcp_config).expanduser() if args.mcp_config else root / "mcp.json"
    if not args.mcp_config and not config.is_file():
        return None
    holder: dict = {"registry": None, "error": None}

    def work() -> None:
        try:
            holder["registry"] = open_mcp(args, root)
        except Exception as exc:            # noqa: BLE001 - startup must never die
            holder["error"] = exc

    thread = threading.Thread(target=work, name="mcp-load", daemon=True)
    thread.start()
    return thread, holder


def join_mcp_loading(job) -> McpRegistry | None:
    """Wait for the loader started earlier and hand back the registry."""
    if job is None:
        return None
    thread, holder = job
    thread.join()
    if holder["error"] is not None:
        print(Ink.warn(f"[mcp] could not load: {holder['error']}"))
        return None
    return holder["registry"]


def _mcp_config_path(args, root: Path) -> Path:
    """Where /mcp-add writes: the same file open_mcp() reads."""
    if args.mcp_config:
        return Path(args.mcp_config).expanduser()
    return root / "mcp.json"


def _read_mcp_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError(f"cannot read {path}: {exc}")
    return data if isinstance(data, dict) else {}


def _write_mcp_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def add_mcp_server(agent: Agent, config_path: Path, name: str, command: str,
                   cmd_args: list[str]) -> tuple[str, list[str]]:
    """Add a stdio server to mcp.json, start it, and make its tools callable now."""
    name = name.strip()
    if not name:
        raise ToolError("the server needs a name")
    if agent.mcp is not None and any(s.name == name for s in agent.mcp.servers):
        raise ToolError(f"an MCP server named {name!r} is already loaded")
    config = _read_mcp_config(config_path)
    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ToolError(f"{config_path} has a broken 'mcpServers' entry")
    spec = {"command": command, "args": list(cmd_args)}
    servers[name] = spec
    _write_mcp_config(config_path, config)
    server = McpServer(name, spec, base_dir=config_path.parent, verbose=Log.enabled)
    try:
        tools = server.start()
    except McpError as exc:
        server.close()
        raise ToolError(f"wrote the config, but the server did not start: {exc}")
    if agent.mcp is None:
        agent.mcp = McpRegistry(verbose=Log.enabled)
    taken = {t.action for t in agent.mcp.tools}
    for tool in tools:
        base = tool.action
        suffix = 0
        while tool.action in taken:
            suffix += 1
            tool.action = f"{base}_{suffix}"
        taken.add(tool.action)
    agent.mcp.servers.append(server)
    agent.mcp.tools.extend(tools)
    agent.tools = Tools(agent.root, agent.mcp)
    return f"added {name}: {len(tools)} tool(s)", [t.action for t in tools]


def add_mcp_http_server(agent: Agent, config_path: Path, name: str,
                        url: str) -> tuple[str, list[str]]:
    """Add a Streamable HTTP server to mcp.json and use its tools right away."""
    name = name.strip()
    if not name:
        raise ToolError("the server needs a name")
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ToolError("the url must start with http:// or https://")
    if agent.mcp is not None and any(s.name == name for s in agent.mcp.servers):
        raise ToolError(f"an MCP server named {name!r} is already loaded")
    config = _read_mcp_config(config_path)
    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ToolError(f"{config_path} has a broken 'mcpServers' entry")
    spec = {"url": url}
    servers[name] = spec
    _write_mcp_config(config_path, config)
    server = McpHttpServer(name, spec, base_dir=config_path.parent, verbose=Log.enabled)
    try:
        tools = server.start()
    except McpError as exc:
        server.close()
        raise ToolError(f"wrote the config, but the server did not start: {exc}")
    if agent.mcp is None:
        agent.mcp = McpRegistry(verbose=Log.enabled)
    taken = {t.action for t in agent.mcp.tools}
    for tool in tools:
        base = tool.action
        suffix = 0
        while tool.action in taken:
            suffix += 1
            tool.action = f"{base}_{suffix}"
        taken.add(tool.action)
    agent.mcp.servers.append(server)
    agent.mcp.tools.extend(tools)
    agent.tools = Tools(agent.root, agent.mcp)
    return f"added {name} (HTTP): {len(tools)} tool(s)", [t.action for t in tools]


def _announce_mcp_added(agent: Agent, summary: str, actions: list[str],
                        config_path: Path) -> None:
    """Print what was added and tell the running model about the new actions."""
    print(Ink.dim(f"[mcp] {summary} -> {config_path}"))
    if not actions:
        return
    print(Ink.dim("  " + ", ".join(actions)))
    note = ("New MCP tools were just added to your action list; call them with a "
            "`tool` block exactly like the built-in actions:\n"
            + "\n".join("  " + action for action in actions)
            + "\nReply with one short sentence confirming; do not call anything yet.")
    try:
        agent._send(note)
    except RuntimeError as exc:
        print(Ink.warn(f"added, but could not tell the model about them: {exc}"))


def remove_mcp_server(agent: Agent, config_path: Path, name: str) -> str:
    """Drop a server from mcp.json and from the running session."""
    removed_config = False
    config = _read_mcp_config(config_path)
    servers = config.get("mcpServers")
    if isinstance(servers, dict) and name in servers:
        del servers[name]
        _write_mcp_config(config_path, config)
        removed_config = True
    removed_live = False
    if agent.mcp is not None:
        for server in list(agent.mcp.servers):
            if server.name == name:
                server.close()
                agent.mcp.servers.remove(server)
                removed_live = True
        if removed_live:
            agent.mcp.tools = [t for t in agent.mcp.tools if t.server != name]
            agent.tools = Tools(agent.root, agent.mcp)
    if not (removed_config or removed_live):
        raise ToolError(f"no MCP server named {name!r}")
    return f"removed {name}" + (" (config only)" if removed_config and not removed_live else "")


def choose_directory(initial: str | None) -> Path:
    if initial:
        path = Path(initial).expanduser()
        if not path.is_dir():
            raise SystemExit(f"[dir] not a directory: {path}")
        return path.resolve()
    cwd = Path.cwd()
    try:
        answer = input(f"Working directory (the agent will be confined to it)\n[{cwd}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("[dir] no directory chosen")
    if not answer:
        return cwd.resolve()
    path = Path(answer).expanduser()
    if not path.is_dir():
        raise SystemExit(f"[dir] not a directory: {path}")
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Terminal chat with file access over DeepSeek.",
        epilog="--verbose logs every step; --test runs a full self test with those logs on.",
    )
    parser.add_argument("--dir", help="working directory for the agent (default: ask)")
    parser.add_argument("--api-port", type=int, default=8765, help="deepseek.py port")
    parser.add_argument("--show-thinking", action="store_true", help="print reasoning traces")
    parser.add_argument("--no-stream", action="store_true",
                        help="wait for the whole answer instead of printing it live")
    parser.add_argument("--mcp-config", metavar="PATH",
                        help="MCP servers JSON (default: <working dir>/mcp.json)")
    parser.add_argument("--no-mcp", action="store_true",
                        help="ignore mcp.json and start without MCP tools")
    parser.add_argument("--mcp-timeout", type=float, default=None, metavar="SECONDS",
                        help="startup budget per MCP server (default 25, env MCP_START_TIMEOUT)")
    parser.add_argument("--max-rounds", type=int, default=MAX_TOOL_ROUNDS)
    parser.add_argument("--no-prime", action="store_true",
                        help="skip sending the tool guide (not recommended)")
    parser.add_argument("--prime-only", action="store_true",
                        help="send the tool guide and exit, to check the protocol")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="debug logs on stderr, including the backend's own log lines")
    parser.add_argument("--test", action="store_true",
                        help="exercise every feature, print every debug log, report pass/fail")
    parser.add_argument("--offline", action="store_true",
                        help="with --test: run only the checks that need no browser")
    args = parser.parse_args(argv)

    if args.verbose or args.test:
        Log.enable()
    Log.dbg(f"cli: argv={argv if argv is not None else sys.argv[1:]}")
    Log.dbg(f"cli: verbose={Log.enabled} test={args.test} offline={args.offline}")

    if args.test:
        print()
        print(Ink.bold("DeepSeek CLI") + Ink.dim(f"  self test{SEP}verbose output"))
        print(rule())
        suite = SelfTest(args.api_port, offline=args.offline)
        code = suite.run()
        if suite.api is not None and suite.api.spawned_by_us:
            Log.dbg("test: stopping the backend this run started")
            suite.api.stop()
        return code

    root = choose_directory(args.dir)
    # MCP servers start in the background: connecting the browser and waiting
    # for a login usually take longer, so this load costs no extra startup time.
    mcp_job = start_mcp_loading(args, root)
    api = Backend(args.api_port, verbose=Log.enabled)
    state = api.ensure()
    if state == "spawned":
        print(Ink.dim(f"[backend] starting deepseek.py on port {args.api_port} (log: server.log)"))
    elif state == "attached":
        Log.dbg("cli: attached to a backend that was already running; its verbosity is "
                "whatever it was started with (restart it with --verbose to see its logs)")
    banner(root, ready=(state != "missing"))
    if state == "missing":
        print(Ink.bad(f"nothing is listening on port {args.api_port}"))
        return 1

    mcp = join_mcp_loading(mcp_job)
    agent = Agent(api, root, show_thinking=args.show_thinking, max_rounds=args.max_rounds,
                  stream=not args.no_stream, mcp=mcp)

    # A new process starts a new conversation: without this the guide and the
    # first message would land in whatever chat the browser tab still had open.
    try:
        agent.new_chat()
    except RuntimeError as exc:
        print(Ink.bad(f"could not start a new chat: {exc}"))
        Log.dbg("cli: server.log tail:\n  " + "\n  ".join(api.log_tail(10)))
        if agent.mcp:
            agent.mcp.close()
        return 1

    if not args.no_prime:
        print(Ink.dim("sending the tool guide to the model…"))
        try:
            confirmation = agent.prime()
        except RuntimeError as exc:
            print(Ink.bad(f"could not prime the conversation: {exc}"))
            Log.dbg("cli: server.log tail:\n  " + "\n  ".join(api.log_tail(10)))
            if agent.mcp:
                agent.mcp.close()
            return 1
        if confirmation:
            print(Ink.dim("model: ") + confirmation)
        if args.prime_only:
            if agent.mcp:
                agent.mcp.close()
            return 0
        print()

    while True:
        try:
            print(rule())
            line = input(Ink.accent(f"{GLYPH_PROMPT} "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        print(rule())

        text = line.strip()
        if not text:
            continue
        Log.dbg(f"cli: user input {len(text)} chars")
        if text in ("/quit", "/exit", "/q"):
            break
        if text == "/help":
            print("/new [dir]  start a fresh conversation (re-sends the tool guide)")
            print("/dir [path] show or change the working directory")
            print("/think      toggle printing the reasoning trace")
            print("/tools      list what the agent can do")
            print("/skills     list the skill files the model can load")
            print("/mcp        list the MCP servers and their tools")
            print("/mcp-add-http add a Streamable HTTP MCP server by URL")
            print("/mcp-add    add a stdio MCP server and use its tools now")
            print("/mcp-remove remove an MCP server by name")
            print("/quit       leave")
            continue
        if text == "/tools":
            print("actions: " + ", ".join(agent.tools.actions))
            print(Ink.dim("files stay inside the working directory; run_command is not sandboxed"))
            continue
        if text == "/skills":
            skills = agent.tools.skills
            if not skills.items:
                print(Ink.dim(f"no skills found; put .txt files in {agent.root / 'skills'}"))
            else:
                print(Ink.dim(f"{len(skills.items)} skill(s)"))
                for item in skills.items.values():
                    print(f"  {item['name']}" +
                          (f"  {item['description']}" if item["description"] else ""))
                    print(Ink.dim(f"      {item['path']}"))
            continue
        if text == "/mcp":
            if agent.mcp is None or not agent.mcp.tools:
                print(Ink.dim("no MCP tools loaded (put an mcp.json in the working directory)"))
            else:
                print(Ink.dim(agent.mcp.summary()))
                for tool in agent.mcp.tools:
                    print(f"  {tool.action}" + (f"({tool.signature()})" if tool.signature() else ""))
            continue
        if text.startswith("/mcp-add-http"):
            rest = text[len("/mcp-add-http"):].strip()
            if not rest:
                print("usage: /mcp-add-http <name> <url>")
                print(Ink.dim("example: /mcp-add-http docs https://example.com/mcp"))
                continue
            try:
                parts = shlex.split(rest)
            except ValueError as exc:
                print(Ink.bad(f"cannot parse the url: {exc}"))
                continue
            if len(parts) < 2:
                print(Ink.bad("need a server name and a url: /mcp-add-http <name> <url>"))
                continue
            name, url = parts[0], parts[1]
            config_path = _mcp_config_path(args, agent.root)
            try:
                summary, actions = add_mcp_http_server(agent, config_path, name, url)
            except (ToolError, McpError, OSError) as exc:
                print(Ink.bad(f"could not add {name}: {exc}"))
                continue
            _announce_mcp_added(agent, summary, actions, config_path)
            continue
        if text.startswith("/mcp-add"):
            rest = text[len("/mcp-add"):].strip()
            if not rest:
                print("usage: /mcp-add <name> <command> [args...]")
                print(Ink.dim('example: /mcp-add fs npx -y @modelcontextprotocol/server-filesystem C:/work'))
                continue
            try:
                parts = shlex.split(rest)
            except ValueError as exc:
                print(Ink.bad(f"cannot parse the command: {exc}"))
                continue
            if len(parts) < 2:
                print(Ink.bad("need a server name and a command: /mcp-add <name> <command> [args...]"))
                continue
            name, command, cmd_args = parts[0], parts[1], parts[2:]
            config_path = _mcp_config_path(args, agent.root)
            try:
                summary, actions = add_mcp_server(agent, config_path, name, command, cmd_args)
            except (ToolError, McpError, OSError) as exc:
                print(Ink.bad(f"could not add {name}: {exc}"))
                continue
            _announce_mcp_added(agent, summary, actions, config_path)
            continue
        if text.startswith("/mcp-remove"):
            name = text[len("/mcp-remove"):].strip()
            if not name:
                print("usage: /mcp-remove <name>")
                continue
            config_path = _mcp_config_path(args, agent.root)
            try:
                summary = remove_mcp_server(agent, config_path, name)
            except (ToolError, OSError) as exc:
                print(Ink.bad(f"could not remove {name}: {exc}"))
                continue
            print(Ink.dim(f"[mcp] {summary}"))
            continue
        if text.startswith("/think"):
            agent.show_thinking = not agent.show_thinking
            print(Ink.dim("reasoning trace " + ("on" if agent.show_thinking else "off")))
            continue
        if text.startswith("/dir"):
            parts = text.split(maxsplit=1)
            if len(parts) == 1:
                print(Ink.dim(f"working directory: {agent.root}"))
            else:
                candidate = Path(parts[1]).expanduser()
                if not candidate.is_dir():
                    print(Ink.bad(f"not a directory: {candidate}"))
                else:
                    agent.root = candidate.resolve()
                    agent.tools = Tools(agent.root, agent.mcp)
                    print(Ink.dim(f"working directory: {agent.root} "
                                  f"(the model's copy is still the old path until /new)"))
            continue
        if text.startswith("/new"):
            parts = text.split(maxsplit=1)
            if len(parts) > 1:
                candidate = Path(parts[1]).expanduser()
                if candidate.is_dir():
                    agent.root = candidate.resolve()
                    agent.tools = Tools(agent.root, agent.mcp)
                else:
                    print(Ink.bad(f"not a directory: {candidate}"))
                    continue
            try:
                agent.new_chat()
                print(Ink.dim(f"new conversation{SEP}re-sending the tool guide"))
                confirmation = agent.prime()
                if confirmation:
                    print(Ink.dim("model: ") + confirmation)
                print()
            except RuntimeError as exc:
                print(Ink.bad(f"could not start a new chat: {exc}"))
            continue

        try:
            agent.ask(text)
        except KeyboardInterrupt:
            print(Ink.warn("interrupted; the answer may still complete in the browser tab"))
        except RuntimeError as exc:
            print(Ink.bad(f"error: {exc}"))
            Log.dbg("cli: server.log tail:\n  " + "\n  ".join(api.log_tail(10)))
        print()

    if agent.mcp:
        agent.mcp.close()
    api.stop()
    print(Ink.dim("bye"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
