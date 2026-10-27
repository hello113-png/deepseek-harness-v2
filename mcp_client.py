#!/usr/bin/env python3
"""
mcp_client.py -- a small MCP client for the DeepSeek CLI.

    python cli.py --mcp-config path/to/mcp.json
    python cli.py                      # uses ./mcp.json if it exists

The config is the same shape Claude Desktop uses:

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/work"]
        }
      }
    }

A server is either stdio (`command`/`args`/`env`/`cwd`) or Streamable HTTP
(`url`, optional `headers` and `timeout`):

    {
      "mcpServers": {
        "docs": {
          "url": "https://example.com/mcp",
          "headers": {"Authorization": "Bearer ..."}
        }
      }
    }

The HTTP transport POSTs JSON-RPC to the URL, accepts both `application/json`
and `text/event-stream` replies, and carries the `Mcp-Session-Id` handed out by
`initialize`. Each server is started once, its tools are listed (following
pagination), and every tool becomes an action the model can call exactly like
the built-in ones:

    ```tool
    {"action": "filesystem__read_file", "path": "C:/work/notes.md"}
    ```

MCP's stdio framing is newline-delimited JSON-RPC 2.0, which is implemented
here on the standard library. Nothing is proxied over the network.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "deepseek-cli", "version": "1.0"}
START_TIMEOUT = float(os.environ.get("MCP_START_TIMEOUT") or 25.0)
CALL_TIMEOUT = float(os.environ.get("MCP_CALL_TIMEOUT") or 180.0)
MAX_PARALLEL = int(os.environ.get("MCP_MAX_PARALLEL") or 4)
MAX_PAGES = 20            # tools/list pagination stops here at worst
MAX_GUIDE_LINES = 60      # tools described in the tool guide
MAX_TOOL_TEXT = 60_000    # characters returned by one MCP tool
MAX_HTTP_BYTES = 8_000_000  # one HTTP reply body, in bytes
USER_AGENT = "deepseek-cli/1.0"


class McpError(RuntimeError):
    """The server refused, died, or answered nonsense."""


def clean_action_part(name: str) -> str:
    """Make a server or tool name safe inside a JSON action string."""
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "")).strip("_")
    return text or "tool"


@dataclass
class McpTool:
    action: str
    server: str
    name: str
    description: str = ""
    schema: dict = field(default_factory=dict)

    def signature(self) -> str:
        properties = (self.schema or {}).get("properties") or {}
        required = set((self.schema or {}).get("required") or [])
        bits = []
        for key, spec in list(properties.items())[:12]:
            kind = (spec or {}).get("type") or "any"
            bits.append(f"{key}: {kind}" + ("*" if key in required else ""))
        return ", ".join(bits)


def tools_from_listing(server_name: str, listed: list[dict]) -> list[McpTool]:
    """Turn tools/list entries into action-addressable McpTool objects."""
    tools: list[McpTool] = []
    for item in listed:
        tool_name = str(item.get("name") or "").strip()
        if not tool_name:
            continue
        tools.append(McpTool(
            action=f"{clean_action_part(server_name)}__{clean_action_part(tool_name)}",
            server=server_name,
            name=tool_name,
            description=str(item.get("description") or ""),
            schema=item.get("inputSchema") or {},
        ))
    return tools


def content_text(result: dict) -> str:
    """Flatten an MCP CallToolResult into text for the model."""
    chunks = []
    for item in result.get("content") or []:
        kind = item.get("type")
        if kind == "text":
            chunks.append(item.get("text") or "")
        elif kind == "resource":
            resource = item.get("resource") or {}
            chunks.append(str(resource.get("text") or resource.get("uri") or ""))
        else:
            chunks.append(f"[{kind} content omitted]")
    text = "\n".join(chunks).strip()
    if len(text) > MAX_TOOL_TEXT:
        text = text[:MAX_TOOL_TEXT] + f"\n… truncated at {MAX_TOOL_TEXT} characters …"
    return text or "(the tool returned no text)"


class McpServer:
    """One stdio MCP server and its JSON-RPC plumbing."""

    def __init__(self, name: str, config: dict, base_dir: Path | None = None,
                 verbose: bool = False):
        self.name = name
        self.config = config or {}
        self.base_dir = base_dir
        self.verbose = verbose
        self.proc: subprocess.Popen | None = None
        self.tools: list[McpTool] = []
        self.start_seconds = 0.0
        self.call_timeout = float(self.config.get("timeout") or CALL_TIMEOUT)
        self.stderr_tail: list[str] = []
        self._waiters: dict[int, tuple[threading.Event, dict]] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._closed = False

    # -- logging -----------------------------------------------------------
    def log(self, message: str) -> None:
        if self.verbose:
            sys.stderr.write(f"[mcp:{self.name}] {message}\n")
            sys.stderr.flush()

    def describe(self) -> str:
        command = str(self.config.get("command") or "")
        args = " ".join(str(a) for a in (self.config.get("args") or []))
        return (command + (" " + args if args else "")).strip() or "(no command)"

    # -- process -----------------------------------------------------------
    def command_line(self) -> list[str]:
        command = str(self.config.get("command") or "").strip()
        if not command:
            raise McpError(f"{self.name}: the config needs a 'command'")
        args = [str(a) for a in (self.config.get("args") or [])]
        exe = shutil.which(command) or command
        if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
            # CreateProcess cannot run a batch file directly; cmd.exe can.
            return [os.environ.get("COMSPEC", "cmd.exe"), "/c", exe, *args]
        return [exe, *args]

    @staticmethod
    def _left(deadline: float) -> float:
        """Seconds left in the startup budget, never zero or negative."""
        return max(1.0, deadline - time.time())

    def start(self, timeout: float | None = None) -> list[McpTool]:
        """Launch the process and list its tools inside one deadline.

        Previously `initialize` and `tools/list` each got a full timeout, so a
        server that was never going to answer cost two of them in a row, one
        server at a time. Now the whole handshake shares one budget.
        """
        started = time.time()
        timeout = float(START_TIMEOUT if timeout is None else timeout)
        deadline = started + timeout
        argv = self.command_line()
        env = dict(os.environ)
        for key, value in (self.config.get("env") or {}).items():
            env[str(key)] = str(value)
        cwd = self.config.get("cwd")
        cwd = str(Path(cwd).expanduser()) if cwd else (str(self.base_dir) if self.base_dir else None)
        self.log(f"starting: {' '.join(argv)} (budget {timeout:.0f}s)")
        flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
        try:
            self.proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", bufsize=1, cwd=cwd, env=env,
                creationflags=flags,
            )
        except OSError as exc:
            raise McpError(f"{self.name}: cannot start {argv[0]!r}: {exc}") from exc

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        }, timeout=self._left(deadline))
        self.notify("notifications/initialized")
        self.tools = self._list_tools(deadline)
        self.start_seconds = time.time() - started
        self.log(f"ready: {len(self.tools)} tool(s) in {self.start_seconds:.1f}s, protocol "
                 f"{result.get('protocolVersion') or 'unknown'}")
        return self.tools

    def _list_tools(self, deadline: float) -> list[McpTool]:
        """Walk every page of tools/list (servers may paginate)."""
        tools: list[McpTool] = []
        cursor = None
        for _page in range(MAX_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params, timeout=self._left(deadline))
            tools.extend(tools_from_listing(self.name, result.get("tools") or []))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def _read_stdout(self) -> None:
        for raw in self.proc.stdout or []:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self.log(f"non-JSON output: {line[:160]}")
                continue
            if not isinstance(message, dict):
                continue
            if "id" in message and message["id"] in self._waiters:
                event, slot = self._waiters.pop(message["id"])
                slot["response"] = message
                event.set()
            elif "method" in message:
                if "id" in message:                 # server -> client request
                    self._write({"jsonrpc": "2.0", "id": message["id"],
                                 "error": {"code": -32601,
                                           "message": "client does not implement this"}})
                else:
                    self.log(f"notification: {message.get('method')}")

    def _read_stderr(self) -> None:
        for raw in self.proc.stderr or []:
            line = raw.rstrip()
            if line:
                self.stderr_tail.append(line)
                del self.stderr_tail[:-40]
                self.log("stderr: " + line[:200])

    # -- JSON-RPC ----------------------------------------------------------
    def _write(self, message: dict) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise McpError(f"{self.name}: the server is not running"
                           + (f" ({self.stderr_tail[-1][:160]})" if self.stderr_tail else ""))
        if self.proc.stdin is None:
            raise McpError(f"{self.name}: the server has no stdin")
        with self._lock:
            self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

    def request(self, method: str, params: dict | None, timeout: float = CALL_TIMEOUT) -> dict:
        request_id = self._next_id
        self._next_id += 1
        event = threading.Event()
        slot: dict = {}
        self._waiters[request_id] = (event, slot)
        try:
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method,
                         "params": params or {}})
            deadline = time.time() + timeout
            # Wait in slices so a server that died fails in 0.1s, not `timeout`.
            while not event.wait(0.1):
                proc = self.proc
                if proc is None or proc.poll() is not None:
                    code = None if proc is None else proc.returncode
                    tail = self.stderr_tail[-1][:160] if self.stderr_tail else ""
                    raise McpError(
                        f"{self.name}: the server exited"
                        f"{'' if code is None else f' (code {code})'}"
                        + (f": {tail}" if tail else "")
                    )
                if time.time() >= deadline:
                    raise McpError(f"{self.name}: no reply to {method} in {timeout:.0f}s")
        finally:
            self._waiters.pop(request_id, None)
        message = slot.get("response") or {}
        if "error" in message:
            error = message["error"] or {}
            raise McpError(f"{self.name}: {error.get('message') or error}")
        return message.get("result") or {}

    def notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        self._write(payload)

    # -- tools -------------------------------------------------------------
    def restart(self) -> list[McpTool]:
        """Bring a dead server back once, keeping its last stderr lines."""
        self.log("restarting")
        tail = self.stderr_tail[-3:]
        self.close()
        self._closed = False
        self._waiters.clear()
        self.proc = None
        self.stderr_tail = list(tail)
        self.tools = []
        try:
            return self.start()
        except McpError as exc:
            raise McpError(f"{self.name}: restart failed: {exc}") from exc

    def call_tool(self, name: str, arguments: dict,
                  timeout: float | None = None) -> tuple[bool, str]:
        result = self.request("tools/call", {"name": name, "arguments": arguments},
                              timeout=self.call_timeout if timeout is None else timeout)
        return bool(result.get("isError")), content_text(result)

    # -- shutdown ----------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self.proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    try:
                        self.request("shutdown", {}, timeout=1.0)
                    except McpError:
                        pass
                    try:
                        self.notify("exit")
                    except McpError:
                        pass
                    time.sleep(0.05)
                    if proc.poll() is None:
                        proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except OSError:
                pass
            finally:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    try:
                        if stream:
                            stream.close()
                    except OSError:
                        pass


class McpHttpServer:
    """A Streamable HTTP MCP server: one URL, JSON or SSE replies.

    JSON-RPC messages are POSTed to a single endpoint, the server answers with
    `application/json` or a `text/event-stream` stream, and the session id it
    hands out on `initialize` is echoed on every later request (and released
    with DELETE). No process is involved.
    """

    def __init__(self, name: str, config: dict, base_dir: Path | None = None,
                 verbose: bool = False):
        self.name = name
        self.config = config or {}
        self.base_dir = base_dir
        self.verbose = verbose
        self.url = str(self.config.get("url") or "").strip()
        self.extra_headers = {str(key): str(value)
                              for key, value in (self.config.get("headers") or {}).items()}
        self.proc = None                  # the registry treats both kinds alike
        self.tools: list[McpTool] = []
        self.start_seconds = 0.0
        self.call_timeout = float(self.config.get("timeout") or CALL_TIMEOUT)
        self.stderr_tail: list[str] = []
        self.session_id: str | None = None
        self.protocol_version = PROTOCOL_VERSION
        self._next_id = 1
        self._closed = False
        self._lock = threading.Lock()

    # -- logging -----------------------------------------------------------
    def log(self, message: str) -> None:
        if self.verbose:
            sys.stderr.write(f"[mcp:{self.name}] {message}\n")
            sys.stderr.flush()

    def describe(self) -> str:
        return self.url or "(no url)"

    @staticmethod
    def _left(deadline: float) -> float:
        return max(1.0, deadline - time.time())

    # -- lifecycle ---------------------------------------------------------
    def start(self, timeout: float | None = None) -> list[McpTool]:
        if not self.url:
            raise McpError(f"{self.name}: the config needs a 'url'")
        started = time.time()
        timeout = float(START_TIMEOUT if timeout is None else timeout)
        deadline = started + timeout
        self.session_id = None
        self._closed = False
        self.log(f"connecting: {self.url} (budget {timeout:.0f}s)")
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        }, timeout=self._left(deadline))
        self.protocol_version = str(result.get("protocolVersion") or PROTOCOL_VERSION)
        self.notify("notifications/initialized")
        self.tools = self._list_tools(deadline)
        self.start_seconds = time.time() - started
        self.log(f"ready: {len(self.tools)} tool(s) in {self.start_seconds:.1f}s")
        return self.tools

    def _list_tools(self, deadline: float) -> list[McpTool]:
        tools: list[McpTool] = []
        cursor = None
        for _page in range(MAX_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = self.request("tools/list", params, timeout=self._left(deadline))
            tools.extend(tools_from_listing(self.name, result.get("tools") or []))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    # -- transport ---------------------------------------------------------
    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
            "MCP-Protocol-Version": self.protocol_version,
            **self.extra_headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _post(self, payload: dict, timeout: float):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.url, data=data, method="POST",
                                         headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self.session_id = session
                kind = (response.headers.get("Content-Type") or "").lower()
                body = response.read(MAX_HTTP_BYTES).decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()[:300]
            raise McpError(f"{self.name}: HTTP {exc.code} from {self.url}"
                           + (f": {detail}" if detail else "")) from exc
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise McpError(f"{self.name}: {self.url} is unreachable: {exc}") from exc
        if not body.strip():
            return None
        if "text/event-stream" in kind:
            return self._read_events(body, payload.get("id"))
        try:
            return json.loads(body)
        except ValueError:
            raise McpError(f"{self.name}: the reply was not JSON: {body.strip()[:200]!r}")

    def _read_events(self, text: str, want_id):
        """Pick the JSON-RPC reply with our id out of an SSE stream.

        The interleaved progress notifications are normal and simply logged;
        only the message whose id matches the request is the answer.
        """
        answer = None
        for block in text.replace("\r\n", "\n").split("\n\n"):
            data = [line[5:].lstrip(" ") for line in block.splitlines()
                    if line.startswith("data:")]
            if not data:
                continue
            try:
                parsed = json.loads("\n".join(data))
            except ValueError:
                self.log(f"ignored a non-JSON event: {data[0][:120]}")
                continue
            for item in parsed if isinstance(parsed, list) else [parsed]:
                if not isinstance(item, dict):
                    continue
                if "method" in item:
                    self.log(f"notification: {item.get('method')}")
                    continue
                if want_id is not None and item.get("id") == want_id:
                    answer = item
        if want_id is not None and answer is None:
            raise McpError(f"{self.name}: the event stream had no reply to request {want_id}")
        return answer

    def request(self, method: str, params: dict | None,
                timeout: float | None = None) -> dict:
        request_id = self._next_id
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method,
                   "params": params or {}}
        result = self._post(payload, self.call_timeout if timeout is None else timeout)
        if isinstance(result, list):
            result = next((item for item in result
                           if isinstance(item, dict) and item.get("id") == request_id),
                          result[0] if result else None)
        if not isinstance(result, dict):
            raise McpError(f"{self.name}: no reply to {method} "
                           f"(the server returned no JSON-RPC message)")
        if "error" in result:
            error = result["error"] or {}
            raise McpError(f"{self.name}: {error.get('message') or error}")
        return result.get("result") or {}

    def notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        self._post(payload, timeout=min(15.0, self.call_timeout))

    # -- tools -------------------------------------------------------------
    def call_tool(self, name: str, arguments: dict,
                  timeout: float | None = None) -> tuple[bool, str]:
        result = self.request("tools/call", {"name": name, "arguments": arguments},
                              timeout=self.call_timeout if timeout is None else timeout)
        return bool(result.get("isError")), content_text(result)

    def restart(self) -> list[McpTool]:
        self.log("restarting the session")
        self.close()
        self.session_id = None
        self.tools = []
        return self.start()

    # -- shutdown ----------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.session_id:
            return
        try:
            request = urllib.request.Request(self.url, method="DELETE",
                                             headers=self._headers())
            with urllib.request.urlopen(request, timeout=5):
                pass
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            self.log(f"DELETE failed: {exc}")
        self.session_id = None


McpConnection = McpServer | McpHttpServer


class McpRegistry:
    """All configured servers, flattened into one action namespace."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.servers: list[McpConnection] = []
        self.tools: list[McpTool] = []
        self.errors: list[str] = []

    def log(self, message: str) -> None:
        if self.verbose:
            sys.stderr.write(f"[mcp] {message}\n")
            sys.stderr.flush()

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, config_path: Path | str | None, verbose: bool = False,
             start_timeout: float | None = None) -> "McpRegistry | None":
        registry = cls(verbose=verbose)
        if not config_path:
            return None
        path = Path(config_path).expanduser()
        if not path.is_file():
            registry.errors.append(f"{path} does not exist")
            return registry
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            registry.errors.append(f"cannot read {path}: {exc}")
            return registry
        servers = (config or {}).get("mcpServers")
        if not isinstance(servers, dict):
            registry.errors.append(f"{path} has no 'mcpServers' object")
            return registry
        specs: list[tuple[str, dict]] = []
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                registry.errors.append(f"{name}: config must be an object")
                continue
            if spec.get("disabled"):
                continue
            if spec.get("url") and spec.get("command"):
                registry.errors.append(f"{name}: use either 'command' (stdio) or "
                                       f"'url' (HTTP), not both")
                continue
            if not spec.get("url") and not spec.get("command"):
                registry.errors.append(f"{name}: needs a 'command' (stdio) or "
                                       f"a 'url' (Streamable HTTP)")
                continue
            specs.append((str(name), spec))
        if not specs:
            return registry

        # Servers start in parallel: `npx` unpacking a package used to add its
        # whole startup time to the next server's, one after another.
        workers = max(1, min(MAX_PARALLEL, len(specs)))
        outcomes: dict[str, tuple[McpServer | None, list[McpTool], str | None]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(cls._start_server, name, spec, path.parent,
                                   verbose, start_timeout): name
                       for name, spec in specs}
            for future in as_completed(futures):
                outcomes[futures[future]] = future.result()
        for name, _spec in specs:                  # report in the config's order
            server, tools, error = outcomes.get(name, (None, [], "did not start"))
            if error or server is None:
                registry.errors.append(error or f"{name}: did not start")
                if server is not None:
                    server.close()
                continue
            registry.servers.append(server)
            registry.tools.extend(tools)
        registry._dedupe_actions()
        return registry

    @staticmethod
    def _start_server(name: str, spec: dict, base_dir: Path, verbose: bool,
                      start_timeout: float | None):
        kind = McpHttpServer if spec.get("url") else McpServer
        server = kind(name, spec, base_dir=base_dir, verbose=verbose)
        try:
            return server, server.start(timeout=start_timeout), None
        except (McpError, OSError) as exc:
            return server, [], str(exc)

    def _dedupe_actions(self) -> None:
        """Action names must be unique even when two servers share a tool name."""
        seen: dict[str, int] = {}
        for tool in self.tools:
            if tool.action in seen:
                seen[tool.action] += 1
                tool.action = f"{tool.action}_{seen[tool.action]}"
            else:
                seen[tool.action] = 0

    def _rebuild(self) -> None:
        """Re-flatten the tool list after a server restarted."""
        self.tools = [tool for server in self.servers for tool in server.tools]
        self._dedupe_actions()

    # -- lookups -----------------------------------------------------------
    def actions(self) -> list[str]:
        return [tool.action for tool in self.tools]

    def find(self, action: str) -> tuple[McpConnection, McpTool] | None:
        for server in self.servers:
            for tool in server.tools:
                if tool.action == action:
                    return server, tool
        return None

    def call(self, action: str, arguments: dict,
             timeout: float | None = None) -> tuple[str, str, bool]:
        """Return (summary, text, is_error), restarting a dead server once."""
        found = self.find(action)
        if not found:
            raise McpError(f"unknown MCP action {action!r}")
        server, tool = found
        started = time.time()
        try:
            is_error, text = server.call_tool(tool.name, arguments, timeout=timeout)
        except McpError as exc:
            if not self._revive(server, exc):
                return (f"{action} failed after {time.time() - started:.1f}s", str(exc), True)
            try:
                is_error, text = server.call_tool(tool.name, arguments, timeout=timeout)
            except McpError as exc2:
                return (f"{action} failed after {time.time() - started:.1f}s",
                        f"{exc2} (after restarting the server)", True)
        elapsed = time.time() - started
        summary = (f"{action} -> {'error' if is_error else 'ok'} "
                   f"({len(text)} chars, {elapsed:.1f}s)")
        return summary, text, is_error

    def _revive(self, server: McpConnection, error: McpError) -> bool:
        """Restart a server only when the failure means the connection is gone."""
        text = str(error).lower()
        dead = ("exited" in text or "not running" in text or "no reply" in text
                or "broken pipe" in text or "has no stdin" in text
                or "unreachable" in text or "http 404" in text or "http 400" in text)
        if not dead:
            return False
        self.log(f"{server.name} is gone ({str(error)[:80]}); restarting it")
        try:
            server.restart()
        except McpError as exc:
            self.errors.append(str(exc))
            return False
        self._rebuild()
        return True

    # -- prompt ------------------------------------------------------------
    def guide_lines(self) -> list[str]:
        lines = []
        for tool in self.tools[:MAX_GUIDE_LINES]:
            description = " ".join((tool.description or "").split())[:140]
            signature = tool.signature()
            entry = f"  {tool.action}({signature})"
            lines.append(entry + (f"  {description}" if description else ""))
        if len(self.tools) > MAX_GUIDE_LINES:
            lines.append(f"  … {len(self.tools) - MAX_GUIDE_LINES} more MCP tools not listed …")
        return lines

    def summary(self) -> str:
        if not self.servers:
            return "no MCP servers configured"
        bits = []
        for server in self.servers:
            timing = f" in {server.start_seconds:.1f}s" if server.start_seconds else ""
            bits.append(f"{server.name}: {len(server.tools)} tool(s){timing}")
        return "; ".join(bits)

    def close(self) -> None:
        for server in self.servers:
            server.close()
