"""PTY proxy: bridges docker-exec stdin/stdout pipes to a Claude PTY.

Run inside the container via ``docker exec -i <cid> python3 /path/to/pty_proxy.py``.

Protocol
--------
The host sends **line-delimited JSON** commands on the proxy's stdin:

    {"cmd": "start", "args": ["claude", "--session-id", "..."]}
    {"cmd": "write", "data": "Say hello\\r"}
    {"cmd": "interrupt"}
    {"cmd": "exit"}

The proxy writes **line-delimited JSON** events on stdout:

    {"event": "ready"}
    {"event": "jsonl", "data": {...}}     # JSONL event from inside container
    {"event": "claude_exited", "code": 0}
    {"event": "proxy_done"}

All stdout output is flushed immediately.
"""

from __future__ import annotations

import json
import os
import pty
import select
import sys
import time

_POLL_INTERVAL = 0.5
# Wrap multi-line / large writes in bracketed-paste so Claude's Ink TUI
# treats the whole thing as a single paste event instead of dropping bytes.
_BRACKETED_PASTE_START = "\x1b[200~"
_BRACKETED_PASTE_END = "\x1b[201~"


def _emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def _safe_write(fd: int, data: bytes) -> None:
    """Write all bytes to fd, handling partial writes."""
    total = len(data)
    written = 0
    while written < total:
        try:
            n = os.write(fd, data[written:])
            written += n
        except BlockingIOError:
            # PTY buffer full — wait briefly and retry.
            time.sleep(0.01)
        except OSError:
            break


def _drain_fd(fd: int, bufsize: int = 65536) -> str:
    chunks: list[str] = []
    while True:
        try:
            data = os.read(fd, bufsize)
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))
        except (BlockingIOError, OSError):
            break
    return "".join(chunks)


def _compute_jsonl_path(args: list[str]) -> str:
    """Derive the JSONL path from the claude args and environment."""
    session_id = ""
    for i, a in enumerate(args):
        if a in ("--session-id", "--resume") and i + 1 < len(args):
            session_id = args[i + 1]
            break
    if not session_id:
        return ""

    home = os.environ.get("HOME", "/root")
    cwd = os.environ.get("PWD", os.getcwd())
    dir_name = cwd.replace("/", "-")
    return f"{home}/.claude/projects/{dir_name}/{session_id}.jsonl"


class _JsonlPoller:
    """Polls a JSONL file from inside the container and emits events."""

    def __init__(self, path: str, *, skip_existing: bool = False) -> None:
        self._path = path
        self._offset = 0
        if skip_existing and path:
            try:
                self._offset = os.path.getsize(path)
            except OSError:
                pass

    def poll(self) -> None:
        if not self._path:
            return
        try:
            with open(self._path, "r") as f:
                f.seek(self._offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        _emit({"event": "jsonl", "data": data})
                    except json.JSONDecodeError:
                        pass
                self._offset = f.tell()
        except OSError:
            pass


class _SubagentPoller:
    """Scans for subagent JSONL files and polls new content.

    Claude Code stores subagent transcripts at:
        <project_dir>/<session_id>/subagents/agent-<shortId>.jsonl

    The subagent directory only appears when the main session spawns
    subagents, so ``poll()`` silently skips until the directory exists.
    """

    def __init__(self, project_dir: str, session_id: str) -> None:
        self._subagent_dir = os.path.join(project_dir, session_id, "subagents")
        self._offsets: dict[str, int] = {}
        self._found_logged = False

    def poll(self) -> None:
        if not self._subagent_dir:
            return
        if not os.path.isdir(self._subagent_dir):
            return
        if not self._found_logged:
            _emit({"event": "debug", "msg": f"subagent_dir found: {self._subagent_dir}"})
            self._found_logged = True
        try:
            entries = sorted(os.listdir(self._subagent_dir))
        except OSError:
            return
        for name in entries:
            if not name.startswith("agent-") or not name.endswith(".jsonl"):
                continue
            path = os.path.join(self._subagent_dir, name)
            offset = self._offsets.get(name, 0)
            # Use stem (e.g. "agent-abc123") without .jsonl suffix for consistency
            # with host-side file polling that uses Path.stem
            subagent_stem = name[:-6] if name.endswith(".jsonl") else name
            try:
                with open(path, "r") as f:
                    f.seek(offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            _emit({"event": "jsonl", "data": data, "_subagent": subagent_stem})
                        except json.JSONDecodeError:
                            pass
                    self._offsets[name] = f.tell()
            except OSError:
                pass


def main() -> None:
    start_cmd = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("cmd") == "start":
            start_cmd = msg.get("args", ["claude"])
            break

    if start_cmd is None:
        _emit({"event": "proxy_done", "error": "no start command received"})
        return

    master_fd, slave_fd = pty.openpty()

    # Set a reasonable terminal size so Claude's TUI can render.
    # pty.openpty() defaults to 0x0 which prevents the Textual/Ink
    # renderer from initialising and blocks JSONL output.
    import struct
    import fcntl
    import termios
    winsize = struct.pack("HHHH", 50, 160, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    pid = os.fork()

    if pid == 0:
        os.close(master_fd)
        os.setsid()
        # Make the slave PTY the controlling terminal.
        # Opening /dev/tty after setsid() would fail; instead we
        # use the TIOCSCTTY ioctl on the slave fd.
        import fcntl
        import termios
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        except OSError:
            pass
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        if slave_fd > 2:
            os.close(slave_fd)
        try:
            os.execvp(start_cmd[0], start_cmd)
        except Exception:
            os._exit(127)

    os.close(slave_fd)
    os.set_blocking(master_fd, False)
    _emit({"event": "ready", "pid": pid})

    jsonl_path = _compute_jsonl_path(start_cmd)
    is_resume = "--resume" in start_cmd
    poller = _JsonlPoller(jsonl_path, skip_existing=is_resume)

    session_id = ""
    for i, a in enumerate(start_cmd):
        if a in ("--session-id", "--resume") and i + 1 < len(start_cmd):
            session_id = start_cmd[i + 1]
            break
    home = os.environ.get("HOME", "/root")
    cwd = os.environ.get("PWD", os.getcwd())
    dir_name = cwd.replace("/", "-")
    project_dir = f"{home}/.claude/projects/{dir_name}"
    subagent_poller = _SubagentPoller(project_dir, session_id)

    claude_alive = True
    stdin_eof = False
    last_poll = time.monotonic()
    pty_seen = False
    last_pty_sample = time.monotonic()

    while True:
        readers = [master_fd]
        if not stdin_eof:
            readers.append(sys.stdin)

        timeout = 0.1
        try:
            ready, _, _ = select.select(readers, [], [], timeout)
        except (ValueError, OSError):
            break

        # PTY output: drain everything pending. We don't forward the raw
        # bytes (JSONL events replace them), but we emit a "pty_output"
        # event on the first drain so host-side debugging can see claude's
        # startup banner or error output.  Also sample PTY output
        # periodically so we can see dialog screens that appear after
        # startup.
        if master_fd in ready:
            chunk = _drain_fd(master_fd)
            if chunk and not pty_seen:
                _emit({"event": "pty_output", "data": chunk[:2000]})
                pty_seen = True
            elif chunk and time.monotonic() - last_pty_sample > 5.0:
                # Periodic sample: strip ANSI escape sequences for
                # readability, keep only printable text.
                import re
                clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', chunk)
                clean = re.sub(r'\x1b\].*?\x07', '', clean)
                clean = ''.join(c for c in clean if c.isprintable() or c in '\n\r\t')
                clean = clean.strip()[:500]
                if clean:
                    _emit({"event": "pty_output", "data": clean})
                last_pty_sample = time.monotonic()

        if not stdin_eof and sys.stdin in ready:
            line = sys.stdin.readline()
            if not line:
                stdin_eof = True
                continue
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("cmd", "")
            if cmd == "write":
                data = msg.get("data", "")
                raw = bool(msg.get("raw", False))
                _emit({"event": "debug", "msg": f"write cmd received, data_len={len(data)}, raw={raw}"})
                if raw:
                    # Raw write: no bracketed-paste wrapping. Required
                    # for control sequences like Down+Enter used to
                    # dismiss claude's startup dialog — wrapping them
                    # in paste markers makes Ink treat them as literal
                    # text and the dialog does not accept them.
                    _safe_write(master_fd, data.encode())
                    continue
                # Strip a trailing "\r" if present so we can re-append it
                # after the paste-end marker, otherwise Ink would see
                # "text\r<paste-end>" and the Enter arrives before the
                # paste is terminated.
                trailing_cr = data.endswith("\r")
                if trailing_cr:
                    data = data[:-1]
                payload = _BRACKETED_PASTE_START + data + _BRACKETED_PASTE_END
                if trailing_cr:
                    payload += "\r"
                _safe_write(master_fd, payload.encode())
            elif cmd == "interrupt":
                try:
                    os.write(master_fd, b"\x03")
                except OSError:
                    pass
            elif cmd == "exit":
                try:
                    os.write(master_fd, b"\x03")
                    time.sleep(0.3)
                    os.write(master_fd, b"/exit\r")
                except OSError:
                    pass
                break

        now = time.monotonic()
        if now - last_poll >= _POLL_INTERVAL:
            poller.poll()
            subagent_poller.poll()
            last_poll = now

        if claude_alive:
            try:
                wpid, status = os.waitpid(pid, os.WNOHANG)
                if wpid != 0:
                    exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                    _emit({"event": "claude_exited", "code": exit_code})
                    claude_alive = False
            except ChildProcessError:
                _emit({"event": "claude_exited", "code": -1})
                claude_alive = False

        if not claude_alive:
            poller.poll()
            subagent_poller.poll()
            break

    os.close(master_fd)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass

    _emit({"event": "proxy_done"})


if __name__ == "__main__":
    main()
