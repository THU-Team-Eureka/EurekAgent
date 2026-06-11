"""Stream adapter: wraps ``claude -p --output-format stream-json`` as async sessions.

Each session is a subprocess that runs a single Claude Code turn: the prompt
is passed on argv, events stream from stdout as NDJSON, and the process
exits when the turn is done.

Pause-and-send uses a kill-and-resume strategy: SIGTERM the current process
(so Claude flushes session state to disk), then start a new process with
``--resume <session_id> -p "<user_message>"``.  The ``stream()`` generator
automatically switches to the new process via a per-session ``asyncio.Event``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from .protocol import SessionEvent, SessionRequest, SessionResult
from .pty_adapter import _RESUME_PROMPT

log = logging.getLogger(__name__)

_GRACEFUL_EXIT_TIMEOUT = 5.0  # seconds to wait after SIGTERM before SIGKILL
_STREAM_LIMIT = 2**22  # 4 MB readline buffer; Claude CLI NDJSON lines can exceed the default 64 KB


class StreamAdapter:
    """Spawns claude CLI subprocesses and streams their NDJSON output."""

    def __init__(self, claude_command: str = "claude") -> None:
        self._claude = claude_command
        self._sessions: dict[str, asyncio.subprocess.Process] = {}
        self._session_ids: dict[str, str] = {}  # our key -> actual session_id
        self._log_handles: dict[str, open] = {}
        self._session_requests: dict[str, SessionRequest] = {}  # for resume
        self._pending_resume: dict[str, asyncio.Event] = {}  # kill→resume signal
        self._tmp_dir = Path(__file__).resolve().parents[2] / ".tmp"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, request: SessionRequest) -> str:
        """Start a new session and return its session key."""
        session_key = request.session_id or str(uuid.uuid4())
        # Always pass --session-id so the in-container process carries a
        # unique fingerprint we can use for tree-kill on timeout.
        if not request.session_id:
            request = SessionRequest(
                prompt=request.prompt,
                model=request.model,
                cwd=request.cwd,
                session_id=session_key,
                permissions=request.permissions,
                log_path=request.log_path,
                resume=request.resume,
                env=request.env,
            )

        # On resume, replace the full stage brief with a minimal continue
        # message — re-sending the full brief causes the agent to
        # re-process old context from the loaded conversation history.
        if request.resume:
            request = SessionRequest(
                prompt=_RESUME_PROMPT,
                model=request.model,
                cwd=request.cwd,
                session_id=request.session_id,
                permissions=request.permissions,
                log_path=request.log_path,
                resume=True,
                env=request.env,
            )

        cmd = self._build_command(request, session_key)
        log.info("Starting session %s: %s", session_key, " ".join(cmd[:6]))
        env = self._subprocess_env()
        if request.env:
            env.update(request.env)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=request.cwd or None,
            start_new_session=True,
            limit=_STREAM_LIMIT,
            env=env,
        )
        self._sessions[session_key] = process
        self._session_ids[session_key] = request.session_id
        self._session_requests[session_key] = request

        if request.log_path:
            log_path = Path(request.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handles[session_key] = open(log_path, "a")

        return session_key

    async def send(self, session_key: str, message: str) -> None:
        """Interrupt the running agent and send a follow-up user message.

        Uses kill-and-resume: SIGTERM the current process (Claude flushes
        session state on graceful exit, verified with exit code 143), then
        start a new process with ``--resume <session_id> -p "<message>"``.
        The ``stream()`` generator picks up events from the new process
        automatically via ``_pending_resume``.
        """
        old_process = self._sessions.get(session_key)
        if not old_process or old_process.returncode is not None:
            raise RuntimeError(f"Session {session_key} is not running")

        # Signal stream() that a new process is coming so it doesn't return.
        resume_event = asyncio.Event()
        self._pending_resume[session_key] = resume_event

        # Gracefully kill the current process.
        actual_id = self._session_ids.get(session_key, "")
        await self._kill_process_tree(session_key, old_process, actual_id)

        try:
            await asyncio.wait_for(old_process.wait(), timeout=_GRACEFUL_EXIT_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning("Session %s: SIGTERM timed out, SIGKILL", session_key)
            try:
                old_process.kill()
            except ProcessLookupError:
                pass
            await old_process.wait()

        log.info("Session %s: old process exited (%s), resuming with user message",
                 session_key, old_process.returncode)

        # Build resume request reusing original parameters.
        orig = self._session_requests[session_key]
        resume_request = SessionRequest(
            prompt=message,
            model=orig.model,
            cwd=orig.cwd,
            session_id=actual_id,
            permissions=orig.permissions,
            log_path=orig.log_path,
            resume=True,
            env=orig.env,
        )

        cmd = self._build_command(resume_request, session_key)
        env = self._subprocess_env()
        if orig.env:
            env.update(orig.env)
        new_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=orig.cwd or None,
            start_new_session=True,
            limit=_STREAM_LIMIT,
            env=env,
        )
        self._sessions[session_key] = new_process

        # Notify stream() that the new process is ready.
        resume_event.set()

    async def stream(self, session_key: str, max_recoveries: int | None = None) -> AsyncIterator[SessionEvent]:
        """Stream NDJSON events from the session's stdout.

        Supports process switching: when ``send()`` replaces the subprocess,
        the outer loop detects the ``_pending_resume`` event and continues
        reading from the new process instead of returning.

        max_recoveries is accepted for interface compatibility with PtyAdapter
        but is ignored — stream mode has no recovery budget (the subprocess
        exits naturally on stop_sequence/max_tokens).
        """
        while True:
            process = self._sessions.get(session_key)
            if process is None or process.stdout is None:
                return

            log_handle = self._log_handles.get(session_key)

            while True:
                try:
                    line = await process.stdout.readline()
                except ValueError:
                    # StreamReader buffer overrun — the NDJSON line exceeded
                    # even the expanded limit. Drain the oversized line so we
                    # can continue reading subsequent events.
                    log.warning("Session %s: oversized line, draining", session_key)
                    try:
                        await process.stdout.read(process.stdout._limit * 2)
                    except Exception:
                        pass
                    continue
                if not line:
                    break

                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                if log_handle:
                    log_handle.write(text + "\n")
                    log_handle.flush()

                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON line: %.100s", text)
                    continue

                event_type = data.get("type", "unknown")
                event = SessionEvent(type=event_type, data=data, timestamp=time.time())

                if event_type == "init" and "session_id" in data:
                    self._session_ids[session_key] = data["session_id"]

                yield event

            # stdout closed. Check whether send() is resuming with a new process.
            resume_event = self._pending_resume.pop(session_key, None)
            if resume_event is not None:
                await resume_event.wait()
                continue  # Loop back to read from the new process.

            # No resume pending — session truly ended.
            return

    async def stop(self, session_key: str, grace_seconds: float = 5.0) -> None:
        """Stop a session — tree-kill the process group, then close handles."""
        process = self._sessions.get(session_key)
        if not process or process.returncode is not None:
            self._cleanup(session_key)
            return

        actual_id = self._session_ids.get(session_key, "")
        await self._kill_process_tree(session_key, process, actual_id)
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

        self._cleanup(session_key)

    async def is_alive(self, session_key: str) -> bool:
        """Check if the session subprocess is still running."""
        process = self._sessions.get(session_key)
        return process is not None and process.returncode is None

    async def collect_result(
        self, session_key: str, started_at: float
    ) -> SessionResult:
        """Wait for process exit and build a SessionResult."""
        process = self._sessions.get(session_key)
        exit_code = None
        stderr_text = None

        if process:
            await process.wait()
            exit_code = process.returncode
            if process.stderr:
                raw = await process.stderr.read()
                stderr_text = raw.decode("utf-8", errors="replace").strip() or None

        elapsed = time.time() - started_at
        log_path = self._log_path_for(session_key)
        final_result = self._read_final_result(log_path)

        self._cleanup(session_key)

        return SessionResult(
            session_id=self._session_ids.get(session_key, session_key),
            exit_code=exit_code,
            elapsed_seconds=elapsed,
            log_path=log_path,
            final_result=final_result,
            error=stderr_text if exit_code and exit_code != 0 else None,
            recovery_abort_reason="",
        )

    # ------------------------------------------------------------------
    # Override points for subclasses
    # ------------------------------------------------------------------

    def _build_command(self, request: SessionRequest, session_key: str) -> list[str]:
        """Build the argv for ``claude -p ...``.

        When ``request.resume`` is True, uses ``--resume <session_id>``
        instead of ``--session-id <session_id>`` so the new process picks
        up the existing conversation history.
        """
        cmd = [self._claude, "-p", "--output-format", "stream-json", "--verbose"]
        if request.model:
            cmd.extend(["--model", request.model])
        if request.resume and request.session_id:
            cmd.extend(["--resume", request.session_id])
        elif request.session_id:
            cmd.extend(["--session-id", request.session_id])
        if request.permissions == "bypassPermissions":
            cmd.append("--dangerously-skip-permissions")
        cmd.append(request.prompt)
        return cmd

    def _subprocess_env(self) -> dict[str, str] | None:
        """Return env dict for subprocess.

        GPU policy: default-deny — CUDA_VISIBLE_DEVICES is always cleared.
        Agents acquire GPUs via gpu_helpers.gpu_session() (in their own
        process), which sets the env var on entry and restores it on exit.
        """
        from ..runtime import get_config
        try:
            _config = get_config()  # noqa: F841 — kept for forward compatibility
        except Exception:
            pass
        env = dict(os.environ)
        # Default-deny: agents start with no GPU visibility regardless of --gpus mode.
        # Agents must acquire GPUs via gpu_helpers.gpu_session() which sets
        # CUDA_VISIBLE_DEVICES on entry and restores it on exit.
        env["CUDA_VISIBLE_DEVICES"] = ""
        # Expose gpu_helpers via PYTHONPATH
        internal_dir = str(self._workspace.resolve() / ".eureka_internal")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{internal_dir}:{existing}" if existing else internal_dir
        return env

    async def _kill_process_tree(
        self,
        session_key: str,
        process: asyncio.subprocess.Process,
        session_id: str,
    ) -> None:
        """SIGTERM the host-side process group for `process`.

        Subclasses extend this to also clean up processes inside a container.
        """
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    # ------------------------------------------------------------------
    # Private internals
    # ------------------------------------------------------------------

    def _log_path_for(self, session_key: str) -> Path:
        handle = self._log_handles.get(session_key)
        if handle and hasattr(handle, "name"):
            return Path(handle.name)
        return self._tmp_dir / f"eureka_session_{session_key}.jsonl"

    def _read_final_result(self, log_path: Path) -> dict | None:
        """Read the last 'result' event from the NDJSON log file."""
        if not log_path.exists():
            return None
        try:
            for line in reversed(log_path.read_text().splitlines()):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("type") == "result":
                    return data
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _cleanup(self, session_key: str) -> None:
        handle = self._log_handles.pop(session_key, None)
        if handle:
            handle.close()
        self._sessions.pop(session_key, None)
        self._session_requests.pop(session_key, None)
        self._pending_resume.pop(session_key, None)
