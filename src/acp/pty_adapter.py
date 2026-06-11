"""PTY-based adapter: wraps ``claude`` interactive mode with JSONL file tailing.

Each session is a Claude Code interactive process started via a pseudo-terminal
(PTY).  Messages are sent by writing to the PTY master; the interrupt signal
(Ctrl+C / ``\\x03``) causes Claude to stop generating but keep the process
alive — a *true pause* that avoids the kill-and-resume overhead of
:class:`StreamAdapter`.

Structured output is read from the session transcript JSONL file that Claude
Code writes to ``~/.claude/projects/<project-dir>/<session_id>.jsonl``.
Events are written per-API-call (thinking, text, and tool_use content blocks
appear together when a streaming response finishes; tool_result events appear
as each tool completes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pty
import signal
import termios
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from collections.abc import Callable

from .protocol import SessionEvent, SessionRequest, SessionResult

log = logging.getLogger(__name__)

_POLL_INTERVAL = 0.5  # seconds between JSONL file polls
_END_DETECTION_IDLE = 10.0  # seconds of no new events after a result before ending stream
_RECOVERY_DELAY = 5.0  # seconds to wait before sending recovery message
_RUNTIME_SILENCE_SECONDS = 300.0  # seconds of no JSONL before runtime silence recovery
_MAX_CONSECUTIVE_SAME = 20  # consecutive same-type stop_sequence recoveries before abort
# Time to let Claude's Ink/React TUI mount its input widget before we write
# the first prompt. Empirical: < 1.5s loses keystrokes on a cold start.
_TUI_RENDER_DELAY = 2.0
# Time to let Claude's Ink TUI finish unmounting its "generating" view and
# remount the input widget after Ctrl+C. Without this, the follow-up
# message lands in a component that is not ready to accept input and is
# silently dropped.
_POST_INTERRUPT_DELAY = 0.5
# Size of each read when continuously draining the PTY master fd during
# the session. The child's Ink TUI redraws produce large bursts of ANSI
# sequences; if we do not read them, the kernel PTY buffer (typically
# 4-64KB) fills and the child blocks in write(), stalling its input
# loop so Ctrl+C and new messages stop working.
_PTY_DRAIN_CHUNK = 65536
# Wrap multi-line / large writes in bracketed-paste so Claude's Ink TUI
# treats the whole input as a single paste event instead of dropping bytes.
# Without this, prompts above ~200 chars are silently truncated.
_BRACKETED_PASTE_START = "\x1b[200~"
_BRACKETED_PASTE_END = "\x1b[201~"
_STAGE_CONTINUE_PROMPT = (
    "Continue with your task. You have not yet produced the required output files. "
    "If you were summarizing or researching, proceed to writing the deliverables now."
)
_RESUME_PROMPT = (
    "This session was interrupted and has been resumed. "
    "Review your skill instructions, task goals, budget limits, and current progress, "
    "then continue working from where you left off."
)


class PtyAdapter:
    """Spawns Claude Code in interactive mode via PTY, tails JSONL for events."""

    def __init__(self, claude_command: str = "claude", workspace: Path | None = None) -> None:
        self._claude = claude_command
        self._sessions: dict[str, asyncio.subprocess.Process] = {}
        self._master_fds: dict[str, int] = {}
        self._session_ids: dict[str, str] = {}
        self._jsonl_offsets: dict[str, int] = {}
        self._log_handles: dict[str, object] = {}
        # Background tasks that continuously drain each session's PTY
        # master fd. Without these, the child's TUI redraws fill the
        # kernel PTY buffer and the child blocks in write(), which
        # stalls its input event loop — Ctrl+C still gets through (one
        # byte) but the follow-up message is dropped.
        self._drain_tasks: dict[str, asyncio.Task] = {}
        # Path of each session's user-provided log file, remembered so
        # collect_result can read it after _cleanup closes the handle.
        self._last_log_paths: dict[str, Path] = {}
        self._workspace = workspace
        self._project_dir = self._compute_project_dir()
        self._completion_checks: dict[str, Callable[[], bool]] = {}
        self._persist_flags: dict[str, bool] = {}
        self._pause_checks: dict[str, Callable[[], bool]] = {}
        self._on_pause: dict[str, Callable[[], None]] = {}
        self._pause_fired: set[str] = set()
        self._session_is_resume: dict[str, bool] = {}
        self._recovery_abort_reasons: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API (same interface as StreamAdapter)
    # ------------------------------------------------------------------

    async def start(self, request: SessionRequest) -> str:
        """Start an interactive Claude session via PTY and return its key.

        We sleep briefly after spawning to let the Ink/React TUI mount its
        input widget — without this delay, the first prompt is consumed
        before the input handler is ready and Claude sees no user input.
        We do NOT poll the JSONL file before writing: the interactive TUI
        only creates that file after receiving user input, so polling first
        would deadlock.
        """
        session_key = request.session_id or str(uuid.uuid4())
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

        cmd = self._build_command(request, session_key)
        log.info("Starting PTY session %s: %s", session_key, " ".join(cmd[:6]))

        master_fd, slave_fd = pty.openpty()

        # Set a reasonable terminal size so Claude's TUI can render.
        import struct
        import fcntl
        winsize = struct.pack("HHHH", 50, 160, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        try:
            env = self._subprocess_env()
            if request.env:
                env.update(request.env)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=request.cwd or None,
                start_new_session=True,
                env=env,
            )
        except Exception:
            os.close(slave_fd)
            os.close(master_fd)
            raise
        os.close(slave_fd)
        os.set_blocking(master_fd, False)

        self._sessions[session_key] = process
        self._session_ids[session_key] = session_key
        self._master_fds[session_key] = master_fd
        self._session_is_resume[session_key] = request.resume

        if request.log_path:
            log_path = Path(request.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handles[session_key] = open(log_path, "a")
            self._last_log_paths[session_key] = log_path
        # On resume, skip past existing transcript content in the
        # Claude-side JSONL so only new events are forwarded. Past
        # history is pushed separately via push_resume_history.
        if request.resume:
            jsonl_path = self._jsonl_path_for(session_key)
            if jsonl_path.exists():
                self._jsonl_offsets[session_key] = jsonl_path.stat().st_size

        # Let the TUI render its initial frame, then drain whatever escape
        # sequences it emitted so they don't get echoed back to claude.
        await asyncio.sleep(_TUI_RENDER_DELAY)
        self._drain_pty(master_fd)

        # Install a continuous drainer so the child's subsequent TUI
        # redraws do not fill the kernel PTY buffer. Must be running
        # before we send the first prompt — once the child starts
        # generating, output volume is high.
        self._start_pty_drain(session_key)

        # Dismiss any startup mode-selection screen.  Claude Code's TUI may
        # show a "bypass permissions on (shift+tab to cycle)" screen at
        # startup.  Sending a raw Enter dismisses it; if no startup screen
        # is present the Enter is harmlessly absorbed by the chat input
        # (empty submit is a no-op).
        try:
            os.write(master_fd, b"\r")
        except OSError:
            pass
        await asyncio.sleep(1.0)

        # Send the initial prompt.  On resume, send a minimal continue
        # message — interactive mode requires user input to start
        # generating, and re-sending the full stage brief causes the
        # agent to re-process old recovery exchanges.
        if request.resume:
            self._write_pty(master_fd, _RESUME_PROMPT + "\r")
        else:
            self._write_pty(master_fd, request.prompt + "\r")

        if request.completion_check is not None:
            self._completion_checks[session_key] = request.completion_check
        if request.persist_until_timeout:
            self._persist_flags[session_key] = True
        if request.pause_check is not None:
            self._pause_checks[session_key] = request.pause_check
        if request.on_pause is not None:
            self._on_pause[session_key] = request.on_pause

        return session_key

    async def send(self, session_key: str, message: str) -> None:
        """Send a follow-up message to the interactive session.

        Unlike :class:`StreamAdapter`, this simply writes to the PTY — no
        kill-and-resume needed.  The caller (TUI) should call
        ``interrupt()`` first if the agent is currently generating.
        """
        fd = self._master_fds.get(session_key)
        if fd is None:
            raise RuntimeError(f"Session {session_key} has no PTY master fd")
        self._write_pty(fd, message + "\r")

    async def interrupt(self, session_key: str) -> None:
        """Interrupt the current generation (Ctrl+C) without killing the process.

        After writing Ctrl+C we must wait for Claude's Ink TUI to
        unmount its "generating" view and remount the input widget —
        this is an async React re-render that takes ~hundreds of ms.
        Without the wait, a follow-up send() lands in a component that
        is not ready to accept input and the message is silently
        dropped (the symptom: "I hit Ctrl+C, typed a new message, and
        nothing happened").
        """
        fd = self._master_fds.get(session_key)
        if fd is not None:
            # Raw write — control chars must NOT be paste-wrapped.
            try:
                os.write(fd, b"\x03")
            except OSError:
                pass
            await asyncio.sleep(_POST_INTERRUPT_DELAY)

    async def stream(self, session_key: str, max_recoveries: int | None = None) -> AsyncIterator[SessionEvent]:
        """Yield SessionEvents by tailing the session's JSONL file.

        The stream ends when either:
        - The process exits (killed by timeout or explicit :meth:`stop`)
        - The agent finished its turn (``stop_reason == "end_turn"``) and
          then stayed idle for ``_END_DETECTION_IDLE`` seconds
        - A non-end stop (``stop_sequence`` / ``max_tokens``) occurs and
          recovery budget is exhausted, then idle for ``_END_DETECTION_IDLE``
        """
        # Interactive mode does not emit a ``system/init`` JSONL event
        # (unlike ACP / ``-p`` mode).  Synthesise one so the TUI can
        # display "Session started" / "Session resumed" and track state.
        is_resume = self._session_is_resume.get(session_key, False)
        yield SessionEvent(
            type="init",
            data={"type": "system", "subtype": "init",
                  "session_id": self._session_ids.get(session_key, session_key),
                  "resume": is_resume},
            timestamp=time.time(),
        )

        last_event_time = time.monotonic()
        saw_end_turn = False
        saw_non_end_stop = False
        turn_recoveries = 0
        _consecutive_type = ""
        _consecutive_count = 0

        while True:
            events = self._poll_jsonl_with_results(session_key)
            for event in events:
                last_event_time = time.monotonic()
                if event.type == "result":
                    saw_end_turn = True
                elif event.type in ("tool_use", "tool_result", "init", "user"):
                    pass
                else:
                    saw_end_turn = False
                yield event

                # Auto-recover from stop_sequence / max_tokens.
                if event.type == "assistant":
                    msg = event.data.get("message", {})
                    if not isinstance(msg, dict):
                        continue
                    sr = msg.get("stop_reason")
                    if sr == "end_turn":
                        saw_end_turn = True
                        # A clean end_turn means whatever recovery streak
                        # we were tracking is over; reset so unrelated
                        # later failures don't accumulate against it.
                        _consecutive_type = ""
                        _consecutive_count = 0
                    elif sr in ("stop_sequence", "max_tokens"):
                        msg = event.data.get("message", {})
                        is_api_err = bool(event.data.get("isApiErrorMessage", False))
                        api_err_val = (msg.get("apiError") if isinstance(msg, dict) else "") or ""

                        # Context exhaustion: end stream immediately
                        if is_api_err and api_err_val == "max_output_tokens":
                            log.warning("Session %s: context exhausted (max_output_tokens), "
                                        "ending stream for continuation", session_key)
                            self._recovery_abort_reasons[session_key] = "context_exhausted"
                            return

                        # Classify recovery type
                        if is_api_err:
                            rtype = "api_error"
                        else:
                            text = "".join(c.get("text", "") for c in msg.get("content", [])
                                           if isinstance(c, dict) and c.get("type") == "text")
                            if text.strip() == "No response requested.":
                                rtype = "no_response"
                            elif sr == "max_tokens":
                                rtype = "max_tokens"
                            else:
                                rtype = "other"

                        # Track consecutive same-type recoveries
                        if rtype == _consecutive_type:
                            _consecutive_count += 1
                        else:
                            _consecutive_type = rtype
                            _consecutive_count = 1

                        # Early termination for repeated api_error / no_response:
                        # this many consecutive same-type failures indicates a
                        # systemic issue (API outage, model fault) — end the
                        # stream so SessionManager surfaces the abort reason
                        # to the caller rather than burning time on recovery.
                        if rtype in ("api_error", "no_response") and _consecutive_count >= _MAX_CONSECUTIVE_SAME:
                            log.warning("Session %s: %d consecutive '%s' recoveries, aborting stream",
                                        session_key, _consecutive_count, rtype)
                            self._recovery_abort_reasons[session_key] = rtype
                            return
                        elif max_recoveries is None or turn_recoveries < max_recoveries:
                            budget_str = f"({turn_recoveries + 1}/{max_recoveries})" if max_recoveries else f"({turn_recoveries + 1})"
                            log.warning(
                                "Session %s: stop_reason=%s, sending recovery %s",
                                session_key, sr, budget_str,
                            )
                            await asyncio.sleep(_RECOVERY_DELAY)
                            fd = self._master_fds.get(session_key)
                            if fd is not None:
                                self._write_pty(
                                    fd,
                                    "Please continue from where you left off.\r",
                                )
                            turn_recoveries += 1
                            saw_non_end_stop = False
                            saw_end_turn = False
                        else:
                            saw_non_end_stop = True
                            self._recovery_abort_reasons[session_key] = "budget_exhausted"
                            log.warning(
                                "Session %s: stop_reason=%s but recovery budget exhausted",
                                session_key, sr,
                            )

            # Runtime silence recovery: no JSONL for a long time.
            # Send "continue" WITHOUT Ctrl+C to avoid interrupting a
            # legitimate long API response. Skip if the agent is waiting
            # for user input (pause_check).
            if max_recoveries is None or turn_recoveries < max_recoveries:
                idle = time.monotonic() - last_event_time
                if idle >= _RUNTIME_SILENCE_SECONDS:
                    pause = self._pause_checks.get(session_key)
                    if pause and pause():
                        last_event_time = time.monotonic()
                    else:
                        budget_str = f"({turn_recoveries + 1}/{max_recoveries})" if max_recoveries else f"({turn_recoveries + 1})"
                        log.warning(
                            "Session %s: no JSONL for %.0fs, sending continue %s",
                            session_key, idle, budget_str,
                        )
                        fd = self._master_fds.get(session_key)
                        if fd is not None:
                            self._write_pty(
                                fd,
                                "Please continue from where you left off.\r",
                            )
                        last_event_time = time.monotonic()
                        turn_recoveries += 1
                        saw_non_end_stop = False
                        saw_end_turn = False

            alive = await self.is_alive(session_key)

            if not alive:
                final = self._poll_jsonl_with_results(session_key)
                for e in final:
                    yield e
                return

            # Idle detection: end stream after _END_DETECTION_IDLE seconds
            # once the agent finished a turn or hit a non-end stop.
            # If a completion_check is registered and the stage artifact
            # has not been produced yet, send a continue prompt instead
            # of ending the session — the agent returned to the user
            # prompt prematurely.
            # EXCEPTION: if pause_check signals the agent is waiting for
            # user input (e.g. question.json), suppress auto-continue and
            # just wait.
            if saw_end_turn or saw_non_end_stop:
                idle = time.monotonic() - last_event_time
                if idle > _END_DETECTION_IDLE:
                    check = self._completion_checks.get(session_key)
                    pause = self._pause_checks.get(session_key)
                    if check and not check():
                        if pause and pause():
                            # Agent is waiting for user input — don't auto-continue.
                            # Fire the one-shot on_pause callback the first time.
                            if session_key not in self._pause_fired:
                                self._pause_fired.add(session_key)
                                on_pause = self._on_pause.get(session_key)
                                if on_pause:
                                    on_pause()
                            last_event_time = time.monotonic()
                            saw_end_turn = False
                            continue
                        if max_recoveries is None or turn_recoveries < max_recoveries:
                            budget_str = f"({turn_recoveries + 1}/{max_recoveries})" if max_recoveries else f"({turn_recoveries + 1})"
                            log.warning(
                                "Session %s: end_turn but stage artifact not "
                                "produced, sending continue %s",
                                session_key, budget_str,
                            )
                            fd = self._master_fds.get(session_key)
                            if fd is not None:
                                self._write_pty(
                                    fd, _STAGE_CONTINUE_PROMPT + "\r",
                                )
                            turn_recoveries += 1
                            saw_end_turn = False
                            saw_non_end_stop = False
                            last_event_time = time.monotonic()
                            continue
                        else:
                            self._recovery_abort_reasons[session_key] = "budget_exhausted"
                            log.warning(
                                "Session %s: artifact not produced but "
                                "recovery budget exhausted",
                                session_key,
                            )
                    if saw_non_end_stop and not saw_end_turn:
                        log.warning(
                            "Session %s: no activity after non-end stop, "
                            "ending stream",
                            session_key,
                        )
                    return

            # Check if the stage was externally marked complete (e.g. /skip-prepare
            # wrote complete.json). End the stream immediately in that case.
            # Skip for persist_until_timeout sessions (e.g. implement: agent
            # keeps iterating even after producing best_result.jsonl).
            check = self._completion_checks.get(session_key)
            if check and check() and not self._persist_flags.get(session_key):
                log.info("Session %s: completion_check True, ending stream", session_key)
                return

            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self, session_key: str, grace_seconds: float = 5.0) -> None:
        """Gracefully stop a session: Ctrl+C → /exit → SIGTERM → SIGKILL."""
        fd = self._master_fds.get(session_key)
        process = self._sessions.get(session_key)

        if fd is not None:
            try:
                os.write(fd, b"\x03")  # Ctrl+C — raw, not paste-wrapped
                await asyncio.sleep(0.3)
                self._write_pty(fd, "/exit\r")
            except OSError:
                pass

        if process and process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    try:
                        process.send_signal(signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

        self._cleanup(session_key)

    async def is_alive(self, session_key: str) -> bool:
        process = self._sessions.get(session_key)
        return process is not None and process.returncode is None

    async def collect_result(
        self, session_key: str, started_at: float
    ) -> SessionResult:
        process = self._sessions.get(session_key)
        exit_code = None
        if process:
            await process.wait()
            exit_code = process.returncode

        elapsed = time.time() - started_at
        log_path = self._log_path_for(session_key)
        final_result = self._read_final_result(log_path)

        self._cleanup(session_key)
        self._last_log_paths.pop(session_key, None)

        recovery_abort_reason = self._recovery_abort_reasons.pop(session_key, "")

        return SessionResult(
            session_id=self._session_ids.get(session_key, session_key),
            exit_code=exit_code,
            elapsed_seconds=elapsed,
            log_path=log_path,
            final_result=final_result,
            error=None,
            recovery_abort_reason=recovery_abort_reason,
        )

    # ------------------------------------------------------------------
    # Override points for subclasses (e.g. DockerPtyAdapter)
    # ------------------------------------------------------------------

    def _build_command(self, request: SessionRequest, _session_key: str) -> list[str]:
        cmd = [self._claude]
        if request.resume and request.session_id:
            cmd.extend(["--resume", request.session_id])
        elif request.session_id:
            cmd.extend(["--session-id", request.session_id])
        if request.model:
            cmd.extend(["--model", request.model])
        # We intentionally do NOT pass any bypass CLI flag
        # (--dangerously-skip-permissions or --permission-mode
        # bypassPermissions). In interactive PTY mode both flags open
        # a blocking startup dialog ("1. No, exit / 2. Yes, I accept")
        # which is hard to dismiss programmatically from here.
        # Bypass is instead enabled via the per-session
        # settings.local.json written by DockerContainer — set
        # `skipDangerousModePermissionPrompt: true` (suppresses the
        # dialog) and `permissions.defaultMode: "bypassPermissions"`
        # (enables bypass). See src/docker/container.py.
        return cmd

    def _subprocess_env(self) -> dict[str, str] | None:
        """Return env dict for subprocess.

        GPU policy: default-deny — CUDA_VISIBLE_DEVICES is always cleared.
        Agents acquire GPUs via gpu_helpers.gpu_session() (in their own
        process), which sets the env var on entry and restores it on exit.
        We still resolve `config` for future use (e.g. token injection in
        the implement/prepare nodes) but no longer branch on `config.gpus`.
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

    def _compute_project_dir(self) -> Path:
        # Claude Code writes JSONL to ~/.claude/projects/<dir-name>/<session>.jsonl
        # where <dir-name> is the CWD of the claude process with "/" → "-"
        # AND "_" → "-".  The underscore conversion is easy to miss — it
        # only matters when the workspace path contains underscores (e.g.
        # "00_Eureka" or run timestamps like "20260516_105256").
        cwd = str(self._workspace.resolve()) if self._workspace else os.getcwd()
        dir_name = cwd.replace("/", "-").replace("_", "-")
        return Path.home() / ".claude" / "projects" / dir_name

    # ------------------------------------------------------------------
    # Private internals
    # ------------------------------------------------------------------

    @staticmethod
    def _write_pty(fd: int, data: str) -> None:
        """Write user input to the PTY master, wrapping in bracketed-paste.

        Ink's input widget processes a paste event atomically; without the
        paste markers it has to process each character through its keyboard
        reducer, and prompts above ~200 chars are silently truncated.
        A trailing "\\r" (Enter) is moved outside the paste so the agent
        receives the submit signal after the paste ends, not before.
        """
        if not data:
            return
        trailing_cr = data.endswith("\r")
        body = data[:-1] if trailing_cr else data
        payload = _BRACKETED_PASTE_START + body + _BRACKETED_PASTE_END
        if trailing_cr:
            payload += "\r"
        try:
            os.write(fd, payload.encode())
        except OSError:
            pass

    @staticmethod
    def _drain_pty(fd: int) -> None:
        """Read and discard all pending PTY output. The TUI's startup
        banner can easily exceed a single 64KB read, so we loop until EAGAIN."""
        while True:
            try:
                data = os.read(fd, 65536)
                if not data:
                    return
            except (BlockingIOError, OSError):
                return

    def _start_pty_drain(self, session_key: str) -> None:
        """Start a background task that continuously drains the PTY
        master fd for this session.

        The child (claude's Ink TUI) writes a large volume of ANSI
        redraw sequences. If nobody reads them, the kernel PTY buffer
        fills, the child blocks in write(), and its input event loop
        stalls — so Ctrl+C and new messages appear to do nothing.
        """
        fd = self._master_fds.get(session_key)
        if fd is None:
            return
        # Avoid double-starting.
        existing = self._drain_tasks.get(session_key)
        if existing is not None and not existing.done():
            return

        loop = asyncio.get_running_loop()
        done = asyncio.Event()

        def _reader() -> None:
            try:
                while True:
                    try:
                        data = os.read(fd, _PTY_DRAIN_CHUNK)
                    except BlockingIOError:
                        return  # Shouldn't happen with add_reader, but be safe.
                    except OSError:
                        # fd was closed (session ending).
                        done.set()
                        try:
                            loop.remove_reader(fd)
                        except (ValueError, KeyError):
                            pass
                        return
                    if not data:
                        # EOF — child closed its end.
                        done.set()
                        try:
                            loop.remove_reader(fd)
                        except (ValueError, KeyError):
                            pass
                        return
                    # Discard. We don't need the TUI output — structured
                    # events come from the JSONL file on disk.
            except Exception:
                log.exception("PTY drain reader crashed for %s", session_key)
                done.set()

        async def _waiter() -> None:
            try:
                loop.add_reader(fd, _reader)
            except Exception:
                log.exception("Could not install PTY drain reader for %s", session_key)
                return
            try:
                await done.wait()
            finally:
                try:
                    loop.remove_reader(fd)
                except (ValueError, KeyError, OSError):
                    pass

        self._drain_tasks[session_key] = loop.create_task(_waiter())

    def _jsonl_path_for(self, session_key: str) -> Path:
        actual_id = self._session_ids.get(session_key, session_key)
        return self._project_dir / f"{actual_id}.jsonl"

    def _log_path_for(self, session_key: str) -> Path:
        handle = self._log_handles.get(session_key)
        if handle and hasattr(handle, "name"):
            return Path(handle.name)
        # Fallback when the handle has already been closed (e.g. after
        # _cleanup). We remember the last-known path so collect_result,
        # which runs post-stop, can still locate the transcript file.
        cached = self._last_log_paths.get(session_key)
        if cached is not None:
            return cached
        return self._jsonl_path_for(session_key)

    def _poll_jsonl(self, session_key: str) -> list[SessionEvent]:
        path = self._jsonl_path_for(session_key)
        if not path.exists():
            return []

        events: list[SessionEvent] = []
        offset = self._jsonl_offsets.get(session_key, 0)

        try:
            with open(path, "r") as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        for event in self._jsonl_to_session_events(data):
                            events.append(event)
                        self._write_log(session_key, data)
                    except json.JSONDecodeError:
                        pass
                self._jsonl_offsets[session_key] = f.tell()
        except OSError:
            pass

        return events

    @staticmethod
    def _jsonl_to_session_events(data: dict) -> list[SessionEvent]:
        """Convert one JSONL line into zero or more SessionEvents.

        A single ``assistant`` message may contain multiple content blocks
        (thinking, text, tool_use).  We emit the parent assistant event
        plus a dedicated ``tool_use`` event per tool invocation so
        downstream consumers can track tool activity without parsing the
        nested content array.  Similarly, ``user`` messages containing
        ``tool_result`` blocks emit separate ``tool_result`` events.
        """
        etype = data.get("type", "")
        events: list[SessionEvent] = []

        if etype == "system" and data.get("subtype") == "init":
            events.append(SessionEvent(type="init", data=data, timestamp=time.time()))
            return events

        if etype == "assistant":
            msg = data.get("message", {})
            events.append(SessionEvent(type="assistant", data=data, timestamp=time.time()))
            for cb in msg.get("content", []):
                if isinstance(cb, dict) and cb.get("type") == "tool_use":
                    events.append(SessionEvent(
                        type="tool_use",
                        data={**cb, "session_id": data.get("session_id", "")},
                        timestamp=time.time(),
                    ))
            return events

        if etype == "user":
            events.append(SessionEvent(type="user", data=data, timestamp=time.time()))
            msg = data.get("message", {})
            if isinstance(msg, dict):
                for cb in msg.get("content", []):
                    if isinstance(cb, dict) and cb.get("type") == "tool_result":
                        events.append(SessionEvent(
                            type="tool_result",
                            data={**cb, "session_id": data.get("session_id", "")},
                            timestamp=time.time(),
                        ))
            return events

        return events

    def _poll_jsonl_with_results(self, session_key: str) -> list[SessionEvent]:
        """Like _poll_jsonl but also emits synthetic 'result' events."""
        raw_events = self._poll_jsonl(session_key)
        result: list[SessionEvent] = []
        for event in raw_events:
            result.append(event)
            if event.type == "assistant":
                msg = event.data.get("message", {})
                sr = msg.get("stop_reason")
                if sr == "end_turn":
                    content = msg.get("content", [])
                    texts = [
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    result_text = "\n".join(texts) if texts else ""
                    result.append(
                        SessionEvent(
                            type="result",
                            data={
                                "type": "result",
                                "result": result_text,
                                "session_id": event.data.get("session_id", ""),
                            },
                            timestamp=time.time(),
                        )
                    )
        return result

    def _write_log(self, session_key: str, data: dict) -> None:
        handle = self._log_handles.get(session_key)
        if handle is not None:
            handle.write(json.dumps(data) + "\n")
            handle.flush()

    def _read_final_result(self, log_path: Path) -> dict | None:
        if not log_path.exists():
            return None
        try:
            # Walk the file looking for the last assistant message.
            last_assistant: dict | None = None
            for line in log_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "assistant":
                        last_assistant = data
                except json.JSONDecodeError:
                    pass
            if last_assistant:
                msg = last_assistant.get("message", {})
                content = msg.get("content", [])
                texts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                return {
                    "type": "result",
                    "result": "\n".join(texts) if texts else "",
                    "session_id": last_assistant.get("session_id", ""),
                }
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _cleanup(self, session_key: str) -> None:
        handle = self._log_handles.pop(session_key, None)
        if handle:
            handle.close()
        # Stop the PTY drain task BEFORE closing the fd; otherwise its
        # reader callback sees a closed fd and may raise. cancel() is
        # fine because the reader tolerates OSError.
        drain_task = self._drain_tasks.pop(session_key, None)
        if drain_task is not None and not drain_task.done():
            drain_task.cancel()
        self._sessions.pop(session_key, None)
        self._session_ids.pop(session_key, None)
        self._session_is_resume.pop(session_key, None)
        self._jsonl_offsets.pop(session_key, None)
        self._completion_checks.pop(session_key, None)
        self._persist_flags.pop(session_key, None)
        self._pause_checks.pop(session_key, None)
        self._on_pause.pop(session_key, None)
        self._pause_fired.discard(session_key)
        fd = self._master_fds.pop(session_key, None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
