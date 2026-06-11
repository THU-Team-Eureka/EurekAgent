"""Docker-aware adapters: routes claude commands through docker exec.

Claude and any children it spawns run inside `setsid --wait` so they share a
POSIX session inside the container. At stop time we send SIGTERM to the whole
session via `docker exec <cid> kill -TERM -- -<leader_pid>`, matching the
cleanup native Claude Code performs when the user interrupts a tool.

The `--wait` flag is load-bearing: `docker exec` spawns its command as a
session leader, which forces `setsid` to fork. Without `--wait`, the setsid
parent exits the instant it forks; `docker exec` sees its direct child exit
and tears down the stdout pipe, so we only ever see the first `init` event.
`setsid --wait` keeps the parent alive until the child finishes, preserving
the pipe for the full session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import AsyncIterator

from ..acp.stream_adapter import StreamAdapter
from ..acp.protocol import SessionEvent, SessionRequest
from ..acp.pty_adapter import PtyAdapter, _POST_INTERRUPT_DELAY, _POLL_INTERVAL, _END_DETECTION_IDLE, _TUI_RENDER_DELAY, _RESUME_PROMPT
from .container import DockerContainer

log = logging.getLogger(__name__)

_STREAM_LIMIT = 2**22  # 4 MB readline buffer; proxy NDJSON lines can exceed the default 64 KB
_GRACE_SECONDS = 15.0  # seconds to wait for first JSONL event before recovery
_RECOVERY_DELAY = 5.0  # seconds to wait before sending recovery message
_MAX_CONSECUTIVE_SAME = 20  # consecutive same-type stop_sequence recoveries before abort
_RUNTIME_SILENCE_SECONDS = 300.0  # seconds of no JSONL + no pty_output before recovery
_PTY_ACTIVE_SECONDS = 60.0  # pty_output within this window means agent is active


class DockerStreamAdapter(StreamAdapter):
    """StreamAdapter subclass that runs claude inside a Docker container."""

    def __init__(self, claude_command: str, container: DockerContainer) -> None:
        super().__init__(claude_command)
        self._container = container

    async def start(self, request: SessionRequest) -> str:
        """Start a session inside the Docker container.

        Overrides cwd to "" since the container's workdir is /workspace.
        This applies to both initial and resume requests.
        """
        request = SessionRequest(
            prompt=request.prompt,
            model=request.model,
            cwd="",
            session_id=request.session_id,
            permissions=request.permissions,
            log_path=request.log_path,
            resume=request.resume,
            env=request.env,
        )
        return await super().start(request)

    def _build_command(self, request: SessionRequest, session_key: str) -> list[str]:
        """Wrap the base claude command with `setsid --wait` and `docker exec`."""
        base_cmd = super()._build_command(request, session_key)
        if not self._container.container_id:
            raise RuntimeError("Container not started")
        cmd = ["docker", "exec"]
        for key, value in request.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._container.container_id, "setsid", "--wait", *base_cmd])
        return cmd

    async def _kill_process_tree(
        self,
        session_key: str,
        process: asyncio.subprocess.Process,
        session_id: str,
    ) -> None:
        """SIGTERM every descendant of the claude process inside the container.

        Claude was launched under `setsid --wait` so it and its Bash children
        share one POSIX session. `kill -TERM -- -<pid>` sends to the entire
        session in one syscall. We locate the leader by matching the unique
        session UUID on the command line — both ``--session-id <uuid>`` (fresh
        start) and ``--resume <uuid>`` (kill-and-resume) must match so that
        the timeout killer and ``stop()`` can clean up after a resume.
        """
        if session_id:
            script = (
                f"pid=$(pgrep -f 'claude.*({session_id})' | head -n1); "
                f"[ -n \"$pid\" ] && kill -TERM -- -\"$pid\" || true"
            )
            kill_cmd = self._container.exec_command(["bash", "-c", script])
            try:
                proc = await asyncio.create_subprocess_exec(
                    *kill_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except Exception as e:
                log.debug("In-container tree kill failed for %s: %s", session_key, e)

        # Also SIGTERM the host-side `docker exec` client so its stdout pipe
        # closes promptly — without this, stream() stays blocked on readline
        # until the container-side process fully exits.
        await super()._kill_process_tree(session_key, process, session_id)


class DockerPtyAdapter(PtyAdapter):
    """PtyAdapter subclass that runs interactive Claude inside a Docker container.

    Uses a PTY proxy script (``pty_proxy.py``) running inside the container
    to create a real pseudo-terminal for Claude's interactive mode.  The host
    communicates with the proxy via ``docker exec -i`` stdin/stdout pipes using
    a JSON command/event protocol.  JSONL events are polled from inside the
    container by the proxy and forwarded over stdout — this avoids bind-mount
    caching issues where JSONL files written inside the container are not
    immediately visible on the host.
    """

    def __init__(
        self,
        claude_command: str,
        container: DockerContainer,
        workspace_dir: Path,
    ) -> None:
        self._container = container
        self._workspace_dir = workspace_dir
        self._host_project_dir = self._compute_host_project_dir()
        # Buffer for JSONL events received from the proxy, consumed by stream().
        self._proxy_events: dict[str, list[SessionEvent]] = {}
        # Prompts saved by start() for stream() grace-period recovery.
        self._pending_prompts: dict[str, str] = {}
        # Sessions where a blocker dialog was detected in pty_output.
        self._blocker_detected: set[str] = set()
        # Last pty_output timestamp per session (for liveness detection).
        self._last_pty_update: dict[str, float] = {}
        # Stage completion checks copied from SessionRequest; used by stream()
        # to decide whether end_turn means "task done" or "agent stopped early".
        self._completion_checks: dict[str, Callable[[], bool]] = {}
        # Sessions where completion_check=True should NOT terminate the stream
        # (e.g. implement stage: agent keeps iterating until timeout).
        self._persist_flags: dict[str, bool] = {}
        # Pause checks: when True, suppress auto-continue (agent waiting for user input).
        self._pause_checks: dict[str, Callable[[], bool]] = {}
        # One-shot callbacks called the first time pause_check returns True.
        self._on_pause: dict[str, Callable[[], None]] = {}
        self._pause_fired: set[str] = set()
        self._session_is_resume: dict[str, bool] = {}
        super().__init__(claude_command, workspace=workspace_dir)

    async def start(self, request: SessionRequest) -> str:
        """Start an interactive Claude session inside the Docker container.

        Flow:
            1. Spawn proxy via docker exec, send ``{"cmd":"start"}``.
            2. Wait for the proxy's ``ready`` event (claude process forked).
            3. Sleep briefly to let the Ink/React TUI finish its first paint;
               before this, keystrokes piped to the PTY are dropped.
            3.5. Send Enter to dismiss any startup selector (permission mode,
                 workspace trust).  Uses raw mode so the selector receives a
                 keystroke, not a bracketed-paste event.
            4. Write the initial prompt.

        Grace-period recovery (step 5) is handled inside :meth:`stream`:
        if no JSONL events appear within a grace period the stream loop
        sends Ctrl+C and re-sends the prompt.  This keeps ``start()``
        fast and avoids blocking the caller.
        """
        import uuid as _uuid

        session_key = request.session_id or str(_uuid.uuid4())
        if not request.session_id:
            request = SessionRequest(
                prompt=request.prompt,
                model=request.model,
                cwd="",
                session_id=session_key,
                permissions=request.permissions,
                log_path=request.log_path,
                resume=request.resume,
                env=request.env,
            )

        cmd = self._build_proxy_command(request.env)
        log.info("Starting Docker PTY session %s via proxy", session_key)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            limit=_STREAM_LIMIT,
        )
        log.info("Proxy subprocess started, pid=%s", process.pid)

        self._sessions[session_key] = process
        self._session_ids[session_key] = session_key
        self._session_is_resume[session_key] = request.resume
        self._proxy_events.setdefault(session_key, [])

        if request.log_path:
            log_path = Path(request.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handles[session_key] = open(log_path, "a")
            self._last_log_paths[session_key] = log_path

        # 1. Send the "start" command with claude args.
        claude_args = self._build_claude_args(request)
        start_msg = json.dumps({"cmd": "start", "args": claude_args}) + "\n"
        process.stdin.write(start_msg.encode())
        await process.stdin.drain()
        log.info("Sent start command to proxy, args=%s", claude_args)

        # 2. Wait for the proxy's "ready" event (claude process forked).
        log.info("Waiting for proxy ready event...")
        await self._read_proxy_until(session_key, "ready", timeout=10.0)
        log.info("Got proxy ready event")

        # 3. Give the TUI a moment to render its initial frame. Without this
        #    sleep, the first prompt is often consumed before the input widget
        #    is mounted and claude sees no user input.
        log.info("Sleeping %.1fs for TUI render...", _TUI_RENDER_DELAY)
        await asyncio.sleep(_TUI_RENDER_DELAY)
        await self._raise_if_proxy_exited(session_key, "before startup-screen dismiss")

        # 4. Dismiss any startup mode-selection screen.  Claude Code's TUI
        #    may show a "bypass permissions on (shift+tab to cycle)" screen
        #    at startup.  This screen does NOT accept text input — it expects
        #    keyboard navigation.  Sending a raw Enter dismisses it (the
        #    configured mode from settings.local.json is pre-selected).
        #    If no startup screen is present, the Enter is harmlessly absorbed
        #    by the chat input widget (empty submit is a no-op).
        dismiss_msg = json.dumps({"cmd": "write", "data": "\r", "raw": True}) + "\n"
        process.stdin.write(dismiss_msg.encode())
        await process.stdin.drain()
        log.info("Sent startup-screen dismiss (raw Enter)")
        await asyncio.sleep(1.0)
        await self._raise_if_proxy_exited(session_key, "before prompt send")

        # 5. Send the initial prompt.  On resume, send a minimal continue
        # message — interactive mode requires user input to start
        # generating, and re-sending the full stage brief causes the
        # agent to re-process old recovery exchanges.
        if request.resume:
            write_msg = json.dumps({"cmd": "write", "data": _RESUME_PROMPT + "\r"}) + "\n"
            process.stdin.write(write_msg.encode())
            await process.stdin.drain()
            log.info("Resume prompt sent to proxy, session_key=%s", session_key)
        else:
            write_msg = json.dumps({"cmd": "write", "data": request.prompt + "\r"}) + "\n"
            process.stdin.write(write_msg.encode())
            await process.stdin.drain()
            log.info("Prompt sent to proxy, session_key=%s", session_key)

        # Save prompt for stream() grace-period recovery.
        # On resume, use the same resume prompt as the initial send.
        self._pending_prompts[session_key] = (
            _RESUME_PROMPT if request.resume else request.prompt
        )

        if request.completion_check is not None:
            self._completion_checks[session_key] = request.completion_check
        if request.persist_until_timeout:
            self._persist_flags[session_key] = True
        if request.pause_check is not None:
            self._pause_checks[session_key] = request.pause_check
        if request.on_pause is not None:
            self._on_pause[session_key] = request.on_pause

        return session_key

    async def _raise_if_proxy_exited(self, session_key: str, context: str) -> None:
        """Surface early proxy/Claude exits before writes hit a closed stdin."""
        await self._drain_proxy_stdout(session_key)
        process = self._sessions.get(session_key)
        if process is None:
            return
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=0.01)
            except asyncio.TimeoutError:
                return
        raise RuntimeError(
            f"Docker PTY proxy exited {context} for session {session_key}. "
            "Check the preceding pty_output/claude_exited entries in run.log "
            "for the underlying CLI error."
        )

    async def send(self, session_key: str, message: str) -> None:
        """Write a follow-up message to the interactive session.

        The caller (TUI) should call ``interrupt()`` first if the agent is
        currently generating — this method just writes to the proxy pipe.
        """

        process = self._sessions.get(session_key)
        if process is None or process.returncode is not None:
            raise RuntimeError(f"Session {session_key} is not running")
        write_msg = json.dumps({"cmd": "write", "data": message + "\r"}) + "\n"
        process.stdin.write(write_msg.encode())
        await process.stdin.drain()

    async def interrupt(self, session_key: str) -> None:

        process = self._sessions.get(session_key)
        if process is None or process.returncode is not None:
            return
        msg = json.dumps({"cmd": "interrupt"}) + "\n"
        process.stdin.write(msg.encode())
        await process.stdin.drain()
        # Wait for Claude's Ink TUI to unmount its "generating" view and
        # remount the input widget.  Without this delay, a follow-up send()
        # lands in a component that is not ready to accept input and the
        # message is silently dropped.  See PtyAdapter.interrupt for the
        # native-PTY equivalent and the empirical justification.
        await asyncio.sleep(_POST_INTERRUPT_DELAY)

    async def _write_raw(self, session_key: str, data: str) -> None:
        """Write a raw keystroke to the session PTY.

        Unlike :meth:`send`, this does not append a newline and is written
        in raw mode, so it lands as a literal keystroke.  Used to dismiss
        startup selectors / confirmation dialogs that expect keyboard
        navigation (e.g. a raw Enter accepts the pre-selected permission
        mode) rather than text input.
        """
        process = self._sessions.get(session_key)
        if process is None or process.returncode is not None:
            return
        msg = json.dumps({"cmd": "write", "data": data, "raw": True}) + "\n"
        process.stdin.write(msg.encode())
        await process.stdin.drain()

    async def stream(self, session_key: str, max_recoveries: int | None = None) -> AsyncIterator[SessionEvent]:
        """Yield SessionEvents read from the proxy's stdout.

        Recovery mechanisms:
            A. Grace-period: if no JSONL events appear within
               ``_GRACE_SECONDS`` of session start, Ctrl+C + re-send prompt.
            B. stop_sequence / max_tokens: send "continue" so the agent
               retries the failed API call or completes a truncated response.
            C. pty_output blocker: if a blocking dialog is detected in
               pty_output, Ctrl+C + "continue" to dismiss it.
            D. Runtime silence: if no JSONL for ``_RUNTIME_SILENCE_SECONDS``
               and no spinner activity in pty_output, Ctrl+C + "continue".

        B/C/D share a ``turn_recoveries`` budget (None=unlimited, int=cap).
        Consecutive same-type B recoveries trigger early termination.
        A is one-shot (``recovery_done`` flag).
        """
        # Interactive mode does not emit a ``system/init`` JSONL event (unlike
        # ACP / ``-p`` mode).  Synthesise one so the TUI can display
        # "Session started" / "Session resumed" and track state.
        is_resume = self._session_is_resume.get(session_key, False)
        yield SessionEvent(
            type="init",
            data={"type": "system", "subtype": "init",
                  "session_id": self._session_ids.get(session_key, session_key),
                  "resume": is_resume},
            timestamp=time.time(),
        )

        last_event_time = time.monotonic()
        session_start = time.monotonic()
        saw_end_turn = False
        saw_non_end_stop = False
        recovery_done = False
        first_jsonl_seen = False
        turn_recoveries = 0
        _consecutive_type = ""
        _consecutive_count = 0

        while True:
            # First, drain any buffered events from proxy stdout reads.
            events = self._drain_proxy_events(session_key)
            for event in events:
                last_event_time = time.monotonic()
                if event.type in ("assistant", "user", "tool_use", "tool_result", "result"):
                    first_jsonl_seen = True
                if event.type == "result":
                    saw_end_turn = True
                elif event.type in ("tool_use", "tool_result", "init", "user"):
                    pass
                else:
                    saw_end_turn = False
                yield event
                # Interactive mode does not emit a "result" JSONL event.
                # Synthesize one from an assistant message with
                # stop_reason == "end_turn" so the idle detector fires.
                if event.type == "assistant":
                    msg = event.data.get("message", {})
                    if not isinstance(msg, dict):
                        continue
                    sr = msg.get("stop_reason")
                    if sr == "end_turn":
                        saw_end_turn = True
                        last_event_time = time.monotonic()
                        # A clean end_turn means whatever recovery streak
                        # we were tracking is over; reset so unrelated
                        # later failures don't accumulate against it.
                        _consecutive_type = ""
                        _consecutive_count = 0
                        yield SessionEvent(
                            type="result",
                            data={
                                "type": "result",
                                "result": "\n".join(
                                    c.get("text", "")
                                    for c in msg.get("content", [])
                                    if isinstance(c, dict) and c.get("type") == "text"
                                ),
                                "session_id": event.data.get("session_id", ""),
                            },
                            timestamp=time.time(),
                        )
                    elif sr in ("stop_sequence", "max_tokens"):
                        # Recovery B: auto-recover from API errors / truncation.
                        msg_data = event.data.get("message", {})
                        is_api_err = bool(event.data.get("isApiErrorMessage", False))
                        api_err_val = (msg_data.get("apiError") if isinstance(msg_data, dict) else "") or ""

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
                            text = "".join(c.get("text", "") for c in msg_data.get("content", [])
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
                            await self.send(
                                session_key,
                                "Please continue from where you left off.",
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

            # Recovery A: grace-period — prompt swallowed by startup dialog.
            if not recovery_done and not first_jsonl_seen:
                elapsed = time.monotonic() - session_start
                if elapsed >= _GRACE_SECONDS:
                    log.warning(
                        "Session %s: no JSONL after %.0fs, sending Ctrl+C + re-prompt",
                        session_key, _GRACE_SECONDS,
                    )
                    await self.interrupt(session_key)
                    await asyncio.sleep(1.0)
                    # Use get(), not pop(): if the prompt was swallowed by a
                    # still-present startup selector, Recovery-C below needs
                    # to re-send it after dismissing the selector.
                    prompt = self._pending_prompts.get(session_key, "")
                    if prompt:
                        await self.send(session_key, prompt)
                    recovery_done = True

            # Recovery C: pty_output blocker detected (permission/mode dialog).
            # These selectors (e.g. "bypass permissions on (shift+tab to
            # cycle)") ignore Ctrl+C and free text — a raw Enter accepts the
            # pre-selected option and dismisses them.  This path is
            # re-armed every time the blocker reappears in pty_output, so it
            # retries until the screen clears (bounded by max_recoveries).
            if (session_key in self._blocker_detected
                    and (max_recoveries is None or turn_recoveries < max_recoveries)):
                self._blocker_detected.discard(session_key)
                log.warning(
                    "Session %s: UI blocker detected via pty_output, "
                    "sending raw Enter to dismiss",
                    session_key,
                )
                await self._write_raw(session_key, "\r")
                await asyncio.sleep(1.0)
                if not first_jsonl_seen:
                    # Startup: the initial prompt was swallowed by the
                    # selector.  Re-send it now that the chat input should
                    # be active again.
                    prompt = self._pending_prompts.get(session_key, "")
                    if prompt:
                        await self.send(session_key, prompt)
                else:
                    await self.send(
                        session_key,
                        "Please continue from where you left off.",
                    )
                turn_recoveries += 1
                saw_non_end_stop = False
                saw_end_turn = False

            # Recovery D: runtime silence — no JSONL for a long time and
            # no pty_output activity, meaning the agent is stuck (not just slow).
            # Send "continue" WITHOUT Ctrl+C first — if the agent is just
            # slow, this is harmless.  If it's blocked by a dialog, recovery
            # C (pty_output blocker) will handle it on the next sample.
            # Skip recovery when pause_check indicates the agent is waiting
            # for user input (e.g. question.json exists).
            if (first_jsonl_seen
                    and (max_recoveries is None or turn_recoveries < max_recoveries)):
                idle = time.monotonic() - last_event_time
                if idle >= _RUNTIME_SILENCE_SECONDS:
                    pause = self._pause_checks.get(session_key)
                    if pause and pause():
                        last_event_time = time.monotonic()
                    else:
                        last_pty = self._last_pty_update.get(session_key, 0)
                        agent_active = (time.monotonic() - last_pty) < _PTY_ACTIVE_SECONDS
                        if not agent_active:
                            budget_str = f"({turn_recoveries + 1}/{max_recoveries})" if max_recoveries else f"({turn_recoveries + 1})"
                            log.warning(
                                "Session %s: no JSONL for %.0fs and no pty_output, "
                                "sending continue %s",
                                session_key, idle, budget_str,
                            )
                            await self.send(
                                session_key,
                                "Please continue from where you left off.",
                            )
                            last_event_time = time.monotonic()
                            turn_recoveries += 1
                            saw_non_end_stop = False
                            saw_end_turn = False
                        else:
                            log.info(
                                "Session %s: no JSONL for %.0fs but pty_output active",
                                session_key, idle,
                            )
                            last_event_time = time.monotonic()

            alive = await self.is_alive(session_key)

            if not alive:
                # Process exited — drain remaining proxy events.
                await self._drain_proxy_stdout(session_key)
                for event in self._drain_proxy_events(session_key):
                    yield event
                return

            # Idle detection: end stream after _END_DETECTION_IDLE seconds
            # of no new events once the agent has finished a turn or hit a
            # non-end stop (stop_sequence / max_tokens with budget exhausted).
            # If a completion_check is registered and the stage artifact
            # has not been produced yet, send a continue prompt instead
            # of ending the session — the agent returned to the user
            # prompt prematurely.
            # EXCEPTION: if pause_check signals the agent is waiting for
            # user input (e.g. question.json exists), suppress auto-continue
            # and just wait.
            if saw_end_turn or saw_non_end_stop:
                idle = time.monotonic() - last_event_time
                if idle > _END_DETECTION_IDLE:
                    check = self._completion_checks.get(session_key)
                    pause = self._pause_checks.get(session_key)
                    if check and not check():
                        if pause and pause():
                            # Agent is waiting for user input — don't auto-continue.
                            # Fire the one-shot on_pause callback (e.g. push
                            # prepare_question event to TUI) the first time.
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
                            await self.send(
                                session_key,
                                "Continue with your task. You have not yet "
                                "produced the required output files. If you "
                                "were summarizing or researching, proceed to "
                                "writing the deliverables now.",
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

            # Read more events from proxy stdout.
            await self._drain_proxy_stdout(session_key)
            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self, session_key: str, grace_seconds: float = 5.0) -> None:

        process = self._sessions.get(session_key)
        if not process or process.returncode is not None:
            self._cleanup(session_key)
            return

        try:
            msg = json.dumps({"cmd": "exit"}) + "\n"
            process.stdin.write(msg.encode())
            await process.stdin.drain()
        except OSError:
            pass

        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            actual_id = self._session_ids.get(session_key, "")
            await self._kill_in_container(session_key, actual_id)
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

        self._cleanup(session_key)

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def _compute_project_dir(self) -> Path:
        return self._host_project_dir

    def _compute_host_project_dir(self) -> Path:
        agent_home = (self._workspace_dir / ".agent_home").resolve()
        return agent_home / ".claude" / "projects" / "-workspace"

    # ------------------------------------------------------------------
    # Docker PTY proxy internals
    # ------------------------------------------------------------------

    def _build_proxy_command(self, env: dict[str, str] | None = None) -> list[str]:
        if not self._container.container_id:
            raise RuntimeError("Container not started")
        proxy_path = "/workspace/.eureka_internal/pty_proxy.py"
        home = str(Path.home())
        path_entries = [
            "/usr/local/bin",
            "/usr/local/sbin",
            "/workspace/.venv/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ]
        if platform.system() == "Linux":
            path_entries.append(f"{home}/.local/bin")
        cmd = [
            "docker", "exec", "-i",
            "-e", f"PATH={':'.join(path_entries)}",
            "-e", "TERM=xterm-256color",
            "-e", "VIRTUAL_ENV=/workspace/.venv",
        ]
        for key, value in (env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._container.container_id, "python3", proxy_path])
        return cmd

    def _build_claude_args(self, request: SessionRequest) -> list[str]:
        cmd = ["claude"]
        if request.resume and request.session_id:
            cmd.extend(["--resume", request.session_id])
        elif request.session_id:
            cmd.extend(["--session-id", request.session_id])
        if request.model:
            cmd.extend(["--model", request.model])
        # No bypass CLI flag — see PtyAdapter._build_command for the
        # rationale. Bypass is enabled via the settings.local.json
        # written into the workspace by DockerContainer.
        return cmd

    def _drain_proxy_events(self, session_key: str) -> list[SessionEvent]:
        """Return and clear buffered SessionEvents from the proxy."""
        return self._proxy_events.pop(session_key, [])

    async def _drain_proxy_stdout(self, session_key: str) -> None:
        """Read available lines from proxy stdout and buffer as SessionEvents."""

        process = self._sessions.get(session_key)
        if not process or process.stdout is None:
            return

        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=0.05)
            except asyncio.TimeoutError:
                return
            except ValueError:
                log.warning("Session %s: oversized proxy line, draining", session_key)
                try:
                    await process.stdout.read(process.stdout._limit * 2)
                except Exception:
                    pass
                continue
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            self._handle_proxy_event(session_key, data)

    # Patterns that indicate a startup dialog is blocking input.
    _BLOCKER_PATTERNS = (
        "shift+tab to cycle",  # permission mode selector
        "1. No, exit",         # bypass confirmation dialog
    )

    def _handle_proxy_event(self, session_key: str, data: dict) -> None:
        """Dispatch a single proxy event. Shared by the draining paths."""
        event_name = data.get("event")
        if event_name == "jsonl":
            jsonl_data = data.get("data", {})
            subagent_id = data.get("_subagent")  # set by _SubagentPoller
            for event in self._jsonl_to_session_events(jsonl_data):
                # Tag subagent events so the TUI can distinguish them
                if subagent_id:
                    event.data["_subagent"] = subagent_id
                self._proxy_events.setdefault(session_key, []).append(event)
            self._write_log(session_key, jsonl_data)
        elif event_name == "debug":
            log.info("Session %s proxy debug: %s", session_key, data.get("msg"))
        elif event_name == "pty_output":
            # Sample of raw PTY output — startup banner, dialogs, or error
            # messages.  Logged at INFO so silent failures are visible in
            # run.log without needing DEBUG level.
            raw = str(data.get("data", ""))
            # Track liveness (any pty_output means the TUI/agent is alive)
            # and detect startup blocker dialogs so stream()'s Recovery-C
            # path can dismiss them.  Patterns are matched against the full
            # raw string, not the truncated sample that is logged below.
            self._last_pty_update[session_key] = time.monotonic()
            if any(p in raw for p in self._BLOCKER_PATTERNS):
                self._blocker_detected.add(session_key)
            log.info("Session %s pty_output: %r", session_key, raw[:500])
        elif event_name == "claude_exited":
            log.info("Session %s claude exited code=%s", session_key, data.get("code"))

    async def _read_proxy_until(
        self, session_key: str, target_event: str, timeout: float = 10.0
    ) -> None:
        """Read proxy stdout until we see a specific event type.

        Events that arrive before the target (jsonl, pty_output, etc.) are
        dispatched normally so they are not dropped.
        """

        process = self._sessions.get(session_key)
        if not process or process.stdout is None:
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=max(0.1, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                continue
            except ValueError:
                log.warning("Session %s: oversized proxy line in read_until, draining", session_key)
                try:
                    await process.stdout.read(process.stdout._limit * 2)
                except Exception:
                    pass
                continue
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if data.get("event") == target_event:
                return
            self._handle_proxy_event(session_key, data)
        log.warning("Session %s: proxy event '%s' not seen within %.0fs",
                     session_key, target_event, timeout)

    async def _kill_in_container(self, session_key: str, session_id: str) -> None:
        if session_id:
            # Match both --session-id (fresh) and --resume (resumed) processes.
            script = (
                f"pid=$(pgrep -f 'claude.*({session_id})' | head -n1); "
                f"[ -n \"$pid\" ] && kill -TERM -- -\"$pid\" || true"
            )
            kill_cmd = self._container.exec_command(["bash", "-c", script])
            try:
                proc = await asyncio.create_subprocess_exec(
                    *kill_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except Exception as e:
                log.debug("In-container tree kill failed for %s: %s", session_key, e)

    def _cleanup(self, session_key: str) -> None:
        self._proxy_events.pop(session_key, None)
        self._pending_prompts.pop(session_key, None)
        self._blocker_detected.discard(session_key)
        self._last_pty_update.pop(session_key, None)
        self._completion_checks.pop(session_key, None)
        self._persist_flags.pop(session_key, None)
        self._pause_checks.pop(session_key, None)
        self._on_pause.pop(session_key, None)
        self._pause_fired.discard(session_key)
        self._session_is_resume.pop(session_key, None)
        super()._cleanup(session_key)
