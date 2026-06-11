"""Session lifecycle manager: spawn, monitor, timeout, kill.

This module is generic — it consumes an adapter but is not
adapter-specific. Any adapter implementing the same interface (start, stream,
stop, is_alive, collect_result, send) can be used.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections.abc import Callable
from typing import Any

from ..acp.protocol import SessionRequest, SessionResult
from ..config import Config

log = logging.getLogger(__name__)

# If a session produces zero events for this many seconds, run an
# in-container diagnostic to help diagnose silent hangs.
_DIAG_SILENCE_SECONDS = 30.0
_DIAG_COOLDOWN = 60.0  # only run once per session
_WARN_THRESHOLD_SECONDS = 300.0  # fire warning this many seconds before deadline


class SessionManager:
    """Manages one or more agent sessions with timeout and cost-limit enforcement."""

    def __init__(
        self,
        adapter: Any,
        config: Config,
        event_queue: asyncio.Queue | None = None,
        cost_limit: float | None = None,
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._event_queue = event_queue
        self._cost_limit = cost_limit

    async def run_single(
        self, request: SessionRequest, timeout: float | None = None
    ) -> SessionResult:
        """Run one session to completion or timeout. Used by propose stage."""
        started_at = time.time()
        session_key = await self._adapter.start(request)

        try:
            result = await self._monitor_session(
                session_key, started_at, timeout,
                completion_check=request.completion_check,
                warning_prompt=request.warning_prompt,
                max_recoveries=request.max_recoveries,
            )
        except Exception:
            await self._adapter.stop(session_key)
            raise

        self._report_nonzero_exit(session_key, result)
        return result

    async def run_parallel(
        self,
        requests: list[SessionRequest],
        timeout: float | None = None,
    ) -> dict[str, SessionResult]:
        """Run N sessions in parallel, each with per-session timeout.

        Returns a dict mapping session_key -> SessionResult.
        """
        started_at = time.time()

        # Launch all sessions
        session_keys: list[str] = []
        for request in requests:
            key = await self._adapter.start(request)
            session_keys.append(key)

        # Monitor all in parallel
        tasks = [
            self._monitor_session(
                key, started_at, timeout,
                completion_check=requests[i].completion_check,
                warning_prompt=requests[i].warning_prompt,
                max_recoveries=requests[i].max_recoveries,
            )
            for i, key in enumerate(session_keys)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict, handling any exceptions
        output: dict[str, SessionResult] = {}
        for key, result in zip(session_keys, results):
            if isinstance(result, Exception):
                log.error("Session %s failed with exception: %s", key, result)
                await self._adapter.stop(key)
                output[key] = SessionResult(
                    session_id=key,
                    exit_code=-1,
                    elapsed_seconds=time.time() - started_at,
                    log_path=self._adapter._log_path_for(key),
                    error=str(result),
                )
            else:
                output[key] = result
                self._report_nonzero_exit(key, result)

        return output

    async def _monitor_session(
        self,
        session_key: str,
        started_at: float,
        timeout: float | None,
        completion_check: Callable[[], bool] | None = None,
        warning_prompt: str | None = None,
        max_recoveries: int | None = None,
    ) -> SessionResult:
        """Monitor one session: drain events, enforce timeout and cost limit.

        Timeout and cost-limit enforcement run in separate coroutines so they
        fire even when the agent is silent. Without the background killer, a
        silent agent would overrun its budget indefinitely.
        """
        deadline = (started_at + timeout) if timeout else None
        killer: asyncio.Task | None = None
        if deadline is not None:
            killer = asyncio.create_task(self._kill_at_deadline(session_key, deadline))
        cost_killer: asyncio.Task | None = None
        if self._cost_limit is not None:
            cost_killer = asyncio.create_task(
                self._kill_at_cost_limit(session_key, self._cost_limit)
            )
        warner: asyncio.Task | None = None
        if deadline is not None and warning_prompt is not None and timeout > _WARN_THRESHOLD_SECONDS:
            warner = asyncio.create_task(
                self._warn_at_threshold(session_key, deadline, completion_check, warning_prompt)
            )

        first_event_time: float | None = None
        last_event_time = started_at
        diag_run = False

        try:
            async for event in self._adapter.stream(session_key, max_recoveries=max_recoveries):
                if self._event_queue is not None:
                    self._event_queue.put_nowait((session_key, event))
                now = time.time()
                if first_event_time is None:
                    first_event_time = now
                    log.info("Session %s: first event at +%.1fs",
                             session_key, now - started_at)
                last_event_time = now
                # Fast path: if the deadline has already passed by the time an
                # event arrives, stop draining. The background killer has (or
                # will shortly) terminate the subprocess.
                if deadline is not None and now >= deadline:
                    break
                # Fast path: cost limit exceeded
                if self._cost_exceeded():
                    break

                # Diagnostic: if no events for a long time, probe the container
                if not diag_run and (now - started_at) > _DIAG_SILENCE_SECONDS:
                    silence = now - last_event_time
                    if silence > _DIAG_SILENCE_SECONDS:
                        await self._run_silence_diagnostic(session_key)
                        diag_run = True
        finally:
            if killer is not None:
                killer.cancel()
            if cost_killer is not None:
                cost_killer.cancel()
            if warner is not None:
                warner.cancel()

        # Log whether we ever saw an event
        if first_event_time is None:
            log.warning("Session %s: ZERO events during entire lifetime (%.1fs)",
                        session_key, time.time() - started_at)

        # Ensure termination even if the stream ended without the killer running.
        if await self._adapter.is_alive(session_key):
            await self._adapter.stop(session_key)

        return await self._adapter.collect_result(session_key, started_at)

    async def _kill_at_deadline(self, session_key: str, deadline: float) -> None:
        """Sleep until the deadline, then stop the session subprocess."""
        delay = max(0.0, deadline - time.time())
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        log.warning("Session %s: timeout reached, stopping", session_key)
        await self._adapter.stop(session_key)

    async def _kill_at_cost_limit(self, session_key: str, cost_limit: float) -> None:
        """Poll cost and stop the session when the limit is exceeded."""
        try:
            while True:
                await asyncio.sleep(5.0)
                if self._cost_exceeded():
                    log.warning(
                        "Session %s: cost limit $%.2f exceeded, stopping",
                        session_key, cost_limit,
                    )
                    await self._adapter.stop(session_key)
                    return
        except asyncio.CancelledError:
            return

    async def _warn_at_threshold(
        self,
        session_key: str,
        deadline: float,
        completion_check: Callable[[], bool] | None,
        warning_prompt: str,
    ) -> None:
        """Sleep until 5 minutes before deadline, then inject a warning prompt.

        Skips the warning if completion_check indicates the deliverable already
        exists. Uses adapter.interrupt() + adapter.send() so the agent stops
        its current work and processes the warning immediately.
        """
        warn_time = deadline - _WARN_THRESHOLD_SECONDS
        delay = max(0.0, warn_time - time.time())
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        # Check if deliverable already produced
        if completion_check is not None and completion_check():
            log.info("Session %s: warning skipped, deliverable already produced", session_key)
            return
        log.warning("Session %s: 5-minute warning, injecting prompt", session_key)
        try:
            await self._adapter.interrupt(session_key)
            await self._adapter.send(session_key, warning_prompt)
        except Exception:
            log.warning("Session %s: failed to inject time warning", session_key, exc_info=True)

    def _cost_exceeded(self) -> bool:
        """Check if the global cost limit has been exceeded."""
        if self._cost_limit is None:
            return False
        try:
            from ..runtime import get_token_tracker
            tracker = get_token_tracker()
            cost = tracker.calculate_cost()
            return cost is not None and cost >= self._cost_limit
        except Exception:
            return False

    async def _run_silence_diagnostic(self, session_key: str) -> None:
        """Docker-exec into the container to capture process state when a
        session is producing no output. Helps diagnose silent hangs."""
        log.warning("Session %s: no events for %.0fs, running container diagnostic",
                     session_key, _DIAG_SILENCE_SECONDS)
        try:
            container_id = self._get_container_id()
            if not container_id:
                log.info("Session %s: no Docker container, skipping diagnostic",
                         session_key)
                return
            script = (
                'echo "=== PS ==="; ps auxf 2>/dev/null | head -30; '
                'echo "=== MEM ==="; free -h 2>/dev/null; '
                'echo "=== CLAUDE_PROC ==="; '
                'cpid=$(pgrep -f claude | head -1); '
                'if [ -n "$cpid" ]; then '
                'echo "claude pid=$cpid"; '
                'ls -la /proc/$cpid/fd 2>/dev/null | wc -l; '
                'echo "=== CLAUDE_JSON_SIZE ==="; '
                'wc -c $HOME/.claude.json 2>/dev/null; '
                'echo "=== SETTINGS_LOCAL ==="; '
                'cat /workspace/.claude/settings.local.json 2>/dev/null | head -5; '
                'fi; '
                'echo "=== ZOMBIES ==="; '
                'ps aux | grep -c "Z" 2>/dev/null'
            )
            cmd = ["docker", "exec", container_id, "bash", "-c", script]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            log.warning("Session %s diagnostic output:\n%s",
                        session_key, stdout.decode("utf-8", errors="replace"))
        except Exception as e:
            log.warning("Session %s diagnostic failed: %s", session_key, e)

    def _get_container_id(self) -> str | None:
        """Best-effort retrieval of the current Docker container ID."""
        try:
            from ..runtime import get_container
            container = get_container()
            if container and container.container_id:
                return container.container_id
        except Exception:
            pass
        return None

    @staticmethod
    def _report_nonzero_exit(session_key: str, result: SessionResult) -> None:
        """Surface non-zero exit codes and any captured stderr to the log.

        Without this, a session that dies instantly (bad CLI argument, auth
        failure, etc.) appears silent: no events stream, no error propagates,
        and upstream code sees only an empty log file.
        """
        if result.exit_code in (0, None):
            return
        log.warning(
            "Session %s exited with code %s: %s",
            session_key,
            result.exit_code,
            (result.error or "").strip() or "<no stderr captured>",
        )
