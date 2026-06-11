"""EurekAgent TUI application — Textual-based terminal interface."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Static

from ..acp import PtyAdapter
from ..config import Config
from ..pipeline import _reconstruct_state, resume_pipeline, run_pipeline
from ..resume_preflight import check_resume_preflight
from ..runtime import (
    get_active_adapter,
    get_active_session,
    push_session_resuming_event,
    push_session_sending_event,
    set_event_queue,
    set_token_tracker,
    set_user_message_queue,
)
from ..token_tracker import (
    TokenTracker, format_token_count, hydrate_tracker_from_run,
    scan_jsonl_usage_since,
)
from .screens.overview import OverviewScreen
from .screens.session import SessionScreen
from .widgets.logo import LogoBanner
from .widgets.resume_extra_time_dialog import ResumeExtraTimeDialog

log = logging.getLogger(__name__)

_CSS_PATH = Path(__file__).parent / "theme.tcss"

_EUREKA_THEME = Theme(
    name="eureka",
    primary="#B39DDB",
    secondary="#CE93D8",
    accent="#B39DDB",
    background="#000000",
    surface="#000000",
    panel="#1a1a1a",
    dark=True,
)


class EurekAgentApp(App):
    """Main TUI application for EurekAgent."""

    TITLE = "EurekAgent"
    token_summary: reactive[str] = reactive("")
    CSS_PATH = str(_CSS_PATH) if _CSS_PATH.exists() else None
    # Textual's built-in palette steals Ctrl+P and its "Keys" help screen has
    # no obvious dismiss path — we turn it off so our own key hints are the
    # only story the user sees.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+c", "init_quit", show=False),
        Binding("q", "confirm_quit", "Quit"),
        Binding("ctrl+s", "toggle_selection_mode", "Select text", show=False),
    ]

    def __init__(
        self,
        *,
        config: Config,
        resume_id: str | None = None,
        problem: Path | None = None,
        initial_code: Path | None = None,
        monitor_port: int | None = None,
        resume_extra_seconds: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.register_theme(_EUREKA_THEME)
        self.theme = "eureka"
        self._config = config
        self._resume_id = resume_id
        self._problem = problem
        self._initial_code = initial_code
        self._monitor_port = monitor_port
        self._resume_extra_seconds = resume_extra_seconds
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._user_message_queue: asyncio.Queue = asyncio.Queue()
        self._pipeline_task: asyncio.Task | None = None
        self._session_approach_map: dict[str, str] = {}  # session_key → approach_id
        # Per-approach formatted-event buffer. Populated on every event; replayed
        # into SessionScreen on mount so the user sees full history on entry.
        self._approach_events: dict[str, list[str]] = {}
        # When True, Textual's mouse reporting is disabled so the terminal
        # emulator can handle native drag-to-select / copy.
        self._selection_mode: bool = False
        # Session keys that have already produced an init event; used to
        # distinguish "Session started" from "Session resumed" on kill-and-resume.
        self._session_init_seen: set[str] = set()
        # Token tracking
        self._token_tracker: Any = None  # TokenTracker, set in _start_pipeline
        self._session_actual_ids: dict[str, str] = {}  # session_key -> Claude session_id
        self._agent_home_offsets: dict[str, int] = {}  # session_key -> file read offset
        self._jsonl_baseline_sizes: dict[Path, int] = {}
        # Subagent tracking: offsets for subagent JSONL files
        self._subagent_offsets: dict[str, int] = {}  # "session_key:subagent_id" -> offset
        self._subagent_event_seen: set[str] = set()  # dedup keys: "session_key:subagent_id:message_id"
        self._workspace_dir: Path | None = None
        # Quit confirmation state: Ctrl+C sets the flag and shows "Press q to
        # quit"; a subsequent q press exits. A 3-second timer clears the flag
        # so an accidental Ctrl+C doesn't linger.
        self._quit_pending: bool = False

    def compose(self) -> ComposeResult:
        # Splash: block logo shown once on startup
        yield LogoBanner(style="block", id="splash-logo")
        yield Static(
            f"Starting run... ({self._config.max_loops} loops, "
            f"{self._config.max_num_approaches} max approaches/loop)",
            id="splash-info",
        )

    async def on_mount(self) -> None:
        """Show splash briefly, then switch to overview screen."""
        # Install the persistent overview screen. SessionScreen is created
        # fresh each time the user enters an approach — this guarantees
        # on_mount fires and the RichLog is properly initialized.
        self.install_screen(
            OverviewScreen(
                stage="propose",
                round_idx=1,
                max_rounds=self._config.max_loops,
                max_num_approaches=self._config.max_num_approaches,
            ),
            name="overview",
        )

        # Brief splash delay, then start pipeline from a Textual worker. Modal
        # screens that wait for dismissal must be opened from a worker.
        self.set_timer(4.0, self._launch_pipeline_startup)

    def _launch_pipeline_startup(self) -> None:
        self.run_worker(
            self._start_pipeline(),
            name="pipeline-startup",
            group="pipeline-startup",
            exit_on_error=False,
            exclusive=True,
        )

    async def _start_pipeline(self) -> None:
        """Switch to overview and launch the pipeline as a background task."""
        try:
            self.push_screen("overview")
            # Create the token tracker here and register it as the global
            # singleton so both the TUI and pipeline share the same instance.
            # Without this, the pipeline creates its own empty tracker whose
            # totals are always zero when _save_summary reads them.
            self._token_tracker = TokenTracker(
                _input_price=self._config.input_token_price,
                _cache_creation_price=self._config.cache_creation_token_price,
                _cache_read_price=self._config.cache_read_token_price,
                _output_price=self._config.output_token_price,
            )
            set_token_tracker(self._token_tracker)
            if self._resume_id:
                run_dir = Path(self._config.runs_dir) / self._resume_id
                hydrate_tracker_from_run(run_dir, self._token_tracker)
                self._jsonl_baseline_sizes = self._snapshot_existing_jsonl(run_dir / "workspace")
                state = _reconstruct_state(self._resume_id, run_dir, self._config)
                self._hydrate_resume_overview(state)
                preflight = check_resume_preflight(run_dir, state, self._config)
                if preflight.needs_extra_time and self._resume_extra_seconds is None:
                    extra = await self.push_screen_wait(ResumeExtraTimeDialog(preflight))
                    if extra is None:
                        self.exit(
                            return_code=0,
                            message="Resume cancelled: no extra time granted.",
                        )
                        return
                    self._resume_extra_seconds = extra
            # Initialize token summary with zero values so it's visible from the start
            self._update_token_summary()
            self._pipeline_task = asyncio.create_task(self._run_pipeline())
            asyncio.create_task(self._consume_events())
            asyncio.create_task(self._consume_user_messages())
            # Poll .agent_home JSONL for stream-mode token usage
            self.set_interval(1.0, self._poll_agent_home_usage)
        except Exception as exc:
            log.exception("Pipeline startup failed")
            self._notify_error(str(exc))

    async def _run_pipeline(self) -> None:
        """Run the eureka-loop pipeline in the background."""
        set_event_queue(self._event_queue)
        set_user_message_queue(self._user_message_queue)

        try:
            if self._resume_id:
                await resume_pipeline(
                    self._resume_id,
                    runs_dir=Path(self._config.runs_dir),
                    config=self._config,
                    resume_extra_seconds=self._resume_extra_seconds,
                )
            else:
                await run_pipeline(
                    problem=self._problem,
                    initial_code=self._initial_code,
                    config=self._config,
                )
        except Exception as e:
            log.error("Pipeline error: %s", e, exc_info=True)
            self._notify_error(str(e))
        except BaseException as e:
            # CancelledError (when TUI exits) and other BaseException
            # subclasses bypass "except Exception" — log them here so they
            # aren't silently swallowed by the asyncio task machinery.
            log.error(
                "Pipeline terminated with BaseException (%s): %s",
                type(e).__name__, e, exc_info=True,
            )

    async def _consume_events(self) -> None:
        """Read events from the queue and dispatch to the active screen.

        A dispatch failure for one event must not kill the whole consumer —
        otherwise subsequent terminal events (e.g. `run_ended`) never land and
        the TUI freezes with no indication the pipeline has stopped.
        """
        while True:
            try:
                session_key, event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except Exception:
                log.exception("_consume_events: queue get failed, stopping consumer")
                break

            try:
                self._dispatch_event(session_key, event)
            except Exception:
                log.exception("Event dispatch failed; continuing")

    async def _consume_user_messages(self) -> None:
        """Drain user-typed messages from the queue into adapter.send().

        Kept in a separate task so a slow send() never blocks event dispatch
        or the TUI's own input loop. Errors are surfaced to the user as
        notifications; they never kill the consumer.
        """
        while True:
            try:
                context_key, text = await self._user_message_queue.get()
            except Exception:
                break

            adapter = get_active_adapter()
            session_key = get_active_session(context_key)
            if adapter is None or session_key is None:
                self.notify(
                    "No active session to receive the message.",
                    severity="warning",
                )
                continue

            # Echo the user turn into the TUI and the per-approach buffer so
            # replays (SessionScreen re-mount) and live viewers see it.
            display = f"[b cyan][You] {text}[/b cyan]"
            approach_id = self._session_approach_map.get(session_key)
            if approach_id:
                self._approach_events.setdefault(approach_id, []).append(display)
            overview = self._get_overview_screen()
            if overview is not None:
                overview.append_event(display)

            # PTY mode: interrupt current generation first so the agent
            # stops what it's doing and processes the new message.
            # ACP mode: send() does kill-and-resume internally.
            is_pty = isinstance(adapter, PtyAdapter)
            if is_pty:
                await adapter.interrupt(session_key)

            # Push a synthetic event so the TUI shows feedback immediately.
            # ACP mode says "Resuming session" (kill+resume takes 30-60s);
            # PTY mode says "Sending message" (write is near-instant).
            if not is_pty:
                push_session_resuming_event(session_key)
            else:
                push_session_sending_event(session_key)

            try:
                await adapter.send(session_key, text)
                # If the user answered a prepare-stage question, delete
                # question.json so the adapter's pause_check resumes and
                # the _monitor_questions loop can proceed.
                if context_key == "prepare" and self._workspace_dir is not None:
                    q_path = self._workspace_dir / "prepare" / "question.json"
                    try:
                        q_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            except RuntimeError as exc:
                # Session no longer running — surface to user without crashing.
                log.warning("adapter.send rejected for %s: %s", session_key, exc)
                overview = self._get_overview_screen()
                if overview is not None:
                    overview.append_event(
                        f"[dim yellow]Message not delivered: {exc}[/dim yellow]"
                    )
            except Exception as exc:
                log.exception("adapter.send failed for %s", session_key)
                self.notify(
                    f"Send failed: {exc}",
                    severity="error",
                )

    def _dispatch_event(self, session_key: str, event: Any) -> None:
        """Route an event to the appropriate screen widget.

        System events (stage_change, approaches_registered, run_ended) are
        always delivered to the OverviewScreen instance, even when a modal
        or focused SessionScreen is currently on top. Using `self.screen`
        (the active screen) instead would silently drop events whenever any
        overlay is showing.
        """
        screen = self.screen
        event_type = event.type if hasattr(event, "type") else ""
        event_data = event.data if hasattr(event, "data") else {}

        # Forward to web monitor (thread-safe, non-blocking)
        self._push_monitor_event(event_type, event_data)

        # Handle synthetic stage-change events
        if event_type == "stage_change":
            ws_dir = event_data.get("workspace_dir", "")
            if ws_dir:
                self._workspace_dir = Path(ws_dir)
            overview = self._get_overview_screen()
            if overview is not None:
                overview.set_stage(
                    stage=event_data.get("stage", ""),
                    round_idx=event_data.get("loop_index", 0),
                    max_rounds=self._config.max_loops,
                    max_num_approaches=self._config.max_num_approaches,
                    time_limit_seconds=event_data.get("time_limit_seconds"),
                    prior_elapsed_seconds=event_data.get("prior_elapsed_seconds", 0),
                )
            return

        # Handle cost warning events
        if event_type == "cost_warning":
            level = event_data.get("level", "")
            current = event_data.get("current_cost", 0)
            limit = event_data.get("limit", 0)
            self.notify(
                f"Cost ${current:.2f} approaching limit ${limit:.2f} ({level})",
                severity="warning",
            )
            return

        # Handle resume history summary
        if event_type == "resume_history":
            summary = event_data.get("summary", "")
            if summary:
                overview = self._get_overview_screen()
                if overview is not None:
                    overview.append_event(summary)
            return

        # Handle prepare-stage questions from the agent
        if event_type == "prepare_question":
            q = event_data.get("question", "")
            opts = event_data.get("options", [])
            lines = [f"[bold yellow]AGENT QUESTION:[/bold yellow] {q}"]
            for i, opt in enumerate(opts, 1):
                label = opt.get("label", "") if isinstance(opt, dict) else str(opt)
                desc = opt.get("description", "") if isinstance(opt, dict) else ""
                lines.append(f"  {i}. {label}" + (f" - {desc}" if desc else ""))
            lines.append("[dim]Type your answer in the input bar below.[/dim]")
            text = "\n".join(lines)
            overview = self._get_overview_screen()
            if overview is not None:
                overview.append_event(text)
            return

        # Handle pipeline termination — banner always lands on OverviewScreen.
        if event_type == "run_ended":
            overview = self._get_overview_screen()
            if overview is not None:
                overview.mark_run_ended(event_data)
            if isinstance(self.screen, SessionScreen):
                self.screen.mark_ended()
                self.pop_screen()
            return

        # Handle approach registration (implement stage start)
        if event_type == "approaches_registered":
            self._session_approach_map = event_data.get("session_map", {})
            overview = self._get_overview_screen()
            if overview is not None:
                overview.register_approaches(
                    event_data.get("approaches", []),
                    budget_seconds=event_data.get("time_limit_seconds"),
                    prior_elapsed_seconds=event_data.get("prior_elapsed_seconds", 0),
                )
            return

        # Handle per-approach results (completed/failed + scores)
        if event_type == "approach_results":
            overview = self._get_overview_screen()
            if overview is not None:
                overview.update_approach_results(event_data.get("results", {}))
            return

        # Handle session resuming (kill-and-resume in progress — ACP mode only)
        # Handle session sending (interrupt + write — PTY mode only)
        feedback_text = None
        if event_type == "session_resuming":
            feedback_text = "[dim yellow]⏳ Resuming session with your message…[/dim yellow]"
        elif event_type == "session_sending":
            feedback_text = "[dim yellow]⏳ Interrupted, sending your message…[/dim yellow]"
        if feedback_text is not None:
            approach_id = self._session_approach_map.get(session_key)
            if approach_id:
                self._approach_events.setdefault(approach_id, []).append(feedback_text)
            overview = self._get_overview_screen()
            if overview is not None:
                if approach_id:
                    overview.append_preview(approach_id, feedback_text)
                else:
                    overview.append_event(feedback_text)
            if isinstance(screen, SessionScreen):
                if approach_id and approach_id == screen.approach_id:
                    screen.append_event(feedback_text)
            return

        # Format event for display
        # Check if this is a subagent event (forwarded by proxy or read from disk)
        subagent_id = event_data.pop("_subagent", None)

        # Subagent tokens use a SEPARATE session key so that
        # cache_read_input_tokens (which is cumulative per session) is
        # tracked independently per subagent instead of being merged
        # into the parent session's cumulative counter.
        token_session_key = session_key
        if subagent_id:
            token_session_key = f"{session_key}:subagent:{subagent_id}"

        # ALWAYS extract token usage before any dedup check.
        # Streaming updates for the same message_id carry progressively
        # higher token counts; the token tracker handles dedup via
        # _last_per_message.  Skipping token extraction on dedup would
        # lose these streaming deltas entirely.
        self._extract_token_usage(event_type, event_data, token_session_key)

        # Dedup subagent DISPLAY events between proxy and file-polling paths.
        # This only prevents duplicate display — tokens are already tracked above.
        if subagent_id and event_type == "assistant":
            msg = event_data.get("message", {})
            msg_id = msg.get("id", "") if isinstance(msg, dict) else ""
            dedup_key = f"{session_key}:{subagent_id}:{msg_id}" if msg_id else ""
            if dedup_key and dedup_key in self._subagent_event_seen:
                return
            if dedup_key:
                self._subagent_event_seen.add(dedup_key)

        text = self._format_event(event_type, event_data, session_key)
        if not text:
            return

        # Add subagent prefix if this event came from a subagent
        if subagent_id:
            text = f"[dim magenta]↳ {subagent_id}[/dim magenta]\n{text}"

        # Distinguish "Session started" from "Session resumed" on kill-and-resume.
        # A second init event for the same session_key means the process was
        # replaced via adapter.send().
        is_resume = False
        if (event_type == "system" and event_data.get("subtype") == "init") or event_type == "init":
            is_resume = session_key in self._session_init_seen
            self._session_init_seen.add(session_key)
            # Record actual Claude session_id for .agent_home lookups
            actual_id = event_data.get("session_id", "")
            if actual_id:
                self._session_actual_ids[session_key] = actual_id
        if is_resume:
            text = text.replace("Session started:", "Session resumed:", 1)

        approach_id = self._session_approach_map.get(session_key)

        # Always buffer per-approach events so SessionScreen can replay on mount.
        if approach_id:
            self._approach_events.setdefault(approach_id, []).append(text)

        if isinstance(screen, OverviewScreen):
            if approach_id:
                screen.append_preview(approach_id, text)
            else:
                screen.append_event(text)
        elif isinstance(screen, SessionScreen):
            # Only route events belonging to the approach currently being viewed.
            if approach_id and approach_id == screen.approach_id:
                screen.append_event(text)

    @staticmethod
    def _push_monitor_event(event_type: str, event_data: dict) -> None:
        """Forward an event to the web monitor bridge (thread-safe, non-blocking)."""
        try:
            from ..monitor.server import push_event
            push_event(event_type, event_data)
        except Exception:
            pass

    def _format_event(self, event_type: str, data: dict, session_key: str = "") -> str:
        """Format a raw NDJSON event into display text.

        Claude CLI stream-json emits:
          - type="system" with subtype (hook_started, init, etc.)
          - type="assistant" with message.content = [{type:"text",...}, {type:"tool_use",...}]
          - type="user" with message.content (tool results)
          - type="result" with total_cost_usd, duration_ms, etc.
        """
        if event_type == "assistant":
            msg = data.get("message", {})
            content = msg.get("content", []) if isinstance(msg, dict) else []
            parts = []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type", "")
                if bt == "text":
                    text = block.get("text", "").strip()
                    if text:
                        parts.append(text)
                elif bt == "thinking":
                    text = block.get("thinking", "").strip()
                    if text:
                        parts.append(f"[dim italic]💭 {text}[/dim italic]")
                elif bt == "tool_use":
                    tool = block.get("name", "tool")
                    inp = block.get("input", {})
                    if tool in ("Edit", "Write", "Read") and isinstance(inp, dict):
                        preview = inp.get("file_path", str(inp)[:100])
                    elif tool == "Bash" and isinstance(inp, dict):
                        preview = inp.get("command", str(inp)[:100])
                    else:
                        preview = str(inp)[:120]
                    parts.append(f"┌ Tool: {tool}\n│ {preview}\n└")
            return "\n".join(parts) if parts else ""
        elif event_type == "result":
            duration = data.get("duration_ms", 0) / 1000
            usage = data.get("usage") or {}
            # Use locally calculated cost when available, fallback to API cost
            cost_str = ""
            if self._token_tracker is not None:
                cost = self._token_tracker.session_cost(session_key)
                if cost is not None:
                    cost_str = f"${cost:.2f}"
            if not cost_str:
                cost_usd = data.get("total_cost_usd")
                if cost_usd is not None:
                    cost_str = f"${cost_usd:.4f}"
                else:
                    cost_str = "$N/A"
            tokens = (
                f" · in {(usage.get('input_tokens') or 0):,}"
                f" · out {(usage.get('output_tokens') or 0):,}"
                f" · cache_r {(usage.get('cache_read_input_tokens') or 0):,}"
                f" · cache_c {(usage.get('cache_creation_input_tokens') or 0):,}"
                if usage else ""
            )
            return f"✓ Session complete ({duration:.1f}s, {cost_str}{tokens})"
        elif event_type in ("system", "init"):
            subtype = data.get("subtype", "")
            if subtype == "init" or event_type == "init":
                sid = data.get("session_id", "")
                is_resume = data.get("resume", False)
                label = "Session resumed" if is_resume else "Session started"
                return f"[dim]{label}: {sid}[/dim]"
            return ""
        return ""

    def _extract_token_usage(self, event_type: str, event_data: dict, session_key: str) -> None:
        """Extract token usage from assistant and result events, update tracker."""
        if self._token_tracker is None:
            return
        usage = None
        message_id = ""
        if event_type == "assistant":
            msg = event_data.get("message", {})
            usage = msg.get("usage") if isinstance(msg, dict) else None
            message_id = msg.get("id", "") if isinstance(msg, dict) else ""
        elif event_type == "result":
            usage = event_data.get("usage")
        if not usage or not isinstance(usage, dict):
            return
        inp = usage.get("input_tokens") or 0
        out = usage.get("output_tokens") or 0
        cache_r = usage.get("cache_read_input_tokens") or 0
        cache_c = usage.get("cache_creation_input_tokens") or 0
        if inp > 0 or out > 0 or cache_r > 0 or cache_c > 0:
            changed = self._token_tracker.update_session(session_key, usage, message_id=message_id)
            if changed:
                self._update_token_summary()
                self._update_approach_token(session_key)

    def _update_token_summary(self) -> None:
        """Recompute the reactive token_summary string from the tracker."""
        if self._token_tracker is None:
            self.token_summary = ""
            return
        t = self._token_tracker.totals
        cost = self._token_tracker.calculate_cost()
        cost_str = f" · ${cost:.2f}" if cost is not None else " · $N/A"
        self.token_summary = (
            f"{format_token_count(t.input_tokens)} in · "
            f"{format_token_count(t.output_tokens)} out"
            f"{cost_str}"
        )
        # Write token usage to pipeline state so the monitor can display it.
        from ..runtime import write_pipeline_state
        write_pipeline_state(token_usage={
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "cache_read_input_tokens": t.cache_read_input_tokens,
            "cache_creation_input_tokens": t.cache_creation_input_tokens,
        })

    def _update_approach_token(self, session_key: str) -> None:
        """Update the approach list with per-approach token info.

        Aggregates the main session plus any subagent sessions
        (keyed as ``session_key:subagent:<id>``) so the displayed
        total includes subagent token usage.
        """
        approach_id = self._session_approach_map.get(session_key)
        if not approach_id or self._token_tracker is None:
            return
        # Start with the main session usage
        main_usage = self._token_tracker.session_usage(session_key)
        if main_usage is None:
            return
        total_in = main_usage.input_tokens
        total_out = main_usage.output_tokens
        # Add subagent sessions that belong to this approach
        prefix = f"{session_key}:subagent:"
        for key, usage in self._token_tracker._sessions.items():
            if key.startswith(prefix):
                total_in += usage.input_tokens
                total_out += usage.output_tokens
        summary = f"{format_token_count(total_in)} in {format_token_count(total_out)} out"
        overview = self._get_overview_screen()
        if overview is not None:
            from .widgets.approach_list import ApproachList
            approach_list = overview.query_one("#approach-list", ApproachList)
            approach_list.update_approach(approach_id, token_summary=summary)

    def _poll_agent_home_usage(self) -> None:
        """Timer callback: scan JSONL for token usage.

        Wrapped in try/except to prevent a crash in token polling from
        killing the entire TUI during the implement stage.
        """
        try:
            self._poll_agent_home_usage_inner()
        except Exception:
            log.exception("Token polling error (non-fatal)")

    def _poll_agent_home_usage_inner(self) -> None:
        if self._token_tracker is None or self._workspace_dir is None:
            return

        search_dirs: list[Path] = []

        # .agent_home is bind-mounted as HOME inside the Docker container.
        # Files may be delayed by bind-mount caching but eventually become
        # visible so stream mode can discover subagent transcripts.
        agent_home_dir = self._workspace_dir / ".agent_home" / ".claude" / "projects" / "-workspace"
        if agent_home_dir.is_dir():
            search_dirs.append(agent_home_dir)

        if not search_dirs:
            return

        # --- Main session JSONL scanning (existing logic) ---
        for session_key, actual_id in self._session_actual_ids.items():
            for search_dir in search_dirs:
                jsonl_path = search_dir / f"{actual_id}.jsonl"
                if not jsonl_path.exists():
                    continue
                offset = self._agent_home_offsets.get(
                    session_key,
                    self._jsonl_baseline_sizes.get(jsonl_path.resolve(), 0),
                )
                new_offset, usages = self._scan_jsonl_for_usage(jsonl_path, offset)
                self._agent_home_offsets[session_key] = new_offset
                for usage, message_id in usages:
                    self._token_tracker.update_session(session_key, usage, message_id=message_id)
                break  # Found the file, no need to check other dirs

        # --- Subagent JSONL scanning (host-only dirs) ---
        self._poll_subagent_files(search_dirs)

        self._update_token_summary()
        for session_key in self._session_approach_map:
            self._update_approach_token(session_key)

    def _poll_subagent_files(self, search_dirs: list[Path]) -> None:
        """Discover and read new events from subagent JSONL files.

        For each known main session, looks for a ``subagents/`` directory
        under the session's project dir.  New events are dispatched to
        the TUI; token usage is merged into the parent session.
        """
        if self._token_tracker is None:
            return

        for session_key, actual_id in self._session_actual_ids.items():
            for search_dir in search_dirs:
                # Nested layout: <search_dir>/<sessionId>/subagents/agent-*.jsonl
                subagent_dir = search_dir / actual_id / "subagents"
                if subagent_dir.is_dir():
                    self._scan_subagent_dir(session_key, subagent_dir)
                    break  # Found in this search_dir, skip others

                # Fallback: flat layout — agent-*.jsonl directly in search_dir
                # Only check if nested dir doesn't exist
                flat_agents = list(search_dir.glob("agent-*.jsonl"))
                if flat_agents:
                    self._scan_subagent_dir_flat(session_key, search_dir)
                    break

    def _scan_subagent_dir(self, session_key: str, subagent_dir: Path) -> None:
        """Read new events from subagent JSONL files in a nested layout dir."""
        for jsonl_path in sorted(subagent_dir.glob("agent-*.jsonl")):
            subagent_id = jsonl_path.stem  # e.g. "agent-abc123"
            self._read_subagent_jsonl(session_key, subagent_id, jsonl_path)

    def _scan_subagent_dir_flat(self, session_key: str, search_dir: Path) -> None:
        """Read new events from flat-layout agent-*.jsonl files."""
        for jsonl_path in sorted(search_dir.glob("agent-*.jsonl")):
            subagent_id = jsonl_path.stem
            self._read_subagent_jsonl(session_key, subagent_id, jsonl_path)

    def _read_subagent_jsonl(
        self, session_key: str, subagent_id: str, jsonl_path: Path,
    ) -> None:
        """Read new lines from a subagent JSONL file, dispatch events and tokens."""
        offset_key = f"{session_key}:{subagent_id}"
        offset = self._subagent_offsets.get(
            offset_key,
            self._jsonl_baseline_sizes.get(jsonl_path.resolve(), 0),
        )

        events: list[tuple[str, dict]] = []
        usages_last: dict[str, dict] = {}  # message_id -> last usage (streaming: keep last)

        try:
            with open(jsonl_path, "r") as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_type = data.get("type", "")
                    if event_type in ("assistant", "user", "result", "system"):
                        events.append((event_type, data))
                    if event_type == "assistant":
                        msg = data.get("message", {})
                        usage = msg.get("usage") if isinstance(msg, dict) else None
                        message_id = msg.get("id", "") if isinstance(msg, dict) else ""
                        if usage and isinstance(usage, dict):
                            inp = usage.get("input_tokens") or 0
                            out = usage.get("output_tokens") or 0
                            cache_r = usage.get("cache_read_input_tokens") or 0
                            cache_c = usage.get("cache_creation_input_tokens") or 0
                            if inp > 0 or out > 0 or cache_r > 0 or cache_c > 0:
                                if message_id:
                                    usages_last[message_id] = usage
                                # events without message_id are not expected for subagents
                new_offset = f.tell()
        except OSError:
            new_offset = offset

        self._subagent_offsets[offset_key] = new_offset

        # Dispatch events to TUI
        for event_type, event_data in events:
            self._dispatch_subagent_event(session_key, subagent_id, event_type, event_data)

        # Merge token usage into a SEPARATE subagent session so that
        # cache_read_input_tokens (cumulative per session) is tracked
        # independently per subagent, not overwritten by the parent.
        # The token tracker handles its own dedup via _last_per_message
        # and _seen_message_ids — no need to check _subagent_event_seen
        # here (doing so would cause self-dedup: _dispatch_subagent_event
        # adds keys to _subagent_event_seen before this loop runs, so all
        # token updates would be skipped).
        subagent_session_key = f"{session_key}:subagent:{subagent_id}"
        for message_id, usage in usages_last.items():
            if self._token_tracker is None:
                continue
            self._token_tracker.update_session(subagent_session_key, usage, message_id=message_id)

    def _dispatch_subagent_event(
        self, session_key: str, subagent_id: str, event_type: str, event_data: dict,
    ) -> None:
        """Format and route a subagent event to the TUI.

        Subagent events are deduplicated against inline events in the
        parent session (same message_id) and displayed with a distinctive
        prefix.  Only ``assistant`` events carry a message_id suitable
        for dedup — ``result`` and ``system`` events from subagents do
        NOT appear in the parent session's inline JSONL, so they are
        always dispatched.
        """
        # Dedup: only assistant events have a message_id and also appear
        # inline in the parent session JSONL.  result/system/user events
        # from subagents are subagent-only and never duplicated.
        msg_id = ""
        if event_type == "assistant":
            msg = event_data.get("message", {})
            msg_id = msg.get("id", "") if isinstance(msg, dict) else ""
        dedup_key = f"{session_key}:{subagent_id}:{msg_id}" if msg_id else ""
        if dedup_key and dedup_key in self._subagent_event_seen:
            return
        if dedup_key:
            self._subagent_event_seen.add(dedup_key)

        # Format using existing _format_event, then add subagent prefix
        base_text = self._format_event(event_type, event_data, session_key)
        if not base_text:
            return
        text = f"[dim magenta]↳ {subagent_id}[/dim magenta]\n{base_text}"

        # Route to the correct approach — same logic as _dispatch_event
        approach_id = self._session_approach_map.get(session_key)

        if approach_id:
            self._approach_events.setdefault(approach_id, []).append(text)

        screen = self.screen
        if isinstance(screen, OverviewScreen):
            if approach_id:
                screen.append_preview(approach_id, text)
            else:
                screen.append_event(text)
        elif isinstance(screen, SessionScreen):
            if approach_id and approach_id == screen.approach_id:
                screen.append_event(text)

    @staticmethod
    def _scan_jsonl_for_usage(path: Path, offset: int) -> tuple[int, list[tuple[dict, str]]]:
        """Read new lines from a JSONL file starting at offset, extract (usage, message_id) pairs.

        For streaming, multiple assistant events share the same message_id.
        We keep the LAST usage per message_id (it has the final token counts).
        """
        return scan_jsonl_usage_since(path, offset)

    @staticmethod
    def _snapshot_existing_jsonl(workspace_dir: Path) -> dict[Path, int]:
        """Record existing Claude JSONL sizes so resume polling only counts new data."""
        sizes: dict[Path, int] = {}
        search_dirs = [
            workspace_dir / ".agent_home" / ".claude" / "projects" / "-workspace",
        ]
        for root in search_dirs:
            if not root.is_dir():
                continue
            for path in root.rglob("*.jsonl"):
                try:
                    sizes[path.resolve()] = path.stat().st_size
                except OSError:
                    continue
        return sizes

    def _notify_error(self, msg: str) -> None:
        self.notify(f"Pipeline error: {escape(msg)}", severity="error")

    @property
    def event_queue(self) -> asyncio.Queue:
        return self._event_queue

    def _get_overview_screen(self) -> "OverviewScreen | None":
        """Return the installed OverviewScreen, or None if not installed.

        System events target this screen by identity (not by "is it visible
        right now") so a modal overlay cannot drop stage/approach/run events.
        """
        try:
            screen = self.get_screen("overview")
        except KeyError:
            return None
        return screen if isinstance(screen, OverviewScreen) else None

    def _hydrate_resume_overview(self, state: dict) -> None:
        """Show the reconstructed resume stage before any node can abort."""
        overview = self._get_overview_screen()
        if overview is None:
            return
        stage = str(state.get("next_stage", "") or "")
        if stage not in {"prepare", "propose", "implement"}:
            return
        loop_index = int(state.get("loop_index", 0) or 0)
        round_idx = loop_index + 1 if stage == "propose" else loop_index
        overview.set_stage(
            stage=stage,
            round_idx=round_idx,
            max_rounds=self._config.max_loops,
            max_num_approaches=self._config.max_num_approaches,
        )

    def get_approach_events(self, approach_id: str) -> list[str]:
        """Return the buffered formatted-event lines for an approach.

        Used by SessionScreen.on_mount to replay history when the user enters
        or re-enters the focused view for an approach.
        """
        return list(self._approach_events.get(approach_id, ()))

    # -- Quit actions (Ctrl+C → q) --

    def action_quit(self) -> None:
        """Safety net: redirect Textual's default quit to our two-step flow."""
        self.action_init_quit()

    def action_init_quit(self) -> None:
        """First step of quit: Ctrl+C blurs the Input and shows confirmation."""
        if self._quit_pending:
            # Double Ctrl+C — exit immediately.
            self.exit()
            return
        # Blur any focused Input so a subsequent 'q' press is not swallowed.
        from textual.widgets import Input
        focused = self.focused
        if isinstance(focused, Input):
            self.set_focus(None)
        self._quit_pending = True
        self.notify("Press q to quit")
        self.set_timer(3.0, self._clear_quit_pending)

    def action_confirm_quit(self) -> None:
        """Second step of quit: q exits only after Ctrl+C set the flag."""
        if self._quit_pending:
            self.exit()

    def _clear_quit_pending(self) -> None:
        self._quit_pending = False

    def action_toggle_selection_mode(self) -> None:
        """Toggle Textual's mouse capture so the user can select/copy text.

        Textual uses private driver methods for mouse reporting; if they are
        unavailable in a future version this action degrades to a no-op
        rather than crashing the TUI.
        """
        from .screens.overview import OverviewScreen
        if isinstance(self.screen, OverviewScreen) and self.screen._stage == "implement" and not self.screen._run_ended:
            return
        driver = getattr(self, "_driver", None)
        enable = getattr(driver, "_enable_mouse_support", None)
        disable = getattr(driver, "_disable_mouse_support", None)
        if driver is None or not callable(enable) or not callable(disable):
            self.notify("Selection toggle unavailable on this Textual version", severity="warning")
            return
        if self._selection_mode:
            enable()
            self._selection_mode = False
            self.notify("Mouse capture re-enabled")
        else:
            disable()
            self._selection_mode = True
            self.notify("Selection mode: drag to select · Ctrl+S to resume")
