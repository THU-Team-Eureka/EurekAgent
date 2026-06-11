"""Overview screen: main monitoring view that adapts to pipeline stage."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, OptionList, Static

from ...runtime import enqueue_user_message
from ..terminal_report import TerminalReport, build_report
from ..widgets.approach_list import ApproachList, ApproachState
from ..widgets.command_palette import CommandPalette
from ..widgets.logo import LogoBanner
from ..widgets.smart_rich_log import SmartRichLog
from ..widgets.terminal_banner import TerminalBanner


class OverviewScreen(Screen):
    """Main run monitoring screen.

    During propose stage: shows full streamed output (single session).
    During implement stage: shows approach list with progress bars.
    """

    BINDINGS = [
        Binding("slash", "focus_input", "Commands", show=False),
        Binding("enter", "enter_session", "Enter session", show=False),
    ]

    DEFAULT_CSS = """
    OverviewScreen {
        layout: vertical;
    }
    #model-header {
        text-align: center;
        color: $text-muted;
        padding: 0 1;
        height: 1;
    }
    #stage-header {
        text-align: center;
        color: $text-muted;
        padding: 0 1;
    }
    #key-hints {
        text-align: center;
        color: $text-disabled;
        padding: 0 1;
    }
    #event-log {
        height: 1fr;
        border: none;
        padding: 0 2;
    }
    #command-input {
        dock: bottom;
        margin: 0 1;
    }
    #preview-label {
        color: $text-muted;
        padding: 0 2;
        height: 1;
    }
    #preview-log {
        height: 1fr;
        border: none;
        padding: 0 2;
    }
    """

    def __init__(
        self,
        stage: str = "propose",
        round_idx: int = 1,
        max_rounds: int = 5,
        max_num_approaches: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._stage = stage
        self._round = round_idx
        self._max_rounds = max_rounds
        # Stage timer state. set_stage() refreshes these; _tick_progress uses
        # them to update the header each second.
        self._propose_start_time: float | None = None
        self._propose_budget_seconds: float | None = None
        self._prepare_start_time: float | None = None
        # Terminal-state memo so a re-mount after the run ends still renders
        # the banner (e.g. user was on a SessionScreen when the event fired).
        self._run_ended: bool = False
        self._run_ended_report: TerminalReport | None = None

    def compose(self) -> ComposeResult:
        yield LogoBanner(style="minimal", id="logo")
        yield Static("", id="model-header")
        yield Static("", id="stage-header")
        yield Static("", id="key-hints")
        yield TerminalBanner(id="terminal-banner")
        yield ApproachList(id="approach-list")
        yield SmartRichLog(id="event-log", wrap=True, highlight=True, markup=True)
        yield Static("", id="preview-label")
        yield SmartRichLog(id="preview-log", wrap=True, highlight=True, markup=True)
        yield CommandPalette(id="command-palette")
        yield Input(
            placeholder="Type / for commands…",
            id="command-input",
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "command-input":
            return
        palette = self.query_one("#command-palette", CommandPalette)
        if event.value.startswith("/"):
            palette.show(filter_text=event.value, stage=self._stage)
        else:
            palette.hide()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        list_id = event.option_list.id
        if list_id == "cmd-options":
            cmd = event.option.id or ""
            inp = self.query_one("#command-input", Input)
            inp.value = cmd
            self.query_one("#command-palette", CommandPalette).hide()
            inp.focus()
        elif list_id == "approach-list" and self._stage == "implement":
            # Enter on the highlighted approach opens its session screen.
            aid = event.option.id
            if aid:
                self._open_session_screen(aid)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "command-input":
            return
        cmd = event.value.strip()
        self.query_one("#command-palette", CommandPalette).hide()
        event.input.clear()
        if not cmd:
            return
        if cmd == "/quit":
            self.app.exit()
        elif cmd == "/skip-prepare":
            self._skip_prepare()
        elif not cmd.startswith("/"):
            self._send_to_active_agent(cmd)
        else:
            self.append_event(f"[dim]Unknown command: {cmd}[/dim]")

    def _send_to_active_agent(self, text: str) -> None:
        """Dispatch a non-slash message to the stage-appropriate session.

        Propose: single session, route to context "propose". Implement:
        ambiguous on overview — tell the user to enter the approach.
        """
        if self._stage in ("propose", "prepare"):
            if enqueue_user_message(self._stage, text):
                self.append_event("[dim]⏸ Delivering message to agent…[/dim]")
            else:
                self.append_event(
                    "[dim]No active message channel — is the pipeline running?[/dim]"
                )
            return

        if self._stage == "implement":
            self.append_event(
                "[dim]Enter an approach session (⏎) to send a message to "
                "that approach's agent.[/dim]"
            )
            return

        self.append_event("[dim]No active session to receive the message.[/dim]")

    def _skip_prepare(self) -> None:
        """Skip the prepare stage by writing complete.json, cleaning up
        question.json, and interrupting the running agent."""
        if self._stage != "prepare":
            self.append_event("[dim]/skip-prepare is only available during the prepare stage.[/dim]")
            return

        app = self.app
        if not hasattr(app, "_workspace_dir") or app._workspace_dir is None:
            self.append_event("[dim]Workspace not yet initialized.[/dim]")
            return

        import json as _json
        workspace = app._workspace_dir
        complete_path = workspace / "prepare" / "complete.json"
        question_path = workspace / "prepare" / "question.json"

        # Write complete.json so the adapter's completion_check returns True
        # and the prepare_node detects "ready" status.
        complete_path.write_text(
            _json.dumps({"status": "skipped"}) + "\n",
            encoding="utf-8",
        )
        # Clean up any pending question so the adapter's pause_check resumes.
        try:
            question_path.unlink(missing_ok=True)
        except OSError:
            pass

        # Interrupt the running agent so it stops generating immediately.
        # The adapter's next poll will see completion_check=True and end the stream.
        from ...runtime import get_active_adapter, get_active_session
        adapter = get_active_adapter()
        session_key = get_active_session("prepare")
        if adapter is not None and session_key is not None:
            import asyncio
            try:
                asyncio.get_event_loop().create_task(adapter.interrupt(session_key))
            except RuntimeError:
                pass

        self.append_event("[bold green]Prepare stage skipped. Moving to propose…[/bold green]")

    def on_mount(self) -> None:
        self._update_stage_display()
        # Drive per-approach elapsed-time display once per second.
        self.set_interval(1.0, self._tick_progress)

    def _tick_progress(self) -> None:
        """Refresh the elapsed-time indicator for the current stage."""
        if self._run_ended:
            return
        if self._stage == "propose":
            self._refresh_propose_header()
            return
        if self._stage == "prepare":
            self._refresh_prepare_header()
            return
        if self._stage != "implement":
            return
        approach_list = self.query_one("#approach-list", ApproachList)
        now = time.monotonic()
        for state in approach_list.iter_approaches():
            if state.start_time is None:
                continue
            approach_list.update_approach(
                state.approach_id,
                elapsed_seconds=now - state.start_time,
            )
        # Refresh implement header with updated token summary
        header = self.query_one("#stage-header", Static)
        header.update(self._format_implement_header())
        model_header = self.query_one("#model-header", Static)
        model_header.update(self._format_model_header())

    def set_stage(
        self,
        stage: str,
        round_idx: int,
        max_rounds: int,
        max_num_approaches: int = 0,
        time_limit_seconds: float | None = None,
        prior_elapsed_seconds: float = 0.0,
    ) -> None:
        """Update the displayed stage info.

        When entering the propose stage, we stamp a start time so the header
        can show Xm:XXs / Ym. The implement stage gets its per-approach bars
        via register_approaches instead.
        """
        self._stage = stage
        self._round = round_idx
        self._max_rounds = max_rounds
        self._run_ended = False
        prior = max(0.0, float(prior_elapsed_seconds))
        if stage == "propose":
            self._propose_start_time = time.monotonic() - prior
            self._propose_budget_seconds = time_limit_seconds
        if stage == "prepare":
            self._prepare_start_time = time.monotonic() - prior
        if self.is_mounted:
            self._update_stage_display()

    def _update_stage_display(self) -> None:
        model_header = self.query_one("#model-header", Static)
        header = self.query_one("#stage-header", Static)
        hints = self.query_one("#key-hints", Static)
        approach_list = self.query_one("#approach-list", ApproachList)
        event_log = self.query_one("#event-log", SmartRichLog)
        preview_log = self.query_one("#preview-log", SmartRichLog)
        command_input = self.query_one("#command-input", Input)

        if self._stage == "propose":
            header.update(self._format_propose_header())
            model_header.update(self._format_model_header())
            hints.update("/ Commands  Ctrl+S Select  Ctrl+C q Quit")
            approach_list.display = False
            event_log.display = True
            command_input.display = True
            self.query_one("#preview-label", Static).display = False
            preview_log.display = False
            preview_log.can_focus = True
        elif self._stage == "prepare":
            header.update(self._format_prepare_header())
            model_header.update(self._format_model_header())
            hints.update("/ Commands  Ctrl+S Select  Ctrl+C q Quit")
            approach_list.display = False
            event_log.display = True
            command_input.display = True
            self.query_one("#preview-label", Static).display = False
            preview_log.display = False
            preview_log.can_focus = True
        else:
            header.update(self._format_implement_header())
            model_header.update(self._format_model_header())
            hints.update("↑↓ Navigate  ⏎ Enter session  Ctrl+C q Quit")
            approach_list.display = True
            event_log.display = False
            command_input.display = False
            approach_list.focus()
            self.query_one("#preview-label", Static).display = True
            preview_log.display = True
            preview_log.can_focus = False

        # If the run ended while this screen was not mounted (user on
        # SessionScreen), apply the terminal banner on re-mount so the user
        # sees the closing state rather than a stale stage header.
        if self._run_ended:
            self._apply_terminal_banner()

    def _refresh_propose_header(self) -> None:
        """Ticker callback: rewrite the propose header with the current elapsed time."""
        if not self.is_mounted:
            return
        self.query_one("#stage-header", Static).update(self._format_propose_header())
        self.query_one("#model-header", Static).update(self._format_model_header())

    def _refresh_prepare_header(self) -> None:
        """Ticker callback: rewrite the prepare header with the current elapsed time."""
        if not self.is_mounted:
            return
        self.query_one("#stage-header", Static).update(self._format_prepare_header())
        self.query_one("#model-header", Static).update(self._format_model_header())

    def _format_propose_header(self) -> str:
        """Render the propose-stage header, including an elapsed/budget clock."""
        base = f"Round {self._round} of {self._max_rounds} · Propose · generating approaches"
        if self._propose_start_time is None:
            return base
        elapsed = max(0.0, time.monotonic() - self._propose_start_time)
        em, es = divmod(int(elapsed), 60)
        budget = self._propose_budget_seconds
        if budget and budget > 0:
            bm = int(budget) // 60
            em_clamped = min(em, bm)
            header = f"{base} · {em_clamped}m{es:02d}s / {bm}m"
        else:
            header = f"{base} · {em}m{es:02d}s"
        return header + self._monitor_hint()

    def _format_prepare_header(self) -> str:
        """Render the prepare-stage header with elapsed clock (no round number)."""
        base = "Prepare · validating setup"
        if self._prepare_start_time is None:
            return base + self._monitor_hint()
        elapsed = max(0.0, time.monotonic() - self._prepare_start_time)
        em, es = divmod(int(elapsed), 60)
        return f"{base} · {em}m{es:02d}s" + self._monitor_hint()

    def _maybe_append_tokens(self, header: str) -> str:
        ts = getattr(self.app, "token_summary", "")
        if ts:
            return f"{header} · {ts}"
        return header

    def _format_model_header(self) -> str:
        """Render the model + token info line."""
        model = getattr(self.app, "_config", None)
        model_name = getattr(model, "model", "") if model else ""
        ts = getattr(self.app, "token_summary", "")
        parts = []
        if model_name:
            parts.append(f"{model_name}")
        if ts:
            parts.append(ts)
        return " · ".join(parts) if parts else ""

    def _monitor_hint(self) -> str:
        """Return a short monitor URL hint string, or empty if not available."""
        port = getattr(self.app, "_monitor_port", None)
        if port:
            return f" · Monitor: {port}"
        return ""

    def _format_implement_header(self) -> str:
        base = f"Round {self._round} of {self._max_rounds} · Implement"
        return base + self._monitor_hint()

    def mark_run_ended(self, payload: dict) -> None:
        """Switch the overview into a terminal state when the graph has ended.

        `payload` is the raw `run_ended` event dict. We build a TerminalReport
        once and remember it, so a later on_mount (user returning from a
        SessionScreen) can replay the banner without re-deriving it.
        """
        self._run_ended = True
        self._run_ended_report = build_report(payload)
        self._finalize_running_approaches()
        if self.is_mounted:
            self._apply_terminal_banner()

    def _apply_terminal_banner(self) -> None:
        """Render the run-ended banner, slim the header, update the hint line."""
        report = self._run_ended_report
        if report is None:
            return
        self.query_one("#terminal-banner", TerminalBanner).show(report)
        self.query_one("#stage-header", Static).update(
            f"Round {self._round} of {self._max_rounds} · {report.stage_label} · run ended"
        )
        hints = "Run ended.  Ctrl+S Select  Ctrl+C q Quit"
        self.query_one("#key-hints", Static).update(hints)

    def _finalize_running_approaches(self) -> None:
        """Flip any approach still marked 'running' to 'failed' at run end.

        Approaches that received a result via update_approach_results will
        already be 'completed' or 'failed', so this only catches approaches
        that never got a result event (e.g. propose-stage abort).
        """
        try:
            approach_list = self.query_one("#approach-list", ApproachList)
        except Exception:
            return
        for state in list(approach_list.iter_approaches()):
            if state.status == "running":
                approach_list.update_approach(state.approach_id, status="failed")

    def update_approach_results(self, results: dict) -> None:
        """Update approach list with per-approach outcome from implement_node.

        ``results`` maps approach_id → {"status": "completed"|"failed",
        "score": float|None}.
        """
        try:
            approach_list = self.query_one("#approach-list", ApproachList)
        except Exception:
            return
        for approach_id, info in results.items():
            if not isinstance(info, dict):
                continue
            status = info.get("status", "failed")
            score = info.get("score")
            update = {"status": status}
            if score is not None:
                update["score"] = score
            approach_list.update_approach(approach_id, **update)

    def register_approaches(
        self,
        approaches: list[dict],
        budget_seconds: float | None = None,
        prior_elapsed_seconds: float = 0.0,
    ) -> None:
        """Populate the approach list from manifest data.

        budget_seconds, when provided, overrides ApproachState's default budget
        so the progress bar and `Xm/Ym` label reflect the actual session time
        limit from config.
        """
        approach_list = self.query_one("#approach-list", ApproachList)
        prior = max(0.0, float(prior_elapsed_seconds))
        now = time.monotonic()
        states: list[ApproachState] = []
        for a in approaches:
            if not (isinstance(a, dict) and a.get("id")):
                continue
            state = ApproachState(
                approach_id=a["id"],
                status="running",
                start_time=now - prior,
                elapsed_seconds=prior,
            )
            if budget_seconds is not None and budget_seconds > 0:
                state.budget_seconds = budget_seconds
            states.append(state)
        approach_list.set_approaches(states)
        approach_list.focus()

    def append_event(self, text: str) -> None:
        """Append text to the event log (propose stage stream)."""
        log = self.query_one("#event-log", SmartRichLog)
        log.write(text)

    def append_preview(self, approach_id: str, text: str) -> None:
        """Append text to the preview area for the specified approach.

        Only displays events for the currently highlighted approach so
        the preview doesn't interleave output from parallel sessions.
        """
        approach_list = self.query_one("#approach-list", ApproachList)
        highlighted = approach_list.get_highlighted_id()
        if highlighted is not None and approach_id != highlighted:
            return
        label = self.query_one("#preview-label", Static)
        label.update(f"Preview ({approach_id}):")
        preview = self.query_one("#preview-log", SmartRichLog)
        preview.write(text)

    def clear_preview(self) -> None:
        self.query_one("#preview-log", SmartRichLog).clear()

    def _refresh_preview(self, approach_id: str | None = None) -> None:
        """Rebuild the preview log from the buffered events for an approach."""
        if approach_id is None:
            approach_list = self.query_one("#approach-list", ApproachList)
            approach_id = approach_list.get_highlighted_id()
        preview = self.query_one("#preview-log", SmartRichLog)
        preview.clear()
        label = self.query_one("#preview-label", Static)
        if approach_id:
            label.update(f"Preview ({approach_id}):")
            getter = getattr(self.app, "get_approach_events", None)
            if callable(getter):
                for line in getter(approach_id):
                    preview.write(line)
        else:
            label.update("Preview:")

    def on_approach_list_highlight_changed(
        self, event: ApproachList.HighlightChanged
    ) -> None:
        """Refresh preview when the highlighted approach changes."""
        if self._stage != "implement":
            return
        self._refresh_preview(event.approach_id)

    def on_show(self) -> None:
        """Refresh preview when returning from a SessionScreen."""
        if self._stage != "implement":
            return
        self._refresh_preview()

    # -- Actions --

    def action_focus_input(self) -> None:
        inp = self.query_one("#command-input", Input)
        if inp.display:
            inp.focus()
            if not inp.value.startswith("/"):
                inp.value = "/" + inp.value

    def action_enter_session(self) -> None:
        if self._stage != "implement":
            return
        approach_list = self.query_one("#approach-list", ApproachList)
        aid = approach_list.get_highlighted_id()
        if aid:
            self._open_session_screen(aid)

    def _open_session_screen(self, approach_id: str) -> None:
        """Push a fresh session screen for the given approach.

        Creating a new instance each time guarantees on_mount fires and
        the RichLog is properly initialized — avoiding stale content from
        a previously viewed approach.
        """
        from .session import SessionScreen
        approach_list = self.query_one("#approach-list", ApproachList)
        state = approach_list.get_approach(approach_id)
        self.app.push_screen(SessionScreen(
            approach_id=approach_id,
            budget_seconds=state.budget_seconds if state else None,
            start_time=state.start_time if state else None,
        ))

