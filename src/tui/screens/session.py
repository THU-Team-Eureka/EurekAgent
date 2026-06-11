"""Session screen: focused view for one approach with live event stream."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Input, Static

from ...runtime import enqueue_user_message
from ...token_tracker import format_token_count
from ..widgets.logo import LogoBanner
from ..widgets.smart_rich_log import SmartRichLog


class SessionScreen(Screen):
    """Focused view for a single approach session with message input."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=False),
    ]

    DEFAULT_CSS = """
    SessionScreen {
        layout: vertical;
    }
    #session-breadcrumb {
        text-align: center;
        color: $text-muted;
        padding: 0 1;
    }
    #session-hints {
        text-align: center;
        color: $text-disabled;
        padding: 0 1;
    }
    #session-status {
        text-align: center;
        color: $accent;
        padding: 0 1;
    }
    #session-log {
        height: 1fr;
        border: none;
        padding: 0 2;
    }
    #session-input {
        dock: bottom;
        margin: 0 1;
    }
    """

    def __init__(
        self,
        approach_id: str = "",
        budget_seconds: float | None = None,
        start_time: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._approach_id = approach_id
        self._budget_seconds = budget_seconds
        self._start_time = start_time
        self._ended: bool = False

    def compose(self) -> ComposeResult:
        yield LogoBanner(style="minimal", id="session-logo")
        yield Static("", id="session-breadcrumb")
        yield Static("Esc Back  ↵ Send message  Ctrl+C q Quit  Ctrl+S Select", id="session-hints")
        yield Static("⠧ Agent is working...", id="session-status")
        yield SmartRichLog(id="session-log", wrap=True, highlight=True, markup=True)
        yield Input(
            placeholder="Type message to agent…",
            id="session-input",
        )

    def on_mount(self) -> None:
        try:
            self.query_one("#session-breadcrumb", Static).update(
                self._format_breadcrumb()
            )
            self._replay_buffered_events()
            self._tick_timer_handle = self.set_interval(1.0, self._tick_timer)
        except Exception:
            import logging
            logging.getLogger(__name__).exception("SessionScreen.on_mount failed")

    def _format_breadcrumb(self) -> str:
        base = f"EurekAgent › {self._approach_id}"
        if self._budget_seconds and self._start_time:
            elapsed = max(0.0, time.monotonic() - self._start_time)
            em, es = divmod(int(elapsed), 60)
            bm = int(self._budget_seconds) // 60
            em = min(em, bm)
            base += f" · {em}m{es:02d}s / {bm}m"
        token_str = self._get_token_summary()
        if token_str:
            base += f" · {token_str}"
        return base

    def _get_token_summary(self) -> str:
        tracker = getattr(self.app, "_token_tracker", None)
        if tracker is None:
            return ""
        approach_map = getattr(self.app, "_session_approach_map", {})
        session_key = None
        for sk, aid in approach_map.items():
            if aid == self._approach_id:
                session_key = sk
                break
        if session_key is None:
            return ""
        usage = tracker.session_usage(session_key)
        if usage is None:
            return ""
        return f"{format_token_count(usage.input_tokens)} in {format_token_count(usage.output_tokens)} out"

    def _tick_timer(self) -> None:
        if self._ended:
            return
        self.query_one("#session-breadcrumb", Static).update(
            self._format_breadcrumb()
        )

    def mark_ended(self) -> None:
        """Stop the timer when the session/run has ended."""
        self._ended = True
        if hasattr(self, "_tick_timer_handle") and self._tick_timer_handle is not None:
            self._tick_timer_handle.stop()

    def _replay_buffered_events(self) -> None:
        """Populate the session log with events received before this screen
        was opened (e.g. events that arrived while the user was on the
        overview screen).
        """
        getter = getattr(self.app, "get_approach_events", None)
        if not callable(getter) or not self._approach_id:
            return
        log = self.query_one("#session-log", SmartRichLog)
        log.clear()
        for line in getter(self._approach_id):
            log.write(line)

    @property
    def approach_id(self) -> str:
        """The approach id currently displayed by this screen."""
        return self._approach_id

    def append_event(self, text: str) -> None:
        """Append text to the session event log."""
        self.query_one("#session-log", SmartRichLog).write(text)

    def set_status(self, text: str) -> None:
        self.query_one("#session-status", Static).update(text)

    # -- Input handling --

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Send a typed message to the approach's running agent session."""
        if event.input.id != "session-input":
            return
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        context_key = f"implement::{self._approach_id}"
        if enqueue_user_message(context_key, text):
            self.append_event(f"[b cyan][You] {text}[/b cyan]")
        else:
            self.append_event("[dim]No active message channel.[/dim]")

    # -- Actions --

    def action_go_back(self) -> None:
        self.app.pop_screen()
