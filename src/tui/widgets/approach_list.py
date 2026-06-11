"""Navigable approach list with inline progress bars."""

from __future__ import annotations

from dataclasses import dataclass, field
from textual import on
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import OptionList
from textual.widgets.option_list import Option

_BAR_WIDTH = 14
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class ApproachState:
    """Tracks the state of one approach for display."""
    approach_id: str
    elapsed_seconds: float = 0.0
    budget_seconds: float = 1200.0  # 20 min default; overridden from config at register time
    score: float | None = None
    status: str = "queued"  # queued, running, thinking, completed, failed, paused
    last_activity: str = ""
    token_summary: str = ""
    spinner_frame: int = 0
    start_time: float | None = None  # monotonic clock at session start; None = not yet running


class ApproachList(OptionList):
    """Arrow-key navigable list of approaches with progress info."""

    class Selected(Message):
        """Fired when the user presses Enter on an approach."""
        def __init__(self, approach_id: str) -> None:
            super().__init__()
            self.approach_id = approach_id

    class HighlightChanged(Message):
        """Fired when the highlighted approach changes (up/down navigation)."""
        def __init__(self, approach_id: str | None) -> None:
            super().__init__()
            self.approach_id = approach_id

    DEFAULT_CSS = """
    ApproachList {
        height: auto;
        max-height: 12;
        border: none;
        padding: 0 1;
    }
    ApproachList > .option-list--option {
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._approaches: dict[str, ApproachState] = {}

    def set_approaches(self, approaches: list[ApproachState]) -> None:
        """Replace the full approach list."""
        self._approaches = {a.approach_id: a for a in approaches}
        self.clear_options()
        for a in approaches:
            self.add_option(Option(self._format_line(a), id=a.approach_id))
        if approaches:
            self.highlighted = 0

    def update_approach(self, approach_id: str, **fields) -> None:
        """Update fields on one approach and refresh its display."""
        state = self._approaches.get(approach_id)
        if not state:
            return
        for k, v in fields.items():
            if hasattr(state, k):
                setattr(state, k, v)
        # Refresh the option text. Textual's replace_option_prompt takes an
        # option ID (str); the index-based variant is required here.
        idx = self._option_index(approach_id)
        if idx is not None:
            self.replace_option_prompt_at_index(idx, self._format_line(state))

    def get_highlighted_id(self) -> str | None:
        """Return the approach_id of the currently highlighted option."""
        idx = self.highlighted
        if idx is not None and idx < len(self._approaches):
            return list(self._approaches.keys())[idx]
        return None

    def get_approach(self, approach_id: str) -> ApproachState | None:
        return self._approaches.get(approach_id)

    def iter_approaches(self):
        """Yield all tracked ApproachState objects (read-only view)."""
        return self._approaches.values()

    def watch_highlighted(self, highlighted: int | None) -> None:
        """Notify parent when the highlighted approach changes."""
        aid = None
        if highlighted is not None and highlighted < len(self._approaches):
            keys = list(self._approaches.keys())
            aid = keys[highlighted]
        self.post_message(self.HighlightChanged(aid))

    def _option_index(self, approach_id: str) -> int | None:
        keys = list(self._approaches.keys())
        if approach_id in keys:
            return keys.index(approach_id)
        return None

    @staticmethod
    def _format_line(a: ApproachState) -> str:
        # Progress bar
        if a.budget_seconds > 0:
            ratio = min(a.elapsed_seconds / a.budget_seconds, 1.0)
        else:
            ratio = 0.0
        filled = int(ratio * _BAR_WIDTH)
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)

        # Time — clamp the elapsed minute display to the budget so we don't
        # show nonsensical values like "5m/3m" while the process is being
        # stopped.
        bm = int(a.budget_seconds) // 60
        em = min(int(a.elapsed_seconds) // 60, bm) if bm > 0 else 0
        time_str = f"{em}m/{bm}m"

        # Score / status
        if a.score is not None:
            info = f"{a.score:.3f}"
        elif a.status == "thinking":
            frame = _SPINNER_FRAMES[a.spinner_frame % len(_SPINNER_FRAMES)]
            info = f"{frame} thinking"
        elif a.status == "paused":
            info = "⏸ paused"
        else:
            info = a.status

        line = f"  {a.approach_id:<16} {bar} {time_str:>8}  {info}"
        if a.token_summary:
            line += f"  {a.token_summary}"
        return line
