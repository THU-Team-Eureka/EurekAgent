"""Modal prompt for granting extra time to an exhausted resume stage."""

from __future__ import annotations

import textwrap

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ...resume_preflight import MIN_RESUME_EXTRA_SECONDS, ResumePreflightResult


class ResumeExtraTimeDialog(ModalScreen[float | None]):
    """Ask the user for extra minutes before resuming an exhausted stage."""

    DEFAULT_CSS = """
    ResumeExtraTimeDialog {
        align: center middle;
    }
    ResumeExtraTimeDialog > Vertical {
        width: 72;
        padding: 1 2;
        border: round $primary;
        background: $panel;
    }
    ResumeExtraTimeDialog Input {
        margin-top: 1;
    }
    ResumeExtraTimeDialog Horizontal {
        height: auto;
        margin-top: 1;
    }
    ResumeExtraTimeDialog .wrapped {
        width: 100%;
        height: auto;
        text-wrap: wrap;
        text-overflow: fold;
    }
    """

    def __init__(self, result: ResumePreflightResult) -> None:
        super().__init__()
        self._result = result

    def compose(self) -> ComposeResult:
        min_minutes = int(MIN_RESUME_EXTRA_SECONDS / 60)
        missing = "\n".join(
            _wrap_line(str(path), width=62, initial_indent="  ", subsequent_indent="  ")
            for path in self._result.missing_artifacts[:3]
        )
        more = len(self._result.missing_artifacts) - 3
        if more > 0:
            missing += f"\n  ... and {more} more"
        yield Vertical(
            Label("[b]Resume Needs Extra Time[/b]"),
            Static(
                _wrap_line(
                    f"Round {self._result.loop_index} {self._result.stage} already used "
                    f"{self._result.elapsed_seconds / 60:.1f} / "
                    f"{self._result.budget_seconds / 60:.1f} minutes.",
                ),
                classes="wrapped",
            ),
            Static(_wrap_line(self._result.message), classes="wrapped"),
            Static(
                f"Missing:\n{missing}" if missing else "Missing required artifact.",
                classes="wrapped",
            ),
            Label(f"Extra minutes (minimum {min_minutes}):"),
            Input(value="10", id="extra-minutes"),
            Label("", id="extra-error"),
            Horizontal(
                Button("Resume", variant="primary", id="resume"),
                Button("Cancel", id="cancel"),
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id == "resume":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        input_widget = self.query_one("#extra-minutes", Input)
        error = self.query_one("#extra-error", Label)
        try:
            minutes = float(input_widget.value.strip())
        except ValueError:
            error.update("[red]Enter a number of minutes.[/red]")
            return
        seconds = minutes * 60
        if seconds < MIN_RESUME_EXTRA_SECONDS:
            error.update(
                f"[red]Extra time must be at least {MIN_RESUME_EXTRA_SECONDS / 60:.0f} minutes.[/red]"
            )
            return
        self.dismiss(seconds)


def _wrap_line(
    text: str,
    *,
    width: int = 62,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> str:
    """Pre-wrap modal text so narrow terminals don't truncate it."""
    return textwrap.fill(
        text,
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=True,
        break_on_hyphens=False,
    )
