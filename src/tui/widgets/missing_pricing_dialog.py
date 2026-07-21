"""Modal warning when live token pricing is unavailable."""

from __future__ import annotations

import textwrap

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class MissingPricingDialog(ModalScreen[bool]):
    """Ask whether to continue when token cost cannot be calculated."""

    DEFAULT_CSS = """
    MissingPricingDialog {
        align: center middle;
    }
    MissingPricingDialog > Vertical {
        width: 82;
        padding: 1 2;
        border: round $primary;
        background: $panel;
    }
    MissingPricingDialog Horizontal {
        height: auto;
        margin-top: 1;
    }
    MissingPricingDialog .wrapped {
        width: 100%;
        height: auto;
        text-wrap: wrap;
        text-overflow: fold;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("[b]Token Pricing Unavailable[/b]"),
            Static(_wrap(self._message), classes="wrapped"),
            Horizontal(
                Button("Continue without live cost", variant="primary", id="continue"),
                Button("Exit", id="exit"),
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "continue")


def _wrap(text: str, *, width: int = 72) -> str:
    paragraphs = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            paragraphs.append("")
        elif paragraph.startswith("  "):
            paragraphs.append(paragraph)
        else:
            paragraphs.append(textwrap.fill(paragraph, width=width))
    return "\n".join(paragraphs)
