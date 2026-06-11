"""Popup command list shown when user types '/' in the input bar."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import OptionList
from textual.widgets.option_list import Option

# Stage-aware command registry: stage → [(command, description), ...]
STAGE_COMMANDS: dict[str, list[tuple[str, str]]] = {
    "prepare": [
        ("/quit", "Exit EurekAgent"),
        ("/skip-prepare", "Skip the prepare stage"),
    ],
    "propose": [
        ("/quit", "Exit EurekAgent"),
    ],
    "implement": [
        ("/quit", "Exit EurekAgent"),
    ],
}

_DEFAULT_COMMANDS: list[tuple[str, str]] = [
    ("/quit", "Exit EurekAgent"),
]


def _commands_for_stage(stage: str) -> list[tuple[str, str]]:
    return STAGE_COMMANDS.get(stage, _DEFAULT_COMMANDS)


class CommandPalette(Container):
    """Filterable command list that appears above the input bar."""

    DEFAULT_CSS = """
    CommandPalette {
        dock: bottom;
        height: auto;
        max-height: 8;
        margin: 0 1;
        display: none;
    }
    CommandPalette OptionList {
        height: auto;
        max-height: 8;
        border: tall $accent;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        yield OptionList(id="cmd-options")

    def show(self, filter_text: str = "/", stage: str = "") -> None:
        """Populate with commands matching *filter_text* for the given *stage*."""
        ol = self.query_one("#cmd-options", OptionList)
        ol.clear_options()
        prefix = filter_text.lower()
        for cmd, desc in _commands_for_stage(stage):
            if cmd.startswith(prefix):
                ol.add_option(Option(f"{cmd}  {desc}", id=cmd))
        if ol.option_count:
            self.display = True
        else:
            self.display = False

    def hide(self) -> None:
        self.display = False
