"""EurekAgent ASCII logo banner widget."""

from __future__ import annotations

from rich.style import Style
from rich.text import Text
from textual.widgets import Static

LOGO_BLOCK = r"""
███████╗██╗   ██╗██████╗ ███████╗██╗  ██╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔════╝██║   ██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
█████╗  ██║   ██║██████╔╝█████╗  █████╔╝ ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██╔═██╗ ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
███████╗╚██████╔╝██║  ██║███████╗██║  ██╗██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝
""".strip("\n")

LOGO_MINIMAL = "· · E u r e k A g e n t · ·"
TAGLINE = "AI-for-Research auto-experiment loop"

# Block logo requires ~85 columns
_BLOCK_MIN_WIDTH = 85

# Purple gradient: deep → light (one per logo row)
_GRADIENT = ["#7C3AED", "#9C4DCC", "#B39DDB", "#CE93D8", "#E1BEE7", "#F3E5F5"]
_TAGLINE_COLOR = "#CE93D8"
_MINIMAL_COLOR = "#B39DDB"


class LogoBanner(Static):
    """Displays the EurekAgent logo in block or minimal style."""

    DEFAULT_CSS = """
    LogoBanner {
        text-align: center;
        padding: 1 0;
    }
    """

    def __init__(self, style: str = "minimal", **kwargs) -> None:
        super().__init__(**kwargs)
        self._logo_style = style

    def render(self) -> Text:
        text = Text(justify="center")

        if self._logo_style == "block":
            try:
                w = self.app.size.width
            except Exception:
                w = 80

            lines = LOGO_BLOCK.splitlines()
            width = max(map(len, lines), default=0)

            if w >= max(_BLOCK_MIN_WIDTH, width):
                for i, line in enumerate(lines):
                    text.append(line.ljust(width), Style(color=_GRADIENT[i % len(_GRADIENT)]))
                    text.append("\n")
                text.append(TAGLINE, Style(color=_TAGLINE_COLOR, bold=True))
                return text

        text.append(LOGO_MINIMAL, Style(color=_MINIMAL_COLOR, bold=True))
        text.append("\n")
        text.append(TAGLINE, Style(color=_TAGLINE_COLOR))
        return text

    def set_style(self, style: str) -> None:
        self._logo_style = style
        self.refresh()
