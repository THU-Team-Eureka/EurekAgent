"""RichLog that pauses auto-scroll when the user scrolls up."""

from __future__ import annotations

from textual.widgets import RichLog


class SmartRichLog(RichLog):
    """RichLog that disables auto-scroll when the user scrolls up.

    When the user is at the bottom of the log, new writes auto-scroll
    as normal. When the user scrolls up to read history, new writes
    append content but don't force the view back down. Scrolling back
    to the bottom re-enables auto-scroll.
    """

    def on_scroll(self, event: RichLog.ScrollEvent) -> None:
        distance_from_bottom = self.max_scroll_y - self.scroll_y
        self.auto_scroll = distance_from_bottom <= 3
