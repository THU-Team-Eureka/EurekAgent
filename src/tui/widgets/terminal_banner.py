"""Bordered panel the OverviewScreen shows when a run terminates.

Severity drives the border colour; everything else comes from the
`TerminalReport` passed in via `show(report)`. The widget is hidden until
`show` is called.
"""

from __future__ import annotations

from textual.widgets import Static

from ..terminal_report import TerminalReport


class TerminalBanner(Static):
    """Prominent panel that reports why the run ended and where to look."""

    DEFAULT_CSS = """
    TerminalBanner {
        display: none;
        margin: 1 2;
        padding: 1 2;
        border: round $primary;
        background: $panel;
    }
    TerminalBanner.-error {
        border: round red;
    }
    TerminalBanner.-warning {
        border: round yellow;
    }
    TerminalBanner.-success {
        border: round green;
    }
    """

    _SEVERITY_CLASSES = ("-error", "-warning", "-success")

    def show(self, report: TerminalReport) -> None:
        """Render the report and make the banner visible."""
        self._apply_severity(report.severity)
        self.update(self._render_report(report))
        self.display = True

    def _apply_severity(self, severity: str) -> None:
        for cls in self._SEVERITY_CLASSES:
            self.remove_class(cls)
        self.add_class(f"-{severity}")

    @staticmethod
    def _render_report(r: TerminalReport) -> str:
        """Produce Rich-markup text for the panel body."""
        lines: list[str] = [f"[b]{r.headline}[/b]", ""]
        lines.append(f"[b]Stage:[/b]   {r.stage_label}")
        lines.append(f"[b]Reason:[/b]  {r.reason}")
        if r.detail:
            lines.append(f"[b]Detail:[/b]  {r.detail}")

        if r.best_score is not None or r.baseline_score is not None:
            lines.append("")
            if r.best_score is not None:
                aid = f" ({r.best_approach_id})" if r.best_approach_id else ""
                lines.append(f"[b]Best:[/b]    {r.best_score:.4f}{aid}")
            if r.baseline_score is not None:
                aid = f" ({r.baseline_approach_id})" if r.baseline_approach_id else ""
                lines.append(f"[b]First:[/b]   {r.baseline_score:.4f}{aid}")

        if r.evidence_paths:
            lines.append("")
            lines.append("[b]Inspect logs for more details:[/b]")
            for path in r.evidence_paths:
                lines.append(f"  [cyan]{path}[/cyan]")

        if r.suggestions:
            lines.append("")
            lines.append("[b]Suggested next steps:[/b]")
            for s in r.suggestions:
                lines.append(f"  • {s}")

        if r.token_summary:
            lines.append("")
            lines.append(f"[b]Token usage:[/b]  {r.token_summary}")

        return "\n".join(lines)
