"""Structured report shown in the TUI when a run terminates.

The pipeline hands the TUI a small dict via `push_run_end_event`; this module
turns that dict into a `TerminalReport` that the banner widget renders. Kept
separate from the widget itself so it can be unit-tested without Textual.

Today the report surfaces only basic info (stage, reason, detail, evidence
paths). A `suggestions` field is reserved for a later iteration that maps
specific failure modes to concrete fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_SEVERITY_BY_STATUS = {
    "success": "success",
    "all_succeeded": "success",
    "partial_succeeded": "success",
    "abort": "error",
    "error": "error",
    "interrupted": "warning",
}

_HEADLINE_BY_STATUS = {
    "success": "✅  RUN COMPLETED",
    "all_succeeded": "✅  RUN COMPLETED",
    "partial_succeeded": "✅  RUN COMPLETED",
    "abort": "❌  RUN ABORTED",
    "error": "💥  PIPELINE ERROR",
    "interrupted": "⏸  RUN INTERRUPTED",
}


@dataclass
class TerminalReport:
    """What the banner needs to render a terminal-state panel."""
    headline: str            # e.g. "❌  RUN ABORTED"
    severity: str            # "error" | "warning" | "success"
    stage_label: str         # e.g. "implement (round 1/2)"
    reason: str              # one-line human explanation
    detail: str              # measured one-liner, may be empty
    evidence_paths: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    token_summary: str = ""
    best_score: float | None = None
    best_approach_id: str = ""
    baseline_score: float | None = None
    baseline_approach_id: str = ""


def build_report(data: dict[str, Any]) -> TerminalReport:
    """Construct a `TerminalReport` from the `run_ended` event payload.

    The event payload is `{status, reason, stage, round, detail, evidence_paths}`.
    Unknown keys are tolerated so the TUI never crashes on an older sender.
    """
    status = str(data.get("status", "unknown"))
    severity = _SEVERITY_BY_STATUS.get(status, "error")
    headline = _HEADLINE_BY_STATUS.get(status, f"RUN {status.upper()}")

    stage = str(data.get("stage", "") or "")
    round_label = str(data.get("round", "") or "")
    if stage and round_label:
        stage_label = f"{stage} (round {round_label})"
    elif stage:
        stage_label = stage
    else:
        stage_label = "—"

    evidence = [str(p) for p in data.get("evidence_paths", []) if p]

    return TerminalReport(
        headline=headline,
        severity=severity,
        stage_label=stage_label,
        reason=str(data.get("reason", "") or "—"),
        detail=str(data.get("detail", "") or ""),
        evidence_paths=evidence,
        suggestions=[],
        token_summary=str(data.get("token_summary", "") or ""),
        best_score=data.get("best_score"),
        best_approach_id=str(data.get("best_approach_id", "") or ""),
        baseline_score=data.get("baseline_score"),
        baseline_approach_id=str(data.get("baseline_approach_id", "") or ""),
    )
