"""Load run data from filesystem into the dict shape the monitor HTML expects."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..json_utils import json_safe
from ..stage_results import scan_stage_results
from ..token_tracker import find_transcripts_dir

log = logging.getLogger(__name__)

_APPROACH_ROUND_RE = re.compile(r"^round_(\d+)_")

# In-memory file cache: avoids re-reading unchanged files on every poll.
# Keyed by resolved path; value is (mtime_ns, parsed_content).
_file_cache: dict[Path, tuple[int, Any]] = {}

# Short-lived cache for expensive scan results (invalidated after 2 s).
_scan_cache: dict[str, tuple[float, Any]] = {}
_SCAN_TTL = 2.0


def _find_transcripts_dir(workspace: Path) -> Path:
    """Return session_transcripts dir, preferring new location under session_data/."""
    return find_transcripts_dir(workspace)


def load_run(run_dir: Path) -> dict[str, Any]:
    """Build the full RUN dict from a run directory on disk."""
    return _load_run(run_dir, detail_level="full")


def load_run_overview(run_dir: Path) -> dict[str, Any]:
    """Build a lightweight RUN dict with only overview data.

    Heavy fields (code files, conversation transcripts, experiment logs)
    are replaced with ``has_*`` flags and counts.  Use
    :func:`load_approach_detail` to load full data for a single approach
    on demand.
    """
    return _load_run(run_dir, detail_level="overview")


def load_approach_detail(run_dir: Path, approach_id: str) -> dict[str, Any] | None:
    """Load full detail for a single approach on demand."""
    workspace = run_dir.resolve() / "workspace"
    approach_dir = workspace / "approach_details" / approach_id
    if not approach_dir.is_dir():
        return None
    return _load_approach_detail(approach_dir, approach_id, {}, None, detail_level="full")


def load_conversation_detail(
    run_dir: Path, conversation_key: str, *, since: int | None = None
) -> dict | None:
    """Load a single conversation transcript on demand.

    *conversation_key* can be:
      - ``prepare`` for the prepare session
      - ``loop_N_propose`` for a loop's propose session
      - an approach_id (e.g. ``round_1_custom_cnn``)

    *since* (event offset) enables incremental loading: when given, only events
    at index ``>= since`` are returned (plus a ``total`` cursor). ``None`` loads
    the full conversation (legacy callers).
    """
    workspace = run_dir.resolve() / "workspace"

    if conversation_key == "prepare":
        path = _find_transcripts_dir(workspace) / "prepare" / "prepare.jsonl"
        return _load_conversation(path, "prepare", since=since)

    m = re.match(r"^loop_(\d+)_propose$", conversation_key)
    if m:
        transcript_dir = _find_transcripts_dir(workspace) / f"loop_{m.group(1)}"
        path = transcript_dir / "propose.jsonl"
        return _load_conversation(path, "propose", since=since)

    # Treat as approach_id
    for loop_dir in sorted(_find_transcripts_dir(workspace).glob("loop_*")):
        path = loop_dir / f"{conversation_key}.jsonl"
        if path.exists():
            return _load_conversation(path, conversation_key, since=since)
    return None


def _load_run(run_dir: Path, *, detail_level: str = "full") -> dict[str, Any]:
    """Internal: build the RUN dict with the given detail level."""
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_id = run_dir.name
    workspace = run_dir / "workspace"

    summary = _load_summary(run_dir)
    metadata = _load_metadata(run_dir, summary=summary)
    loops = _load_loops(run_dir, workspace, detail_level=detail_level)

    return json_safe({
        "run_id": run_id,
        "metadata": metadata,
        "summary": summary,
        "loops": loops,
    })


def load_execution_log(run_dir: Path) -> list[dict[str, Any]]:
    """Build timeline events from run artifacts."""
    run_dir = run_dir.resolve()
    workspace = run_dir / "workspace"
    summary = _load_summary(run_dir)
    timeline: list[dict[str, Any]] = []

    # Read created_at directly from metadata file (avoid heavy _load_metadata)
    raw_meta = _read_json_or_jsonl(run_dir / "run_metadata.json") or {}
    created_at = raw_meta.get("created_at", "")
    if created_at:
        timeline.append({
            "event": "pipeline_start",
            "timestamp": created_at,
            "run_id": run_dir.name,
        })

    stage_results = _scan_stage_results(workspace)

    # Prepare stage event (loop_index 0, before any rounds)
    prepare_result = stage_results.get(0, {}).get("prepare")
    if prepare_result:
        prepare_timestamp = created_at or ""
        prepare_result_file = workspace / "prepare_result.json"
        if prepare_result_file.exists():
            try:
                pr = json.loads(prepare_result_file.read_text(encoding="utf-8"))
                if pr.get("completed_at"):
                    prepare_timestamp = pr["completed_at"]
            except (OSError, json.JSONDecodeError):
                pass
        timeline.append({
            "event": "stage_complete",
            "timestamp": prepare_timestamp,
            "stage": "prepare",
            "loop_index": 0,
            "status": prepare_result.get("status", ""),
        })

    for loop_idx in sorted(stage_results):
        results = stage_results[loop_idx]
        propose = results.get("propose")
        implement = results.get("implement")

        if propose:
            timeline.append({
                "event": "stage_complete",
                "timestamp": created_at or "",
                "stage": "propose",
                "loop_index": loop_idx,
                "status": propose.get("status", ""),
                "summary": propose.get("summary", ""),
            })

        if implement:
            impl_status = implement.get("status", "")
            # Use "stage_progress" for running stages to avoid misleading
            # "stage_complete" label when the stage is still in progress.
            event_name = "stage_progress" if impl_status == "running" else "stage_complete"
            # Try to get timestamp from the first completed approach
            impl_timestamp = created_at or ""
            details_dir = workspace / "approach_details"
            if details_dir.is_dir():
                for ad in sorted(details_dir.iterdir()):
                    if not ad.is_dir() or not _approach_belongs_to_loop(ad.name, loop_idx):
                        continue
                    br = _read_json_or_jsonl(ad / "best_result.jsonl")
                    if br and br.get("evaluated_at"):
                        impl_timestamp = br["evaluated_at"]
                        break
            timeline.append({
                "event": event_name,
                "timestamp": impl_timestamp,
                "stage": "implement",
                "loop_index": loop_idx,
                "status": impl_status,
            })
            # Add approach_scored events from approach best_results
            for ad in sorted(details_dir.iterdir()) if details_dir.is_dir() else []:
                if not ad.is_dir() or not _approach_belongs_to_loop(ad.name, loop_idx):
                    continue
                br = _read_json_or_jsonl(ad / "best_result.jsonl")
                if not br:
                    continue
                timeline.append({
                    "event": "approach_scored",
                    "timestamp": br.get("evaluated_at") or impl_timestamp,
                    "loop_index": loop_idx,
                    "approach_id": ad.name,
                    "name": ad.name,
                    "score": br.get("score"),
                    "approach_status": "evaluated" if br.get("valid", True) else "failed",
                })

    final_status = summary.get("final_status")
    if final_status and final_status != "running":
        timeline.append({
            "event": "pipeline_end",
            "timestamp": summary.get("completed_at") or datetime.now(tz=timezone.utc).isoformat(),
            "final_status": final_status,
        })

    return json_safe(timeline)


def _coerce_finite_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _chart_score(approach: dict[str, Any]) -> float | None:
    final = approach.get("final_result") or {}
    if final and not final.get("valid", True):
        return None
    return _coerce_finite_score(approach.get("score"))


def build_score_chart(
    loops: list[dict[str, Any]],
    *,
    is_better: Callable[[float, float], bool] | None = None,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    """Build score chart data from loops.

    Returns (chart, global_best, global_worst). When is_better is provided,
    best/worst/improved/delta are pre-computed per loop and per agent.
    """
    from typing import Callable

    chart: list[dict[str, Any]] = []
    global_best: float | None = None
    global_worst: float | None = None

    for loop in loops:
        agents = []
        for a in loop.get("approaches", []):
            score = _chart_score(a)
            agents.append({
                "name": a.get("name") or a.get("approach_id", "unknown"),
                "score": score,
                "parent": a.get("parent_approach_id"),
                "is_best": False,
                "is_worst": False,
            })
        valid_scores = [a["score"] for a in agents if a["score"] is not None]
        if not valid_scores:
            continue

        # Compute loop best/worst using is_better
        loop_best = valid_scores[0]
        loop_worst = valid_scores[0]
        for s in valid_scores[1:]:
            if is_better is not None:
                if is_better(s, loop_best):
                    loop_best = s
                if is_better(loop_worst, s):
                    loop_worst = s

        # Update global best/worst
        if is_better is not None:
            if global_best is None or is_better(loop_best, global_best):
                global_best = loop_best
            if global_worst is None or is_better(global_worst, loop_worst):
                global_worst = loop_worst

        # Mark per-agent best/worst
        for a in agents:
            if a["score"] is not None and is_better is not None:
                a["is_best"] = a["score"] == loop_best
                a["is_worst"] = a["score"] == loop_worst and a["score"] != loop_best

        # Compute improvement vs previous loop
        prev_best = chart[-1]["best"] if chart else loop_best
        improved = is_better(loop_best, prev_best) if is_better and chart else False

        chart.append({
            "iter": loop["loop_index"],
            "agents": agents,
            "best": loop_best,
            "worst": loop_worst,
            "improved": improved,
            "delta": loop_best - prev_best if chart else 0.0,
        })

    return chart, global_best, global_worst


def _load_metadata(run_dir: Path, *, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {"_current_loop_index": 0, "_current_stage": "", "_pipeline_status": "unknown"}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}

    if summary is None:
        summary = _load_summary(run_dir)
    final_status = summary.get("final_status", "running")

    # Derive current pipeline state
    workspace = run_dir / "workspace"
    stage_results = _scan_stage_results(workspace)

    # Prefer live pipeline state file (written by runtime.write_pipeline_state)
    pipeline_state = _read_pipeline_state(workspace)
    current_loop = pipeline_state.get("current_loop_index", 0)
    current_stage = pipeline_state.get("current_stage", "")
    pipeline_status = pipeline_state.get("pipeline_status", "")

    # When pipeline_state reports a terminal status but run_summary.json is
    # missing (pipeline crashed before writing it), promote pipeline_status to
    # final_status so the frontend shows the correct state instead of "running".
    if pipeline_status in ("abort", "interrupted", "error") and final_status == "running":
        final_status = pipeline_status

    # Conversely, when a run is resumed the pipeline is alive again but the
    # stale run_summary.json still carries the old terminal status.  Trust the
    # live pipeline_state over the on-disk summary.
    if pipeline_status == "running" and final_status in ("abort", "interrupted", "error"):
        final_status = "running"

    is_finished = final_status in ("success", "abort", "interrupted", "error")

    if not pipeline_state:
        # Fallback: derive from stage result files
        current_loop = max(stage_results) if stage_results else 0
        current_stage = ""
        pipeline_status = "running" if not is_finished else final_status

        if not is_finished and stage_results:
            latest = stage_results[current_loop]
            if latest.get("implement") is None and latest.get("propose") is not None:
                current_stage = "implement"
            elif latest.get("propose") is None and latest.get("prepare") is not None:
                current_stage = "propose"
                current_loop += 1
            else:
                current_stage = "prepare"

    # Token usage from pipeline state (live), summary, or aggregated
    token_usage = pipeline_state.get("token_usage") or summary.get("token_usage", {})
    data.setdefault("input_tokens", token_usage.get("input_tokens"))
    data.setdefault("output_tokens", token_usage.get("output_tokens"))
    data.setdefault("cache_read_input_tokens", token_usage.get("cache_read_input_tokens"))
    data.setdefault("cache_creation_input_tokens", token_usage.get("cache_creation_input_tokens"))

    # Build formatted token_usage string for the header display
    from ..token_tracker import format_token_count
    inp = data.get("input_tokens") or 0
    out = data.get("output_tokens") or 0
    if inp or out:
        cost_str = ""
        try:
            from ..token_tracker import TokenTracker
            tracker = TokenTracker(
                _input_price=data.get("input_token_price"),
                _cache_creation_price=data.get("cache_creation_token_price"),
                _cache_read_price=data.get("cache_read_token_price"),
                _output_price=data.get("output_token_price"),
            )
            tracker.update_session("_total", {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": data.get("cache_read_input_tokens") or 0,
                "cache_creation_input_tokens": data.get("cache_creation_input_tokens") or 0,
            })
            cost = tracker.calculate_cost()
            if cost is not None:
                cost_str = f" · ${cost:.2f}"
        except Exception:
            pass
        data["token_usage"] = (
            f"{format_token_count(inp)} in · "
            f"{format_token_count(out)} out"
            f"{cost_str}"
        )
    else:
        data["token_usage"] = None

    # Time and cost limits
    propose_limit = data.get("propose_time_limit_per_session") or "--"
    implement_limit = data.get("implement_time_limit_per_session") or "--"
    data.setdefault(
        "time_limit",
        f"propose {propose_limit} / implement {implement_limit}",
    )
    cost_limit = data.get("cost_limit")
    if cost_limit is not None:
        data["cost_limit"] = cost_limit
    else:
        data.setdefault("cost_limit", None)

    data.setdefault("created_at", data.get("created_at", ""))
    data["_current_loop_index"] = current_loop
    data["_current_stage"] = current_stage
    data["_pipeline_status"] = _normalize_status(pipeline_status)

    return data


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        # No summary yet — check pipeline_state for a terminal status
        # (pipeline may have crashed before writing run_summary.json).
        pipeline_state = _read_pipeline_state(run_dir / "workspace")
        ps = _normalize_status(pipeline_state.get("pipeline_status", ""))
        if ps in ("abort", "interrupted", "error"):
            return {"final_status": ps}
        return {"final_status": "running"}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pipeline_state = _read_pipeline_state(run_dir / "workspace")
        ps = _normalize_status(pipeline_state.get("pipeline_status", ""))
        if ps in ("abort", "interrupted", "error"):
            return {"final_status": ps}
        return {"final_status": "running"}

    pipeline_state = _read_pipeline_state(run_dir / "workspace")
    if pipeline_state.get("pipeline_status") == "running":
        data["final_status"] = "running"
        return data

    # Derive final_status from state if not present
    if "final_status" not in data:
        status = _normalize_status(data.get("status", ""))
        if status in ("success", "abort", "interrupted", "error"):
            data["final_status"] = status
        else:
            # Check pipeline state first — it knows if pipeline aborted
            pipeline_state = _read_pipeline_state(run_dir / "workspace")
            ps = _normalize_status(pipeline_state.get("pipeline_status", ""))
            if ps in ("abort", "interrupted", "error"):
                data["final_status"] = ps
            elif pipeline_state.get("pipeline_status", "") == "ready" and data.get("status") in ("unknown", ""):
                # "ready" is the initial state — if summary says "unknown",
                # the pipeline ended abnormally (e.g. cost limit hit before
                # a proper status was written)
                data["final_status"] = "abort"
            else:
                # Try to derive from history
                impl_status = ""
                for entry in reversed(data.get("history", [])):
                    if entry.get("stage") == "implement":
                        impl_status = _normalize_status(entry.get("status", ""))
                        break
                if impl_status == "success":
                    data["final_status"] = "success"
                elif impl_status == "abort":
                    data["final_status"] = "abort"
                elif data.get("status") == "error":
                    data["final_status"] = "error"
                else:
                    data["final_status"] = "running"

    return data


def _load_loops(run_dir: Path, workspace: Path, *, detail_level: str = "full") -> list[dict[str, Any]]:
    stage_results = _scan_stage_results(workspace)
    loops: list[dict[str, Any]] = []

    for loop_idx in sorted(stage_results):
        results = stage_results[loop_idx]
        loop: dict[str, Any] = {"loop_index": loop_idx}

        # Prepare result (only on loop 0)
        prepare = results.get("prepare")
        if prepare:
            loop["prepare_result"] = {
                "status": prepare.get("status", ""),
                "summary": prepare.get("summary", ""),
            }

        # Propose result
        propose = results.get("propose")
        if propose:
            loop["propose_result"] = {
                "status": propose.get("status", ""),
                "summary": propose.get("summary", ""),
                "notes": propose.get("notes", ""),
            }

        # Implement result
        implement = results.get("implement")
        if implement:
            loop["implement_result"] = {
                "status": implement.get("status", ""),
                "summary": implement.get("summary", ""),
                "notes": implement.get("notes", ""),
            }

        # Approaches
        loop["approaches"] = _load_approaches(workspace, loop_idx, detail_level=detail_level)

        # When implement stage is still running, mark completed approaches as
        # "evaluated" so the UI doesn't give the impression the stage is done.
        if implement and implement.get("status") == "running":
            for a in loop["approaches"]:
                if a.get("status") == "completed":
                    a["status"] = "evaluated"

        # Propose conversation (skip full content in overview mode)
        transcript_dir = _find_transcripts_dir(workspace) / f"loop_{loop_idx}"
        propose_transcript = transcript_dir / "propose.jsonl"
        if detail_level == "overview":
            loop["has_propose_conversation"] = propose_transcript.exists()
        else:
            loop["propose_conversation"] = _load_conversation(propose_transcript, "propose")

        loops.append(loop)

    # Prepare conversation (always loaded, independent of loop index)
    prepare_transcript = _find_transcripts_dir(workspace) / "prepare" / "prepare.jsonl"
    if detail_level == "overview":
        has_prepare_conv = prepare_transcript.exists()
        if has_prepare_conv or stage_results.get(0, {}).get("prepare"):
            loop0 = next((l for l in loops if l["loop_index"] == 0), None)
            if loop0 is None:
                loop0 = {"loop_index": 0, "approaches": []}
                loops.insert(0, loop0)
            loop0["has_prepare_conversation"] = has_prepare_conv
    else:
        prepare_conv = _load_conversation(prepare_transcript, "prepare")
        if prepare_conv or stage_results.get(0, {}).get("prepare"):
            loop0 = next((l for l in loops if l["loop_index"] == 0), None)
            if loop0 is None:
                loop0 = {"loop_index": 0, "approaches": []}
                loops.insert(0, loop0)
            loop0["prepare_conversation"] = prepare_conv
            # Load prepare summary.md
            summary_path = workspace / "prepare" / "summary.md"
            if summary_path.exists():
                try:
                    loop0["prepare_summary"] = summary_path.read_text(encoding="utf-8")
                except OSError:
                    pass

    # Surface an in-progress propose round even before it has result files,
    # so the UI can stream its conversation live instead of showing the static
    # "stage in progress" placeholder. A loop_N/propose.jsonl that exists
    # without a stage_result entry marks the round currently being proposed.
    transcripts = _find_transcripts_dir(workspace)
    existing = {l["loop_index"] for l in loops}
    for loop_dir in sorted(transcripts.glob("loop_*"), key=lambda p: p.name):
        m = re.match(r"loop_(\d+)$", loop_dir.name)
        if not m:
            continue
        li = int(m.group(1))
        if li in existing:
            continue
        propose_transcript = loop_dir / "propose.jsonl"
        if not propose_transcript.exists():
            continue
        loop = {
            "loop_index": li,
            "approaches": [],
            "propose_result": {"status": "running", "summary": "Proposing approaches…", "notes": ""},
        }
        if detail_level == "overview":
            loop["has_propose_conversation"] = True
        else:
            loop["propose_conversation"] = _load_conversation(propose_transcript, "propose")
        loops.append(loop)

    loops.sort(key=lambda l: l["loop_index"])
    return loops


def _approach_belongs_to_loop(approach_id: str, loop_idx: int) -> bool:
    """Check if an approach_id belongs to the given loop by matching round_N prefix."""
    m = _APPROACH_ROUND_RE.match(approach_id)
    return m is not None and int(m.group(1)) == loop_idx


def _load_approaches(workspace: Path, loop_idx: int, *, detail_level: str = "full") -> list[dict[str, Any]]:
    """Load approach data for a specific loop."""
    approaches: list[dict[str, Any]] = []

    # Read the round-specific proposal snapshot first; fall back for old runs.
    manifest_path = workspace / "round_state" / f"round_{loop_idx}_approaches.jsonl"
    if not manifest_path.exists():
        manifest_path = workspace / "round_state" / f"loop_{loop_idx}_approaches.jsonl"
    if not manifest_path.exists():
        manifest_path = workspace / "round_state" / "current_round_approaches.jsonl"
    if not manifest_path.exists():
        manifest_path = workspace / "round_state" / "current_round_approaches.json"
    manifest = _read_json_or_jsonl(manifest_path)
    manifest_approaches = manifest.get("approaches", []) if manifest else []

    # Build lookup by id
    approach_meta: dict[str, dict] = {}
    for a in manifest_approaches:
        if isinstance(a, dict):
            aid = str(a.get("id", a.get("approach_id", ""))).strip()
            if aid:
                approach_meta[aid] = a

    # Collect results from ranked history or approach_details
    ranked_path = workspace / "round_state" / "ranked_past_best_solutions.jsonl"
    if not ranked_path.exists():
        ranked_path = workspace / "round_state" / "ranked_past_approaches.jsonl"
    if not ranked_path.exists():
        ranked_path = workspace / "round_state" / "ranked_past_approaches.json"
    ranked = _read_json_or_jsonl(ranked_path)
    ranked_entries = ranked.get("entries", []) if ranked else []

    # Scan approach_details directly, filtered by loop_idx
    details_dir = workspace / "approach_details"
    if details_dir.is_dir():
        for ad in sorted(details_dir.iterdir()):
            if not ad.is_dir():
                continue
            aid = ad.name
            if not _approach_belongs_to_loop(aid, loop_idx):
                continue
            meta = approach_meta.get(aid, {})
            entry = next((e for e in ranked_entries if e.get("approach_id") == aid), None)

            approach = _load_approach_detail(ad, aid, meta, entry, detail_level=detail_level)
            approaches.append(approach)

    # Add approaches from manifest that don't have detail dirs yet, filtered by loop_idx
    seen_ids = {a["approach_id"] for a in approaches}
    for a in manifest_approaches:
        if not isinstance(a, dict):
            continue
        aid = str(a.get("id", a.get("approach_id", ""))).strip()
        if not aid or aid in seen_ids:
            continue
        if not _approach_belongs_to_loop(aid, loop_idx):
            continue
        approaches.append({
            "approach_id": aid,
            "name": a.get("name", aid),
            "description": "",
            "solution_description": "",
            "initial_hypothesis_description": a.get("description", ""),
            "status": "pending",
            "score": None,
        })

    return approaches


def _scored_description(
    best_result: dict[str, Any] | None,
    ranked_entry: dict | None,
) -> str:
    for source in (best_result, ranked_entry):
        if not isinstance(source, dict):
            continue
        value = source.get("description")
        if isinstance(value, str) and value.strip():
            return value.strip()
        # Legacy runs stored the submitted description inside solution.
        solution = source.get("solution")
        if isinstance(solution, dict):
            value = solution.get("description")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _load_approach_detail(
    approach_dir: Path,
    approach_id: str,
    meta: dict,
    ranked_entry: dict | None,
    *,
    detail_level: str = "full",
) -> dict[str, Any]:
    """Load a single approach's detail.

    *detail_level="overview"* omits heavy fields (code files, conversations,
    experiment log) and replaces them with ``has_*`` flags and counts.
    """
    # Score and status from best_result.jsonl
    best_result_path = approach_dir / "best_result.jsonl"
    best_result = _read_json_or_jsonl(best_result_path)

    score = None
    status = "pending"
    final_result = None

    if best_result:
        score = _coerce_finite_score(best_result.get("score"))
        status = "completed" if best_result.get("valid", True) else "failed"
        final_result = json_safe(best_result)
    elif ranked_entry:
        score = _coerce_finite_score(ranked_entry.get("score"))
        status = ranked_entry.get("status", "pending")

    # Overview mode: skip heavy fields, return existence flags
    if detail_level == "overview":
        md_path = approach_dir / "approach.md"
        approach_md = ""
        if md_path.exists():
            try:
                approach_md = md_path.read_text(encoding="utf-8")
            except OSError:
                pass

        code_dir = approach_dir / "code"
        code_file_count = 0
        has_code_files = False
        if code_dir.is_dir():
            try:
                code_files_list = [f for f in code_dir.rglob("*") if f.is_file() and not f.name.startswith(".")]
                code_file_count = len(code_files_list)
                has_code_files = code_file_count > 0
            except OSError:
                pass

        initial_description = meta.get(
            "description",
            ranked_entry.get("initial_hypothesis_description", "") if ranked_entry else "",
        )
        scored_description = _scored_description(best_result, ranked_entry)
        return json_safe({
            "approach_id": approach_id,
            "name": meta.get("name", ranked_entry.get("initial_hypothesis_name", approach_id) if ranked_entry else approach_id),
            "description": scored_description,
            "solution_description": scored_description,
            "initial_hypothesis_description": initial_description,
            "status": status,
            "score": score,
            "approach_md": approach_md,
            "final_result": final_result,
            "has_code_files": has_code_files,
            "code_file_count": code_file_count,
            "has_experiment_log": (approach_dir / "experiment_log.md").exists(),
            "has_subagent_conversations": _has_subagent_conversations(approach_dir, approach_id),
            "subagent_conversation_count": _count_subagent_conversations(approach_dir, approach_id),
        })

    # Full mode: original behavior
    # Approach markdown
    approach_md = ""
    md_path = approach_dir / "approach.md"
    if md_path.exists():
        try:
            approach_md = md_path.read_text(encoding="utf-8")
        except OSError:
            pass

    # Experiment log
    experiment_log = ""
    log_path = approach_dir / "experiment_log.md"
    if log_path.exists():
        try:
            experiment_log = log_path.read_text(encoding="utf-8")
        except OSError:
            pass

    # Code files (cached by directory mtime)
    code_files: dict[str, str] = {}
    code_dir = approach_dir / "code"
    if code_dir.is_dir():
        code_dir_resolved = code_dir.resolve()
        try:
            code_dir_stat = code_dir_resolved.stat()
            code_cache_key = ("code", code_dir_resolved)
            code_cached = _file_cache.get(code_cache_key)  # type: ignore[arg-type]
            if code_cached is not None and code_cached[0] == code_dir_stat.st_mtime_ns:
                code_files = code_cached[1]
            else:
                for f in code_dir.rglob("*"):
                    if f.is_file() and not f.name.startswith("."):
                        try:
                            rel = str(f.relative_to(code_dir))
                            code_files[rel] = f.read_text(encoding="utf-8", errors="replace")
                        except OSError:
                            pass
                _file_cache[code_cache_key] = (code_dir_stat.st_mtime_ns, code_files)  # type: ignore[assignment]
        except OSError:
            for f in code_dir.rglob("*"):
                if f.is_file() and not f.name.startswith("."):
                    try:
                        rel = str(f.relative_to(code_dir))
                        code_files[rel] = f.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        pass

    # Session conversations
    subagent_conversations = _load_subagent_conversations(approach_dir, approach_id)

    initial_description = meta.get(
        "description",
        ranked_entry.get("initial_hypothesis_description", "") if ranked_entry else "",
    )
    scored_description = _scored_description(best_result, ranked_entry)
    return json_safe({
        "approach_id": approach_id,
        "name": meta.get("name", ranked_entry.get("initial_hypothesis_name", approach_id) if ranked_entry else approach_id),
        "description": scored_description,
        "solution_description": scored_description,
        "initial_hypothesis_description": initial_description,
        "status": status,
        "score": score,
        "approach_md": approach_md,
        "experiment_log": experiment_log,
        "code_files": code_files,
        "final_result": final_result,
        "subagent_conversations": subagent_conversations,
    })


def _has_subagent_conversations(approach_dir: Path, approach_id: str) -> bool:
    """Quick check if any conversation transcripts exist (without reading them)."""
    workspace = approach_dir.parent.parent
    for loop_dir in _find_transcripts_dir(workspace).glob("loop_*"):
        if (loop_dir / f"{approach_id}.jsonl").exists():
            return True
    return False


def _count_subagent_conversations(approach_dir: Path, approach_id: str) -> int:
    """Count conversation transcripts (without reading them)."""
    workspace = approach_dir.parent.parent
    count = 0
    for loop_dir in _find_transcripts_dir(workspace).glob("loop_*"):
        if (loop_dir / f"{approach_id}.jsonl").exists():
            count += 1
    return count


def _load_subagent_conversations(approach_dir: Path, approach_id: str) -> list[dict]:
    """Load conversation transcripts for an approach.

    Checks two sources:
    1. session_transcripts/loop_N/{approach_id}.jsonl (always)
    2. Claude Code subagents/ directories (agent-*.jsonl files)
    """
    conversations: list[dict] = []
    workspace = approach_dir.parent.parent  # approach_details -> workspace

    # Source 1: session_transcripts (existing)
    for loop_dir in sorted(_find_transcripts_dir(workspace).glob("loop_*")):
        transcript = loop_dir / f"{approach_id}.jsonl"
        conv = _load_conversation(transcript, approach_id)
        if conv:
            conversations.append(conv)

    # Source 2: Claude Code subagents/ directories
    _load_subagent_dirs(workspace, conversations)

    return conversations


def _load_subagent_dirs(workspace: Path, conversations: list[dict]) -> None:
    """Load subagent conversations from Claude Code's subagents directory."""
    project_dir = workspace / ".agent_home" / ".claude" / "projects" / "-workspace"
    if not project_dir.is_dir():
        return
    # Nested layout: <sessionId>/subagents/agent-*.jsonl
    for session_dir in project_dir.iterdir():
        subagent_dir = session_dir / "subagents"
        if not subagent_dir.is_dir():
            continue
        for jsonl_path in sorted(subagent_dir.glob("agent-*.jsonl")):
            subagent_name = f"subagent:{jsonl_path.stem}"
            conv = _load_conversation(jsonl_path, subagent_name)
            if conv:
                conversations.append(conv)
    # Flat layout: agent-*.jsonl directly in project_dir
    for jsonl_path in sorted(project_dir.glob("agent-*.jsonl")):
        subagent_name = f"subagent:{jsonl_path.stem}"
        conv = _load_conversation(jsonl_path, subagent_name)
        if conv:
            conversations.append(conv)


def _load_conversation(
    transcript_path: Path, session_name: str, *, since: int | None = None
) -> dict | None:
    """Parse a JSONL transcript into the conversation model the HTML expects.

    When *since* is given, return only events at index ``>= since`` plus a
    ``total`` cursor (number of parsed events in the file). This lets the
    client poll for just the new tail of an append-only transcript. The full
    parse is still mtime-cached, so repeated incremental reads are cheap.
    """
    if not transcript_path.exists():
        return None

    events: list[dict[str, Any]] = []

    # Cache parsed transcript by mtime
    resolved = transcript_path.resolve()
    try:
        stat = resolved.stat()
    except OSError:
        _file_cache.pop(resolved, None)
        return None

    cached = _file_cache.get(resolved)
    if cached is not None and cached[0] == stat.st_mtime_ns:
        events = cached[1]  # type: ignore[assignment]
    else:
        try:
            lines = resolved.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")
            parsed = _parse_transcript_event(event_type, data)
            if parsed:
                events.append(parsed)

        _file_cache[resolved] = (stat.st_mtime_ns, events)

    # Full load (legacy callers) — unchanged shape.
    if since is None:
        if not events:
            return None
        return {"session_name": session_name, "events": events}

    total = len(events)
    # File shrank below the client's cursor (e.g. resume truncated it): signal
    # a reset so the client refetches the full conversation.
    if since > total:
        return {"session_name": session_name, "events": [], "total": total, "reset": True}
    return {"session_name": session_name, "events": events[since:], "total": total}


def _parse_transcript_event(event_type: str, data: dict) -> dict | None:
    """Convert a raw transcript event into the HTML conversation model."""
    ts = data.get("timestamp", "")
    if event_type == "user":
        content_items = _extract_content_items(data.get("message", {}), role="user")
        # Skip user events with no renderable content (e.g. tool_result-only
        # turns that the API auto-generates — they produce empty "User"
        # bubbles). Image-only user prompts are renderable and must be kept.
        if not any(ci.get("type") in {"text", "image"} for ci in content_items):
            return None
        result: dict[str, Any] = {"role": "user", "content_items": content_items}
        if ts:
            result["timestamp"] = ts
        return result

    if event_type == "assistant":
        content_items = _extract_content_items(data.get("message", {}), role="assistant")
        result = {"role": "assistant", "content_items": content_items}
        if ts:
            result["timestamp"] = ts
        return result

    return None


def _extract_content_items(message: dict, role: str) -> list[dict]:
    """Extract content_items from a message dict."""
    content = message.get("content", [])
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    items: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        if block_type == "text":
            text = block.get("text", "")
            if text:
                items.append({"type": "text", "text": text})
        elif block_type == "thinking":
            text = block.get("thinking", "")
            if text:
                items.append({"type": "thinking", "text": text})
        elif block_type == "image":
            image_item = _extract_image_item(block)
            if image_item:
                items.append(image_item)
        elif block_type == "tool_use":
            items.append({
                "type": "tool_use",
                "tool_name": block.get("name", ""),
                "tool_input": json.dumps(block.get("input", {}), ensure_ascii=False),
            })
        elif block_type == "tool_result":
            items.append({
                "type": "tool_result",
                "tool_name": block.get("tool_use_id", ""),
                "text": _extract_tool_result_text(block),
            })

    return items


def _extract_image_item(block: dict) -> dict[str, Any] | None:
    """Extract a renderable image item from an Anthropic-style content block."""
    source = block.get("source")
    if not isinstance(source, dict):
        return None

    source_type = source.get("type")
    media_type = str(source.get("media_type") or "image/png")
    title_value = block.get("name") or block.get("title")
    title = str(title_value) if title_value else ""

    if source_type == "base64":
        data = source.get("data")
        if not isinstance(data, str) or not data:
            return None
        return {
            "type": "image",
            "title": title,
            "media_type": media_type,
            "url": f"data:{media_type};base64,{data}",
        }

    if source_type == "url":
        url = source.get("url")
        if isinstance(url, str) and url:
            return {"type": "image", "title": title, "media_type": media_type, "url": url}

    return None


def _extract_tool_result_text(block: dict) -> str:
    """Extract text from a tool_result block."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _scan_stage_results(workspace: Path) -> dict[int, dict[str, dict[str, Any]]]:
    """Derive stage completion from directory structure.

    Results are cached for a short TTL so that multiple callers within the
    same poll cycle (load_metadata, load_loops, load_execution_log) share
    the same scan instead of repeating the directory walk.
    """
    import time as _time
    cache_key = f"scan:{workspace.resolve()}"
    now = _time.monotonic()
    cached = _scan_cache.get(cache_key)
    if cached is not None and now - cached[0] < _SCAN_TTL:
        return cached[1]

    result = scan_stage_results(workspace)
    _scan_cache[cache_key] = (now, result)
    return result


def _read_json_or_jsonl(path: Path) -> dict[str, Any] | None:
    """Read a JSON or JSONL file, returning a dict.  Results are cached by mtime."""
    path = path.resolve()
    try:
        stat = path.stat()
    except OSError:
        _file_cache.pop(path, None)
        return None

    cached = _file_cache.get(path)
    if cached is not None and cached[0] == stat.st_mtime_ns:
        return cached[1]

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    # Try single JSON
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            _file_cache[path] = (stat.st_mtime_ns, payload)
            return payload
    except json.JSONDecodeError:
        pass

    # Try JSON Lines
    approaches: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                approaches.append(obj)
        except json.JSONDecodeError:
            continue
    if approaches:
        result = {"approaches": approaches}
        _file_cache[path] = (stat.st_mtime_ns, result)
        return result
    _file_cache[path] = (stat.st_mtime_ns, None)
    return None


def _normalize_status(raw: str) -> str:
    """Map pipeline-specific status values to the canonical set.

    The pipeline writes ``all_succeeded`` / ``partial_succeeded`` as status,
    but the monitor expects ``success``.
    """
    _STATUS_MAP = {
        "all_succeeded": "success",
        "partial_succeeded": "success",
        "all_failed": "abort",
    }
    return _STATUS_MAP.get(raw, raw)


def _read_pipeline_state(workspace: Path) -> dict[str, Any]:
    """Read the live pipeline state file written by runtime.write_pipeline_state."""
    path = (workspace / ".pipeline_state.json").resolve()
    try:
        stat = path.stat()
    except OSError:
        _file_cache.pop(path, None)
        return {}

    cached = _file_cache.get(path)
    if cached is not None and cached[0] == stat.st_mtime_ns:
        return cached[1]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        result = {}
    _file_cache[path] = (stat.st_mtime_ns, result)
    return result
