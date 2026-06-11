"""Brief builder for the implement stage (per-approach)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..state import LoopState


def _format_token_pricing(state: LoopState) -> str:
    inp = state.get("input_token_price")
    cc = state.get("cache_creation_token_price")
    cr = state.get("cache_read_token_price")
    out = state.get("output_token_price")
    currency = state.get("cost_currency", "USD")
    if inp is None and out is None and cc is None and cr is None:
        return "N/A"
    parts = []
    if inp is not None:
        parts.append(f"input {inp}/{currency} per 1M tokens")
    if cc is not None:
        parts.append(f"cache_creation {cc}/{currency} per 1M tokens")
    if cr is not None:
        parts.append(f"cache_read {cr}/{currency} per 1M tokens")
    if out is not None:
        parts.append(f"output {out}/{currency} per 1M tokens")
    return ", ".join(parts)


def build_implement_brief(
    state: LoopState, approach: dict[str, Any], time_limit: str
) -> str:
    """Build the markdown brief for a single approach worker session.

    `time_limit` is the stage-resolved human string (e.g. "20 minutes") — the
    caller passes the implement-specific value if one is configured.
    """
    workspace = Path(state["workspace_dir"])
    loop_index = state["loop_index"]

    approach_id = str(approach.get("id", "")).strip()
    initial_code = workspace / "inputs" / "initial_code"
    has_initial_code = initial_code.exists()

    # GPU availability
    gpu_config_path = workspace / ".gpu_config.json"
    if gpu_config_path.is_file():
        try:
            gpu_cfg = json.loads(gpu_config_path.read_text(encoding="utf-8"))
            gpu_ids = gpu_cfg.get("allowed_gpu_ids", [])
            gpu_str = f"{len(gpu_ids)} GPUs (IDs: {', '.join(str(g) for g in gpu_ids)})" if gpu_ids else "No GPUs available"
        except (json.JSONDecodeError, OSError):
            gpu_str = "N/A"
    else:
        gpu_str = "N/A"

    brief = f"""## Run Parameters
- Round: {loop_index}
- Approach id: {approach_id}
- Approach dir: approach_details/{approach_id}
- Initial hypothesis: {approach.get("description", "")}
- Time limit: {time_limit}
- Cost limit: {state.get("cost_limit") or "N/A"}
- Token pricing: {_format_token_pricing(state)}
- Initial/reference code: {"inputs/initial_code/" if has_initial_code else "N/A"}
- Python env: venv already active — use `python3` directly (no `uv run` needed). Pre-installed: numpy, scipy. Install more with `uv pip install <pkg>`.
- GPU availability: {gpu_str} (GPUs hidden — any GPU use must be inside `gpu_session(gpu_ids=[...], approach_id=<assigned_approach_id>)` after `get_gpu_info()`; never set `CUDA_VISIBLE_DEVICES`; see skill)
"""

    brief += f"""
## Experiment Model

Your assigned approach is an initial hypothesis, not a fixed implementation contract.
Start from it, then use official grader feedback to refine, tune, or pivot within your
own approach directory. The final score is attributed to the exact submitted solution,
so every submission JSON MUST include a non-empty `description` that accurately
describes that candidate as a complete standalone solution summary.

You may inspect prior rounds' approach directories and ranked best solutions. Do NOT
read or modify same-round peer approach directories or descriptions.

## Secure Evaluation

This run uses **secure evaluation**. The evaluation script is not available in the workspace.

**Do NOT compute your own score/fitness. Use `eureka_score()` inside your iteration loop.**

The `eureka_score()` helper is defined in your skill — embed it in your code and call it
at regular intervals (every k iterations or every m minutes) to get the official score.
Each call submits a candidate to the grading service and returns `(score, valid)`.
The grading service automatically maintains `intermediate_results.jsonl` and
`best_result.jsonl` — you do NOT write these yourself. Submitted files under
`submissions/` are temporary and will be cleaned after the submitted payload is
recorded in `intermediate_results.jsonl`.

If you are not using an iterative solver, you can also submit directly:
`python3 /workspace/eval/eureka_submit.py --approach-dir approach_details/{approach_id} --submission <path>`

**Critical rules:**
- Do NOT compute your own fitness/score — only the grading service produces valid scores.
- Do NOT attempt to read, recreate, or bypass the hidden grader.
- Score files (`intermediate_results.jsonl`, `best_result.jsonl`) are read-only for you — they are updated automatically by the grading service.
- Only submit valid JSON files matching the format in `inputs/submission_format.md`.
- Every submission must include a complete standalone `description` for the exact candidate being graded; do not use version labels or relative change notes.
"""

    return brief
