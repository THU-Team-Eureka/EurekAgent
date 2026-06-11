"""Brief builder for the propose stage."""

from __future__ import annotations

import json
from pathlib import Path

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


def build_propose_brief(state: LoopState, loop_index: int, time_limit: str) -> str:
    """Build the markdown brief sent to the propose session.

    `time_limit` is the stage-resolved human string (e.g. "5 minutes") — the
    caller passes the propose-specific value if one is configured.
    """
    workspace = Path(state["workspace_dir"])
    max_num = state["max_num_approaches"]
    max_loops = state["max_loops"]

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
- Round: {loop_index}/{max_loops}
- Max approaches: {max_num} (generate as many distinct strategies as you see fit, up to this limit; fewer is fine if fewer viable strategies exist)
- Time limit per session: {time_limit}
- Cost limit: {state.get("cost_limit") or "N/A"}
- Token pricing: {_format_token_pricing(state)}
- Initial/reference code: {"inputs/initial_code/" if has_initial_code else "N/A"}
- GPU availability: {gpu_str} (GPUs hidden — any GPU use must be inside `gpu_session(gpu_ids=[...], approach_id=<assigned_approach_id>)` after `get_gpu_info()`; never set `CUDA_VISIBLE_DEVICES`; see skill)

## Required output paths (enforced by the controller)

Write outputs to these exact paths, relative to the workspace root. Writing
to any other path will cause the run to abort even if the content is valid.

- `round_state/current_round_approaches.jsonl` — JSON Lines manifest, one
  initial hypothesis per line. Each line MUST be a JSON object with at least:
      {{"id": "round_{{N}}_{{short_name}}", "name": "<short name>", "description": "<...>"}}
  The `id` MUST follow the format `round_{{N}}_{{short_name}}` where `N` is the current
  round number and `short_name` is a brief descriptive slug (lowercase, underscores/hyphens).
  Example: `round_1_cmaes_optim`, `round_2_lbfgs_refine`. Do NOT use generic names like
  `round_1_approach_1`. The `id` must match the `approach_details/<id>/` directory you create below.
- `approach_details/<id>/approach.md` — mandatory implementation-ready initial hypothesis + rationale
- `approach_details/<id>/code/` — created automatically by the system
- `approach_details/<id>/best_result.jsonl` — created automatically by the grader after scoring
- `approach_details/<id>/intermediate_results.jsonl` — created automatically by the grader after scoring

Write the manifest early — even with minimal initial-hypothesis descriptions — and
then refine in place. The stage is complete only after every proposed approach
also has a non-empty, implementation-ready `approach.md`.
"""

    brief += """
## Secure Evaluation

This run uses **secure evaluation**. The evaluation script is hidden from agent sessions.
- Future implementation agents must submit candidates through the grading service:
  `python3 /workspace/eval/eureka_submit.py --approach-dir approach_details/<id> --submission <path>`
- Agents cannot directly read or run the hidden grader.
- Submission format is described in `inputs/submission_format.md`.
- When writing initial-hypothesis descriptions, include instructions for using the grading service.
"""

    return brief
