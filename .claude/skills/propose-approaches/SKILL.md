---
name: propose-approaches
description: >
  Propose initial hypotheses for an optimization problem. Reads problem
  and submission format from workspace inputs, generates diverse starting
  strategies, and writes a manifest for parallel grader-driven experiments.
---

# Propose Approaches

## Overview

Given a problem description and submission format in the workspace, this skill:

1. Analyzes previous rounds (if any) by reading ranked history
2. Researches the problem domain via web search
3. Generates diverse executable initial hypotheses
4. Writes `round_state/current_round_approaches.jsonl` with hypothesis metadata
5. Creates `approach_details/` directory structure for each approach

The engine ranks approaches automatically using the `is_better()` function from `evaluate.py`. No ranking script is needed.

## GPU Policy (mandatory when GPUs may be used)

GPUs are **hidden by default**. Do **not** use `nvidia-smi` or set
`CUDA_VISIBLE_DEVICES` in Bash, Python, or subprocess environments. Use
**only** the `gpu_helpers` module:

```python
from gpu_helpers import get_gpu_info, gpu_session

info = get_gpu_info()  # call before every acquire
```

Always call `get_gpu_info()` before `gpu_session()`. Inside the
`with gpu_session(...)` block CUDA renumbers your acquired GPUs starting at 0.
When choosing GPUs, treat `locked_by is None` as "not locked by Eureka" and
also inspect `memory_used_pct`; external processes can still occupy an
unlocked physical GPU.

> Propose-stage does not call the grader. If your proposed approach needs
> GPU during grading, the implement-stage skill explains the full
> `gpu_session(...)` workflow — the grading service authorizes GPU use from
> the lock token inherited inside the session; you do not pass GPU ids around.

## Inputs

All inputs come from files in the workspace:

| Source | Description |
|--------|-------------|
| `inputs/problem.md` | Problem description, goals, constraints |
| `inputs/submission_format.md` | Submission format and scoring description |
| `inputs/initial_code/` | Reference code provided by the user (optionally provided by th euser) |
| `round_state/ranked_past_best_solutions.jsonl` | Prior round best scored solutions (if not first round) |
| Brief (appended to skill invocation) | Loop context: round number, max approaches, time limit |

## Step-by-Step Instructions

### Step 1 — Understand the Problem

Read `inputs/problem.md` and `inputs/submission_format.md` as the primary source of truth for the problem definition. If `prepare/summary.md` exists, also read it to deepen understanding of validated details, clarified ambiguities, and environment constraints discovered during preparation.

Before proceeding, clearly identify:
- The problem description
- The solution space (what kind of code/strategy is expected)
- The submission format and scoring rules (how the hidden grader evaluates candidates)

If `inputs/initial_code/` exists, read the reference code to understand existing baselines or starting points.

**If any input is missing:**
- Use the `AskUserQuestion` tool to explicitly ask for the missing information
- Do NOT use default values or make guesses

### Step 1.5 — GPU Constraints (when relevant)

If approaches may use GPUs, call `get_gpu_info()` (see GPU Policy) and note allowed
GPU IDs, contention, and memory usage in each `approach.md` where it affects strategy.

### Step 2 — Learn from Previous Rounds

Before proposing new approaches, check for previous rounds:

If `round_state/ranked_past_best_solutions.jsonl` exists:
1. Read and analyze which actual scored solutions performed well/poorly
2. Review details at `approach_details/<approach_id>/` for top entries
3. If useful, inspect that approach's `intermediate_results.jsonl` to understand the evolution of a specific solution
4. Identify patterns: which methods worked? Which failed?
5. Use insights to inform new initial hypotheses

To inspect the code for a scored historical result, read `code_git_hash` from
that result record and run:

```bash
git -C approach_details/<approach_id>/code show <code_git_hash>
```

If `code_git_dirty` is `true`, the score used uncommitted changes; treat the
commit hash as an approximate anchor rather than exact reconstruction.

If only the legacy `round_state/ranked_past_approaches.jsonl` exists, read it as a fallback.

If first round: skip this step.

### Step 3 — Generate Diverse Initial Hypotheses

Produce up to the maximum number of diverse approaches specified in the brief. You may generate fewer if fewer distinct strategies are viable, but aim for at least 2. Quality over quantity. Each must be:
- A self-contained initial hypothesis implementable and evaluable independently
- Meaningfully different in strategy
- Described concisely in 3 sentences or fewer

**Key Principle**: Each proposal is an initial hypothesis for a long-running implementation experiment. Implementation agents may refine, tune, or pivot based on official grader feedback, so describe the starting assumption clearly without implying the final submitted solution must stay identical.

Each initial hypothesis should:
- Be comparable directly against other hypotheses
- Enable strategy-level learning across rounds

**Note on Web Search:** During your strategy brainstorm process, you may research the problem via web search to:
- Identify state-of-the-art methods and algorithms
- Find relevant libraries or frameworks
- Identify novel angles worth exploring

**Web Search Sources**: Please prioritize searching from credible sources, for example:
- Source types: academic papers, reputable blogs, official documentation, etc
- Sources: Google Scholar, ArXiv, Semantic Scholar, official docs, tech blogs, etc

**Web Search Tool Routing**: Please use `Web_Search` (may also appear as the `web-search-prime` MCP) to perform general web search from the search engine, and the `Playwright MCP` to obtain actual webpage content.

`./round_state/web_search_history.jsonl` is auto-maintained by the system — every search query and visited URL is recorded there automatically. Read this file before searching to avoid repeating queries that already have results.


### Step 4 — Write Manifest

Output `round_state/current_round_approaches.jsonl`:

```json
{
  "round_number": 1,
  "propose_time_limit_per_session": "20 minutes",
  "previous_insights": "summary of what worked/failed in previous rounds",
  "approaches": [
    {
      "id": "round_1_cmaes_optim",
      "name": "CMA-ES Optimization",
      "description": "Initial hypothesis description in 3 sentences or fewer."
    }
  ]
}
```

Requirements:
- Top-level key is `round_number`
- Each approach `id` MUST follow the format `round_{N}_{short_name}` where `N` is the current round number and `short_name` is a brief descriptive slug (lowercase alphanumeric, underscores, hyphens). Example: `round_2_lbfgs_refine`. Do NOT use generic names like `round_2_approach_1`.
- Carefully reread the written manifest and verify the validity of all required fields before proceeding

### Step 5 — Write Approach Details

For each approach, create:

```
approach_details/<id>/
  approach.md              # initial hypothesis + rationale
```

The following are created automatically by the system or grader — do NOT create them yourself:
- `approach_details/<id>/code/`
- `approach_details/<id>/best_result.jsonl` (after grading)
- `approach_details/<id>/intermediate_results.jsonl` (after grading)

`approach.md` is mandatory. Missing or empty `approach.md` prevents the
implement stage from starting. It must be more detailed than the short
manifest `description` and should be implementation-ready for the assigned
agent.

Each `approach.md` must include:
- Problem restatement
- Initial hypothesis description
- Concrete implementation steps
- Important parameters, kernels, constraints, or search knobs to try
- Expected strengths/weaknesses
- Main evaluation risks and fallback ideas

### Step 6 — Web Search History Review

Search history is automatically recorded — no manual writing needed. If you conducted web searches, review `round_state/web_search_history.jsonl` to verify the results are captured for future rounds to reference.

## Time Budget Awareness

Your session has a time budget. A `.time/` directory in the workspace root contains:

- `.time/time_budget.json` -- session start time and total budget in seconds
- `.time/check_time.py` -- helper script to check elapsed/remaining time

**Stay time-aware**: Run `python3 .time/check_time.py` from time to time to keep
track of how much budget remains. A good habit is to check periodically between
operations -- reading files, web searches, or writing drafts.
- Leave enough time to write your deliverables before the session ends
- If you receive a system time warning, stop all exploration immediately and write deliverables

## Quality Checklist

- [ ] `inputs/problem.md` and `inputs/submission_format.md` read and understood
- [ ] `inputs/initial_code/` reviewed if present
- [ ] Previous rounds analyzed (if any)
- [ ] GPU status queried via `get_gpu_info()` when GPUs may be used
- [ ] Up to max number of diverse initial hypotheses generated, all materially different
- [ ] `round_state/current_round_approaches.jsonl` is valid JSON with `round_number` and `id` fields
- [ ] Each `approach_details/<id>/` has a non-empty, implementation-ready `approach.md`
- [ ] Web search conducted and reviewed against `round_state/web_search_history.jsonl`
