---
name: implement-approach
description: >
  Run a grader-driven experiment from an assigned initial hypothesis. Reads
  the hypothesis brief, writes code, submits candidates through the grading
  service, and iterates on official scores within one approach directory.
---

# Implement Approach

## Overview

You are responsible for running one hypothesis-driven experiment. Start from the assigned initial hypothesis, write code, submit candidates through the grading service, and use official scores to refine, tune, or pivot within the time budget and within your assigned approach directory.

## Inputs

| Source | Description |
|--------|-------------|
| Brief (appended to skill invocation) | Approach ID, initial hypothesis, time limit, file paths |
| `approach_details/<id>/approach.md` | Initial hypothesis and rationale |
| `inputs/problem.md` | Problem description |
| `inputs/submission_format.md` | Submission format and scoring description |
| `inputs/initial_code/` | Reference code provided by the user (optionally provided by the user) |

## File Ownership and Controller-Owned Files

- Write only inside `approach_details/<id>/`
- Do **not** edit `round_state/current_round_approaches.jsonl`
- Do **not** write outside your approach directory
- You may inspect prior rounds, but do **not** read or modify same-round peer approach directories or descriptions
- Do **not** write to `intermediate_results.jsonl` or `best_result.jsonl` — they are controller-owned
- Do **not** write to `eval_feedback/` — grading feedback appears here (do NOT write here)

The grading service automatically maintains:

- `intermediate_results.jsonl` — one line per evaluated submission
- `best_result.jsonl` — best entry so far
- `eval_feedback/latest_feedback.json` — feedback from the most recent submission

You do NOT write these files yourself.

## Step-by-Step Instructions

### Step 1 — Understand the Approach

1. Read `approach_details/<id>/approach.md` for the initial hypothesis
2. Read `inputs/problem.md` for problem context
3. Read `inputs/submission_format.md` to understand the expected JSON submission format and scoring rules
4. If `inputs/initial_code/` exists, read the reference code for baselines or starting points

### Step 2 — Implement

Write your implementation under `approach_details/<id>/code/`. The code must produce output that can be serialized into the JSON submission format described in `inputs/submission_format.md`.

**Verify your workspace**: The following directories inside `approach_details/<id>/` are created by the system before your session starts — run `ls approach_details/<id>/` to confirm they exist:
- `code/` — write your implementation files here
- `submissions/` — temporary staging only; submitted JSON files will be cleaned after grading
- `eval_feedback/` — grading feedback appears here (do NOT write here)
- `logs/` — write experiment logs here

Do NOT create these directories yourself. If any are missing, that is a system error — proceed anyway (the grading service will create them as needed). Historical submitted payloads are preserved in `intermediate_results.jsonl` under `solution`; do not expect files to remain in `submissions/`.

**Environment Info**: You are coding in a **pre-configured Python virtual environment**. The venv is already active — use `python3` and `pip` directly (no `uv run` prefix needed). There are some pre-installed packages in the environment — check with `pip show {package_name}`. To install additional packages, use `uv pip install {package_name}` (NOT bare `pip install` — the venv is managed by uv). The hidden grader service also runs under the same venv. DO NOT read or configure /usr/bin/python3 or ~/.local/lib/python3.x/site-packages; always use the venv environment!

#### Evaluation Helper — Use This Instead of Computing Your Own Score

Your solver MUST use the secure grader to evaluate candidates. Do NOT compute your own fitness/score metric — it may differ from the official one.

Embed this helper in your code and call it inside your iteration loop:

```python
import subprocess, json, time

def _validate_solution_description(candidate: dict) -> None:
    desc = candidate.get("description")
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError("candidate must include a description of the exact submitted solution")
    desc = desc.strip()
    lower = desc.lower()
    vague_starts = ("retry", "rerun", "tune", "tuned", "update", "updated", "fix", "bugfix", "debug", "minor", "small", "delay")
    if len(desc) < 80 or len(desc.split()) < 12 or lower.startswith(vague_starts) or (lower.startswith("v") and len(lower) > 1 and lower[1].isdigit()):
        raise ValueError(
            "candidate description must be a standalone solution summary: describe the core method, "
            "key design choices, and important parameters/kernels/constraints; do not write a version "
            "label or relative change note"
        )

def eureka_score(candidate: dict, approach_dir: str) -> tuple[float, bool] | None:
    """Submit a candidate to the grader, return (score, valid) or None on failure."""
    path = f"{approach_dir}/submissions/_ckpt_{int(time.time()*1000)}.json"
    _validate_solution_description(candidate)
    with open(path, "w") as f:
        json.dump(candidate, f)
    r = subprocess.run(
        ["python3", "/workspace/eval/eureka_submit.py",
         "--approach-dir", approach_dir, "--submission", path],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode == 0:
        d = json.loads(r.stdout)
        return d.get("score", 0.0), d.get("valid", False)
    return None
```

**Scoring direction**: Determined by `is_better()` in `evaluate.py`. For example, if `is_better(new, old)` returns `True` when `new > old`, higher is better; if `True` when `new < old`, lower is better. Read `inputs/submission_format.md` for the problem-specific direction. Do NOT assume any default direction. The engine uses `is_better()` for cross-approach ranking — optimizing in the wrong direction will hurt your ranking.

If you are not using an iterative solver, you can also submit directly:

```bash
python3 /workspace/eval/eureka_submit.py \
  --approach-dir approach_details/<id> \
  --submission approach_details/<id>/submissions/<file>.json
```

#### Version Control — Git Is Your Memory

Your code directory has git initialized. **Git is your memory.** Commit your
code whenever the solution implementation changes — new algorithm, parameter
change, bug fix, or structural refactor. Each commit records a distinct
solution version so you can track what you've tried and what worked.

**When to commit:**
- After writing your initial implementation
- After changing the algorithm or optimization strategy
- After modifying key parameters or hyperparameters
- After fixing a bug that affects the solution output
- After any significant refactoring of the solution code
- Before submitting a candidate after code changes, so `code_git_hash` can reconstruct the scored code

**When NOT to commit:** You don't necessarily have to commit on every `eureka_score()` call. Iterative
solvers may call `eureka_score()` many times
within the same code version — commit only when the code itself changes.

**Commit message format (REQUIRED):**

```
[v<N>] <solution description> | Changed: <what changed from previous version>
```

Each commit message MUST contain two parts separated by ` | Changed: `:
1. **Solution description**: A concise description of the current solution's
   strategy (not just "updated main.py")
2. **What changed**: What is different from the previous commit's solution

**Commands:**
```bash
cd approach_details/<id>/code
git add -A
git commit -m "[v1] <description> | Changed: <what changed>"
```

**Reviewing your own history:**
```bash
git log --oneline     # see all versions at a glance
git show HEAD~1:main.py  # inspect the previous version's code
git diff HEAD~1       # see what changed between versions
```

**Tracing scored code later:**
Read a record in `intermediate_results.jsonl` or `best_result.jsonl`, then inspect `code_git_hash`:

```bash
git -C approach_details/<id>/code show <code_git_hash>
```

If `code_git_dirty` is `true`, the score used uncommitted changes; the commit hash is only an approximate anchor. Commit before submitting whenever code changed.

#### GPU-Aware Experimentation (GPU tasks only)

GPUs are **hidden by default** — sessions start with empty
`CUDA_VISIBLE_DEVICES`. The only sanctioned way to use a GPU is the
`gpu_helpers` module. Do **not** use `nvidia-smi`, do **not** set
`CUDA_VISIBLE_DEVICES` yourself in Bash, Python, or subprocess environments,
and do **not** read or modify anything under `.eureka_internal/` (the hook
blocks this).

**The single workflow — one function does everything you need:**

```python
from gpu_helpers import get_gpu_info, gpu_session, GpuUnavailableError

# 1. Inspect availability (REQUIRED before every acquire).
info = get_gpu_info()
# Each entry: {id, locked_by, memory_used_pct, memory_total_mb}.

# 2. Pick an unlocked, low-load GPU based on your needs.
candidates = [g for g in info if g["locked_by"] is None]
candidates.sort(key=lambda g: g["memory_used_pct"])
chosen = candidates[0]["id"]

# 3. Run all GPU work inside a single gpu_session block.
with gpu_session(gpu_ids=[chosen], approach_id="<your_approach_id>") as g:
    # Inside this block:
    #   - GPU <chosen> is locked to you (other approaches cannot acquire it)
    #   - CUDA_VISIBLE_DEVICES is set to <chosen> automatically
    #   - On exit (normal or exception), the lock is released and the
    #     previous CUDA_VISIBLE_DEVICES is restored
    run_gpu_work()
    eureka_score(candidate, approach_dir)     # GPU-based grading just works
```

**If nothing is free, loop with backoff:**

```python
import time
while True:
    info = get_gpu_info()
    candidates = [g for g in info if g["locked_by"] is None]
    candidates.sort(key=lambda g: g["memory_used_pct"])
    if candidates:
        break
    time.sleep(30)  # back off; another approach may release soon
# then proceed with `with gpu_session(gpu_ids=[candidates[0]["id"]], ...) as g: ...`
```

`gpu_session` itself also waits up to 5 minutes for the requested GPU(s) to
free up, so a single retry after a slow `time.sleep` is usually enough.

**Three concepts — do not confuse them:**
- `get_gpu_info()` — **inspect** which GPUs are free / loaded
- `gpu_session(gpu_ids=[…], approach_id=…)` — **lock + expose + auto-release**
  in one context manager. This is the only acquire you ever need.
- `CUDA_VISIBLE_DEVICES` — managed by `gpu_session` automatically. You must
  never touch it yourself.

**Grading needs a GPU? Just keep your `gpu_session` open across the call.**
The grading service reads `.gpu_locks/` directly to discover which physical
GPUs you currently hold and runs evaluation in a subprocess restricted to
exactly those GPUs. You do not (and cannot) pass GPU ids to `eureka_score`
or `eureka_submit.py` — the lock token inherited from `gpu_session` is the
single source of truth.

**Important rules:**
1. **Do NOT set CUDA_VISIBLE_DEVICES directly** — the system blocks this.
2. **Do NOT use a GPU for any reason without `gpu_session`** — GPU code fails with
   "no CUDA devices" because the env is empty until you enter the block.
3. **Always call `get_gpu_info()` first** to pick a free GPU.
4. If `GpuUnavailableError` is raised, all your chosen GPUs are still locked —
   wait, re-check `get_gpu_info()`, and retry.
5. For CPU-only tasks, skip `gpu_session` entirely; the grader sees no locks
   for you and runs evaluation with `CUDA_VISIBLE_DEVICES=""`.
6. `gpu_session` is a context manager — release is automatic and guaranteed
   on normal exit *or* exception. Do not try to manage locks yourself.
7. **Physical vs logical GPU IDs**: `get_gpu_info()` and `gpu_session` use
   **physical GPU IDs**. Inside the `with` block, CUDA renumbers your GPUs
   starting at 0, so `gpu_session(gpu_ids=[5])` means your code uses
   `torch.device("cuda:0")` for physical GPU 5.

**Contention-aware GPU use:**

Before acquiring a GPU, call `get_gpu_info()` and check `memory_used_pct`:
- `locked_by is None`: no Eureka agent holds the GPU lock, but external
  processes may still be using the physical GPU.
- `memory_used_pct < 1%`: GPU is likely idle.
- `memory_used_pct >= 5%`: Another process is likely using this GPU.
- `memory_used_pct >= 50%`: GPU is heavily occupied; wait for a cleaner GPU.

If no clean GPU is available, wait and re-check before entering `gpu_session`.

#### Time Budget Awareness

Your session has a time budget. A `.time/` directory in the workspace root contains:

- `.time/time_budget.json` -- session start time and total budget in seconds
- `.time/check_time.py` -- helper script to check elapsed/remaining time

**Stay time-aware**: Run `python3 .time/check_time.py` from time to time to keep
track of how much budget remains. A good habit is to check periodically between
iterations, submissions, or code changes.
- Leave enough time to submit candidates through eureka_score() or
  eureka_submit.py and get scored results before the session ends
- If you receive a system time warning, stop modifying your code and ensure
  you have at least one candidate submitted and scored


### Step 3 — Run Experiments and Submit for Grading

Your solver should call `eureka_score()` at regular intervals inside its iteration loop (every k generations or every m minutes, whichever is appropriate for your algorithm's pace).

- Each call writes a temporary submission file, gets the official score back, and the submitted file is cleaned after grading.
- After calling `eureka_score()`, use the returned score to guide your search (e.g., keep the best candidate, adjust parameters).
- If `eureka_score()` returns `None`, the submission failed — log the error and continue with your next iteration.

**Repair and resubmit:**
- If a submission returns `valid: false`, read the feedback for the error and fix your approach
- Do not stop on the first failure — iterate and improve within budget
- Every submitted JSON must include a standalone `description` that accurately describes the complete current candidate. It must explain the core method, key design choices, and important parameters/kernels/constraints so another agent can understand the solution without reading your git history. Do not use version labels or relative change notes like "v32 retry", "tuned params", or "bug fix"; put relative changes in git commit messages.

**Live output:**
- Run Python in unbuffered mode: `python3 -u`
- Pipe stdout/stderr to `logs/agent_run.log`: `python3 -u code/main.py 2>&1 | tee -a logs/agent_run.log`
- Log milestones and best-score updates in `experiment_log.md`

**Important:**
- Score files (`intermediate_results.jsonl`, `best_result.jsonl`) are read-only for you — they are updated automatically by the grading service
- Do NOT attempt to read, recreate, or bypass the hidden grader

### Step 4 — Verify Results

After submitting candidates, verify the grading results:

- Read `approach_details/<id>/eval_feedback/latest_feedback.json` for the latest evaluation
- Read `approach_details/<id>/best_result.jsonl` for the current best result (read-only, updated by grading service)
- Read `approach_details/<id>/intermediate_results.jsonl` for all historical submitted payloads and scores
- Ensure at least one submission received `valid: true`

## Error Handling

| Situation | Action |
|-----------|--------|
| Experiment fails | Log failure, repair code, rerun if budget remains |
| Grading returns `valid: false` | Read `eval_feedback/latest_feedback.json` for the error, fix submission format and resubmit |
| Time budget exhausted | Stop, preserve best result so far |
| No valid result produced | Leave best_result.jsonl absent (controller handles this) |
| No submission made before budget expires | Result is lost — system cannot score or carry it forward |

## Quality Checklist

- [ ] `approach.md`, `inputs/problem.md`, and `inputs/submission_format.md` read before starting
- [ ] `inputs/initial_code/` reviewed if present
- [ ] Code written under `approach_details/<id>/code/`
- [ ] Same-round peer approach directories were not read or modified
- [ ] Each submitted candidate includes a complete standalone solution `description`
- [ ] GPU status queried via `get_gpu_info()` before any `gpu_session()` (GPU tasks)
- [ ] `eureka_score()` called at regular intervals inside the iteration loop — no result counts unless the grading service records it
- [ ] Feedback from `eval_feedback/latest_feedback.json` reviewed after each submission
- [ ] Experiments run with live output to `logs/agent_run.log`
- [ ] All files stay inside `approach_details/<id>/`
