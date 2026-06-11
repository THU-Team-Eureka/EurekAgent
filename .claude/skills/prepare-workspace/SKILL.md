---
name: prepare-workspace
description: >
  Validate problem setup, test the evaluation pipeline, and configure the
  environment before the propose-implement loop begins.
---

# Prepare Workspace

## Overview

You are the preparation agent. Your job is to ensure everything is ready
before the optimization loop starts. You have three responsibilities:

1. Validate the problem description for completeness
2. Test the evaluation pipeline
3. Set up the environment

There is **no time limit**. Take the time you need to get things right.

## Progress Tracking — Your Checkpoint

You MUST maintain `prepare/progress.json` as you work. This file is your
checkpoint — it records which steps are done so you always know where you
are. Update it after completing each step.

Write `progress.json` using this exact template:
```json
{
  "steps_completed": ["<step_names>"],
  "steps_remaining": ["<step_names>"],
  "notes": "<brief status of what was found/done>",
  "last_updated": "<ISO timestamp>"
}
```

Valid step names: `"problem_clarification"`, `"evaluation_verification"`,
`"environment_setup"`, `"write_summary"`.

**Example** — after completing Step 1:
```json
{
  "steps_completed": ["problem_clarification"],
  "steps_remaining": ["evaluation_verification", "environment_setup", "write_summary"],
  "notes": "Problem statement validated. One ambiguity found and resolved via Q&A.",
  "last_updated": "2026-05-15T06:26:52.313Z"
}
```

If `prepare/progress.json` already exists when you start, read it first —
it tells you what has been done and where to resume.

## GPU Policy (mandatory when GPUs may be used)

GPUs are **hidden by default** — sessions start with empty
`CUDA_VISIBLE_DEVICES`. The only sanctioned way to use a GPU is the
`gpu_helpers` module. Do **not** probe GPUs via `nvidia-smi`,
`torch.cuda.device_count()`, or by setting `CUDA_VISIBLE_DEVICES` yourself in
Bash, Python, or subprocess environments, and do **not** read or modify
`.eureka_internal/` (hook blocks both).

The single workflow — one function does everything:

```python
from gpu_helpers import get_gpu_info, gpu_session

info = get_gpu_info()  # call before every acquire
# Each entry: {id, locked_by, memory_used_pct, memory_total_mb}

candidates = [g for g in info if g["locked_by"] is None]
candidates.sort(key=lambda g: g["memory_used_pct"])
chosen = candidates[0]["id"]

with gpu_session(gpu_ids=[chosen], approach_id="prepare") as g:
    # Inside this block CUDA_VISIBLE_DEVICES is set to your acquired GPU,
    # the lock is held, and any eureka_score() call you make is graded
    # with that same GPU exposed to the evaluator (automatically — the
    # grading service reads .gpu_locks/ to discover which GPUs you hold).
    ...  # any GPU-using work belongs inside this block
# Lock released, env restored automatically on exit.
```

Rules: always call `get_gpu_info()` before `gpu_session()`; never set
`CUDA_VISIBLE_DEVICES` yourself; inside the `with` block CUDA renumbers
your GPUs starting at 0, so `gpu_session(gpu_ids=[5])` means your code uses
`torch.device("cuda:0")` for physical GPU 5.
`locked_by is None` only means no Eureka agent holds the GPU lock; high
`memory_used_pct` can still indicate an external process or stale CUDA user.
Prefer low-memory GPUs, and wait if all free GPUs are heavily loaded.

## Q&A Protocol

If you need user input, write a question file and **end your turn**:

```bash
cat > prepare/question.json << 'JSONEOF'
{
  "question": "Your question here?",
  "options": [
    {"label": "Option A", "description": "What A means"},
    {"label": "Option B", "description": "What B means"},
    {"label": "Other", "description": "Type your own answer"}
  ]
}
JSONEOF
```

Rules:
- The `question` key is required. The `options` key is optional.
- Always validate the file after writing: read it back and confirm it is valid JSON.
- If the JSON is malformed, rewrite it immediately.
- If you cannot produce structured options, write a free-form question: `{"question": "..."}`
- After writing `question.json`, **stop generating**. The pipeline will present
  your question to the user and send their answer back.
- The user's answer arrives as a message. You can also read `prepare/answer.json`
  for a structured version.

## Web Search

If you need to search anything from the web, such as to obtain problem context or to aid environment configuration, please use `Web_Search` (may also appear as the `web-search-prime` MCP) to perform general web search from the search engine, and the `Playwright MCP` to obtain actual webpage content.

## Step-by-Step Instructions

### Step 1 — Problem Clarification

Read `inputs/problem.md` and `inputs/submission_format.md`.

Verify these required components are present and clear:

- Problem statement with a clearly stated optimization objective
- Optimization direction — confirm it matches `is_better()`
- Constraints explicitly listed
- Submission JSON format with required keys and their types
- Score semantics (what the score means, what value is better)
- Invalid submission behavior (what happens on validation failure)

If any required component is **missing or genuinely ambiguous**, write
`prepare/question.json` and ask. Do NOT over-ask — only ask about things
that would cause incorrect optimization if misunderstood. Do not ask about
optional preferences or implementation details.

### Step 2 — Environment Setup

1. You are running in a pre-configured virtual uv Python environment. The hidden grader service also runs under the same venv. Read `inputs/initial_code/` for dependency clues (imports, requirements.txt, etc.)
2. You may check the current Python environment directly via `pip list` or `pip show <package>`
3. Install all required packages via: `uv pip install <package>`. DO NOT read or configure /usr/bin/python3 or ~/.local/lib/python3.x/site-packages; always use the venv environment!
4. If installation fails, **troubleshoot autonomously first**:
   - Try alternative versions: `uv pip install <package>==<version>`
   - Check PyPI availability
   - Try `pip install` as fallback
   - Only ask the user for help if ALL self-repair attempts fail
5. Check GPU availability (when the task may use GPUs):
   - Call `get_gpu_info()` (see GPU Policy above)
   - Record allowed GPU IDs, lock status, and memory usage in `prepare/summary.md`
   - If the task requires GPU but `get_gpu_info()` returns no GPUs, warn via `prepare/question.json`
   - **Do NOT use `torch.cuda.is_available()` to decide whether to install CUDA PyTorch.**
     GPU access is managed by a coordination system — `CUDA_VISIBLE_DEVICES` is
     empty by default, so `torch.cuda.is_available()` always returns False even
     when GPUs are present. Instead, check the environment variable
     `EUREKA_HOST_CUDA_DEVICES` (comma-separated GPU IDs, e.g. `0,1,2,3`).
     If GPUs are detected, install the CUDA version of PyTorch.
6. Verify other hardware availability if relevant (memory, disk space)
7. Ensure the environment is ready for all subsequent implement sessions

### Step 3 — Evaluation Script Verification

1. Run a trivial test submission through the grading service:

```python
import subprocess, json

def eureka_score(candidate, approach_dir):
    path = f"{approach_dir}/submissions/_test.json"
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

2. Verify `is_better()` logic: submit two known scores and confirm the
   ranking direction matches the problem description (e.g., if the problem
   says "maximize", then `is_better(3, 2)` should return True).

3. **If the grader needs a GPU**,
   wrap the `eureka_score(...)` call in a `gpu_session(approach_id="prepare")`
   block as shown in the GPU Policy section. The grading service reads
   `.gpu_locks/` to discover which GPU you hold and exposes only that GPU
   to the evaluator. You do not pass any GPU id to `eureka_score`.

4. If the grader fails, report the error and ask the user for guidance
   via `prepare/question.json`.

### Step 4 — Write Summary and Complete

When everything is verified and ready, update `prepare/progress.json` to mark all steps completed, then write two files:

**`prepare/summary.md`** — containing:
- Validated problem understanding (objective, direction, constraints)
- Evaluation test results (grader works, is_better direction confirmed)
- Environment status (installed packages, hardware detected)
- Any issues or warnings the implement agents should know about

**`prepare/complete.json`** — signaling completion:

```bash
cat > prepare/complete.json << 'JSONEOF'
{
  "status": "ready",
  "summary_path": "prepare/summary.md",
  "questions_asked": 0,
  "environment_packages_installed": []
}
JSONEOF
```

## Quality Checklist

- [ ] `inputs/problem.md` and `inputs/submission_format.md` read and validated
- [ ] Optimization objective and direction confirmed
- [ ] Submission format (JSON keys, types, score semantics) verified
- [ ] Evaluation grader tested with a trivial submission
- [ ] `is_better()` direction confirmed
- [ ] Required packages installed and importable
- [ ] GPU status queried via `get_gpu_info()` when GPUs may be used
- [ ] `prepare/summary.md` written
- [ ] `prepare/complete.json` written
