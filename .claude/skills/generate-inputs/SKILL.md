---
name: generate-inputs
description: >
  Interactively generate all required EUREKA problem input files
  (INSTRUCTION.md, SUBMISSION_FORMAT.md, evaluate.py) and the launch
  script (run.sh) from a researcher's natural language description,
  then validate the evaluation pipeline.
---

# Generate Inputs

## Overview

You are the problem generation assistant. Your job is to help researchers create new optimization problems for the EUREKA framework by interactively collecting information in phases, generating the three required input files (INSTRUCTION.md, SUBMISSION_FORMAT.md, evaluate.py) plus a launch script (run.sh), and validating the evaluation pipeline end-to-end. At each phase you show the generated artifact to the researcher, collect feedback, and finalize before moving on.

## Interaction Protocol

Rules for Q&A with the researcher:

- Use the AskUserQuestion tool for structured questions (prefer multiple choice when possible). If the tool is unavailable in your environment, use direct conversation with numbered options instead.
- Use direct conversation for open-ended questions.
- Ask one question at a time.
- Only ask about information critical for correct problem generation.
- Do NOT over-ask about optional details — if something can be reasonably inferred or defaulted, infer or default it.

## Phase 1 — Collect Problem Description

Ask the researcher the following questions one at a time. Use AskUserQuestion with multiple-choice options where specified.

1. **Problem name** — Ask the researcher for a name in kebab-case (e.g., `circle_packing`). This will be used as the directory name under `examples/`.
   - After receiving the name, check whether `examples/<problem_name>/` already exists.
   - If it exists, ask the researcher whether to overwrite the existing directory or use a different name.

2. **Domain** — Based on the problem name and any context the researcher has already provided, propose 3-4 relevant domain options via AskUserQuestion (e.g., if the problem name suggests geometry, offer "Computational Geometry", "Graph Theory", etc.). Always include an "Other" option so the researcher can provide a custom domain.

3. **Optimization objective** — Ask via direct conversation. Request a formula or clear natural language description of what is being optimized. The optimization direction (maximize/minimize/other) should be naturally evident from the objective — do not ask it as a separate question. If the direction is ambiguous, clarify it as part of the objective discussion.

4. **Constraints** — Ask via direct conversation. Collect all hard constraints that solutions must satisfy.

5. **Known best result** — Ask via direct conversation. This is optional — the researcher can leave it empty.

6. **Initial code** — Ask whether the researcher has starting code to provide as `initial.py`. This is optional — if not provided, the generated run.sh will omit the `--initial-code` flag and INSTRUCTION.md will not reference `initial.py`.

After collecting all Phase 1 information, summarize it back to the researcher and confirm before proceeding to Phase 2.

## Phase 2 — Generate INSTRUCTION.md

Generate INSTRUCTION.md using this template. Adapt the placeholders to the information collected in Phase 1.

```markdown
## TASK
Write a program to solve this <domain> problem.

<Mathematical formulation of the objective>

Your objective:
- <derive from the optimization objective — e.g., "Maximize the sum of radii" or "Minimize C5 bound">

Hard constraints:
- <constraint 1>
- <constraint 2>
- ...
- Return valid output in the evaluator contract.
- You should write a program to solve the problem.
<If initial.py is provided:>- Initial code is provided in the file `initial.py`. You can use it as a reference.
- To evaluate your program, we provide a evaluator in the file `evaluate.py`, which provides a final combined score. NEVER MODIFY THE EVALUATOR.

## Contract

<Ask the researcher what the entrypoint function name should be (e.g., `run`, `run_packing`). Default: `run`.>

Implement a function with the following signature:

```python
def <function_name>(seed: int, budget_s: float, **kwargs) -> dict:
    ...
```

The function must return a dictionary matching the submission format (see SUBMISSION_FORMAT.md).

<Problem-specific notes on input/output, e.g. what kwargs are available, what the return dict must contain>

## HINTS and RECOMMENDATIONS

Useful directions:
1. <hint 1 based on domain and objective>
2. <hint 2 based on domain and objective>
3. <hint 3 based on domain and objective>
4. <hint 4 based on domain and objective — include if applicable>
5. <hint 5 based on domain and objective — include if applicable>

Recommendations:
You can use evolution algorithm to solve the problem. Keep a track of the solutions you have tried. And try to improve the solution. You can keep of promising solutions in a directory. And keep mutating, crossing, and evolving the solutions.
You can also keep a note in "NOTES.md" to record your thoughts and ideas, and improvements each iteration.
<If known best result is provided:>
Note that the best known result is <value>. Keep pushing to get a better result.
```

Show the generated INSTRUCTION.md to the researcher. Ask if they want any changes. Apply edits if requested, then finalize and move to Phase 3.

## Phase 3 — Generate SUBMISSION_FORMAT.md

First, ask the researcher about the submission structure through these questions (one at a time):

1. **Required keys and types** — Ask via direct conversation. For each key, collect: name, type (number, integer, string, array), and shape/constraints (e.g., array of length 26, 2D array of shape (N,2)).
2. **Value constraints** — Ask via direct conversation. Collect any bounds or restrictions on values (e.g., values in [0,1], positive numbers only).
3. **Invalid submission score** — Ask via AskUserQuestion with these options:
   - 0.0
   - inf
   - -inf

Then generate SUBMISSION_FORMAT.md using this template:

```markdown
## Submission Format

Write each candidate submission as a temporary JSON file under your approach directory. The file will be cleaned after grading; the full submitted payload is preserved in intermediate_results.jsonl.

### JSON Structure

```json
{
  "description": "standalone solution summary describing the exact candidate",
  "key1": <type and shape description>,
  "key2": <type and shape description>
}
```

### Field Descriptions

| Key | Type | Shape | Constraints | Description |
|-----|------|-------|-------------|-------------|
| description | string | scalar | Required, at least 80 characters and 12 words | Standalone summary of the exact candidate solution |
| key1 | <type> | <shape> | <range/rules> | <what it represents> |
| key2 | <type> | <shape> | <range/rules> | <what it represents> |

### Score Semantics

- Score represents: <what the score measures>
- Better direction: <higher/lower>
- Invalid submission score: <0.0 / inf / -inf>

JSON example:

<Minimal valid JSON example based on the key definitions>

Python snippet for writing a submission:

<Generate a Python snippet showing how to construct the JSON payload and write it to a file>

How to submit for grading:

python3 /workspace/eval/eureka_submit.py \
  --approach-dir approach_details/<approach_id> \
  --submission approach_details/<approach_id>/submissions/<name>.json

Workflow:

1. Run your solver and produce <describe expected outputs>
2. Write them to a temporary JSON file under approach_details/<approach_id>/submissions/
3. Run the submission command above
4. Read approach_details/<approach_id>/eval_feedback/latest_feedback.json
5. If the score improves, best_result.jsonl will be updated automatically. All submitted payloads are recorded in intermediate_results.jsonl under solution.

Notes:

- The hidden grader validates <what it validates> and computes the score as <scoring method>. <Higher/Lower> is better.
- Invalid submissions receive score: <invalid_score> and valid: false.
```

Show the generated SUBMISSION_FORMAT.md to the researcher. Ask if they want any changes. Apply edits if requested, then finalize and move to Phase 4.

## Phase 4 — Generate evaluate.py

This is the most critical file. First ask the researcher how they want to provide the evaluation logic via AskUserQuestion with these options:

- I'll describe the scoring formula and validation rules in natural language
- I'll provide pseudocode or mathematical formulas
- I have reference code I can share
- Please propose evaluation logic based on my problem description

Based on their choice, collect the necessary information through follow-up questions one at a time:

- If they chose natural language: ask them to describe the scoring formula, what makes a submission valid, and edge cases.
- If they chose pseudocode/formulas: ask them to provide the formulas for scoring and the validation conditions.
- If they chose reference code: ask them to share the code, then analyze it to extract scoring and validation logic.
- If they chose "propose": use the Phase 1 information to propose the evaluation logic, then ask them to confirm or correct.

After collecting the information, generate evaluate.py. The file structure should be adapted to the problem's needs, but must include these required interfaces:

### Required entrypoints (must be present in every evaluate.py)

1. **`grade_submission(submission_path: str, context: dict) -> dict`** — The entrypoint called by the grader server. Must:
   - Read and parse the submission JSON file
   - Validate constraints and compute the score
   - If invalid, return `{"score": <invalid_score>, "valid": False, "message": "<error>"}`
   - If valid, return `{"score": <score>, "valid": True, "message": "<summary>"}`

   The engine transports the returned dict via a private result file, so
   `grade_submission` is free to write progress lines, logging messages,
   tqdm bars, or any other diagnostics to stdout/stderr. The engine does
   not parse stdout for the primary result.

2. **`is_better(new_score: float, old_score: float) -> bool`** — Returns `True` if `new_score` is better than `old_score`. The comparison logic must match the optimization objective (e.g., `new_score > old_score` for maximization, `new_score < old_score` for minimization). If the direction is unclear from the objective, confirm with the researcher before implementing.

### Common patterns (adapt, don't copy verbatim)

Read an existing evaluate.py (e.g., `examples/circle_packing/hidden_eval_dir/evaluate.py`) to understand the patterns used in this project. Adapt its structure to the current problem — you may:

- Use the standalone eval harness (`run_eval`, `_run_with_timeout`, etc.) if the problem needs program execution with timeout. Not all problems need this — simpler problems may only need `grade_submission` and `is_better`.
- Use helper functions like `_validate()` and `_compute_score()` if it helps organize the code, but these are not required — the only hard requirement is that `grade_submission` and `is_better` work correctly.
- Include `main_from_program()` and `main_from_result()` for standalone testing, with an `if __name__ == "__main__"` argparse block (`--program_path`, `--results_dir`, `--from-result`).
- Adjust imports, validation logic, scoring formulas, and data transformations to match the problem.

The key principle: **the evaluate.py should be clean and appropriate for the specific problem, not a bloated copy-paste of boilerplate**. If the problem is simple, the evaluator should be simple too.

Show the generated evaluate.py to the researcher. Ask if they want any changes. Apply edits if requested, then finalize and move to Phase 5.

## Phase 5 — Generate run.sh (Launch Script)

Ask the researcher for runtime parameters for the launch script. Use AskUserQuestion with multiple choice where applicable. Provide defaults and let the researcher confirm or override.

1. **Model** — Default: `glm-5.1`. Ask via AskUserQuestion with common options or allow free-form.
2. **Number of approaches** — Default: 3. Ask via direct conversation.
3. **Propose time limit per session** — Default: "20 minutes". Ask via direct conversation.
4. **Implement time limit per session** — Default: "100 minutes". Ask via direct conversation.
5. **Max loops** — Default: 12. Ask via direct conversation.
6. **Docker** — Default: Yes. Ask via AskUserQuestion with options: Yes, No.
7. **Cost limit** — Default: 100. Ask via direct conversation.
8. **Adapter mode** — Default: "pty". Ask via AskUserQuestion with options: "pty", "stream".

Generate run.sh using this template:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Please run from the root directory of the project.
#
# Time budgets use strict either-or mode:
#   (a) shared budget:    --time-limit-per-session "N minutes"
#   (b) per-stage budget: --propose-time-limit-per-session ... \
#                         --implement-time-limit-per-session ...
# Pass --force-low-budget if you want to push any limit below the 5-minute floor.

cd "$(dirname "$0")/../.."

uv run python -m src \
    --model <model> \
    --problem examples/<problem_name>/INSTRUCTION.md \
    --hidden-eval-dir examples/<problem_name>/hidden_eval_dir \
    --submission-format examples/<problem_name>/SUBMISSION_FORMAT.md \
    <initial_code_flag> \
    --propose-time-limit-per-session "<propose_time>" \
    --implement-time-limit-per-session "<implement_time>" \
    --max-num-approaches <num_approaches> \
    <docker_flag> \
    --max-loops <max_loops> \
    --cost-limit <cost_limit> \
    --adapter-mode "<adapter_mode>"
```

Where `<docker_flag>` is `--docker` if Docker is Yes, otherwise the line is omitted entirely. `<initial_code_flag>` is `--initial-code examples/<problem_name>/initial.py` if the researcher provides initial code, otherwise the line is omitted entirely.

Show the generated run.sh to the researcher. Ask if they want any changes. Apply edits if requested, then finalize and move to Phase 6.

## Phase 6 — Write Files and Validate

### Step 1 — Create directory structure

```bash
mkdir -p examples/<problem_name>/hidden_eval_dir
```

### Step 2 — Write all four files

Use the Write tool to create:
- `examples/<problem_name>/INSTRUCTION.md`
- `examples/<problem_name>/SUBMISSION_FORMAT.md`
- `examples/<problem_name>/hidden_eval_dir/evaluate.py`
- `examples/<problem_name>/run.sh`

After writing run.sh, make it executable:

```bash
chmod +x examples/<problem_name>/run.sh
```

### Step 3 — Validate the evaluation pipeline

Run the following validation script, adapting the `test_submission` dict to match the submission format defined in Phase 3:

```bash
cd <project_root>
python3 -c "
import json, sys
sys.path.insert(0, 'examples/<problem_name>/hidden_eval_dir')
from evaluate import grade_submission, is_better

# Create a trivial test submission
test_submission = {<generate based on submission format with minimal/zero values>}
with open('/tmp/test_submission.json', 'w') as f:
    json.dump(test_submission, f)

# Test grade_submission
result = grade_submission('/tmp/test_submission.json', {})
assert 'score' in result, 'grade_submission must return score'
assert 'valid' in result, 'grade_submission must return valid'
print(f'grade_submission result: {result}')

# Test is_better
if result['valid']:
    print(f'is_better(3, 2) = {is_better(3, 2)}')
    print(f'is_better(2, 3) = {is_better(2, 3)}')
else:
    print('Note: trivial submission was invalid - may be expected')
    print('Please verify evaluate.py manually with a valid submission')
"
```

### Step 4 — Handle validation failures

If validation fails, report the error to the researcher and help fix evaluate.py. Common issues:
- Import errors: missing dependencies or incorrect module paths
- Missing keys: the trivial test submission does not match the expected format
- Validation always fails: the trivial values do not satisfy constraints (expected in some cases)
- is_better direction mismatch: the comparison logic does not match the optimization direction

Fix issues iteratively until `grade_submission` runs without errors and `is_better` matches the expected direction.

### Step 5 — Report completion

After successful validation, report completion to the researcher with:

- The directory path: `examples/<problem_name>/`
- List of generated files with their paths
- Validation results summary
- Next steps: the researcher can now run `bash examples/<problem_name>/run.sh` to start the optimization loop, or optionally add an `initial.py` file as starting code and update run.sh to include `--initial-code`.

## Quality Checklist

- [ ] Problem name is kebab-case and directory does not conflict (or conflict resolved with researcher)
- [ ] All Phase 1 information collected (name, domain, objective with direction, constraints, known best result, initial code availability)
- [ ] INSTRUCTION.md generated, shown to researcher, reviewed, and finalized
- [ ] SUBMISSION_FORMAT.md generated, shown to researcher, reviewed, and finalized
- [ ] evaluate.py generated with grade_submission(), is_better(), and all required functions; shown to researcher, reviewed, and finalized
- [ ] run.sh generated with correct paths and parameters; shown to researcher, reviewed, and finalized
- [ ] All files written to examples/<problem_name>/ (directory structure created)
- [ ] grade_submission() tested with trivial submission — returns dict with 'score' and 'valid' keys
- [ ] is_better() direction confirmed to match the optimization objective
- [ ] Completion summary provided to researcher with next steps

## Edge Cases

- **Complex evaluation logic**: If the researcher cannot easily describe the scoring formula, suggest breaking the problem into simpler sub-problems or providing reference implementations that can be adapted.
- **Missing dependencies**: If evaluate.py requires non-standard Python packages, note them in the completion summary but do NOT install them — that is `prepare-workspace`'s responsibility.
