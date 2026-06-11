## Submission Format

Write each candidate submission as a temporary JSON file under your approach directory. The file will be cleaned after grading; the full submitted payload is preserved in `intermediate_results.jsonl`.

Required top-level keys:

- `sequence`: array of nonnegative floats
- `description`: standalone solution summary accurately describing this candidate; at least 80 characters and 12 words

Shape contract:

- `sequence` must be a flat list of floats (1D array)

JSON example:

```json
{
  "description": "Normalized coefficient sequence from projected local search",
  "sequence": [
    0.95, 1.02, 0.88, 1.10, 0.97,
    0.93, 1.05, 0.91, 0.99, 1.01
  ]
}
```

Field notes:

- `sequence` is the nonnegative coefficient array `a`
- All values must be finite, nonnegative floats; booleans are rejected
- Values are clipped to [0, 1000] by the evaluator; sum must be >= 0.01
- Lower objective value is better: `2*n*max(convolve(a,a)) / (sum(a)^2)`
- Use plain JSON numbers, not strings
- Do not include comments, trailing commas, or NumPy-specific syntax

Python snippet for writing a submission:

```python
import json

payload = {
    "description": "Normalized coefficient sequence from projected local search",
    "sequence": [float(x) for x in sequence],
}

with open("approach_details/<approach_id>/submissions/candidate_001.json", "w", encoding="utf-8") as f:
    json.dump(payload, f)
```

How to submit for grading:

```bash
python3 /workspace/eval/eureka_submit.py \
  --approach-dir approach_details/<approach_id> \
  --submission approach_details/<approach_id>/submissions/<name>.json
```

Workflow:

1. Run your solver and produce a `sequence` (list of nonnegative floats)
2. Write it to a temporary JSON file under `approach_details/<approach_id>/submissions/`
3. Run `python3 /workspace/eval/eureka_submit.py --approach-dir approach_details/<approach_id> --submission <path>`
4. Read `approach_details/<approach_id>/eval_feedback/latest_feedback.json`
5. If the score improves, `best_result.jsonl` will be updated automatically. All submitted payloads are recorded in `intermediate_results.jsonl` under `solution`.

Notes:

- The hidden grader validates that `sequence` is a list of finite nonnegative floats, clips values to [0, 1000], checks sum >= 0.01, and computes the objective `2*n*max(convolve(a,a)) / (sum(a)^2)`.
- Score is the raw objective value. Lower is better.
- Invalid submissions receive `score: inf` and `valid: false`.
