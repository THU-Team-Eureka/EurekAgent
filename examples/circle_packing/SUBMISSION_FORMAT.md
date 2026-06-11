## Submission Format

Write each candidate submission as a temporary JSON file under your approach directory. The file will be cleaned after grading; the full submitted payload is preserved in `intermediate_results.jsonl`.

Required top-level keys:

- `description`: standalone solution summary accurately describing this candidate; at least 80 characters and 12 words
- `centers`: array of 26 `[x, y]` pairs
- `radii`: array of 26 radii

Shape contract:

- `centers` must have shape `(26, 2)`
- `radii` must have shape `(26,)`

JSON example:

```json
{
  "description": "Symmetric 26-circle layout refined by local nonlinear optimization",
  "centers": [
    [0.11, 0.11],
    [0.33, 0.11]
  ],
  "radii": [
    0.11,
    0.11
  ]
}
```

Field notes:

- `centers[i]` is the `(x, y)` center of circle `i`
- `radii[i]` is the radius of circle `i`
- Use plain JSON numbers, not strings
- Do not include comments, trailing commas, or NumPy-specific syntax

Python snippet for writing a submission:

```python
import json

payload = {
    "description": "Symmetric 26-circle layout refined by local nonlinear optimization",
    "centers": centers.tolist(),
    "radii": radii.tolist(),
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

1. Run your solver and produce `centers` and `radii`
2. Write them to a temporary JSON file under `approach_details/<approach_id>/submissions/`
3. Run `python3 /workspace/eval/eureka_submit.py --approach-dir approach_details/<approach_id> --submission <path>`
4. Read `approach_details/<approach_id>/eval_feedback/latest_feedback.json`
5. If the score improves, `best_result.jsonl` will be updated automatically. All submitted payloads are recorded in `intermediate_results.jsonl` under `solution`.

Notes:

- The hidden grader validates feasibility inside the unit square and computes the score as `sum(radii)`. Higher is better.
- Invalid submissions receive `score: 0.0` and `valid: false`.
