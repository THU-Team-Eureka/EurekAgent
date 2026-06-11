## Submission Format

Write each candidate submission as a temporary JSON file under your approach directory. The file will be cleaned after grading; the full submitted payload is preserved in `intermediate_results.jsonl`.

Required top-level keys:

- `description`: standalone solution summary accurately describing this candidate; at least 80 characters and 12 words
- `h_values`: array of floats in [0, 1]
- `c5_bound`: float (the claimed C5 overlap value)
- `n_points`: int (number of discretization points)

Shape contract:

- `h_values` must have shape `(n_points,)` — a flat 1D array
- `c5_bound` must be a single float
- `n_points` must be a positive integer

JSON example:

```json
{
  "description": "Discretized balanced h function from Fourier-smoothed descent",
  "h_values": [
    0.50, 0.55, 0.48, 0.52, 0.49,
    0.51, 0.47, 0.53, 0.50, 0.50
  ],
  "c5_bound": 0.423,
  "n_points": 10
}
```

Field notes:

- `h_values` represents the discretized function h on [0, 2) with `n_points` samples
- All values must satisfy 0 <= h[i] <= 1
- `sum(h_values)` must equal `n_points / 2` (i.e., the integral of h equals 1); the evaluator will normalize if the sum is close but not exact
- `c5_bound` must match the computed C5 value within tolerance (1e-4): C5 = max_k sum_i h[i] * (1 - h[(i+k) mod n]) * (2/n)
- Lower C5 is better
- Use plain JSON numbers, not strings
- Do not include comments, trailing commas, or NumPy-specific syntax

Python snippet for writing a submission:

```python
import json
import numpy as np

# Compute C5 from h_values
n_points = len(h_values)
dx = 2.0 / n_points
overlap = np.correlate(h_values, 1.0 - h_values, mode="full") * dx
c5_bound = float(np.max(overlap))

payload = {
    "description": "Discretized balanced h function from Fourier-smoothed descent",
    "h_values": h_values.tolist(),
    "c5_bound": c5_bound,
    "n_points": int(n_points),
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

1. Run your solver and produce `h_values`, compute `c5_bound`, and note `n_points`
2. Write them to a temporary JSON file under `approach_details/<approach_id>/submissions/`
3. Run `python3 /workspace/eval/eureka_submit.py --approach-dir approach_details/<approach_id> --submission <path>`
4. Read `approach_details/<approach_id>/eval_feedback/latest_feedback.json`
5. If the score improves, `best_result.jsonl` will be updated automatically. All submitted payloads are recorded in `intermediate_results.jsonl` under `solution`.

Notes:

- The hidden grader validates feasibility (h in [0,1], sum constraint), verifies that `c5_bound` matches the recomputed C5 within tolerance, and computes the score.
- Score is the raw C5 value. Lower is better.
- Invalid submissions receive `score: inf` and `valid: false`.
- Prefer smaller `n_points` (<1000) for speed; 400 is a common starting point.
