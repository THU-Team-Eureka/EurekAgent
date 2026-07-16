# EurekAgent Results

Each result is a one-line JSONL file. The `solution` field is the evaluator payload; metadata records the source run and reported score.

## Recompute Math Scores

Run these snippets from the repository root. Use `uv run python` or another Python 3.10+ environment with the repository dependencies available.

### Circle packing

```bash
uv run python - <<'PY'
import json, numpy as np
from examples.circle_packing.hidden_eval_dir.evaluate import adapted_validate_packing
r = json.loads(open('results/circle_packing/result.jsonl').readline())
s = r['solution']
centers = np.array(s['centers'])
radii = np.array(s['radii'])
valid, msg = adapted_validate_packing((centers, radii, float(np.sum(radii))), atol=1e-6)
print({'valid': valid, 'score': float(np.sum(radii)), 'message': msg})
PY
```

### AC1

```bash
uv run python - <<'PY'
import json
from examples.ac1.hidden_eval_dir.evaluate import evaluate_sequence
r = json.loads(open('results/ac1/result.jsonl').readline())
print({'score': evaluate_sequence(r['solution']['sequence'])})
PY
```

### Erdos minimum overlap

```bash
uv run python - <<'PY'
import json
from examples.erdos_min_overlap.hidden_eval_dir.evaluate import verify_c5_solution
r = json.loads(open('results/erdos_min_overlap/result.jsonl').readline())
s = r['solution']
print({'score': verify_c5_solution(s['h_values'], s['c5_bound'], s['n_points'])})
PY
```

## Recompute TriMul Kernel Scores

TriMul results are under `results/kernel_engineering_trimul/<approach>/result.jsonl`. Each `solution.kernel_code` is the exact submitted kernel source for the strict evaluator format.

The reported table uses the strict TTT-Discover TriMul evaluator in `examples/kernel_engineering_trimul_strict/hidden_eval_dir/` on an NVIDIA A100-SXM4-80GB. We used 3 warmup rounds (discarded), then 10 measured rounds with deterministic shuffle seed `20260527`. Each measured score is the evaluator geometric mean over 7 benchmark cases; the table reports median and mean over the 10 measured scores.

To rerun one kernel, extract `solution` to a submission JSON and submit it through the strict evaluator setup used by `examples/kernel_engineering_trimul_strict`. The JSONL metadata records all measured scores, median, mean, per-round correctness counts, and the source ranking file.
