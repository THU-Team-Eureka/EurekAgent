## Submission Format

Write each candidate submission as a temporary JSON file under your approach directory. The file will be cleaned after grading; the full submitted payload is preserved in `intermediate_results.jsonl`.

### JSON Structure

```json
{
  "kernel_code": "string — full Python source of your custom_kernel implementation",
  "description": "string — standalone solution summary for this candidate"
}
```

### Field Descriptions

| Key | Type | Constraints | Description |
|-----|------|-------------|-------------|
| `kernel_code` | string | Must define `custom_kernel(data)`; must contain at least one `@triton.jit` function; UTF-8 source code. | Complete Python source for the Triton kernel implementation. |
| `description` | string | Required; at least 80 characters and 12 words. | Standalone summary of the exact candidate being submitted. |

### Score Semantics

- **Score**: geometric mean of the 7 benchmark runtimes, in microseconds.
- **Better direction**: lower is better.
- **Invalid submission score**: `inf` (`valid: false`).

Evaluation runs through the **unmodified TTT-Discover pipeline** bundled under `hidden_eval_dir/_ttt_lib/`:
1. 18 correctness test cases (rtol=atol=2e-2 against PyTorch reference; each case in an isolated subprocess via `multiprocessing.spawn`).
2. If all correctness tests pass, 7 benchmark cases are timed using CUDA events (3-100 iterations each; stop when relative standard error of the mean < 0.1% or after 30 s per case; benchmarks use `recheck=True` with per-iteration re-seeding and re-validation).
3. Final score = geometric mean (ns), converted to microseconds.

JSON example:

```json
{
  "kernel_code": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef my_kernel(...):\n    ...\n\ndef custom_kernel(data):\n    input_tensor, mask, weights, config = data\n    # ... your optimized implementation ...\n    return out\n",
  "description": "Fused TriMul with Triton matmul + LayerNorm fusion"
}
```

Python snippet for writing a submission:

```python
import json

# After writing your kernel to my_kernel.py:
with open("my_kernel.py") as f:
    code = f.read()

payload = {
    "kernel_code": code,
    "description": "Fused TriMul with Triton kernels",
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

1. Write your Triton kernel code and test it locally.
2. Wrap the code in a JSON file with the `kernel_code` and `description` keys.
3. Submit via `python3 /workspace/eval/eureka_submit.py`.
4. Read `approach_details/<approach_id>/eval_feedback/latest_feedback.json` for the score.
5. If the score improves, `best_result.jsonl` is updated automatically. All submitted payloads are recorded in `intermediate_results.jsonl` under `solution`.

Notes:

- The hidden grader validates that `kernel_code` is a non-empty string containing `@triton.jit`, then hands the code to the unmodified TTT-Discover evaluator. Lower score (us) is better.
- Invalid submissions (parsing errors, correctness failures, timeouts) receive `score: inf` and `valid: false`.
- Best known results (A100-SXM4-80GB): TTT-Discover = **2198 us**, best human = 4531 us, PyTorch starter (`initial.py`) = ~20700 us.
