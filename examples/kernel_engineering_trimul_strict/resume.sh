#!/usr/bin/env bash
set -euo pipefail

# Please run from the root directory of the project.

cd "$(dirname "$0")/../.."

# Python interpreter with CUDA-enabled torch + triton (3.3.1) for kernel evaluation.
# This is forwarded to TTT-Discover's run_program() via TRIMUL_EVAL_PYTHON; the
# evaluator drops a `python3` shim symlink to this interpreter into PATH so the
# subprocess hard-coded `python3` resolves correctly. Defaults to the current
# `python3` on PATH.
export TRIMUL_EVAL_PYTHON="${TRIMUL_EVAL_PYTHON:-$(uv run python -c 'import sys; print(sys.executable)')}"

uv run python -m src \
    --model glm-5.1 \
    --problem examples/kernel_engineering_trimul_strict/INSTRUCTION.md \
    --hidden-eval-dir examples/kernel_engineering_trimul_strict/hidden_eval_dir \
    --submission-format examples/kernel_engineering_trimul_strict/SUBMISSION_FORMAT.md \
    --initial-code examples/kernel_engineering_trimul_strict/initial.py \
    --propose-time-limit-per-session "20 minutes" \
    --implement-time-limit-per-session "160 minutes" \
    --max-num-approaches 3 \
    --max-loops 10 \
    --gpus 5,6,7 \
    --adapter-mode "pty" \
    --resume "20260526_001013"
