# Please run from the root directory of the project.
#
# Time budgets are required per stage:
#   --propose-time-limit-per-session ... \
#   --implement-time-limit-per-session ...
# Pass --force-low-budget if you want to push any limit below the stage floor.

cd "$(dirname "$0")/../.."

uv run python -m src \
    --model glm-5.1 \
    --problem examples/circle_packing/INSTRUCTION.md \
    --hidden-eval-dir examples/circle_packing/hidden_eval_dir \
    --submission-format examples/circle_packing/SUBMISSION_FORMAT.md \
    --initial-code examples/circle_packing/initial.py \
    --propose-time-limit-per-session "20 minutes" \
    --implement-time-limit-per-session "120 minutes" \
    --max-num-approaches 3 \
    --max-loops 5 \
    --cost-limit 30 \
    --adapter-mode "pty" \
    --gpus 0,1,2,3,4 \
    --resume "20260525_225346"
