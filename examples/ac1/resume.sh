# Please run from the root directory of the project.
#
# Time budgets are required per stage:
#   --propose-time-limit-per-session ... \
#   --implement-time-limit-per-session ...
# Pass --force-low-budget if you want to push any limit below the stage floor.

cd "$(dirname "$0")/../.."

uv run python -m src \
    --model glm-5.1 \
    --problem examples/ac1/INSTRUCTION.md \
    --hidden-eval-dir examples/ac1/hidden_eval_dir \
    --submission-format examples/ac1/SUBMISSION_FORMAT.md \
    --initial-code examples/ac1/initial.py \
    --propose-time-limit-per-session "30 minutes" \
    --implement-time-limit-per-session "120 minutes" \
    --max-num-approaches 3 \
    --max-loops 8 \
    --cost-limit 30 \
    --adapter-mode "pty" \
    --gpus 0,1,2,3,4 \
    --resume "20260526_142753"
