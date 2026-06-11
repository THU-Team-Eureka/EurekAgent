"""Detect context window exhaustion from Claude Code session transcripts."""

import re
import uuid
from pathlib import Path
from typing import Optional, Union

from src.acp.protocol import SessionRequest

# Tail-only scanning keeps large transcripts cheap.
_TAIL_BYTES = 64 * 1024

_PREPARE_PREAMBLE = """\
This is a continuation of a previously interrupted session. The previous session
ran out of context window space and has been resumed with a fresh context.

You are in the **prepare** stage. The workspace is at `{workspace}`.
This is continuation #{continuation_count}.

**Resume instructions:**
1. Read `{workspace}/prepare/progress.json` FIRST to understand what has already been done.
2. Read `{workspace}/prepare/summary.md` for any intermediate summaries written so far.
3. Read `{workspace}/prepare/question.json` for the problem definition.
4. Continue the prepare stage from where the previous session left off.
5. Do NOT re-do work that is already completed — pick up from the last checkpoint.
"""

_PROPOSE_PREAMBLE_TEMPLATE = """\
This is a continuation of a previously interrupted session. The previous session
ran out of context window space and has been resumed with a fresh context.

You are in the **propose** stage, round {loop_index}. The workspace is at `{workspace}`.
This is continuation #{continuation_count}.

**Resume instructions:**
1. Read `{workspace}/round_state/current_round_approaches.jsonl` to see initial hypotheses already proposed this round.
2. Read `{workspace}/round_state/ranked_past_best_solutions.jsonl` for prior best scored solutions if it exists.
3. Read `{workspace}/round_state/web_search_history.jsonl` for prior search history.
4. Continue proposing approaches for round {loop_index} from where the previous session left off.
5. Do NOT re-propose approaches that are already in the manifest.
"""

_IMPLEMENT_PREAMBLE_TEMPLATE = """\
This is a continuation of a previously interrupted session. The previous session
ran out of context window space and has been resumed with a fresh context.

You are in the **implement** stage for approach `{approach_id}`. The workspace is at `{workspace}`.
This is continuation #{continuation_count}.

**Resume instructions:**
1. Check recent git history: `git -C {workspace}/approach_details/{approach_id}/code log --oneline`
2. Read any eval_feedback files in `{workspace}/approach_details/{approach_id}/` for prior evaluation results.
3. Read intermediate_results and best_result files to understand current progress.
4. Continue the hypothesis-driven experiment for `{approach_id}` from where the previous session left off.
5. Do NOT read same-round peer approach directories; prior rounds are okay.
6. Do NOT discard or overwrite work that has already been done.
"""


def detect_context_exhaustion(transcript_path: Path) -> bool:
    """Return True if the transcript signals context window exhaustion.

    Scans the tail of a Claude Code JSONL transcript for two signals:
      1. Primary: isApiErrorMessage=True AND apiError="max_output_tokens"
      2. Secondary: assistant message with stop_reason="stop_sequence"
         and text content containing "context window limit"

    Args:
        transcript_path: Path to the .jsonl transcript file.

    Returns:
        True if an exhaustion signal is found, False otherwise
        (including when the file does not exist).
    """
    path = Path(transcript_path)
    if not path.is_file():
        return False

    try:
        file_size = path.stat().st_size
        read_start = max(0, file_size - _TAIL_BYTES)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(read_start)
            raw = f.read()
    except OSError:
        return False

    # Drop a partial first line when the tail starts mid-record.
    if read_start > 0:
        newline_pos = raw.find("\n")
        if newline_pos != -1:
            raw = raw[newline_pos + 1 :]

    # Newer transcript lines are more likely to contain the exhaustion signal.
    import json

    lines = raw.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Primary signal emitted by Claude Code for max-output/context exhaustion.
        if obj.get("isApiErrorMessage") is True and obj.get("apiError") == "max_output_tokens":
            return True

        # Secondary signal preserved in assistant message content.
        if obj.get("type") == "assistant":
            msg = obj.get("message")
            if isinstance(msg, dict) and msg.get("stop_reason") == "stop_sequence":
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and "context window limit" in block.get("text", "")
                        ):
                            return True

    return False


def build_resume_preamble(
    stage: str,
    workspace: str,
    continuation_count: int,
    approach_id: Optional[str] = None,
    loop_index: Optional[int] = None,
) -> str:
    """Return a stage-specific resume preamble string.

    Args:
        stage: One of "prepare", "propose", "implement".
        workspace: Absolute path to the run workspace directory.
        continuation_count: How many continuations have occurred (1-based).
        approach_id: Required when stage="implement".
        loop_index: Required when stage="propose".

    Returns:
        Formatted preamble string for injection into a continuation prompt.
    """
    if stage == "prepare":
        return _PREPARE_PREAMBLE.format(
            workspace=workspace,
            continuation_count=continuation_count,
        )
    elif stage == "propose":
        if loop_index is None:
            raise ValueError("loop_index is required for propose stage")
        return _PROPOSE_PREAMBLE_TEMPLATE.format(
            workspace=workspace,
            continuation_count=continuation_count,
            loop_index=loop_index,
        )
    elif stage == "implement":
        if approach_id is None:
            raise ValueError("approach_id is required for implement stage")
        return _IMPLEMENT_PREAMBLE_TEMPLATE.format(
            workspace=workspace,
            continuation_count=continuation_count,
            approach_id=approach_id,
        )
    else:
        raise ValueError(f"Unknown stage: {stage!r}")


def build_continuation_request(
    original_request: SessionRequest,
    resume_preamble: str,
    remaining_budget_seconds: Union[int, float],
    workspace: str,
) -> SessionRequest:
    """Build a new SessionRequest for a context-continuation session.

    Extracts the ``## Run Context`` and ``## Instruction`` sections from the
    original prompt, then assembles a new prompt consisting of:
      1. The resume preamble
      2. The run context section (preserved verbatim)
      3. A ``## Time Remaining`` section
      4. The instruction section (preserved verbatim)

    The returned SessionRequest has a fresh session_id (uuid4), resume=False,
    and preserves model, cwd, permissions, log_path, completion_check, and
    warning_prompt from the original request.

    Args:
        original_request: The previous (exhausted) session's request.
        resume_preamble: Stage-specific preamble produced by build_resume_preamble.
        remaining_budget_seconds: Approximate seconds left in the budget.
        workspace: Absolute path to the run workspace directory.

    Returns:
        A new SessionRequest ready to launch a continuation session.
    """
    # Preserve the original prompt contract while inserting resume context.
    original_prompt = original_request.prompt

    # Keep Run Context separate from the following top-level section.
    run_context = ""
    run_ctx_match = re.search(
        r"(## Run Context\n.*)", original_prompt, re.DOTALL
    )
    if run_ctx_match:
        run_context = run_ctx_match.group(1)
        next_section = re.search(r"\n## (?!Run Context)", run_context)
        if next_section:
            run_context = run_context[: next_section.start()]

    # Keep Instruction separate from later top-level sections.
    instruction = ""
    instr_match = re.search(
        r"(## Instruction\n.*)", original_prompt, re.DOTALL
    )
    if instr_match:
        instruction = instr_match.group(1)
        next_section = re.search(r"\n## (?!Instruction)", instruction)
        if next_section:
            instruction = instruction[: next_section.start()]

    # Order matters: resume guidance, preserved context, budget, instruction.
    parts = [resume_preamble.rstrip()]
    if run_context.strip():
        parts.append(run_context.rstrip())
    parts.append(f"## Time Remaining\n\nApproximately {int(remaining_budget_seconds)} seconds remain in the budget for this run. Work efficiently.")
    if instruction.strip():
        parts.append(instruction.rstrip())

    new_prompt = "\n\n".join(parts) + "\n"

    return SessionRequest(
        prompt=new_prompt,
        model=original_request.model,
        cwd=original_request.cwd,
        session_id=str(uuid.uuid4()),
        permissions=original_request.permissions,
        log_path=original_request.log_path,
        resume=False,
        completion_check=original_request.completion_check,
        warning_prompt=original_request.warning_prompt,
        env=original_request.env,
    )
