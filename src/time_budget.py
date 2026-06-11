"""Write time budget metadata and helper script into the workspace."""

from __future__ import annotations

import json
import time
from pathlib import Path

CHECK_TIME_SCRIPT = r'''"""Check time remaining in the current session budget."""
import json
import time
import sys
from pathlib import Path

def _fmt(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"

def main():
  clock_path = Path(__file__).parent / "stage_clock.json"
  budget_path = Path(__file__).parent / "time_budget.json"

  if clock_path.exists():
      try:
          data = json.loads(clock_path.read_text())
      except (json.JSONDecodeError, OSError) as e:
          print(f"ERROR: Failed to read stage_clock.json: {e}", file=sys.stderr)
          sys.exit(1)
      started_at = data.get("started_at", 0)
      now = time.time()
      elapsed = max(0.0, now - started_at)
      if data.get("has_deadline"):
          total = float(data.get("total_seconds") or 0)
          deadline_at = data.get("deadline_at", started_at + total)
          remaining = max(0.0, float(deadline_at) - now)
          pct = (elapsed / total * 100) if total > 0 else 0
          print(
              f"Time Budget: {_fmt(total)} total | "
              f"Elapsed: {_fmt(elapsed)} | "
              f"Remaining: {_fmt(remaining)} | "
              f"{pct:.0f}% used"
          )
      else:
          print(f"Stage: {data.get('stage', '?')} | Elapsed: {_fmt(elapsed)} (no time limit)")
      return

  if not budget_path.exists():
      print("ERROR: No .time/stage_clock.json or time_budget.json found.", file=sys.stderr)
      sys.exit(1)
  try:
      data = json.loads(budget_path.read_text())
  except (json.JSONDecodeError, OSError) as e:
      print(f"ERROR: Failed to read time_budget.json: {e}", file=sys.stderr)
      sys.exit(1)

  started_at = data.get("started_at", 0)
  total_seconds = data.get("total_seconds", 0)
  deadline_at = data.get("deadline_at", started_at + total_seconds)
  now = time.time()
  elapsed = now - started_at
  remaining = max(0.0, deadline_at - now)
  pct_used = (elapsed / total_seconds * 100) if total_seconds > 0 else 0

  print(
      f"Time Budget: {_fmt(total_seconds)} total | "
      f"Elapsed: {_fmt(elapsed)} | "
      f"Remaining: {_fmt(remaining)} | "
      f"{pct_used:.0f}% used"
  )

if __name__ == "__main__":
    main()
'''


def write_stage_clock(
    workspace: Path,
    *,
    stage: str,
    started_at: float,
    elapsed_seconds: float,
    total_seconds: float | None = None,
    deadline_at: float | None = None,
) -> None:
    """Write .time/stage_clock.json for agent-visible elapsed (with or without limit)."""
    time_dir = workspace / ".time"
    time_dir.mkdir(parents=True, exist_ok=True)
    has_deadline = total_seconds is not None and total_seconds > 0
    payload: dict = {
        "stage": stage,
        "started_at": started_at,
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "has_deadline": has_deadline,
    }
    if has_deadline:
        payload["total_seconds"] = float(total_seconds)
        payload["deadline_at"] = (
            deadline_at if deadline_at is not None
            else started_at + float(total_seconds)
        )
    (time_dir / "stage_clock.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    (time_dir / "check_time.py").write_text(CHECK_TIME_SCRIPT, encoding="utf-8")


def write_time_budget(
    workspace: Path,
    *,
    started_at: float,
    total_seconds: float,
    stage: str,
    deadline_at: float | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Write timed stage budget files (.time/time_budget.json + stage_clock.json)."""
    if deadline_at is None:
        deadline_at = started_at + total_seconds

    elapsed = (
        max(0.0, float(elapsed_seconds))
        if elapsed_seconds is not None
        else max(0.0, time.time() - started_at)
    )

    time_dir = workspace / ".time"
    time_dir.mkdir(parents=True, exist_ok=True)

    budget_data = {
        "started_at": started_at,
        "total_seconds": total_seconds,
        "stage": stage,
        "deadline_at": deadline_at,
    }
    (time_dir / "time_budget.json").write_text(
        json.dumps(budget_data, indent=2) + "\n", encoding="utf-8",
    )
    write_stage_clock(
        workspace,
        stage=stage,
        started_at=started_at,
        elapsed_seconds=elapsed,
        total_seconds=total_seconds,
        deadline_at=deadline_at,
    )
