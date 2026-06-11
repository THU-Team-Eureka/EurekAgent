#!/usr/bin/env python3
"""PreToolUse hook: block writes to score files and access to ~/.claude/."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROTECTED_HOME_CLAUDE_DIR = (Path.home() / ".claude").resolve()
PROTECTED_SETTINGS_FILE = PROTECTED_HOME_CLAUDE_DIR / "settings.json"
PROTECTED_RESULT_FILES = {"intermediate_results.jsonl", "best_result.jsonl"}
# .eureka_internal/ holds controller-owned coordination modules (gpu_helpers.py,
# pty_proxy.py). Agents must not be able to read its source (information
# leakage) or modify it (would let them bypass the GPU lock protocol).
PROTECTED_INTERNAL_DIR = ".eureka_internal"


def _implement_isolation_active() -> bool:
    return os.environ.get("EUREKA_STAGE") == "implement" and bool(
        os.environ.get("EUREKA_CURRENT_APPROACH_ID", "").strip()
    )


def _current_loop_index() -> str:
    return os.environ.get("EUREKA_CURRENT_LOOP_INDEX", "").strip()


def _emit_allow() -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }))
    sys.exit(0)


def _emit_deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def _resolve_tool_path(raw_path: str, *, cwd: str) -> Path | None:
    if not raw_path:
        return None
    expanded = Path(raw_path).expanduser()
    if not expanded.is_absolute():
        expanded = Path(cwd) / expanded
    try:
        return expanded.resolve(strict=False)
    except OSError:
        return expanded


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_protected_result_path(path: Path) -> bool:
    return path.name in PROTECTED_RESULT_FILES


def _is_inside_internal_dir(path: Path) -> bool:
    """True if *path* (or any ancestor) is named '.eureka_internal'."""
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path
    return PROTECTED_INTERNAL_DIR in resolved.parts


def _is_same_round_peer_path(path: Path) -> str | None:
    if not _implement_isolation_active():
        return None
    parts = path.resolve(strict=False).parts
    loop_index = _current_loop_index()
    current = os.environ.get("EUREKA_CURRENT_APPROACH_ID", "").strip()
    if not loop_index or "approach_details" not in parts:
        return None
    for idx, part in enumerate(parts):
        if part != "approach_details" or idx + 1 >= len(parts):
            continue
        aid = parts[idx + 1]
        if aid.startswith(f"round_{loop_index}_") and aid != current:
            return aid
    return None


def _is_same_round_manifest_path(path: Path) -> bool:
    if not _implement_isolation_active():
        return False
    loop_index = _current_loop_index()
    name = path.name
    blocked = {"current_round_approaches.jsonl", "current_round_approaches.json"}
    if loop_index:
        blocked.update({
            f"round_{loop_index}_approaches.jsonl",
            f"loop_{loop_index}_approaches.jsonl",
        })
    return "round_state" in path.resolve(strict=False).parts and name in blocked


def _deny_for_file_tool(tool_name: str, file_path: Path) -> str | None:
    if file_path.resolve(strict=False) == PROTECTED_SETTINGS_FILE:
        return (
            f"{tool_name} access to ~/.claude/settings.json is blocked. "
            "This file contains API keys and must not be exposed to agent sessions."
        )
    if _is_inside_internal_dir(file_path):
        if tool_name == "Read":
            return (
                f"{tool_name} access to .eureka_internal/ is blocked. "
                "Internal coordination modules are private. Use the documented "
                "gpu_helpers API: from gpu_helpers import get_gpu_info, gpu_session."
            )
        if tool_name in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
            return (
                f"{tool_name} access to .eureka_internal/ is blocked. "
                "These files are controller-owned and read-only."
            )
    peer_id = _is_same_round_peer_path(file_path)
    if peer_id:
        return (
            f"{tool_name} access to same-round peer approach {peer_id} is blocked. "
            "Use your assigned approach directory and prior-round results only."
        )
    if _is_same_round_manifest_path(file_path):
        return (
            f"{tool_name} access to same-round proposal manifests is blocked during implement. "
            "Use the initial hypothesis in your prompt and your own approach directory."
        )
    if tool_name in {"Write", "Edit", "MultiEdit", "NotebookEdit"} and _is_protected_result_path(file_path):
        return (
            f"{tool_name} access to {file_path.name} is blocked. "
            "These result files are controller-owned; submit candidates through "
            "`python3 /workspace/eval/eureka_submit.py` and let the grading service update them."
        )
    return None


def _contains_settings_json_reference(command: str) -> bool:
    home_dir = str(PROTECTED_HOME_CLAUDE_DIR)
    patterns = [
        "~/.claude/settings.json",
        "$HOME/.claude/settings.json",
        "${HOME}/.claude/settings.json",
        f"{home_dir}/settings.json",
    ]
    return any(pattern in command for pattern in patterns)


def _contains_protected_result_reference(command: str) -> bool:
    return any(filename in command for filename in PROTECTED_RESULT_FILES)


def _is_write_to_protected_result(command: str) -> bool:
    """Detect Bash WRITE operations targeting protected result files.

    Read commands (cat, head, tail, grep, wc, stat, etc.) are allowed so
    agents can review past scores.  Only write-redirect and write-utility
    patterns are blocked.
    """
    import re
    filenames = r"(?:best_result|intermediate_results)\.jsonl"
    # Shell redirect write: > or >> followed by path containing a protected filename
    if re.search(r">{1,2}\s*[^|;&\n]*" + filenames, command):
        return True
    # tee to protected file
    if re.search(r"\btee\b[^|;&\n]*" + filenames, command):
        return True
    # Write utilities: cp, mv, ln, install, dd, truncate, sed -i
    if re.search(
        r"(?:^|[\s|;&`(])(?:cp|mv|ln|install|dd|truncate)\b[^|;&\n]*" + filenames,
        command,
    ):
        return True
    if re.search(r"\bsed\b\s+-\w*i\w*\b[^|;&\n]*" + filenames, command):
        return True
    return False


def _contains_cuda_override(command: str) -> bool:
    """Detect direct CUDA_VISIBLE_DEVICES assignment in shell or code text."""
    import re
    patterns = [
        r'(?:^|[\s|;&`(])(?:export\s+)?CUDA_VISIBLE_DEVICES\s*=',
        r'\bos\.environ\s*\[\s*([\'"])CUDA_VISIBLE_DEVICES\1\s*\]\s*=',
        r'\bos\.putenv\s*\(\s*([\'"])CUDA_VISIBLE_DEVICES\1\s*,',
        r'[{,(]\s*([\'"])CUDA_VISIBLE_DEVICES\1\s*:',
    ]
    return any(re.search(pattern, command) for pattern in patterns)


def _written_text_for_file_tool(tool_name: str, tool_input: dict) -> str:
    """Return only text that the file tool is trying to write."""
    if tool_name == "Write":
        return str(tool_input.get("content", "") or "")
    if tool_name in {"Edit", "NotebookEdit"}:
        return str(
            tool_input.get("new_string")
            or tool_input.get("new_source")
            or ""
        )
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits") or []
        if not isinstance(edits, list):
            return ""
        return "\n".join(
            str(edit.get("new_string", "") or "")
            for edit in edits
            if isinstance(edit, dict)
        )
    return ""


# Shell utilities that read or modify files. We match these as whole words
# (preceded by start/space/pipe/semicolon) followed by an argument containing
# `.eureka_internal` to catch attempts to inspect or modify internal modules
# while still allowing benign uses like `PYTHONPATH=.eureka_internal`.
_INTERNAL_DIR_SHELL_PATTERN = (
    r'(?:^|[\s|;&`(])'
    r'(cat|head|tail|less|more|tac|nl|od|xxd|hexdump|strings|grep|egrep|fgrep|'
    r'rg|ack|awk|sed|cut|tr|sort|uniq|wc|diff|cmp|file|stat|find|ls|'
    r'vi|vim|nvim|nano|emacs|ed|'
    r'rm|cp|mv|ln|chmod|chown|chgrp|touch|truncate|install|dd|tar|zip|unzip)\b'
    r'[^|;&\n]*\.eureka_internal'
)


def _contains_internal_dir_shell_op(command: str) -> bool:
    """Detect shell-level read/write attempts against .eureka_internal/."""
    import re
    # Also catch redirection writes: `... > .eureka_internal/...`, `>> ...`, `tee ...`
    if re.search(r'(?:>{1,2}|\btee\b)\s*[^|;&\n]*\.eureka_internal', command):
        return True
    return bool(re.search(_INTERNAL_DIR_SHELL_PATTERN, command))


def _contains_same_round_peer_shell_op(command: str) -> str | None:
    if not _implement_isolation_active():
        return None
    import re
    loop_index = _current_loop_index()
    current = os.environ.get("EUREKA_CURRENT_APPROACH_ID", "").strip()
    if not loop_index:
        return None
    pattern = rf"(?:/workspace/)?approach_details/(round_{re.escape(loop_index)}_[A-Za-z0-9_-]+)"
    for match in re.finditer(pattern, command):
        aid = match.group(1)
        if aid != current:
            return aid
    return None


def _contains_same_round_manifest_shell_op(command: str) -> bool:
    if not _implement_isolation_active():
        return False
    import re
    loop_index = _current_loop_index()
    names = ["current_round_approaches\\.jsonl", "current_round_approaches\\.json"]
    if loop_index:
        names.extend([
            f"round_{re.escape(loop_index)}_approaches\\.jsonl",
            f"loop_{re.escape(loop_index)}_approaches\\.jsonl",
        ])
    return bool(re.search(r"round_state/[^|;&\n]*(?:" + "|".join(names) + r")", command))


def _contains_approach_details_wildcard(command: str) -> bool:
    if not _implement_isolation_active():
        return False
    import re
    current = os.environ.get("EUREKA_CURRENT_APPROACH_ID", "").strip()
    for match in re.finditer(r"(?:/workspace/)?approach_details/[^|;&\n\s]*(?:[*?\[])[^|;&\n\s]*", command):
        if current and current in match.group(0):
            continue
        return True
    return False


def _lists_all_approach_details(command: str) -> bool:
    if not _implement_isolation_active():
        return False
    import re
    current = os.environ.get("EUREKA_CURRENT_APPROACH_ID", "").strip()
    if current and current in command:
        return False
    return bool(re.search(
        r"(?:^|[\s|;&`(])(?:ls|find|rg|grep)\b[^|;&\n]*(?:/workspace/)?approach_details/?(?:\s|$)",
        command,
    ))


def _deny_for_bash(command: str) -> str | None:
    if _contains_settings_json_reference(command):
        return (
            "Bash access to ~/.claude/settings.json is blocked for agent sessions. "
            "This file contains API keys."
        )
    if _is_write_to_protected_result(command):
        return (
            "Writing to intermediate_results.jsonl and best_result.jsonl is blocked. "
            "These files are maintained by the grading service — submit candidates "
            "through `python3 /workspace/eval/eureka_submit.py`."
        )
    if _contains_internal_dir_shell_op(command):
        return (
            ".eureka_internal/ is private and read-only. "
            "Use the gpu_helpers Python API instead: "
            "from gpu_helpers import get_gpu_info, gpu_session."
        )
    peer_id = _contains_same_round_peer_shell_op(command)
    if peer_id:
        return (
            f"Bash access to same-round peer approach {peer_id} is blocked. "
            "Use your assigned approach directory and prior-round results only."
        )
    if _contains_same_round_manifest_shell_op(command):
        return (
            "Bash access to same-round proposal manifests is blocked during implement. "
            "Use the initial hypothesis in your prompt and your own approach directory."
        )
    if _contains_approach_details_wildcard(command):
        return (
            "Wildcard access under approach_details is blocked during implement because it can "
            "expose same-round peer work. Use your assigned directory or prior-round results."
        )
    if _lists_all_approach_details(command):
        return (
            "Listing all approach_details is blocked during implement because it can expose "
            "same-round peer work. Use your assigned directory or ranked prior-round results."
        )
    if _contains_cuda_override(command):
        return (
            "Direct CUDA_VISIBLE_DEVICES setting is blocked. "
            "Use the gpu_helpers Python API in your code: "
            "from gpu_helpers import gpu_session"
        )
    return None


def main() -> None:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except Exception:
        _emit_allow()

    tool_name = str(event.get("tool_name", "") or "")
    tool_input = event.get("tool_input") or {}
    cwd = str(event.get("cwd") or os.getcwd())

    if tool_name in {"Read", "Write", "Edit", "MultiEdit", "NotebookEdit"}:
        raw_path = str(
            tool_input.get("file_path")
            or tool_input.get("notebook_path")
            or ""
        )
        file_path = _resolve_tool_path(raw_path, cwd=cwd)
        if file_path is not None:
            reason = _deny_for_file_tool(tool_name, file_path)
            if reason:
                _emit_deny(reason)
        written_text = _written_text_for_file_tool(tool_name, tool_input)
        if written_text and _contains_cuda_override(written_text):
            _emit_deny(
                "Direct CUDA_VISIBLE_DEVICES setting is blocked. "
                "Use the gpu_helpers Python API instead: "
                "from gpu_helpers import get_gpu_info, gpu_session."
            )
        _emit_allow()

    if tool_name == "Bash":
        command = str(tool_input.get("command", "") or "")
        reason = _deny_for_bash(command)
        if reason:
            _emit_deny(reason)
        _emit_allow()

    _emit_allow()


if __name__ == "__main__":
    main()
