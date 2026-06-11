"""Shared workspace setup: permissions, hooks, and agent configuration."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# Default deny rules to prevent agents from reading credentials
_CREDENTIAL_DENY_RULES = [
    "Bash(echo $ANTHROPIC*)",
    "Bash(echo $anthropic*)",
    "Bash(printenv ANTHROPIC*)",
    "Bash(printenv anthropic*)",
    "Bash(env | *ANTHROPIC*)",
    "Bash(env | *anthropic*)",
    "Bash(cat /proc/*/environ*)",
    "Bash(curl *$ANTHROPIC*)",
    "Bash(wget *$ANTHROPIC*)",
]

# Default allow rules for EurekAgent sessions
_DEFAULT_ALLOW_RULES = [
    "Write(./**)", "Edit(./**)", "Read(./**)",
    "Bash(python3 *)", "Bash(python *)", "Bash(ls *)",
    "Bash(cat *)", "Bash(head *)", "Bash(tail *)",
    "Bash(grep *)", "Bash(find *)", "Bash(cp *)", "Bash(mkdir *)",
    "Bash(pip3 *)", "Bash(pip *)",
    "Bash(python3 /workspace/eval/eureka_submit.py *)",
    "Bash(python /workspace/eval/eureka_submit.py *)",
    "Skill(propose-approaches)", "Skill(implement-approach)",
    "WebSearch",
]


def write_workspace_permissions(
    workspace_dir: Path,
    *,
    hook_prefix: str | None = None,
) -> None:
    """Write merged permissions to workspace/.claude/settings.local.json.

    This keeps the agent session in bypassPermissions mode with appropriate
    allow/deny rules. Without it, the interactive claude process sits in
    "default" permission mode and every tool call blocks on an approval dialog
    that nobody can dismiss.

    Args:
        workspace_dir: The workspace root (run_dir/workspace).
        hook_prefix: Absolute path used in hook command strings. Docker mode
            passes "/workspace". Defaults to str(workspace_dir.resolve()).
    """
    if hook_prefix is None:
        hook_prefix = str(workspace_dir.resolve())

    user_permissions = _read_user_permissions()

    merged_allow = list(_DEFAULT_ALLOW_RULES)
    submit_via_real = f"Bash(python3 {hook_prefix}/eval/eureka_submit.py *)"
    if submit_via_real not in merged_allow:
        merged_allow.append(submit_via_real)
    for rule in user_permissions.get("allow", []):
        if rule not in merged_allow:
            merged_allow.append(rule)

    merged_deny = list(_CREDENTIAL_DENY_RULES)
    for rule in user_permissions.get("deny", []):
        if rule not in merged_deny:
            merged_deny.append(rule)

    settings: dict = {
        "skipDangerousModePermissionPrompt": True,
        "permissions": {
            "allow": merged_allow,
            "deny": merged_deny,
            "defaultMode": "bypassPermissions",
        },
    }

    claude_dir = workspace_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.local.json"
    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    log.info("Wrote workspace permissions to %s", settings_path)


def install_workspace_hooks(
    workspace_dir: Path,
    *,
    hook_prefix: str | None = None,
) -> None:
    """Copy hook scripts into the workspace and register them in settings.

    Args:
        workspace_dir: The workspace root.
        hook_prefix: See :func:`write_workspace_permissions`.
    """
    if hook_prefix is None:
        hook_prefix = str(workspace_dir.resolve())

    repo_hooks = Path(__file__).resolve().parent.parent / ".claude" / "hooks"

    ws_hooks = workspace_dir / ".claude" / "hooks"
    ws_hooks.mkdir(parents=True, exist_ok=True)
    for hook_name in ("protect_result_files.py", "log_web_search.py"):
        hook_src = repo_hooks / hook_name
        if hook_src.is_file():
            shutil.copy2(hook_src, ws_hooks / hook_name)
        else:
            log.warning("%s not found at %s; skipping", hook_name, hook_src)

    settings_path = workspace_dir / ".claude" / "settings.local.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    else:
        settings = {}

    hooks_config = settings.setdefault("hooks", {})

    # PreToolUse: protect result files
    pre_tool_use = hooks_config.setdefault("PreToolUse", [])
    already_registered = any(
        any(hk.get("command", "").endswith("protect_result_files.py")
            for hk in h.get("hooks", []))
        for h in pre_tool_use
    )
    if not already_registered:
        pre_tool_use.append({
            "matcher": "Read|Write|Edit|MultiEdit|NotebookEdit|Bash",
            "hooks": [{
                "type": "command",
                "command": f"python3 {hook_prefix}/.claude/hooks/protect_result_files.py",
            }],
        })

    # PostToolUse: auto-log web search results
    post_tool_use = hooks_config.setdefault("PostToolUse", [])
    search_matcher = (
        "mcp__web-search-prime__web_search_prime|WebSearch"
        "|mcp__playwright__browser_navigate|mcp__playwright__browser_snapshot"
    )
    already_registered = any(
        any(hk.get("command", "").endswith("log_web_search.py")
            for hk in h.get("hooks", []))
        for h in post_tool_use
    )
    if not already_registered:
        post_tool_use.append({
            "matcher": search_matcher,
            "hooks": [{
                "type": "command",
                "command": f"python3 {hook_prefix}/.claude/hooks/log_web_search.py",
            }],
        })

    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    log.info("Installed hooks in %s", settings_path)


def _read_user_permissions() -> dict[str, list[str]]:
    """Read user's permission rules from ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        return {}
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        perms = data.get("permissions", {})
        return {
            "allow": perms.get("allow", []),
            "deny": perms.get("deny", []),
        }
    except (json.JSONDecodeError, OSError):
        return {}
