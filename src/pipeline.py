"""Run directory scaffolding, state construction, and graph invocation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import secrets
import shutil
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .artifacts import best_result_is_valid, validate_prepare_artifacts, validate_propose_artifacts
from .config import Config
from .graph import compile_graph
from .gpu_policy import log_gpu_policy_warnings, resolve_gpu_policy
from .history import current_manifest_path, resolve_loop_manifest, round_manifest_path
from .workspace_setup import write_workspace_permissions, install_workspace_hooks
from .monitor.server import set_run_dir as _monitor_set_run_dir
from .runtime import (
    get_docker_container, get_token_tracker, push_run_end_event,
    set_config, set_pipeline_state_path, set_run_dir, set_token_tracker, write_pipeline_state,
)
from .session_map import latest_open_propose_loop
from .stage_results import scan_stage_results
from .run_status import (
    finalize_run, resolve_run_status, round_label, terminal_context,
)
from .resume_preflight import apply_resume_extra_time, check_resume_preflight
from .run_config import config_to_metadata, new_resume_event
from .token_tracker import (
    TokenTracker, aggregate_token_usage, cost_stats, hydrate_tracker_from_run,
    token_usage_dict,
)
from .docker.container import DockerContainer

log = logging.getLogger(__name__)

_EXCLUDED_AGENT_SKILLS = {"generate-inputs"}


def _copy_controller_file(src: Path, target: Path, *, readonly: bool = False) -> None:
    """Refresh a controller-owned file in an agent workspace."""
    if not src.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        os.chmod(target, 0o644)
    shutil.copy2(src, target)
    if readonly:
        os.chmod(target, 0o444)


def _refresh_agent_runtime_files(workspace: Path) -> None:
    """Install the current runtime files that agent sessions import/execute."""
    eval_grader_dir = Path(__file__).resolve().parent / "eval_grader"
    _copy_controller_file(
        eval_grader_dir / "client.py",
        workspace / "eval" / "eureka_submit.py",
    )
    _copy_controller_file(
        eval_grader_dir / "gpu_helpers.py",
        workspace / ".eureka_internal" / "gpu_helpers.py",
        readonly=True,
    )


def _refresh_agent_skills(workspace: Path) -> None:
    """Install the current project skills visible to agent sessions."""
    repo_skills = Path(__file__).resolve().parent.parent / ".claude" / "skills"
    if not repo_skills.is_dir():
        return
    ws_skills = workspace / ".claude" / "skills"
    if ws_skills.is_symlink():
        ws_skills.unlink()
    elif ws_skills.exists() and not ws_skills.is_dir():
        ws_skills.unlink()
    shutil.copytree(
        repo_skills,
        ws_skills,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*_EXCLUDED_AGENT_SKILLS),
    )


@contextmanager
def _attach_run_log(run_dir: Path) -> Iterator[Path]:
    """Route all `logging` output to runs/<id>/run.log for the duration of one run.

    We attach a FileHandler to the root logger so every module's logger lands
    in the run's own log file. The handler is removed on exit so it cannot
    leak into subsequent runs in the same process.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield log_path
    finally:
        root.removeHandler(handler)
        handler.close()


async def run_pipeline(
    *,
    problem: Path,
    initial_code: Path | None = None,
    config: Config,
) -> dict[str, Any]:
    """Start a new run. Returns final state dict."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(config.runs_dir) if hasattr(config, "runs_dir") else Path("runs")
    run_dir = runs_dir / run_id
    workspace = run_dir / "workspace"

    (workspace / "inputs").mkdir(parents=True, exist_ok=True)
    (workspace / "round_state").mkdir(parents=True, exist_ok=True)
    (workspace / "approach_details").mkdir(parents=True, exist_ok=True)
    # Session metadata directories (transcripts + session maps) live at the
    # run level, outside the agent's workspace cwd, so agents cannot
    # accidentally read or modify their own transcripts.
    (run_dir / "session_data" / "session_transcripts").mkdir(parents=True, exist_ok=True)
    (run_dir / "session_data" / "session_maps").mkdir(parents=True, exist_ok=True)
    # Pre-create so the agent's first reference doesn't hit a missing file.
    (workspace / "round_state" / "web_search_history.jsonl").touch(exist_ok=True)

    # Copy project skills into workspace for agent discovery.
    # We copy instead of symlink so skills are available inside Docker containers
    # where the host repo path is not mounted.
    _refresh_agent_skills(workspace)

    shutil.copy2(problem, workspace / "inputs" / "problem.md")
    fmt_src = Path(config.submission_format_path)
    shutil.copy2(fmt_src, workspace / "inputs" / "submission_format.md")
    if initial_code and initial_code.exists():
        dest = workspace / "inputs" / "initial_code"
        if initial_code.is_dir():
            shutil.copytree(initial_code, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(initial_code, dest / initial_code.name)

    # Refresh controller-owned files so resumed workspaces do not keep stale helpers.
    _refresh_agent_runtime_files(workspace)

    (workspace / "prepare").mkdir(parents=True, exist_ok=True)

    allowed_gpu_ids = _write_gpu_config(workspace, config)

    _save_metadata(
        run_dir,
        run_id=run_id,
        config=config,
        source_input_paths={
            "problem": str(problem.expanduser().resolve()),
            "initial_code": str(initial_code.expanduser().resolve()) if initial_code else None,
            "submission_format": str(fmt_src.expanduser().resolve()),
        },
        copied_workspace_paths={
            "problem": "workspace/inputs/problem.md",
            "submission_format": "workspace/inputs/submission_format.md",
            "initial_code": "workspace/inputs/initial_code" if initial_code else None,
        },
        allowed_gpu_ids=allowed_gpu_ids,
    )

    # Write workspace permissions and hooks for agent sessions. Docker startup
    # overwrites these with "/workspace" prefixed paths after the container starts.
    write_workspace_permissions(workspace)
    install_workspace_hooks(workspace)

    initial_state = _build_initial_state(
        run_id=run_id,
        run_dir=run_dir,
        workspace_dir=workspace,
        config=config,
    )

    _print_header("New Run", run_id, run_dir, config)

    # Notify the monitor of the specific run directory so it doesn't just
    # show whichever run happens to be latest under runs/.
    try:
        _monitor_set_run_dir(run_dir)
    except Exception:
        pass

    set_config(config)

    set_pipeline_state_path(workspace / ".pipeline_state.json")
    set_run_dir(run_dir)

    # Reuse the global tracker if one was already registered (by the TUI)
    # so both the TUI and pipeline share the same instance. Otherwise
    # create a new one — this path is used by non-TUI invocations.
    try:
        tracker = get_token_tracker()
    except RuntimeError:
        tracker = TokenTracker(
            _input_price=config.input_token_price,
            _cache_creation_price=config.cache_creation_token_price,
            _cache_read_price=config.cache_read_token_price,
            _output_price=config.output_token_price,
        )
        set_token_tracker(tracker)

    # Ensure .venv dependencies are consistent before starting agents.
    _guard_venv_deps()
    write_pipeline_state(
        pipeline_status="running",
        current_loop_index=initial_state.get("loop_index", 0),
        current_stage=initial_state.get("next_stage", ""),
    )

    # Start grader server BEFORE container so URL/token are available for
    # injection into the Docker environment.  Both the grader health-check
    # loop (time.sleep + urlopen) and container.start (subprocess.run) are
    # synchronous — run them in threads so the TUI event loop is not frozen.
    try:
        grader_process = await asyncio.to_thread(_start_grader_server, workspace, config)
    except Exception as exc:
        _fail_startup(exc)
        raise

    container = get_docker_container()
    if container is None:
        raise RuntimeError("Docker container not configured; Eureka runs in Docker mode only.")
    secure_eval_env = {
        "EUREKA_SECURE_SUBMIT_URL": config._grader_url,
        "EUREKA_SECURE_SUBMIT_TOKEN": config._grader_token,
    }
    try:
        await asyncio.to_thread(container.start, workspace, secure_eval_env)
    except Exception as exc:
        _stop_grader_server(grader_process)
        _fail_startup(exc)
        raise

    compiled = compile_graph()
    # Everything that needs to land in run.log — graph invocation, summary
    # save, terminal event emission — runs inside a single try/except under
    # _attach_run_log. This guarantees exceptions are logged AND the TUI is
    # always notified, instead of silently escaping into app._run_pipeline.
    with _attach_run_log(run_dir):
        try:
            final_state = await compiled.ainvoke(initial_state)
            status, _last_stage = finalize_run(
                run_id=run_id, run_dir=run_dir, final_state=final_state,
                max_loops=config.max_loops, config=config,
                save_summary=_save_summary,
            )
            log.info("Run %s finished: %s", run_id, status)
            return final_state
        except KeyboardInterrupt:
            state = _reconstruct_state(run_id, run_dir, config)
            state["status"] = "interrupted"
            _save_summary(run_dir, state)
            ctx = terminal_context(
                final_state=state, run_dir=run_dir, max_loops=config.max_loops,
                config=config,
            )
            push_run_end_event("interrupted", reason="User interrupt", **ctx)
            log.info("Interrupted. Resume with: python -m src --resume %s", run_id)
            return state
        except Exception as exc:
            log.exception("Pipeline terminated with an exception")
            push_run_end_event(
                "error",
                reason=str(exc)[:200],
                stage=initial_state.get("next_stage", ""),
                round_label=round_label(initial_state, config.max_loops),
                evidence_paths=[str(run_dir / "run.log")],
            )
            raise
        except BaseException as exc:
            # Catch CancelledError (task cancelled when TUI exits) and other
            # BaseException subclasses that bypass "except Exception".
            log.exception(
                "Pipeline terminated with BaseException (%s)", type(exc).__name__
            )
            state = _reconstruct_state(run_id, run_dir, config)
            state["status"] = "error"
            _save_summary(run_dir, state)
            ctx = terminal_context(
                final_state=state, run_dir=run_dir, max_loops=config.max_loops,
                config=config,
            )
            push_run_end_event(
                "error",
                reason=f"{type(exc).__name__}: {exc!s}"[:200],
                **ctx,
            )
            raise
        finally:
            if grader_process:
                _stop_grader_server(grader_process)
            container.stop()
            # Restore .venv if agent installed packages that conflict with
            # pyproject.toml dependencies. Preserves non-conflicting extras.
            _guard_venv_deps()


async def resume_pipeline(
    run_id: str,
    *,
    runs_dir: Path = Path("runs"),
    config: Config,
    resume_extra_seconds: float | None = None,
) -> dict[str, Any]:
    """Resume a prior run from filesystem state."""
    run_dir = runs_dir / run_id
    if not run_dir.is_dir():
        raise SystemExit(
            f"Run directory not found: {run_dir}\n"
            f"Start a new run without --resume, or provide a valid run ID."
        )

    # Validate that session data exists (either new or legacy location)
    session_data_dir = run_dir / "session_data"
    new_transcripts = session_data_dir / "session_transcripts"
    old_transcripts = run_dir / "workspace" / "session_transcripts"
    has_session_data = (
        (new_transcripts.is_dir() and any(new_transcripts.iterdir()))
        or (old_transcripts.is_dir() and any(old_transcripts.iterdir()))
    )
    if not has_session_data:
        raise SystemExit(
            f"No session data found for run {run_id}.\n"
            f"Start a new run without --resume, or provide a valid run ID."
        )

    # On resume, CLI time limits are authoritative (user may shorten budgets).
    workspace = run_dir / "workspace"

    state = _reconstruct_state(run_id, run_dir, config)
    preflight = check_resume_preflight(run_dir, state, config)
    if preflight.needs_extra_time:
        if resume_extra_seconds is None:
            raise RuntimeError(preflight.message)
        apply_resume_extra_time(preflight, resume_extra_seconds)

    if state.get("next_stage") == "end":
        log.info("Run %s is already terminal. Nothing to resume.", run_id)
        return state

    _print_header("Resume", run_id, run_dir, config)

    try:
        _monitor_set_run_dir(run_dir)
    except Exception:
        pass

    set_config(config)

    set_pipeline_state_path(workspace / ".pipeline_state.json")
    set_run_dir(run_dir)

    # Reuse the global tracker if one was already registered (by the TUI)
    try:
        tracker = get_token_tracker()
    except RuntimeError:
        tracker = TokenTracker(
            _input_price=config.input_token_price,
            _cache_creation_price=config.cache_creation_token_price,
            _cache_read_price=config.cache_read_token_price,
            _output_price=config.output_token_price,
        )
        set_token_tracker(tracker)
    hydrate_tracker_from_run(run_dir, tracker)
    current_stage = str(state.get("next_stage", ""))
    current_loop_index = int(state.get("loop_index", 0) or 0)
    if current_stage == "propose":
        current_loop_index += 1
    write_pipeline_state(
        pipeline_status="running",
        current_loop_index=current_loop_index,
        current_stage=current_stage,
    )
    
    # Always refresh controller-owned runtime files and GPU policy on resume.
    # Runs may have been created before the current grader/GPU lock protocol
    # existed, and explicit --gpus changes must rewrite stale allowlists.
    _refresh_agent_runtime_files(workspace)
    allowed_gpu_ids = _write_gpu_config(workspace, config)
    _remove_disallowed_gpu_locks(workspace, allowed_gpu_ids)

    # Update metadata with the effective resume config and runtime GPU policy.
    _save_metadata(
        run_dir,
        run_id=run_id,
        config=config,
        allowed_gpu_ids=allowed_gpu_ids,
        resume_event=new_resume_event(getattr(config, "_resume_config_report", None)),
    )

    # Refresh agent-facing skills, permissions, and hooks on resume so old
    # workspaces pick up the current controller policy.
    _refresh_agent_skills(workspace)
    write_workspace_permissions(workspace)
    install_workspace_hooks(workspace)

    grader_process = None
    if not hasattr(config, "_grader_url"):
        try:
            grader_process = await asyncio.to_thread(_start_grader_server, workspace, config)
        except Exception as exc:
            _fail_startup(exc)
            raise

    container = get_docker_container()
    if container is None:
        raise RuntimeError("Docker container not configured; Eureka runs in Docker mode only.")
    secure_eval_env = {
        "EUREKA_SECURE_SUBMIT_URL": config._grader_url,
        "EUREKA_SECURE_SUBMIT_TOKEN": config._grader_token,
    }
    try:
        await asyncio.to_thread(container.start, workspace, secure_eval_env)
    except Exception as exc:
        if grader_process:
            _stop_grader_server(grader_process)
        _fail_startup(exc)
        raise

    compiled = compile_graph()
    with _attach_run_log(run_dir):
        try:
            final_state = await compiled.ainvoke(state)
            status, _last_stage = finalize_run(
                run_id=run_id, run_dir=run_dir, final_state=final_state,
                max_loops=config.max_loops, config=config,
                save_summary=_save_summary,
            )
            log.info("Run %s finished: %s", run_id, status)
            return final_state
        except KeyboardInterrupt:
            state = _reconstruct_state(run_id, run_dir, config)
            state["status"] = "interrupted"
            _save_summary(run_dir, state)
            ctx = terminal_context(
                final_state=state, run_dir=run_dir, max_loops=config.max_loops,
                config=config,
            )
            push_run_end_event("interrupted", reason="User interrupt", **ctx)
            log.info("Interrupted. Resume with: python -m src --resume %s", run_id)
            return state
        except Exception as exc:
            log.exception("Pipeline terminated with an exception")
            push_run_end_event(
                "error",
                reason=str(exc)[:200],
                stage=state.get("next_stage", ""),
                round_label=round_label(state, config.max_loops),
                evidence_paths=[str(run_dir / "run.log")],
            )
            raise
        except BaseException as exc:
            log.exception(
                "Pipeline terminated with BaseException (%s)", type(exc).__name__
            )
            state = _reconstruct_state(run_id, run_dir, config)
            state["status"] = "error"
            _save_summary(run_dir, state)
            ctx = terminal_context(
                final_state=state, run_dir=run_dir, max_loops=config.max_loops,
                config=config,
            )
            push_run_end_event(
                "error",
                reason=f"{type(exc).__name__}: {exc!s}"[:200],
                **ctx,
            )
            raise
        finally:
            if grader_process:
                _stop_grader_server(grader_process)
            container.stop()
            _guard_venv_deps()


# -- Secure grader server helpers --

def _find_free_port() -> int:
    """Ask the OS for a free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_no_proxy() -> None:
    """Ensure localhost is excluded from proxy for urllib requests."""
    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        entries = set(e.strip() for e in current.split(",") if e.strip())
        for host in ("localhost", "127.0.0.1"):
            if host not in entries:
                entries.add(host)
        os.environ[var] = ",".join(sorted(entries))


def _wait_for_grader_health(port: int, *, container: DockerContainer | None = None) -> str | None:
    """Wait for the grader HTTP health endpoint and return the last error."""
    last_error: str | None = None
    for _ in range(30):
        if container is not None:
            result = subprocess.run(
                container.exec_command([
                    "python3",
                    "-c",
                    (
                        "import urllib.request; "
                        f"urllib.request.urlopen('http://127.0.0.1:{port}/healthz', timeout=1).read()"
                    ),
                ]),
                capture_output=True, text=True, check=False,
            )
            if result.returncode == 0:
                return None
            last_error = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
            time.sleep(0.3)
            continue

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as resp:
                if resp.status == 200:
                    return None
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code} ({e.reason})"
            log.warning("Docker grader health check got %s", last_error)
            time.sleep(0.3)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_error = str(e.reason) if hasattr(e, "reason") else str(e)
            time.sleep(0.3)
        except Exception as e:
            last_error = str(e)
            time.sleep(0.3)
    return last_error


def _start_grader_server(workspace_dir: Path, config: Config) -> DockerContainer:
    """Start the secure grader server in a Docker container."""
    token = secrets.token_hex(16)
    port = _find_free_port()
    container = get_docker_container()
    if container is None:
        raise RuntimeError("Docker container not configured; Eureka runs in Docker mode only.")

    grader = DockerContainer(
        image=config.docker_image,
        network=config.docker_network,
        gpus=config.gpus,
    )
    grader.start_grader(
        workspace_dir=workspace_dir,
        hidden_eval_dir=Path(config.hidden_eval_dir),
        host_port=port,
        token=token,
    )
    _ensure_no_proxy()
    health_container = grader if config.docker_network == "host" else None
    last_error = _wait_for_grader_health(port, container=health_container)
    if last_error is not None:
        grader.stop()
        raise RuntimeError(
            f"Docker grader server failed to start within 10 seconds "
            f"(last error: {last_error})."
        )

    submit_host = "127.0.0.1" if config.docker_network == "host" else "host.docker.internal"
    object.__setattr__(config, "_grader_url", f"http://{submit_host}:{port}")
    object.__setattr__(config, "_grader_token", token)
    log.info("Docker grader server started on port %d", port)
    return grader


def _stop_grader_server(proc: DockerContainer) -> None:
    """Stop the grader server container."""
    proc.stop()
    log.info("Grader server stopped")


def _fail_startup(exc: BaseException) -> None:
    """Record a startup failure so the TUI and filesystem reflect the error."""
    write_pipeline_state(pipeline_status="error")
    push_run_end_event("error", reason=str(exc)[:200])


def _write_gpu_config(workspace: Path, config: Config) -> list[int]:
    """Write .gpu_config.json so gpu_helpers knows which GPUs are allowed."""
    policy = resolve_gpu_policy(config.gpus)
    log_gpu_policy_warnings(policy, log)

    config_data = {
        "requested_gpus": policy.requested,
        "allowed_gpu_ids": policy.allowed_gpu_ids,
        "total_available": len(policy.allowed_gpu_ids),
    }
    config_path = workspace / ".gpu_config.json"
    config_path.write_text(json.dumps(config_data, indent=2) + "\n", encoding="utf-8")
    return policy.allowed_gpu_ids


def _remove_disallowed_gpu_locks(workspace: Path, allowed_gpu_ids: list[int]) -> list[int]:
    """Remove stale GPU locks for devices no longer allowed by config.gpus."""
    allowed = set(allowed_gpu_ids)
    removed: list[int] = []
    lock_dir = workspace / ".gpu_locks"
    if not lock_dir.is_dir():
        return removed
    for lock_path in lock_dir.glob("gpu_*.lock"):
        try:
            gpu_id = int(lock_path.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if gpu_id in allowed:
            continue
        try:
            lock_path.unlink()
            removed.append(gpu_id)
        except OSError:
            pass
    if removed:
        log.warning(
            "Removed stale GPU lock(s) outside current allowlist: %s",
            sorted(removed),
        )
    return sorted(removed)


# -- Internal helpers --


def _build_initial_state(
    *,
    run_id: str,
    run_dir: Path,
    workspace_dir: Path,
    config: Config,
) -> dict[str, Any]:
    return {
        "problem": "inputs/problem.md",
        "propose_time_limit_per_session": config.propose_time_limit_per_session,
        "implement_time_limit_per_session": config.implement_time_limit_per_session,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "workspace_dir": str(workspace_dir),
        "next_stage": "prepare",
        "max_loops": config.max_loops,
        "max_num_approaches": config.max_num_approaches,
        "loop_index": 0,
        "end_reason": "",
        "history": [],
        "model": config.model or "",
        "cost_limit": config.cost_limit,
        "input_token_price": config.input_token_price,
        "cache_creation_token_price": config.cache_creation_token_price,
        "cache_read_token_price": config.cache_read_token_price,
        "output_token_price": config.output_token_price,
        "cost_currency": config.cost_currency,
    }


def _save_metadata(
    run_dir: Path,
    *,
    run_id: str,
    config: Config,
    source_input_paths: dict[str, str | None] | None = None,
    copied_workspace_paths: dict[str, str | None] | None = None,
    allowed_gpu_ids: list[int] | None = None,
    resume_event: dict[str, Any] | None = None,
) -> None:
    path = run_dir / "run_metadata.json"
    # Preserve original created_at when updating metadata on resume
    existing = _load_metadata(run_dir) if path.exists() else {}
    config_payload = config_to_metadata(config)
    source_paths = (
        {k: v for k, v in source_input_paths.items() if v is not None}
        if source_input_paths is not None
        else existing.get("source_input_paths")
    )
    copied_paths = (
        {k: v for k, v in copied_workspace_paths.items() if v is not None}
        if copied_workspace_paths is not None
        else existing.get("copied_workspace_paths")
    )
    resume_events = list(existing.get("resume_events") or [])
    if resume_event:
        resume_events.append(resume_event)
    payload = {
        "run_id": run_id,
        "problem_path": "inputs/problem.md",
        "config": config_payload,
        "model": config.model,
        "propose_time_limit_per_session": config.propose_time_limit_per_session,
        "implement_time_limit_per_session": config.implement_time_limit_per_session,
        "max_loops": config.max_loops,
        "max_num_approaches": config.max_num_approaches,
        "cost_limit": config.cost_limit,
        "input_token_price": config.input_token_price,
        "cache_creation_token_price": config.cache_creation_token_price,
        "cache_read_token_price": config.cache_read_token_price,
        "output_token_price": config.output_token_price,
        "cost_currency": config.cost_currency,
        "docker_image": config.docker_image,
        "docker_network": config.docker_network,
        "gpus": config.gpus,
        "allowed_gpu_ids": allowed_gpu_ids if allowed_gpu_ids is not None else existing.get("allowed_gpu_ids"),
        "hidden_eval_dir": config.hidden_eval_dir,
        "submission_format_path": config.submission_format_path,
        "adapter_mode": config.adapter_mode,
        "skip_prepare": config.skip_prepare,
        "created_at": existing.get("created_at") or datetime.now(tz=timezone.utc).isoformat(),
    }
    if source_paths:
        payload["source_input_paths"] = source_paths
    if copied_paths:
        payload["copied_workspace_paths"] = copied_paths
    if resume_events:
        payload["resume_events"] = resume_events
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_summary(run_dir: Path, state: dict[str, Any]) -> None:
    path = run_dir / "run_summary.json"
    try:
        tracker = get_token_tracker()
        token_usage = token_usage_dict(tracker)
        if not any(token_usage.values()):
            token_usage = aggregate_token_usage(run_dir / "workspace")
            if any(token_usage.values()):
                tracker.update_session("_summary_fallback", token_usage)
                token_usage = token_usage_dict(tracker)
        cost = cost_stats(tracker, currency=state.get("cost_currency", "USD"))
    except RuntimeError:
        token_usage = aggregate_token_usage(run_dir / "workspace")
        tracker = TokenTracker(
            _input_price=state.get("input_token_price"),
            _cache_creation_price=state.get("cache_creation_token_price"),
            _cache_read_price=state.get("cache_read_token_price"),
            _output_price=state.get("output_token_price"),
        )
        tracker.update_session("_summary", token_usage)
        cost = cost_stats(tracker, currency=state.get("cost_currency", "USD"))
    status, _last_stage = resolve_run_status(state)
    payload = {
        "run_id": state.get("run_id", ""),
        "status": status,
        "loop_index": state.get("loop_index", 0),
        "history": state.get("history", []),
        "token_usage": token_usage,
        "cost": cost,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _reconstruct_state(
    run_id: str,
    run_dir: Path,
    config: Config,
) -> dict[str, Any]:
    """Rebuild state from filesystem artifacts."""
    workspace = run_dir / "workspace"
    state = _build_initial_state(
        run_id=run_id,
        run_dir=run_dir,
        workspace_dir=workspace,
        config=config,
    )

    if validate_prepare_artifacts(workspace).ok:
        state["prepare_status"] = "ready"
        state["next_stage"] = "propose"
    else:
        state["prepare_status"] = "pending"
        state["next_stage"] = "prepare"
        return state

    session_data_dir = run_dir / "session_data"
    resume_propose_loop = latest_open_propose_loop(session_data_dir)
    if resume_propose_loop is not None and not _propose_artifacts_ready(
        workspace, resume_propose_loop, config,
    ):
        state["loop_index"] = max(resume_propose_loop - 1, 0)
        state["next_stage"] = "propose"
        state["propose_status"] = "abort"
        state["manifest_path"] = str(current_manifest_path(workspace))
        return state

    stage_results = _scan_stage_results(workspace)
    if not stage_results:
        return state

    resume_impl_loop: int | None = None
    for loop_idx in range(1, config.max_loops + 1):
        if _loop_has_unresolved_implement(session_data_dir, workspace, loop_idx):
            resume_impl_loop = loop_idx
            break

    if resume_impl_loop is not None:
        latest_index = resume_impl_loop
        latest = stage_results.get(latest_index, {})
        latest.setdefault("propose", {"status": "ready"})
        latest.setdefault("implement", {"status": "interrupted"})
    elif resume_propose_loop is not None:
        latest_index = resume_propose_loop
        latest = stage_results.get(latest_index, {})
        latest.setdefault("propose", {"status": "abort"})
    else:
        latest_index = max(stage_results)
        latest = stage_results[latest_index]

    propose_result = latest.get("propose")
    implement_result = latest.get("implement")

    if propose_result is None:
        # Propose never started for this loop; start fresh.
        state["loop_index"] = max(latest_index - 1, 0)
        state["next_stage"] = "propose"
        state["manifest_path"] = str(current_manifest_path(workspace))
        return state

    state["loop_index"] = latest_index
    state["propose_status"] = str(propose_result.get("status", ""))
    if propose_result.get("status") == "abort":
        # propose_node increments loop_index on entry, so resume from previous index.
        state["loop_index"] = max(latest_index - 1, 0)
        state["next_stage"] = "propose"
        state["manifest_path"] = str(current_manifest_path(workspace))
        return state

    manifest_path = (
        resolve_loop_manifest(
            workspace, latest_index, session_data_dir=session_data_dir,
            sync_current=True,
        )
        or current_manifest_path(workspace)
    )
    state["manifest_path"] = str(manifest_path)

    if implement_result is None:
        state["next_stage"] = "implement"
        return state

    state["implement_status"] = str(implement_result.get("status", ""))
    if implement_result.get("status") in ("all_failed", "interrupted", "abort"):
        # Retry incomplete implementation sessions instead of advancing rounds.
        state["next_stage"] = "implement"
        return state

    if latest_index < config.max_loops:
        state["next_stage"] = "propose"
    else:
        state["next_stage"] = "end"

    return state


def _loop_has_unresolved_implement(
    session_data_dir: Path,
    workspace: Path,
    loop_index: int,
) -> bool:
    """True if a loop has open implement sessions without a valid result."""
    maps_dir = session_data_dir / "session_maps"
    if not maps_dir.is_dir():
        return False
    propose_name = f"loop_{loop_index}_propose_session_map.json"
    for path in maps_dir.glob(f"loop_{loop_index}_*_session_map.json"):
        if path.name == propose_name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("stage") != "implement":
            continue
        if payload.get("status") not in ("aborted", "running"):
            continue
        approach_id = str(payload.get("approach_id") or "")
        if not approach_id:
            return True
        result_path = workspace / "approach_details" / approach_id / "best_result.jsonl"
        if not _best_result_is_valid(result_path):
            return True
    return False


def _best_result_is_valid(path: Path) -> bool:
    return best_result_is_valid(path)


def _propose_artifacts_ready(workspace: Path, loop_index: int, config: Config) -> bool:
    for manifest_path in (round_manifest_path(workspace, loop_index), current_manifest_path(workspace)):
        if validate_propose_artifacts(
            workspace, loop_index, config.max_num_approaches,
            manifest_path=manifest_path,
        ).ok:
            return True
    return False




def _scan_stage_results(
    workspace: Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Compatibility wrapper for tests/imports; implementation is shared."""
    return scan_stage_results(workspace)


def _print_header(
    title: str, run_id: str, run_dir: Path, config: Config
) -> None:
    log.info("--- %s --- run_id=%s run_dir=%s propose_time_limit=%s "
             "implement_time_limit=%s max_loops=%d approaches=%d",
             title, run_id, run_dir, config.propose_time_limit_per_session,
             config.implement_time_limit_per_session,
             config.max_loops, config.max_num_approaches)


def _guard_venv_deps() -> None:
    """Ensure .venv dependencies satisfy pyproject.toml.

    Runs ``uv pip check`` and repairs with ``uv sync --inexact`` if conflicts
    are found. ``--inexact`` preserves agent-installed extras (e.g., scipy)
    while still fixing version conflicts in pyproject.toml dependencies.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".venv").is_dir():
        return
    try:
        result = subprocess.run(
            ["uv", "pip", "check"],
            cwd=str(repo_root),
            capture_output=True, text=True, check=False,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning("Dependency conflicts detected, restoring .venv:\n%s", result.stdout)
            subprocess.run(
                ["uv", "sync", "--inexact"],
                cwd=str(repo_root),
                capture_output=True, text=True, check=True,
                timeout=120,
            )
            log.info("Dependencies restored via uv sync --inexact")
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        log.warning("uv pip check timed out, skipping dependency guard")
