"""CLI entrypoint for eureka-loop."""

from __future__ import annotations

import argparse
import asyncio
import ast
import dataclasses
import logging
import os
import sys
from pathlib import Path

from .config import Config
from .docker.container import DockerContainer
from .duration import parse_time_limit
from .gpu_policy import validate_gpu_request
from .monitor.server import start_monitor_server
from .pipeline import resume_pipeline, run_pipeline
from .pricing import fetch_model_pricing, resolve_model_name
from .resume_preflight import MIN_RESUME_EXTRA_SECONDS
from .runtime import set_docker_container
from .run_config import (
    ResumeConfigError,
    build_new_run_config,
    build_resume_config,
    explicit_config_fields,
)


_MIN_PROPOSE_BUDGET_SECONDS = 900.0  # 15 minutes — agent needs time to brainstorm and write approaches
_MIN_IMPLEMENT_BUDGET_SECONDS = 3600.0  # 60 minutes — agent needs time to implement, improve, evaluate, and submit


def main() -> None:
    # Bootstrap-phase logging goes to stderr. Once a run_dir is known, the
    # pipeline attaches a FileHandler at runs/<id>/run.log so all engine logs
    # for a run live together with the rest of that run's artifacts.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="EurekAgent — AI-for-Research auto-experiment optimization loop.",
    )
    parser.add_argument("--problem", type=str, help="Path to problem description file")
    parser.add_argument("--initial-code", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--max-loops", type=int, default=None)
    parser.add_argument("--max-num-approaches", type=int, default=None)
    parser.add_argument(
        "--propose-time-limit-per-session", type=str, default=None,
        help='Required propose-stage time limit, e.g. "20 minutes".',
    )
    parser.add_argument(
        "--implement-time-limit-per-session", type=str, default=None,
        help='Required implement-stage time limit, e.g. "120 minutes".',
    )
    parser.add_argument(
        "--force-low-budget", action="store_true", default=False,
        help=(
            "Allow per-stage time limits below their safety floors "
            f"(propose: {int(_MIN_PROPOSE_BUDGET_SECONDS/60)} minutes, "
            f"implement: {int(_MIN_IMPLEMENT_BUDGET_SECONDS/60)} minutes)."
        ),
    )
    parser.add_argument("--cost-limit", type=float, default=None,
                        help="Total cost limit for the entire run in USD.")
    parser.add_argument(
        "--no-cost-limit", action="store_true", default=False,
        help="Clear the cost limit when resuming a run.",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. claude-sonnet-4-6). Used for pricing lookup and passed to claude CLI.")
    parser.add_argument("--input-token-price", type=float, default=None)
    parser.add_argument("--cache-creation-token-price", type=float, default=None)
    parser.add_argument("--cache-read-token-price", type=float, default=None)
    parser.add_argument("--output-token-price", type=float, default=None)
    parser.add_argument("--cost-currency", type=str, default="USD")
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_ID")
    parser.add_argument(
        "--resume-extra-time", type=str, default=None,
        help=(
            "Extra time for resuming a stage that already exhausted its budget "
            "without writing required artifacts, e.g. '10 minutes'."
        ),
    )
    # Docker isolation
    parser.add_argument("--docker-image", type=str, default="eureka-agent:node22-bookworm", help="Docker image name")
    parser.add_argument("--docker-network", type=str, default="host", help=argparse.SUPPRESS)
    parser.add_argument("--gpus", type=str, default="auto", help='"auto", "none", or CUDA device IDs like "0,1"')
    # Secure evaluation
    parser.add_argument("--hidden-eval-dir", type=str, default=None,
                        help="Path to private grader directory (outside workspace). "
                             "Must contain evaluate.py.")
    parser.add_argument("--submission-format", type=str, default=None,
                        help="Path to SUBMISSION_FORMAT.md describing the expected "
                             "candidate JSON schema.")
    # Adapter mode
    parser.add_argument(
        "--adapter-mode", type=str, default="pty", choices=["pty", "stream"],
        help='Session adapter mode: "pty" (interactive, default) or "stream" (stream-json)',
    )
    parser.add_argument(
        "--skip-prepare", action="store_true", default=False,
        help="Skip the prepare stage entirely (use when you are confident the setup is ready)",
    )
    # Web monitor (default: on)
    parser.add_argument("--no-monitor", action="store_true", default=False,
                        help="Disable the web monitor server")
    parser.add_argument("--monitor-port", type=int, default=0,
                        help="Port for the web monitor server (default: auto-assign)")
    # Hidden debug flag
    parser.add_argument("--no-tui", action="store_true", default=False, help=argparse.SUPPRESS)

    args = parser.parse_args()
    explicit_fields = explicit_config_fields(sys.argv[1:])

    if args.cost_limit is not None and args.no_cost_limit:
        parser.error("--cost-limit and --no-cost-limit cannot be used together")
    try:
        validate_gpu_request(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))

    if args.resume:
        try:
            resolved = build_resume_config(
                args,
                run_dir=Path(args.runs_dir) / args.resume,
                explicit_fields=explicit_fields,
            )
        except ResumeConfigError as exc:
            raise SystemExit(f"Resume configuration is incompatible:\n{exc}") from exc
        config = _finalize_config(resolved.config)
        _setup_docker_container(config)
        _run(args, config, resume=True)
        return

    # Validate required args (time budgets are validated separately below).
    if not args.problem:
        parser.error("Required argument missing: --problem")
    if not args.hidden_eval_dir:
        parser.error("Required argument missing: --hidden-eval-dir")
    if not args.submission_format:
        parser.error("Required argument missing: --submission-format")

    # Validate file paths
    problem = Path(args.problem)
    if not problem.is_file():
        parser.error(f"--problem: file not found: {args.problem}")

    # Validate secure eval dir
    if not Path(args.hidden_eval_dir, "evaluate.py").is_file():
        parser.error(f"--hidden-eval-dir: directory must contain evaluate.py: {args.hidden_eval_dir}")

    # Validate evaluate.py contains required function definitions without
    # importing it on the host. In Docker mode the evaluator runs inside the
    # grader container, so host-side imports would incorrectly require the
    # evaluator's Linux/container dependencies to exist on macOS.
    _eval_path = Path(args.hidden_eval_dir, "evaluate.py")
    try:
        _tree = ast.parse(_eval_path.read_text(encoding="utf-8"), filename=str(_eval_path))
    except (OSError, SyntaxError) as exc:
        parser.error(f"evaluate.py is unreadable or invalid Python: {exc}")
    _functions = {
        node.name for node in _tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "grade_submission" not in _functions:
        parser.error("evaluate.py must define `grade_submission(submission_path, context)`")
    if "is_better" not in _functions:
        parser.error("evaluate.py must define `is_better(new_score, old_score) -> bool`")

    # Validate submission format
    if not Path(args.submission_format).is_file():
        parser.error(f"--submission-format: file not found: {args.submission_format}")

    initial_code = None
    if args.initial_code:
        initial_code = Path(args.initial_code)
        if not initial_code.exists():
            parser.error(f"--initial-code: path not found: {args.initial_code}")

    # Time budgets: each stage must declare its own limit.
    _check_time_budget_mode(parser, args)
    _validate_time_budget(parser, "--propose-time-limit-per-session",
                          args.propose_time_limit_per_session, args.force_low_budget,
                          _MIN_PROPOSE_BUDGET_SECONDS)
    _validate_time_budget(parser, "--implement-time-limit-per-session",
                          args.implement_time_limit_per_session, args.force_low_budget,
                          _MIN_IMPLEMENT_BUDGET_SECONDS)

    config = _finalize_config(build_new_run_config(args))
    if not config.model:
        parser.error("Required argument missing: --model")
    _setup_docker_container(config)

    _run(args, config, resume=False, problem=problem, initial_code=initial_code)


def _start_monitor(
    args: argparse.Namespace,
    config: Config,
    *,
    resume: bool,
) -> int:
    """Start the web monitor server in a background thread. Returns the actual port."""
    runs_dir = Path(config.runs_dir)
    return start_monitor_server(
        runs_dir=runs_dir,
        port=args.monitor_port,
    )


def _run(
    args: argparse.Namespace,
    config: Config,
    *,
    resume: bool,
    problem: Path | None = None,
    initial_code: Path | None = None,
) -> None:
    """Launch the pipeline via TUI or headless mode."""
    resume_extra_seconds = _parse_resume_extra_time(args)
    # Start web monitor in background (default: on, use --no-monitor to disable)
    monitor_port: int | None = None
    if not args.no_monitor:
        monitor_port = _start_monitor(args, config, resume=resume)

    use_tui = sys.stdout.isatty() and not args.no_tui

    if use_tui:
        # The TUI uses the alternate screen buffer; any output to stderr
        # corrupts the rendered widgets. Remove StreamHandlers so logs only
        # go to the FileHandler attached by _attach_run_log, AND redirect
        # stderr so stray log.warning / C-library writes cannot leak into
        # the terminal. Do NOT redirect stdout — Textual needs it as a TTY
        # for cursor management and Input widget event processing.
        for handler in logging.getLogger().handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logging.getLogger().removeHandler(handler)

        # Redirect stderr to a file (NOT /dev/null) so crash tracebacks
        # are preserved for post-mortem debugging. The TUI hides the
        # alternate screen buffer on exit, making terminal-only errors
        # invisible — writing to a file ensures they're never lost.
        if resume and args.resume:
            _stderr_log_path = Path(args.runs_dir) / args.resume / "tui_stderr.log"
        else:
            _stderr_log_path = Path(args.runs_dir) / "tui_stderr.log"
        _stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        _saved_stderr = sys.stderr
        _stderr_log_file = open(_stderr_log_path, "a")
        sys.stderr = _stderr_log_file

        from .tui.app import EurekAgentApp
        app = EurekAgentApp(
            config=config,
            resume_id=args.resume if resume else None,
            problem=problem,
            initial_code=initial_code,
            monitor_port=monitor_port,
            resume_extra_seconds=resume_extra_seconds,
        )
        try:
            app.run()
        finally:
            sys.stderr = _saved_stderr
            _stderr_log_file.close()
    else:
        # Headless mode (debug/CI)
        if resume:
            asyncio.run(resume_pipeline(
                args.resume,
                runs_dir=Path(args.runs_dir),
                config=config,
                resume_extra_seconds=resume_extra_seconds,
            ))
        else:
            asyncio.run(run_pipeline(
                problem=problem,
                initial_code=initial_code,
                config=config,
            ))


def _parse_resume_extra_time(args: argparse.Namespace) -> float | None:
    if not args.resume_extra_time:
        return None
    seconds = parse_time_limit(args.resume_extra_time)
    if seconds is None:
        raise SystemExit(
            f'--resume-extra-time: invalid format "{args.resume_extra_time}". '
            'Use "N minutes" or "N hours".'
        )
    if seconds < MIN_RESUME_EXTRA_SECONDS:
        raise SystemExit(
            f"--resume-extra-time must be at least {int(MIN_RESUME_EXTRA_SECONDS / 60)} minutes."
        )
    return seconds


def _check_time_budget_mode(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Require explicit per-stage time budgets."""
    propose = args.propose_time_limit_per_session
    implement = args.implement_time_limit_per_session

    if propose and not implement:
        parser.error(
            "--propose-time-limit-per-session was set but "
            "--implement-time-limit-per-session is missing. Set both per-stage flags."
        )

    if implement and not propose:
        parser.error(
            "--implement-time-limit-per-session was set but "
            "--propose-time-limit-per-session is missing. Set both per-stage flags."
        )

    if not propose and not implement:
        parser.error(
            "No time budget specified. Set both required per-stage limits:\n"
            '  --propose-time-limit-per-session "N minutes" '
            '--implement-time-limit-per-session "N minutes"'
        )


def _validate_time_budget(
    parser: argparse.ArgumentParser,
    label: str,
    value: str | None,
    force_low: bool,
    min_seconds: float,
) -> None:
    """Reject unparseable or dangerously-low time limits at startup.

    Catching this here means users see a clear error before the TUI launches,
    rather than watching a doomed run time out mid-way.
    """
    if value is None:
        return
    seconds = parse_time_limit(value)
    if seconds is None:
        parser.error(
            f'{label}: invalid format "{value}". '
            'Use "N minutes", "N hours", or "N hours M minutes".'
        )
    if seconds < min_seconds and not force_low:
        parser.error(
            f'{label}={value} is below the {int(min_seconds/60)}-minute safety floor. '
            "The agent typically cannot read inputs, think, and write results under this "
            "budget. Increase the limit, or pass --force-low-budget if you know what you are doing."
        )


def _finalize_config(config: Config) -> Config:
    """Resolve model/pricing after the effective config is known."""
    # Resolve model name (CLI arg → env var fallback) and store back into config.
    resolved_model = resolve_model_name(config.model)
    if resolved_model and config.model is None:
        config = dataclasses.replace(config, model=resolved_model)

    # Auto-fill missing prices from OpenRouter
    needs_pricing = (
        config.input_token_price is None
        or config.cache_creation_token_price is None
        or config.cache_read_token_price is None
        or config.output_token_price is None
    )
    if needs_pricing and config.model:
        prices = fetch_model_pricing(config.model)
        if prices:
            overrides = {}
            if config.input_token_price is None and prices.get("input_price") is not None:
                overrides["input_token_price"] = prices["input_price"]
            if config.cache_creation_token_price is None and prices.get("cache_creation_price") is not None:
                overrides["cache_creation_token_price"] = prices["cache_creation_price"]
            if config.cache_read_token_price is None and prices.get("cache_read_price") is not None:
                overrides["cache_read_token_price"] = prices["cache_read_price"]
            if config.output_token_price is None and prices.get("output_price") is not None:
                overrides["output_token_price"] = prices["output_price"]
            if overrides:
                config = dataclasses.replace(config, **overrides)
                logging.getLogger(__name__).info(
                    "Auto-filled pricing from OpenRouter for model %s: %s",
                    config.model, overrides,
                )

    return config


def _setup_docker_container(config: Config) -> None:
    """Set up the Docker container used by the pipeline."""
    container = DockerContainer(
        image=config.docker_image,
        network=config.docker_network,
        gpus=config.gpus,
    )
    set_docker_container(container)


if __name__ == "__main__":
    main()
