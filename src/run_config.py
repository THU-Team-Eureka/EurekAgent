"""Central startup and resume configuration resolution."""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .history import read_manifest
from .token_tracker import TokenTracker, aggregate_token_usage


_MISSING = object()

_FIELD_BY_OPTION = {
    "--max-loops": "max_loops",
    "--max-num-approaches": "max_num_approaches",
    "--propose-time-limit-per-session": "propose_time_limit_per_session",
    "--implement-time-limit-per-session": "implement_time_limit_per_session",
    "--cost-limit": "cost_limit",
    "--no-cost-limit": "cost_limit",
    "--model": "model",
    "--input-token-price": "input_token_price",
    "--cache-creation-token-price": "cache_creation_token_price",
    "--cache-read-token-price": "cache_read_token_price",
    "--output-token-price": "output_token_price",
    "--token-price-tiers": "token_price_tiers",
    "--cost-currency": "cost_currency",
    "--docker-image": "docker_image",
    "--docker-network": "docker_network",
    "--gpus": "gpus",
    "--hidden-eval-dir": "hidden_eval_dir",
    "--submission-format": "submission_format_path",
    "--adapter-mode": "adapter_mode",
    "--skip-prepare": "skip_prepare",
    "--runs-dir": "runs_dir",
    "--run-id": "run_id",
}

IMMUTABLE_CONFIG_FIELDS = {
    "run_id",
    "docker_image",
    "docker_network",
    "adapter_mode",
    "hidden_eval_dir",
    "submission_format_path",
    "skip_prepare",
}

MUTABLE_CONFIG_FIELDS = {
    field.name for field in dataclasses.fields(Config)
} - IMMUTABLE_CONFIG_FIELDS

_PATH_FIELDS = {
    "hidden_eval_dir",
    "submission_format_path",
    "problem",
    "initial_code",
}


@dataclass(frozen=True)
class ResumeConfigReport:
    defaulted_fields: list[str] = field(default_factory=list)
    overridden_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected_incompatibilities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedRunConfig:
    config: Config
    report: ResumeConfigReport = field(default_factory=ResumeConfigReport)


class ResumeConfigError(ValueError):
    """Raised when resume CLI/config is incompatible with the saved run."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("\n".join(messages))


def explicit_config_fields(argv: list[str]) -> set[str]:
    """Return Config fields explicitly mentioned on the CLI."""
    fields: set[str] = set()
    for arg in argv:
        if not arg.startswith("--"):
            continue
        opt = arg.split("=", 1)[0]
        field_name = _FIELD_BY_OPTION.get(opt)
        if field_name:
            fields.add(field_name)
    return fields


def build_new_run_config(args: Any) -> Config:
    """Build a new-run Config from parsed CLI args and current defaults."""
    defaults = Config()
    cost_limit = None if getattr(args, "no_cost_limit", False) else args.cost_limit
    return Config(
        max_loops=args.max_loops if args.max_loops is not None else defaults.max_loops,
        max_num_approaches=(
            args.max_num_approaches
            if args.max_num_approaches is not None
            else defaults.max_num_approaches
        ),
        propose_time_limit_per_session=(
            args.propose_time_limit_per_session
            or defaults.propose_time_limit_per_session
        ),
        implement_time_limit_per_session=(
            args.implement_time_limit_per_session
            or defaults.implement_time_limit_per_session
        ),
        runs_dir=args.runs_dir,
        run_id=getattr(args, "run_id", None),
        model=args.model,
        cost_limit=cost_limit,
        input_token_price=args.input_token_price,
        cache_creation_token_price=args.cache_creation_token_price,
        cache_read_token_price=args.cache_read_token_price,
        output_token_price=args.output_token_price,
        token_price_tiers=args.token_price_tiers,
        cost_currency=args.cost_currency,
        docker_image=args.docker_image,
        docker_network=getattr(args, "docker_network", defaults.docker_network),
        gpus=args.gpus,
        hidden_eval_dir=args.hidden_eval_dir or "",
        submission_format_path=args.submission_format or "",
        adapter_mode=args.adapter_mode,
        skip_prepare=args.skip_prepare,
    )


def build_resume_config(
    args: Any,
    *,
    run_dir: Path,
    explicit_fields: set[str],
) -> ResolvedRunConfig:
    """Resolve effective resume config and validate it against run metadata."""
    metadata = _load_metadata(run_dir)
    original = _metadata_config(metadata)
    candidate = build_new_run_config(args)

    values = dataclasses.asdict(candidate)
    report = ResumeConfigReport()
    rejected: list[str] = []

    for field_name in IMMUTABLE_CONFIG_FIELDS:
        saved = original.get(field_name, _MISSING)
        new_value = getattr(candidate, field_name)
        if field_name in explicit_fields:
            if saved is _MISSING:
                rejected.append(
                    f"Cannot validate immutable field {field_name}: original "
                    "run metadata is missing this field."
                )
                continue
            if not _values_equal(field_name, saved, new_value):
                rejected.append(_immutable_message(field_name, saved, new_value))
                continue
        elif saved is not _MISSING:
            values[field_name] = saved

    _validate_immutable_input_paths(args, metadata, rejected)
    _validate_resume_constraints(run_dir, values, original, explicit_fields, rejected)

    for field_name in sorted(MUTABLE_CONFIG_FIELDS):
        saved = original.get(field_name, _MISSING)
        new_value = values[field_name]
        if field_name in explicit_fields:
            if saved is not _MISSING and not _values_equal(field_name, saved, new_value):
                report.overridden_fields[field_name] = {
                    "old": saved,
                    "new": new_value,
                }
        else:
            report.defaulted_fields.append(field_name)

    if rejected:
        report.rejected_incompatibilities.extend(rejected)
        raise ResumeConfigError(rejected)

    config = Config(**values)
    object.__setattr__(config, "_resume_config_report", dataclasses.asdict(report))
    return ResolvedRunConfig(config=config, report=report)


def config_to_metadata(config: Config) -> dict[str, Any]:
    """Return only persistent Config fields, excluding runtime attributes."""
    return dataclasses.asdict(config)


def new_resume_event(report: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "resumed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if report:
        payload["config_report"] = report
    return payload


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.is_file():
        raise ResumeConfigError([
            f"Cannot resume {run_dir.name}: run_metadata.json is missing."
        ])
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeConfigError([
            f"Cannot resume {run_dir.name}: run_metadata.json is unreadable: {exc}"
        ]) from exc


def _metadata_config(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("config")
    if isinstance(raw, dict):
        return {k: raw[k] for k in _config_field_names() if k in raw}

    # Legacy fallback: older metadata stored a subset of Config fields at top level.
    return {k: metadata[k] for k in _config_field_names() if k in metadata}


def _config_field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(Config)}


def _validate_immutable_input_paths(
    args: Any,
    metadata: dict[str, Any],
    rejected: list[str],
) -> None:
    source_paths = metadata.get("source_input_paths")
    if not isinstance(source_paths, dict):
        source_paths = {}

    for field_name, arg_name in (("problem", "problem"), ("initial_code", "initial_code")):
        value = getattr(args, arg_name, None)
        if not value:
            continue
        saved = source_paths.get(field_name)
        if saved is None:
            rejected.append(
                f"Cannot validate immutable field {field_name}: original "
                "run metadata is missing this field."
            )
            continue
        if not _values_equal(field_name, saved, value):
            rejected.append(_immutable_message(field_name, saved, value))


def _validate_resume_constraints(
    run_dir: Path,
    values: dict[str, Any],
    original: dict[str, Any],
    explicit_fields: set[str],
    rejected: list[str],
) -> None:
    current_loop = _current_resumable_loop(run_dir)
    max_loops = int(values.get("max_loops") or 0)
    if max_loops and current_loop and max_loops < current_loop:
        rejected.append(
            "Cannot change constrained field max_loops: "
            f"old/current minimum={current_loop!r}, new={max_loops!r}; "
            "max_loops cannot be below the current/resumable loop."
        )

    if _past_prepare_stage(run_dir):
        saved = original.get("max_num_approaches", _MISSING)
        if saved is _MISSING:
            rejected.append(
                "Cannot validate constrained field max_num_approaches: original "
                "run metadata is missing this field."
            )
            return
        if "max_num_approaches" not in explicit_fields:
            values["max_num_approaches"] = saved
            return
        max_num_approaches = int(values.get("max_num_approaches") or 0)
        saved_int = int(saved)
        if max_num_approaches != saved_int:
            rejected.append(
                "Cannot change constrained field max_num_approaches after "
                "propose has started: "
                f"old={saved_int!r}, new={max_num_approaches!r}. "
                f"Update the resume script to use --max-num-approaches {saved_int}."
            )
    elif "max_num_approaches" not in explicit_fields:
        saved = original.get("max_num_approaches", _MISSING)
        if saved is not _MISSING:
            values["max_num_approaches"] = saved


def validate_resume_cost_floor(run_dir: Path, config: Config) -> None:
    """Reject resume configs whose cost limit is below already-spent cost."""
    if config.cost_limit is None:
        return
    spent = _spent_cost(run_dir, config)
    if spent is None or config.cost_limit >= spent:
        return
    raise ResumeConfigError([
        "Cannot change constrained field cost_limit: "
        f"already spent={spent:.6g} {config.cost_currency}, "
        f"new={config.cost_limit:.6g}. "
        "Increase --cost-limit or use --no-cost-limit."
    ])


def _current_resumable_loop(run_dir: Path) -> int:
    loops: list[int] = []
    state_path = run_dir / "workspace" / ".pipeline_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        value = state.get("current_loop_index")
        if isinstance(value, int):
            loops.append(value)
    except (OSError, json.JSONDecodeError):
        pass

    maps_dir = run_dir / "session_data" / "session_maps"
    if maps_dir.is_dir():
        for path in maps_dir.glob("loop_*_session_map.json"):
            match = re.match(r"loop_(\d+)_", path.name)
            if match:
                loops.append(int(match.group(1)))
        for path in maps_dir.glob("loop_*_*_session_map.json"):
            match = re.match(r"loop_(\d+)_", path.name)
            if match:
                loops.append(int(match.group(1)))

    round_state = run_dir / "workspace" / "round_state"
    if round_state.is_dir():
        for path in round_state.glob("round_*_approaches.jsonl"):
            match = re.match(r"round_(\d+)_approaches\.jsonl", path.name)
            if match:
                loops.append(int(match.group(1)))

    return max(loops) if loops else 0


def _past_prepare_stage(run_dir: Path) -> bool:
    state = _read_json(run_dir / "workspace" / ".pipeline_state.json")
    if isinstance(state, dict):
        loop_index = state.get("current_loop_index")
        if isinstance(loop_index, int) and loop_index > 0:
            return True
        if state.get("current_stage") in ("propose", "implement"):
            return True

    if _max_manifest_approach_count(run_dir) > 0:
        return True

    maps_dir = run_dir / "session_data" / "session_maps"
    if maps_dir.is_dir():
        for path in maps_dir.glob("*_session_map.json"):
            payload = _read_json(path)
            if (
                isinstance(payload, dict)
                and payload.get("stage") in ("propose", "implement")
            ):
                return True

    return False


def _max_manifest_approach_count(run_dir: Path) -> int:
    round_state = run_dir / "workspace" / "round_state"
    if not round_state.is_dir():
        return 0
    max_count = 0
    for path in list(round_state.glob("round_*_approaches.jsonl")) + [
        round_state / "current_round_approaches.jsonl"
    ]:
        if not path.is_file():
            continue
        max_count = max(max_count, _manifest_approach_count(path))
    return max_count


def _manifest_approach_count(path: Path) -> int:
    payload = read_manifest(path)
    if not payload:
        return 0
    approaches = payload.get("approaches")
    return len(approaches) if isinstance(approaches, list) else 0


def _spent_cost(run_dir: Path, config: Config) -> float | None:
    costs: list[float] = []
    for path in (
        run_dir / "workspace" / ".pipeline_state.json",
        run_dir / "run_summary.json",
    ):
        payload = _read_json(path)
        if isinstance(payload, dict):
            cost = _total_cost(payload.get("cost"))
            if cost is not None:
                costs.append(cost)
            calculated = _cost_from_usage(payload.get("token_usage"), config)
            if calculated is not None:
                costs.append(calculated)

    if not costs:
        calculated = _cost_from_usage(aggregate_token_usage(run_dir / "workspace"), config)
        if calculated is not None:
            costs.append(calculated)

    return max(costs) if costs else None


def _cost_from_usage(usage: Any, config: Config) -> float | None:
    if not _valid_usage(usage):
        return None
    tracker = TokenTracker(
        _input_price=config.input_token_price,
        _cache_creation_price=config.cache_creation_token_price,
        _cache_read_price=config.cache_read_token_price,
        _output_price=config.output_token_price,
        _price_tiers=config.token_price_tiers,
    )
    tracker.update_session("_resume_prior", usage)
    return tracker.calculate_cost()


def _valid_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    return any(
        isinstance(usage.get(key), int) and usage.get(key, 0) > 0
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )


def _total_cost(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    cost = value.get("total_cost")
    if not isinstance(cost, (int, float)):
        return None
    return float(cost)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _values_equal(field_name: str, old: Any, new: Any) -> bool:
    if field_name in _PATH_FIELDS and old and new:
        try:
            return Path(str(old)).expanduser().resolve() == Path(str(new)).expanduser().resolve()
        except OSError:
            pass
    return old == new


def _immutable_message(field_name: str, old: Any, new: Any) -> str:
    return (
        f"Cannot change immutable field {field_name}: "
        f"old={old!r}, new={new!r}."
    )
