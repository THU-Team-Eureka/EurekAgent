"""Primary evaluator for the AlphaEvolve AC task."""

import argparse
import importlib.util
import json
import multiprocessing
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Standalone eval harness
# ---------------------------------------------------------------------------

DEFAULT_RUN_TIMEOUT = 120  # seconds


def _run_fn_worker(func, kwargs, result_q, error_q):
    try:
        result_q.put(func(**kwargs))
    except Exception as e:
        error_q.put((type(e).__name__, str(e)))


def _run_with_timeout(func, kwargs, timeout):
    result_q = multiprocessing.Queue()
    error_q = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_fn_worker, args=(func, kwargs, result_q, error_q)
    )
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        raise TimeoutError(f"Execution exceeded timeout of {timeout}s")
    if not error_q.empty():
        etype, emsg = error_q.get()
        raise RuntimeError(f"{etype}: {emsg}")
    if result_q.empty():
        raise RuntimeError("Function completed but no result was returned")
    return result_q.get()


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module at {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_json_results(results_dir, metrics, correct, error=None):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": error}, f, indent=4)
    print(f"Correctness and error status saved to {os.path.join(results_dir, 'correct.json')}")
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {os.path.join(results_dir, 'metrics.json')}")


def run_eval(
    program_path: str,
    results_dir: str,
    experiment_fn_name: str,
    num_runs: int,
    get_experiment_kwargs: Optional[Callable[[int], Dict[str, Any]]] = None,
    aggregate_metrics_fn: Optional[Callable[[List[Any]], Dict[str, Any]]] = None,
    validate_fn: Optional[Callable[[Any], Tuple[bool, Optional[str]]]] = None,
    default_metrics_on_error: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Run experiment, validate, aggregate, and save results."""
    effective_timeout = (
        DEFAULT_RUN_TIMEOUT if timeout_seconds is None
        else (None if timeout_seconds <= 0 else timeout_seconds)
    )
    overall_correct = True
    first_error: Optional[str] = None
    all_validation_errors: List[str] = []
    num_valid = num_invalid = 0
    all_results: List[Any] = []
    exec_times: List[float] = []

    try:
        module = _load_program(program_path)
        if not hasattr(module, experiment_fn_name):
            raise AttributeError(
                f"Experiment function '{experiment_fn_name}' not found in {program_path}"
            )
        experiment_fn = getattr(module, experiment_fn_name)

        for i in range(num_runs):
            kwargs = get_experiment_kwargs(i) if get_experiment_kwargs else {"seed": i + 1}
            t0 = time.perf_counter()
            try:
                if effective_timeout is not None:
                    print(f"Running with timeout: {effective_timeout}s")
                    result = _run_with_timeout(experiment_fn, kwargs, effective_timeout)
                else:
                    result = experiment_fn(**kwargs)
                dt = time.perf_counter() - t0
            except (TimeoutError, RuntimeError) as e:
                dt = time.perf_counter() - t0
                err = str(e)
                print(f"Run {i+1}/{num_runs} failed: {err}")
                num_invalid += 1
                overall_correct = False
                first_error = first_error or err
                all_validation_errors.append(err)
                exec_times.append(dt)
                continue

            all_results.append(result)
            exec_times.append(dt)

            if validate_fn:
                is_valid, verr = validate_fn(result)
                if not is_valid:
                    num_invalid += 1
                    overall_correct = False
                    if verr:
                        first_error = first_error or f"Validation failed: {verr}"
                        if verr not in all_validation_errors:
                            all_validation_errors.append(verr)
                else:
                    num_valid += 1
            print(f"Run {i+1}/{num_runs} completed in {dt:.2f} seconds")

        if aggregate_metrics_fn:
            metrics = aggregate_metrics_fn(all_results)
        else:
            metrics = {"num_successful_runs": len(all_results)}

        metrics["execution_time_mean"] = float(np.mean(exec_times)) if exec_times else 0.0
        metrics["execution_time_std"] = float(np.std(exec_times)) if exec_times else 0.0
        if validate_fn:
            metrics["num_valid_runs"] = num_valid
            metrics["num_invalid_runs"] = num_invalid
            metrics["all_validation_errors"] = all_validation_errors

    except Exception as e:
        print(f"Evaluation error: {e}")
        defaults = (default_metrics_on_error or {}).copy()
        metrics = {
            "combined_score": defaults.get("combined_score", 0.0),
            "execution_time_mean": 0.0,
            "execution_time_std": 0.0,
            "num_successful_runs": 0,
            "num_valid_runs": 0,
            "num_invalid_runs": 0,
            "all_validation_errors": [str(e)],
        }
        first_error = str(e)
        overall_correct = False

    _save_json_results(results_dir, metrics, overall_correct, first_error)
    return metrics, overall_correct, first_error


# ---------------------------------------------------------------------------
# Task-specific logic
# ---------------------------------------------------------------------------


def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluate coefficient sequence with security/validity checks.
    Returns np.inf for invalid inputs.
    Lower is better.
    """
    if not isinstance(sequence, list):
        return float(np.inf)
    if not sequence:
        return float(np.inf)

    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return float(np.inf)
        if np.isnan(x) or np.isinf(x):
            return float(np.inf)

    sequence = [float(x) for x in sequence]
    sequence = [max(0.0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = float(np.max(b_sequence))
    sum_a = float(np.sum(sequence))
    if sum_a < 0.01:
        return float(np.inf)

    return float(2.0 * n * max_b / (sum_a**2))


def validate_run_output(run_output: Any) -> Tuple[bool, Optional[str]]:
    """Validate output of run()."""
    try:
        if not isinstance(run_output, list):
            return False, "run() must return list[float]"
        if len(run_output) == 0:
            return False, "run() returned empty list"
        value = evaluate_sequence(run_output)
        if not np.isfinite(value):
            return False, "evaluate_sequence returned inf/nan"
        return True, None
    except Exception as exc:
        return False, str(exc)


def aggregate_alphaevolve_ac_metrics(results: List[list[float]]) -> Dict[str, Any]:
    """Aggregate metrics with best-only ranking."""
    if not results:
        return {
            "combined_score": 0.0,
            "public": {"best_value": None, "num_runs": 0},
            "private": {"all_values": []},
            "text_feedback": "No successful runs.",
        }

    values: List[float] = []
    lengths: List[int] = []
    best_sequence: Optional[list[float]] = None
    best_value = float(np.inf)

    for seq in results:
        val = evaluate_sequence(seq)
        values.append(float(val))
        lengths.append(len(seq))
        if val < best_value:
            best_value = float(val)
            best_sequence = seq

    combined_score = -best_value
    public = {
        "best_value": best_value,
        "best_length": len(best_sequence) if best_sequence is not None else None,
        "num_runs": len(results),
    }
    private = {
        "all_values": values,
        "all_lengths": lengths,
    }

    return {
        "combined_score": combined_score,
        "public": public,
        "private": private,
        "text_feedback": (
            "Lower evaluate_sequence value is better. "
            "combined_score = -best_value."
        ),
    }


def main_from_program(program_path: str, results_dir: str, num_experiment_runs: int = 1):
    """Run evaluation by executing a program file."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    print(f"Number of runs: {num_experiment_runs}")

    metrics, correct, error = run_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run",
        num_runs=num_experiment_runs,
        validate_fn=validate_run_output,
        aggregate_metrics_fn=aggregate_alphaevolve_ac_metrics,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error}")
    print(f"combined_score={metrics.get('combined_score')}")


def main_from_result(result_path: str, results_dir: str):
    """Validate and score a pre-computed result.json."""
    print(f"Evaluating result file: {result_path}")
    print(f"Saving results to: {results_dir}")

    with open(result_path) as f:
        data = json.load(f)

    sequence = data["sequence"]
    is_valid, verr = validate_run_output(sequence)
    if is_valid:
        metrics = aggregate_alphaevolve_ac_metrics([sequence])
        print("Validation passed.")
    else:
        metrics = {"combined_score": 0.0}
        print(f"Validation failed: {verr}")

    _save_json_results(results_dir, metrics, is_valid, verr)
    print(f"combined_score={metrics.get('combined_score')}")


def grade_submission(submission_path: str, context: dict[str, Any]) -> dict[str, Any]:
    """Secure evaluation entrypoint called by the grader server."""
    try:
        with open(submission_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {"score": 0.0, "valid": False, "message": f"Failed to read submission JSON: {exc}"}

    if not isinstance(data, dict):
        return {"score": 0.0, "valid": False, "message": "Submission must be a JSON object."}

    try:
        sequence = data["sequence"]
    except KeyError:
        return {"score": 0.0, "valid": False, "message": "Missing required key: sequence"}

    is_valid, err_msg = validate_run_output(sequence)
    if not is_valid:
        return {"score": float("inf"), "valid": False, "message": err_msg or "Validation failed"}

    value = evaluate_sequence(sequence)
    return {
        "score": float(value),
        "valid": True,
        "opt_target_met": False,
        "message": f"evaluate_sequence={value:.6f}",
        "public_metrics": {"best_value": float(value), "sequence_length": len(sequence)},
    }


def is_better(new_score: float, old_score: float) -> bool:
    return new_score < old_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AlphaEvolve AC task")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--num_experiment_runs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--from-result",
        type=str,
        default=None,
        help="Path to a result.json file to validate directly (skip program execution)",
    )
    args = parser.parse_args()
    if args.from_result:
        main_from_result(args.from_result, args.results_dir)
    else:
        main_from_program(
            program_path=args.program_path,
            results_dir=args.results_dir,
            num_experiment_runs=args.num_experiment_runs,
        )
