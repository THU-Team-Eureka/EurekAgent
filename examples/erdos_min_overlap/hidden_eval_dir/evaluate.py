"""Primary evaluator for Erdos minimum overlap task."""

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


def verify_c5_solution(
    h_values: np.ndarray,
    c5_achieved: float,
    n_points: int,
) -> float:
    """Validate feasibility and C5 consistency."""
    if not isinstance(h_values, np.ndarray):
        h_values = np.array(h_values, dtype=np.float64)

    if h_values.ndim != 1:
        raise ValueError(f"h_values must be 1D, got shape {h_values.shape}")
    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf")

    if np.any(h_values < 0.0) or np.any(h_values > 1.0):
        raise ValueError(
            f"h_values must be in [0,1], got range [{h_values.min()}, {h_values.max()}]"
        )

    target_sum = n_points / 2.0
    current_sum = float(np.sum(h_values))
    if abs(current_sum - target_sum) > 1e-8:
        # Match discover behavior: normalize if needed.
        if current_sum <= 0.0:
            raise ValueError("sum(h_values) must be positive")
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0.0) or np.any(h_values > 1.0):
            raise ValueError(
                "After normalization, h_values fell outside [0,1]"
            )

    dx = 2.0 / n_points
    correlation = np.correlate(h_values, 1.0 - h_values, mode="full") * dx
    computed_c5 = float(np.max(correlation))

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, float(c5_achieved), atol=1e-4):
        raise ValueError(
            f"C5 mismatch: reported={float(c5_achieved):.8f}, computed={computed_c5:.8f}"
        )

    return computed_c5


def evaluate_erdos_solution(
    h_values: np.ndarray,
    c5_bound: float,
    n_points: int,
) -> float:
    return verify_c5_solution(h_values, c5_bound, n_points)


def validate_run_output(
    run_output: Tuple[np.ndarray, float, int],
) -> Tuple[bool, Optional[str]]:
    """Validate run() output format and numerical correctness."""
    try:
        if not isinstance(run_output, (tuple, list)) or len(run_output) != 3:
            return False, "run() must return a tuple/list of (h_values, c5_bound, n_points)"

        h_values, c5_bound, n_points = run_output
        n_points = int(n_points)
        c5_bound = float(c5_bound)

        _ = evaluate_erdos_solution(h_values, c5_bound, n_points)
        return True, None
    except Exception as exc:
        return False, str(exc)


def aggregate_erdos_metrics(
    results: List[Tuple[np.ndarray, float, int]],
) -> Dict[str, Any]:
    """Aggregate metrics across repeated runs."""
    if not results:
        return {
            "combined_score": 0.0,
            "public": {"best_c5": None, "num_runs": 0},
            "private": {"all_c5": []},
            "text_feedback": "No successful runs.",
        }

    c5_values: List[float] = []
    n_points_values: List[int] = []

    for h_values, c5_bound, n_points in results:
        c5 = evaluate_erdos_solution(h_values, float(c5_bound), int(n_points))
        c5_values.append(float(c5))
        n_points_values.append(int(n_points))

    best_c5 = float(np.min(c5_values))

    # Maximizes combined_score, so rank by minimizing best_c5.
    combined_score = -best_c5

    public = {
        "best_c5": best_c5,
        "n_points_mean": float(np.mean(n_points_values)),
        "num_runs": len(results),
    }
    private = {
        "all_c5": c5_values,
        "all_n_points": n_points_values,
        "all_h_values": [res[0].tolist() for res in results],
    }

    return {
        "combined_score": combined_score,
        "public": public,
        "private": private,
        "text_feedback": (
            "Objective is to minimize C5. Higher combined_score means lower C5 "
            "because combined_score = -best_c5."
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
        aggregate_metrics_fn=aggregate_erdos_metrics,
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

    h_values = np.array(data["h_values"], dtype=np.float64)
    c5_bound = float(data["c5_bound"])
    n_points = int(data["n_points"])
    run_output = (h_values, c5_bound, n_points)

    is_valid, verr = validate_run_output(run_output)
    if is_valid:
        metrics = aggregate_erdos_metrics([run_output])
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
        h_values = np.asarray(data["h_values"], dtype=np.float64)
        c5_bound = float(data["c5_bound"])
        n_points = int(data["n_points"])
    except KeyError as exc:
        return {"score": 0.0, "valid": False, "message": f"Missing required key: {exc.args[0]}"}
    except Exception as exc:
        return {"score": 0.0, "valid": False, "message": f"Could not parse h_values/c5_bound/n_points: {exc}"}

    run_output = (h_values, c5_bound, n_points)
    is_valid, err_msg = validate_run_output(run_output)
    if not is_valid:
        return {"score": float("inf"), "valid": False, "message": err_msg or "Validation failed"}

    return {
        "score": float(c5_bound),
        "valid": True,
        "opt_target_met": False,
        "message": f"C5={c5_bound:.8f}, n_points={n_points}",
        "public_metrics": {"best_c5": float(c5_bound), "n_points": n_points},
    }


def is_better(new_score: float, old_score: float) -> bool:
    return new_score < old_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Erdos minimum overlap task"
    )
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
