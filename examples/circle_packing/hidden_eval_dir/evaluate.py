"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import os
import argparse
import importlib.util
import json
import multiprocessing
import time
import numpy as np
from typing import Any, Callable, Tuple, Optional, List, Dict


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


def format_centers_string(centers: np.ndarray) -> str:
    """Formats circle centers into a multi-line string for display."""
    return "\n".join(
        [
            f"  centers[{i}] = ({x_coord:.4f}, {y_coord:.4f})"
            for i, (x_coord, y_coord) in enumerate(centers)
        ]
    )


def adapted_validate_packing(
    run_output: Tuple[np.ndarray, np.ndarray, float],
    atol=1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates circle packing results based on the output of 'run_packing'.

    Args:
        run_output: Tuple (centers, radii, reported_sum) from run_packing.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    centers, radii, reported_sum = run_output
    msg = "The circles are placed correctly. There are no overlaps or any circles outside the unit square."
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    n_expected = 26
    if centers.shape != (n_expected, 2):
        msg = (
            f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}"
        )
        return False, msg
    if radii.shape != (n_expected,):
        msg = f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}"
        return False, msg

    if np.any(radii < 0):
        negative_indices = np.where(radii < 0)[0]
        msg = f"Negative radii found for circles at indices: {negative_indices}"
        return False, msg

    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        msg = (
            f"Sum of radii ({np.sum(radii):.6f}) does not match "
            f"reported ({reported_sum:.6f})"
        )
        return False, msg

    for i in range(n_expected):
        x, y = centers[i]
        r = radii[i]
        is_outside = (
            x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol
        )
        if is_outside:
            msg = (
                f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square."
            )
            return False, msg

    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                msg = (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )
                return False, msg
    return True, msg


def get_circle_packing_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for circle packing runs (none needed)."""
    return {}


def aggregate_circle_packing_metrics(
    results: List[Tuple[np.ndarray, np.ndarray, float]], results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for circle packing. Assumes num_runs=1.
    Saves extra.npz with detailed packing information.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    centers, radii, reported_sum = results[0]

    public_metrics = {
        "centers_str": format_centers_string(centers),
        "num_circles": centers.shape[0],
    }
    private_metrics = {
        "reported_sum_of_radii": float(reported_sum),
    }
    metrics = {
        "combined_score": float(reported_sum),
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(
            extra_file,
            centers=centers,
            radii=radii,
            reported_sum=reported_sum,
        )
        print(f"Detailed packing data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def _print_metrics(metrics: Dict[str, Any]):
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


def main_from_program(program_path: str, results_dir: str):
    """Runs the circle packing evaluation by executing a program file."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = 1

    # Define a nested function to pass results_dir to the aggregator
    def _aggregator_with_context(
        r: List[Tuple[np.ndarray, np.ndarray, float]],
    ) -> Dict[str, Any]:
        return aggregate_circle_packing_metrics(r, results_dir)

    metrics, correct, error_msg = run_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_packing",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_circle_packing_kwargs,
        validate_fn=adapted_validate_packing,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")
    _print_metrics(metrics)


def main_from_result(result_path: str, results_dir: str):
    """Validate and score a pre-computed result.json."""
    print(f"Evaluating result file: {result_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    with open(result_path) as f:
        data = json.load(f)

    centers = np.array(data["centers"])
    radii = np.array(data["radii"])
    reported_sum = float(data["sum_radii"])
    run_output = (centers, radii, reported_sum)

    is_valid, verr = adapted_validate_packing(run_output)
    if is_valid:
        metrics = aggregate_circle_packing_metrics([run_output], results_dir)
        print("Validation passed.")
    else:
        metrics = {"combined_score": 0.0}
        print(f"Validation failed: {verr}")

    _save_json_results(results_dir, metrics, is_valid, verr)
    _print_metrics(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Circle packing evaluator"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_packing')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parser.add_argument(
        "--from-result",
        type=str,
        default=None,
        help="Path to a result.json file to validate directly (skip program execution)",
    )
    parsed_args = parser.parse_args()
    if parsed_args.from_result:
        main_from_result(parsed_args.from_result, parsed_args.results_dir)
    else:
        main_from_program(parsed_args.program_path, parsed_args.results_dir)


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
        centers = np.asarray(data["centers"], dtype=float)
        radii = np.asarray(data["radii"], dtype=float)
    except KeyError as exc:
        return {"score": 0.0, "valid": False, "message": f"Missing required key: {exc.args[0]}"}
    except Exception as exc:
        return {"score": 0.0, "valid": False, "message": f"Could not parse centers/radii: {exc}"}

    reported_sum = data.get("sum_radii")
    if reported_sum is not None:
        try:
            reported_sum = float(reported_sum)
        except Exception:
            return {"score": 0.0, "valid": False, "message": "Optional sum_radii must be numeric."}
    else:
        reported_sum = float(np.sum(radii))

    run_output = (centers, radii, reported_sum)
    is_valid, err_msg = adapted_validate_packing(run_output)

    if is_valid:
        return {
            "score": float(np.sum(radii)),
            "valid": True,
            "opt_target_met": False,
            "message": err_msg,
            "public_metrics": {"num_circles": int(centers.shape[0])},
        }
    return {"score": 0.0, "valid": False, "message": err_msg}


def is_better(new_score: float, old_score: float) -> bool:
    return new_score > old_score
