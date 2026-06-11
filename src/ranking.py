"""Ranking: score-based sorting using is_better from evaluate.py."""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable
from urllib import request

from .history import load_ranked_history, write_json

log = logging.getLogger(__name__)


def _load_is_better(hidden_eval_dir: str) -> Callable[[float, float], bool]:
    """Load is_better from evaluate.py."""
    eval_path = Path(hidden_eval_dir) / "evaluate.py"
    spec = importlib.util.spec_from_file_location("_ranking_is_better", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "is_better", None)
    if not callable(fn):
        raise RuntimeError(
            f"{eval_path} must define `is_better(new_score, old_score) -> bool`"
        )
    return fn


def grader_is_better_client(
    grader_url: str,
    token: str,
) -> Callable[[float, float], bool]:
    """Return an is_better callable backed by the secure grader service."""
    endpoint = grader_url.rstrip("/") + "/is_better"

    def _is_better(new_score: float, old_score: float) -> bool:
        body = json.dumps({
            "new_score": new_score,
            "old_score": old_score,
        }).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        opener = request.build_opener(request.ProxyHandler({}))
        with opener.open(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return bool(payload.get("is_better"))

    return _is_better


def _score_rank(
    entries: list[dict[str, Any]],
    is_better: Callable[[float, float], bool],
) -> list[dict[str, Any]]:
    """Rank entries by score using is_better.

    Order: trusted+completed -> untrusted+completed -> failed.
    Within each tier, best score first per is_better semantics.
    """

    def _compare(a, b):
        def tier(e):
            if e.get("status") != "completed":
                return 2
            return 0 if e.get("trusted") else 1

        ta, tb = tier(a), tier(b)
        if ta != tb:
            return ta - tb
        sa = a.get("score") if a.get("score") is not None else 0.0
        sb = b.get("score") if b.get("score") is not None else 0.0
        if sa == sb:
            return 0
        return -1 if is_better(sa, sb) else 1

    return sorted(entries, key=functools.cmp_to_key(_compare))


def rank_history(
    workspace_dir: Path,
    *,
    history_path: Path,
    is_better: Callable[[float, float], bool],
) -> None:
    """Rank history entries by score using is_better from evaluate.py."""
    entries = load_ranked_history(history_path)
    if not entries:
        return
    write_json(history_path, {"entries": _score_rank(entries, is_better)})
