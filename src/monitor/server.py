"""FastAPI web monitor server for EurekAgent loop runs.

Runs in a background thread alongside the TUI. Events are bridged from the
TUI's asyncio.Queue via a thread-safe queue.Queue so the SSE endpoint can
push real-time updates to the browser.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import queue
import socket
import sys
import threading
import time
import webbrowser

import uvicorn
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .data import (
    build_score_chart,
    load_approach_detail,
    load_conversation_detail,
    load_execution_log,
    load_run,
    load_run_overview,
)


def _try_load_is_better(run_dir: Path) -> Callable[[float, float], bool] | None:
    """Try to load is_better from evaluate.py via run_metadata.json."""
    import importlib.util
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        eval_dir = meta.get("hidden_eval_dir")
        if not eval_dir:
            return None
        eval_path = Path(eval_dir) / "evaluate.py"
        if not eval_path.is_file():
            return None
        spec = importlib.util.spec_from_file_location("_mon_is_better", eval_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "is_better", None)
        return fn if callable(fn) else None
    except Exception:
        return None

log = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
_MONITOR_PORT_CACHE = ".monitor_port"


def _resolve_run_dir(run_dir: Path, runs_dir: Path | None = None) -> Path:
    """Return the effective run directory.

    If *run_dir* is itself a run (contains ``run_metadata.json``), return it.
    Otherwise scan *runs_dir* (falling back to *run_dir*) for the most recent
    sub-directory that looks like a run, selected by name (run dirs are
    timestamped, so lexicographic order matches chronological order). If none
    is found, return *run_dir* unchanged so the caller can surface the error.
    """
    try:
        if run_dir.is_dir() and (run_dir / "run_metadata.json").is_file():
            return run_dir
    except OSError:
        pass
    search_dir = runs_dir or run_dir
    if search_dir.is_dir():
        candidates = []
        for d in search_dir.iterdir():
            try:
                if d.is_dir() and (d / "run_metadata.json").is_file():
                    candidates.append(d)
            except OSError:
                continue
        if candidates:
            candidates.sort(key=lambda d: d.name, reverse=True)
            return candidates[0]
    return run_dir


class MonitorState:
    """Per-instance state for a monitor server, isolating run_dir and events."""

    def __init__(self, run_dir: Path, runs_dir: Path | None = None, poll_interval: float = 1.0) -> None:
        self.run_dir = run_dir.resolve()
        self.runs_dir = runs_dir.resolve() if runs_dir else None
        self.poll_interval = poll_interval
        self.event_bridge: queue.Queue[dict[str, Any]] = queue.Queue()

    def set_run_dir(self, run_dir: Path) -> None:
        """Update the bound run directory (called once the pipeline creates it)."""
        self.run_dir = run_dir.resolve()

    def resolve_run_dir(self) -> Path:
        """Return the effective run directory.

        If run_dir points at a runs/ parent directory, find the latest run.
        Otherwise use run_dir directly.
        """
        return _resolve_run_dir(self.run_dir, self.runs_dir)


# Module-level default state (used by standalone CLI and as fallback).
_default_state: MonitorState | None = None


def push_event(event_type: str, event_data: dict[str, Any]) -> None:
    """Thread-safe: push an event from the TUI into the default monitor bridge."""
    if _default_state is not None:
        _default_state.event_bridge.put({"type": event_type, "data": event_data})


def set_run_dir(run_dir: Path) -> None:
    """Update the default monitor's run directory once the pipeline creates it."""
    if _default_state is not None:
        _default_state.set_run_dir(run_dir)


def create_app(
    run_dir: Path,
    runs_dir: Path | None = None,
    poll_interval: float = 1.0,
) -> FastAPI:
    """Create and configure a FastAPI app with its own isolated state."""
    state = MonitorState(run_dir, runs_dir=runs_dir, poll_interval=poll_interval)

    # Keep a module-level reference so push_event() can find the default instance.
    global _default_state
    _default_state = state

    app = FastAPI(title="EurekAgent Monitor")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        resolved = state.resolve_run_dir()
        return templates.TemplateResponse(
            request,
            "monitor.html",
            {"run_id": resolved.name},
        )

    @app.get("/logo-dark.png", response_class=FileResponse)
    async def logo_dark():
        return FileResponse(_TEMPLATE_DIR / "logo-dark.png", media_type="image/png")

    @app.get("/logo-light.png", response_class=FileResponse)
    async def logo_light():
        return FileResponse(_TEMPLATE_DIR / "logo-light.png", media_type="image/png")

    @app.get("/EurekAgent1.png", response_class=FileResponse)
    async def eureka_agent_logo():
        return FileResponse(_ASSETS_DIR / "EurekAgent1.png", media_type="image/png")

    @app.get("/api/run-data")
    async def run_data():
        """Full run data (backward-compatible, now uses overview)."""
        resolved = state.resolve_run_dir()
        try:
            run = load_run_overview(resolved)
        except FileNotFoundError:
            return {"run_data": {}, "score_chart": [], "global_best": None, "global_worst": None}

        is_better = _try_load_is_better(resolved)
        chart, global_best, global_worst = build_score_chart(
            run.get("loops", []), is_better=is_better
        )
        return {
            "run_data": run,
            "score_chart": chart,
            "global_best": global_best,
            "global_worst": global_worst,
        }

    @app.get("/api/run-overview")
    async def run_overview():
        """Lightweight overview data for the main dashboard poll."""
        return (await run_data())

    @app.get("/api/approach-detail/{approach_id}")
    async def approach_detail(approach_id: str):
        """Full detail for a single approach (loaded on demand)."""
        resolved = state.resolve_run_dir()
        detail = load_approach_detail(resolved, approach_id)
        if detail is None:
            return {"error": "Approach not found", "approach": None}
        return {"approach": detail}

    @app.get("/api/conversation/{conversation_key:path}")
    async def conversation_detail(conversation_key: str, since: int | None = None):
        """Load a single conversation transcript on demand.

        With ``?since=N`` returns only events at index >= N plus a ``total``
        cursor, for incremental polling of the currently-open conversation.
        """
        resolved = state.resolve_run_dir()
        conv = load_conversation_detail(resolved, conversation_key, since=since)
        if conv is None:
            return {"error": "Conversation not found", "conversation": None}
        return {"conversation": conv}

    @app.get("/api/execution-log")
    async def execution_log():
        resolved = state.resolve_run_dir()
        try:
            timeline = load_execution_log(resolved)
        except FileNotFoundError:
            timeline = []
        return {"timeline": timeline}

    @app.get("/api/events")
    async def events(request: Request):
        """SSE endpoint — pushes real-time events from the TUI bridge."""

        async def event_stream():
            while True:
                if await request.is_disconnected():
                    break

                # Drain the bridge queue (non-blocking, thread-safe)
                drained = False
                for _ in range(20):  # batch up to 20 events per tick
                    try:
                        event = state.event_bridge.get_nowait()
                        yield {"data": json.dumps(event, default=str)}
                        drained = True
                    except queue.Empty:
                        break

                if not drained:
                    # No events — send heartbeat keepalive
                    yield {"data": json.dumps({"type": "heartbeat"})}

                await asyncio.sleep(state.poll_interval)

        return EventSourceResponse(event_stream())

    return app


def _is_port_available(host: str, port: int) -> bool:
    """Return True if a TCP port can be bound right now."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_free_port(host: str) -> int:
    """Ask the OS for a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _choose_monitor_port(runs_dir: Path, host: str) -> int:
    """Reuse the cached monitor port while it remains available."""
    cache_path = runs_dir / _MONITOR_PORT_CACHE
    try:
        cached = int(cache_path.read_text(encoding="utf-8").strip())
        if 0 < cached < 65536 and _is_port_available(host, cached):
            return cached
    except (OSError, ValueError):
        pass

    port = _find_free_port(host)
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(f"{port}\n", encoding="utf-8")
    except OSError:
        log.warning("Failed to persist monitor port cache: %s", cache_path, exc_info=True)
    return port


def start_monitor_server(
    runs_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> int:
    """Start the monitor server in a background daemon thread.

    If *port* is 0, reuse the cached preferred port when available, otherwise
    ask the OS for a free port and persist it for future runs.
    Returns the actual port used.
    """
    if port == 0:
        port = _choose_monitor_port(runs_dir, host)

    app = create_app(runs_dir, runs_dir=runs_dir)
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )

    def _serve():
        try:
            asyncio.run(server.serve())
        except Exception:
            pass

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}"
    log.info("Web monitor: %s", url)
    print(f"  Web monitor: {url}")
    return port


def generate_snapshot(run_dir: Path) -> Path | None:
    """Generate a self-contained HTML snapshot of the monitor page.

    Embeds all data inline so the snapshot works without a live server.
    Returns the path to the generated file, or None on failure.
    """
    run_dir = _resolve_run_dir(run_dir.resolve())
    if not run_dir.is_dir():
        log.warning("Snapshot: run_dir not found: %s", run_dir)
        return None

    try:
        from .data import load_run, load_execution_log  # noqa: avoid circular at module level

        run_data = load_run(run_dir)
        is_better = _try_load_is_better(run_dir)
        chart, global_best, global_worst = build_score_chart(
            run_data.get("loops", []), is_better=is_better
        )
        timeline = load_execution_log(run_dir)

        approach_details: dict[str, dict | None] = {}
        for loop in run_data.get("loops", []):
            for a in loop.get("approaches", []):
                aid = a.get("approach_id", "")
                if aid and aid not in approach_details:
                    detail = load_approach_detail(run_dir, aid)
                    approach_details[aid] = detail

        conversations: dict[str, dict | None] = {}
        for loop in run_data.get("loops", []):
            conv_key = f"loop_{loop.get('loop_index', 0)}_propose"
            if conv_key not in conversations:
                conv = load_conversation_detail(run_dir, conv_key)
                conversations[conv_key] = conv
            if loop.get("loop_index") == 0:
                conv = load_conversation_detail(run_dir, "prepare")
                conversations["prepare"] = conv
            for a in loop.get("approaches", []):
                aid = a.get("approach_id", "")
                if aid and aid not in conversations:
                    conv = load_conversation_detail(run_dir, aid)
                    conversations[aid] = conv

        snapshot_data = {
            "run_data": run_data,
            "score_chart": chart,
            "global_best": global_best,
            "global_worst": global_worst,
            "timeline": timeline,
            "approach_details": approach_details,
            "conversations": conversations,
        }

        html_path = _TEMPLATE_DIR / "monitor.html"
        html = html_path.read_text(encoding="utf-8")

        html = html.replace("{{ run_id }}", run_dir.name)

        logo_path = _ASSETS_DIR / "EurekAgent1.png"
        if logo_path.is_file():
            logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            html = html.replace("/EurekAgent1.png", f"data:image/png;base64,{logo_data}")

        # Inject snapshot data INSIDE the main <script> tag, before the
        # snapshot-mode init check runs.  Placing it in a separate <script>
        # would fail because the init code executes synchronously during the
        # first <script> tag — the data wouldn't be visible yet.
        json_data = json.dumps(snapshot_data, default=str, ensure_ascii=False)
        inject_marker = "// ========== Snapshot mode =========="
        inject_script = f"window.__SNAPSHOT_DATA__ = {json_data};\n"
        html = html.replace(inject_marker, inject_script + inject_marker)

        snapshot_path = run_dir / "monitor_snapshot.html"
        snapshot_path.write_text(html, encoding="utf-8")
        log.info("Monitor snapshot saved: %s", snapshot_path)
        return snapshot_path

    except Exception:
        log.error("Failed to generate monitor snapshot", exc_info=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="EurekAgent web monitor server")
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Path to a specific run directory (e.g. runs/20260426_120000)",
    )
    parser.add_argument(
        "--runs-dir", type=str, default=None,
        help="Path to runs/ parent directory; auto-selects latest run",
    )
    parser.add_argument(
        "--snapshot", action="store_true",
        help="Generate a static HTML snapshot instead of starting a server",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        runs_dir = Path(args.runs_dir) if args.runs_dir else None
    elif args.runs_dir:
        run_dir = Path(args.runs_dir)
        runs_dir = run_dir
    else:
        print("Error: specify --run-dir or --runs-dir", file=sys.stderr)
        sys.exit(1)

    if not run_dir.is_dir():
        print(f"Error: directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    if args.snapshot:
        path = generate_snapshot(run_dir)
        if path:
            print(f"Snapshot saved: {path}")
        else:
            print("Failed to generate snapshot", file=sys.stderr)
            sys.exit(1)
        return

    app = create_app(run_dir, runs_dir=runs_dir)
    url = f"http://127.0.0.1:{args.port}"
    print(f"EurekAgent Monitor: {url}")

    def _open():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
