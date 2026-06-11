"""Adapters for the Claude CLI.

``StreamAdapter`` (in ``stream_adapter.py``) wraps ``claude -p --output-format stream-json``.
``PtyAdapter`` (in ``pty_adapter.py``) wraps ``claude`` interactive mode with JSONL tailing.
Both reuse the same ``SessionRequest``/``SessionResult`` protocol.
"""

from .pty_adapter import PtyAdapter
from .stream_adapter import StreamAdapter
from .protocol import SessionEvent, SessionRequest, SessionResult

__all__ = ["StreamAdapter", "PtyAdapter", "SessionEvent", "SessionRequest", "SessionResult"]
