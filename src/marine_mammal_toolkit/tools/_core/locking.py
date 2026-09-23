"""Fail-fast, reentrant single-writer guards for local product workspaces."""

import json
import os
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_held = ContextVar("marine_mammal_workspace_locks", default=frozenset())


@contextmanager
def workspace_write_lock(root: str | Path):
    root = Path(root).expanduser().resolve()
    if root in _held.get():
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".marine-mammals-write.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Workspace has another writer or an interrupted run: {path}. "
            "Remove the lock only after confirming its process has stopped."
        ) from exc
    token = _held.set(_held.get() | {root})
    try:
        with handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        yield
    finally:
        _held.reset(token)
        path.unlink(missing_ok=True)
