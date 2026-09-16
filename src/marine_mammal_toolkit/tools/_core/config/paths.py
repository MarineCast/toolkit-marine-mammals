"""Explicit workspace roots, independent of installation and current directory."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_ROOT: ContextVar[Path | None] = ContextVar("marine_mammals_workspace", default=None)


@contextmanager
def workspace(root: str | Path):
    """Bind a data workspace for a synchronous or asynchronous call context."""
    token = _ROOT.set(Path(root).expanduser().resolve())
    try:
        yield _ROOT.get()
    finally:
        _ROOT.reset(token)


def current_workspace() -> Path | None:
    return _ROOT.get()


def project_root() -> Path:
    root = current_workspace()
    if root is None:
        raise ValueError(
            "An explicit workspace root is required; use workspace(root) or --workspace-root."
        )
    return root


def resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (project_root() / candidate).resolve()
    )


def resolve_config_include(config_path: str | Path, include_path: str | Path) -> Path:
    candidate = Path(include_path).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (Path(config_path).parent / candidate).resolve()
    )
