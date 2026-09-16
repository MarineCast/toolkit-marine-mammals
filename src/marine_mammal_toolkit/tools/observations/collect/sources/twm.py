"""Acquisition of explicitly supplied TWM CSV exports."""

from pathlib import Path
from shutil import copy2
from collections.abc import Iterable


def collect_twm_files(files: Iterable[Path], snapshot: Path) -> tuple[Path, ...]:
    """Copy source files into a new snapshot, rejecting colliding basenames."""
    files = tuple(Path(path) for path in files)
    if not files:
        raise FileNotFoundError("TWM collection requires at least one local CSV")
    if len({path.name for path in files}) != len(files):
        raise ValueError("TWM source filenames must be unique within a snapshot")
    for path in files:
        if not path.is_file() or path.suffix.lower() != ".csv":
            raise ValueError(f"TWM source must be an existing CSV: {path}")
        if (snapshot / path.name).exists():
            raise FileExistsError(snapshot / path.name)
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in files:
        copy2(path, snapshot / path.name)
    return tuple(snapshot / path.name for path in files)
