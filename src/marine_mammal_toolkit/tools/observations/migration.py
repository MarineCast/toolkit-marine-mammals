from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools._core.persistence import checksum_path

V3_PRODUCTS = (
    "source_record_history.parquet",
    "source_records.parquet",
    "observations.parquet",
    "associations.parquet",
    "normalization_audit.parquet",
    "identity_resolution.parquet",
    "identity_aliases.parquet",
    "imputed_retrospective.parquet",
    "imputed_as_of.parquet",
    "facts",
    "counts",
    "dense",
    "manifests",
)

STATE_LAYOUT_MOVES = {
    "source_record_history.parquet": "state/source_history.parquet",
    "source_records.parquet": "state/source_current.parquet",
    "normalization_audit.parquet": "audit.parquet",
    "identity_resolution.parquet": "state/identity/assignments.parquet",
    "identity_aliases.parquet": "state/identity/aliases.parquet",
    "identity_lineage.parquet": "state/identity/lineage.parquet",
}


def migrate_state_layout(data_root: Path) -> tuple[Path, ...]:
    """Move existing sightings state into the canonical internal-state layout."""
    root = data_root / "processed/sightings/normalized"
    moves: list[tuple[Path, Path]] = []
    for old_name, new_name in STATE_LAYOUT_MOVES.items():
        source = root / old_name
        destination = root / new_name
        if not source.exists():
            continue
        if destination.exists():
            raise FileExistsError(
                f"State-layout destination already exists: {destination}"
            )
        moves.append((source, destination))
    moved: list[Path] = []
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        moved.append(destination)
    return tuple(moved)


def archive_v3_products(data_root: Path, migration_run: str | None = None) -> Path:
    source_root = data_root / "processed/sightings/normalized"
    run_id = migration_run or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = source_root / "legacy/v3" / run_id
    if destination.exists():
        raise FileExistsError(f"Migration archive already exists: {destination}")
    destination.mkdir(parents=True)
    moved = []
    for name in V3_PRODUCTS:
        source = source_root / name
        if not source.exists():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = checksum_path(source)
        os.replace(source, target)
        moved.append(
            {"source": str(source), "archive": str(target), "checksum": checksum}
        )
    atomic_write_json(
        destination / "migration_manifest.json",
        {"migration_run": run_id, "schema_from": "3", "schema_to": "4", "moved": moved},
    )
    return destination
