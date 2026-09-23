"""Materialize the consumer-facing killer-whale sightings product."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    resolve_sightings_release_artifact,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    resolve_sightings_release_manifest,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    validate_sightings_release,
)
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path
from marine_mammal_toolkit.tools._core.locking import workspace_write_lock

PRODUCT_SCHEMA_VERSION = "2"
DATE_COLUMN = "SIGHTING_DATE_UTC"
ID_COLUMN = "OBSERVATION_ID"
SIGHTINGS_RAW_RELATIVE = Path("raw")
SIGHTINGS_ROOT_RELATIVE = Path("processed/sightings")
SIGHTINGS_NORMALIZED_RELATIVE = SIGHTINGS_ROOT_RELATIVE / "normalized"
SIGHTINGS_IMPUTED_RELATIVE = SIGHTINGS_ROOT_RELATIVE / "imputed"
SIGHTINGS_FINAL_RELATIVE = SIGHTINGS_ROOT_RELATIVE / "final"
# Compatibility name retained for callers that used the original final-product constant.
SIGHTINGS_PROCESSED_RELATIVE = SIGHTINGS_FINAL_RELATIVE
DEFAULT_MAX_SIGHTINGS_GROWTH_FRACTION = 0.10


def build_sightings_report_html(**kwargs):
    # Reading already-built products does not require the optional report stack.
    from .report import build_sightings_report_html as build

    return build(**kwargs)


def _single_writer(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with workspace_write_lock(kwargs["product_root"]):
            return function(*args, **kwargs)

    return guarded


@dataclass(frozen=True)
class SightingsProductLayout:
    """Filesystem contract owned by the high-level sightings product workflow."""

    product_root: Path
    raw_root: Path
    sightings_root: Path
    normalized_root: Path
    imputed_root: Path
    final_root: Path

    def prepare(self) -> None:
        """Create the durable product directories before stage execution."""

        for path in (
            self.raw_root,
            self.normalized_root,
            self.imputed_root,
            self.final_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def sightings_product_layout(product_root: str | Path) -> SightingsProductLayout:
    """Resolve every durable sightings location from one application data root."""

    root = Path(product_root).expanduser().resolve()
    return SightingsProductLayout(
        product_root=root,
        raw_root=root / SIGHTINGS_RAW_RELATIVE,
        sightings_root=root / SIGHTINGS_ROOT_RELATIVE,
        normalized_root=root / SIGHTINGS_NORMALIZED_RELATIVE,
        imputed_root=root / SIGHTINGS_IMPUTED_RELATIVE,
        final_root=root / SIGHTINGS_FINAL_RELATIVE,
    )


@dataclass(frozen=True)
class KillerWhaleSightingsProduct:
    """Stable aliases and the immutable dated generation they reference."""

    release_manifest: Path
    dated_root: Path
    composite_path: Path
    imputed_path: Path
    model_manifest_path: Path
    report_path: Path


@dataclass(frozen=True)
class SightingsArtifactRetention:
    """Result of the guarded current-generation retention policy."""

    applied: bool
    reason: str
    previous_row_count: int | None
    current_row_count: int
    growth_count: int | None
    growth_fraction: float | None
    max_growth_fraction: float
    removed_paths: tuple[Path, ...]

    def as_dict(self, *, product_root: str | Path) -> dict[str, Any]:
        root = Path(product_root).expanduser().resolve()
        return {
            "applied": self.applied,
            "reason": self.reason,
            "previous_row_count": self.previous_row_count,
            "current_row_count": self.current_row_count,
            "growth_count": self.growth_count,
            "growth_fraction": self.growth_fraction,
            "max_growth_fraction": self.max_growth_fraction,
            "removed_paths": [_relative(path, root) for path in self.removed_paths],
        }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Product artifact escapes its data root: {path}") from exc


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if checksum_path(temporary, logical_name=source.name) != checksum_path(source):
            raise OSError(f"Copied product checksum mismatch: {destination}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _table_identity(path: Path) -> tuple[dict[str, Any], list[str]]:
    table = pq.read_table(path, columns=[ID_COLUMN, DATE_COLUMN])
    identifiers = table[ID_COLUMN].to_pylist()
    if any(value is None for value in identifiers):
        raise ValueError(f"Sightings product contains null {ID_COLUMN}: {path}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Sightings product contains duplicate {ID_COLUMN}: {path}")
    dates = table[DATE_COLUMN].combine_chunks()
    bounds = pc.min_max(dates).as_py() if len(dates) else {"min": None, "max": None}

    def render(value: Any) -> str | None:
        return value.isoformat() if value is not None else None

    return (
        {
            "row_count": table.num_rows,
            "checksum": checksum_path(path),
            "schema": str(pq.read_schema(path)),
            "temporal_grain": "daily",
            "date_column": DATE_COLUMN,
            "observed_start": render(bounds["min"]),
            "observed_through": render(bounds["max"]),
        },
        [str(value) for value in identifiers],
    )


def _validate_existing_generation(root: Path, release_id: str) -> None:
    manifest = root / "imputation-model-manifest.json"
    if not manifest.is_file():
        raise FileExistsError(f"Incomplete dated sightings product exists: {root}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("release_id") != release_id:
        raise FileExistsError(
            f"Dated sightings product has a different release: {root}"
        )
    for name, key in (
        ("composite-sightings.parquet", "composite"),
        ("imputed-sightings.parquet", "imputed"),
    ):
        path = root / name
        expected = str(payload["tables"][key]["checksum"])
        if not path.is_file() or checksum_path(path) != expected:
            raise ValueError(f"Dated sightings product checksum mismatch: {path}")
    if payload.get("report"):
        report = root / "sightings-report.html"
        if (
            not report.is_file()
            or checksum_path(report) != payload["report"]["checksum"]
        ):
            raise ValueError(f"Dated sightings report checksum mismatch: {report}")


def _current_product_metadata(
    product_root: str | Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]] | None:
    root = Path(product_root).expanduser().resolve()
    pointer_path = root / SIGHTINGS_PROCESSED_RELATIVE / "latest.json"
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    generation_value = pointer.get("generation")
    if not generation_value:
        raise ValueError(f"Sightings product pointer has no generation: {pointer_path}")
    generation = (root / str(generation_value)).resolve()
    _relative(generation, root)
    manifest_path = generation / "imputation-model-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Current sightings product manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        pointer.get("manifest_checksum")
        and checksum_path(manifest_path) != pointer["manifest_checksum"]
    ):
        raise ValueError(
            f"Current sightings manifest checksum mismatch: {manifest_path}"
        )
    if manifest.get("release_id") != pointer.get("release_id"):
        raise ValueError(
            f"Current sightings product release does not match {pointer_path}"
        )
    return pointer_path, pointer, generation, manifest


def current_sightings_row_count(product_root: str | Path) -> int | None:
    """Return the current stable composite row count, or ``None`` before first build."""

    metadata = _current_product_metadata(product_root)
    if metadata is None:
        return None
    _, _, _, manifest = metadata
    try:
        row_count = int(manifest["tables"]["composite"]["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Current sightings product has no valid composite row count"
        ) from exc
    if row_count < 0:
        raise ValueError("Current sightings product row count cannot be negative")
    return row_count


def resolve_sightings_product(
    product_root: str | Path, *, verify_report: bool = True
) -> KillerWhaleSightingsProduct:
    """Read the authoritative pointer once and resolve one immutable generation.

    Flat files are compatibility copies. Concurrent readers must use this resolver
    (or the generation paths in latest.json), never combine independent flat reads.
    """
    root = Path(product_root).expanduser().resolve()
    metadata = _current_product_metadata(root)
    if metadata is None:
        raise FileNotFoundError(f"No current sightings product in {root}")
    _, pointer, generation, manifest = metadata
    _validate_existing_generation(generation, str(pointer["release_id"]))
    report = root / str(pointer.get("report", ""))
    _relative(report, root)
    if verify_report and (
        not report.is_file() or checksum_path(report) != pointer.get("report_checksum")
    ):
        raise ValueError("Current sightings report checksum mismatch")
    release = root / str(manifest["release_manifest"])
    _relative(release, root)
    return KillerWhaleSightingsProduct(
        release,
        generation,
        generation / "composite-sightings.parquet",
        generation / "imputed-sightings.parquet",
        generation / "imputation-model-manifest.json",
        report,
    )


def _retention_decision(
    *,
    previous_row_count: int | None,
    current_row_count: int,
    max_growth_fraction: float,
) -> tuple[bool, str, int | None, float | None]:
    if max_growth_fraction < 0:
        raise ValueError("max_growth_fraction must be non-negative")
    if previous_row_count is None:
        return False, "no_previous_product", None, None
    if previous_row_count < 0:
        raise ValueError("previous_row_count must be non-negative")
    growth_count = current_row_count - previous_row_count
    if growth_count < 0:
        return False, "current_count_below_previous", growth_count, None
    if previous_row_count == 0:
        if current_row_count == 0:
            return True, "retention_gate_passed", growth_count, 0.0
        return False, "previous_count_zero", growth_count, None
    growth_fraction = growth_count / previous_row_count
    if growth_fraction > max_growth_fraction:
        return False, "growth_exceeds_threshold", growth_count, growth_fraction
    return True, "retention_gate_passed", growth_count, growth_fraction


def _remove_retained_artifact(path: Path, *, expected_parent: Path) -> None:
    expected_parent = expected_parent.resolve()
    if path.parent.resolve() != expected_parent:
        raise ValueError(
            f"Refusing to remove artifact outside {expected_parent}: {path}"
        )
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise ValueError(f"Expected retained artifact directory: {path}")


@_single_writer
def prune_prior_sightings_artifacts(
    *,
    product_root: str | Path,
    previous_row_count: int | None,
    max_growth_fraction: float = DEFAULT_MAX_SIGHTINGS_GROWTH_FRACTION,
    keep_generations: int = 3,
    pinned_release_ids: tuple[str, ...] = (),
) -> SightingsArtifactRetention:
    """Keep one accepted sightings generation after conservative count checks.

    The current product must contain at least as many observations as the prior
    accepted product, and its relative increase must not exceed the configured
    threshold. A failed gate preserves every prior generation for inspection.
    Raw provider snapshots and compact stage manifests are outside this policy.
    """

    root = Path(product_root).expanduser().resolve()
    if keep_generations < 1:
        raise ValueError("keep_generations must be at least 1")
    pins_path = root / SIGHTINGS_PROCESSED_RELATIVE / "pins.json"
    existing_pins = json.loads(pins_path.read_text()) if pins_path.exists() else []
    if not isinstance(existing_pins, list) or not all(
        isinstance(item, str) for item in existing_pins
    ):
        raise ValueError(f"Invalid release pin list: {pins_path}")
    pinned_release_ids = tuple(sorted(set(existing_pins) | set(pinned_release_ids)))
    metadata = _current_product_metadata(root)
    if metadata is None:
        raise FileNotFoundError(
            "Cannot apply retention before a sightings product exists"
        )
    _, pointer, current_generation, manifest = metadata
    current_row_count = current_sightings_row_count(root)
    assert current_row_count is not None
    apply_retention, reason, growth_count, growth_fraction = _retention_decision(
        previous_row_count=previous_row_count,
        current_row_count=current_row_count,
        max_growth_fraction=max_growth_fraction,
    )

    processed = root / SIGHTINGS_PROCESSED_RELATIVE
    if pinned_release_ids:
        known = {
            path.name for path in (processed / "by-date").glob("*/*") if path.is_dir()
        }
        if set(pinned_release_ids) - known:
            raise ValueError(
                f"Pinned releases do not exist: {sorted(set(pinned_release_ids) - known)}"
            )
        atomic_write_json(pins_path, list(pinned_release_ids), overwrite=True)
    stable_composite = current_generation / "composite-sightings.parquet"
    expected_composite_checksum = str(manifest["tables"]["composite"]["checksum"])
    if (
        not stable_composite.is_file()
        or checksum_path(stable_composite) != expected_composite_checksum
    ):
        raise ValueError(
            f"Stable composite does not match the current generation: {stable_composite}"
        )

    removed: list[Path] = []
    if apply_retention:
        release_manifest = (root / str(manifest["release_manifest"])).resolve()
        _relative(release_manifest, root)
        release_report = validate_sightings_release(release_manifest)
        release_report.require_valid()

        release_id = str(pointer["release_id"])
        releases_root = release_manifest.parents[2]
        release_generations = releases_root / "generations"
        current_release = release_generations / release_id
        if release_manifest.parent != current_release:
            raise ValueError(
                f"Current release manifest is not in the expected generation: {release_manifest}"
            )

        model_root = processed.parent / "imputed/models/sighting_imputation"
        fit_run_id = str(manifest["imputation_model"].get("fit_run_id") or "")
        current_model = model_root / fit_run_id
        model_pointer = model_root / "latest.json"
        if not fit_run_id or not current_model.is_dir() or not model_pointer.is_file():
            raise FileNotFoundError(
                "Current mutable imputation model run is unavailable; prior models were retained"
            )
        model_pointer_payload = json.loads(model_pointer.read_text(encoding="utf-8"))
        if model_pointer_payload.get("fit_run_id") != fit_run_id:
            raise ValueError(
                "Mutable imputation model pointer does not match the current product"
            )

        by_date_root = processed / "by-date"
        generations = sorted(
            (
                path
                for day in by_date_root.iterdir()
                if day.is_dir()
                for path in day.iterdir()
                if path.is_dir()
            ),
            key=lambda path: (path.parent.name, path.stat().st_mtime_ns),
            reverse=True,
        )
        existing_ids = {path.name for path in generations}
        unknown_pins = set(pinned_release_ids) - existing_ids
        if unknown_pins:
            raise ValueError(f"Pinned releases do not exist: {sorted(unknown_pins)}")
        atomic_write_json(pins_path, list(pinned_release_ids), overwrite=True)
        recent_ids = {release_id}
        for path in generations:
            if len(recent_ids) >= keep_generations:
                break
            recent_ids.add(path.name)
        protected_ids = recent_ids | set(pinned_release_ids)
        protected_models = {fit_run_id}
        for path in generations:
            if path.name in protected_ids:
                saved = json.loads(
                    (path / "imputation-model-manifest.json").read_text()
                )
                model_id = saved.get("imputation_model", {}).get("fit_run_id")
                if model_id:
                    protected_models.add(str(model_id))
        for date_root in sorted(
            path for path in by_date_root.iterdir() if path.is_dir()
        ):
            for generation in sorted(
                path for path in date_root.iterdir() if path.is_dir()
            ):
                if generation.name in protected_ids:
                    continue
                _remove_retained_artifact(generation, expected_parent=date_root)
                removed.append(generation)
            if not any(path.is_dir() for path in date_root.iterdir()):
                shutil.rmtree(date_root)

        for generation in sorted(
            path for path in release_generations.iterdir() if path.is_dir()
        ):
            if generation.name in protected_ids:
                continue
            _remove_retained_artifact(generation, expected_parent=release_generations)
            removed.append(generation)

        for model_run in sorted(path for path in model_root.iterdir() if path.is_dir()):
            if model_run.name in protected_models:
                continue
            _remove_retained_artifact(model_run, expected_parent=model_root)
            removed.append(model_run)

        validate_sightings_release(release_manifest).require_valid()

    retention = SightingsArtifactRetention(
        applied=apply_retention,
        reason=reason,
        previous_row_count=previous_row_count,
        current_row_count=current_row_count,
        growth_count=growth_count,
        growth_fraction=growth_fraction,
        max_growth_fraction=max_growth_fraction,
        removed_paths=tuple(removed),
    )
    payload = {
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_id": pointer["release_id"],
        **retention.as_dict(product_root=root),
    }
    atomic_write_json(processed / "retention.json", payload, overwrite=True)
    return retention


@_single_writer
def record_sightings_report(
    *, product_root: str | Path, report_path: str | Path
) -> None:
    """Attach a rebuilt stable report to current product pointers when present."""

    root = Path(product_root).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    pointer_path = root / SIGHTINGS_PROCESSED_RELATIVE / "latest.json"
    if not pointer_path.is_file():
        return
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["report"] = _relative(report, root)
    pointer["report_checksum"] = checksum_path(report)

    generation = pointer.get("generation")
    if not generation:
        atomic_write_json(pointer_path, pointer, overwrite=True)
        return
    generation_path = (root / str(generation)).resolve()
    _relative(generation_path, root)
    dated_pointer = generation_path.parent / "latest.json"
    if not dated_pointer.is_file() or dated_pointer == pointer_path:
        atomic_write_json(pointer_path, pointer, overwrite=True)
        return
    dated_payload = json.loads(dated_pointer.read_text(encoding="utf-8"))
    if dated_payload.get("release_id") != pointer.get("release_id"):
        raise ValueError(
            f"Dated sightings pointer release does not match {pointer_path}"
        )
    dated_payload["report"] = pointer["report"]
    dated_payload["report_checksum"] = pointer["report_checksum"]
    atomic_write_json(dated_pointer, dated_payload, overwrite=True)
    atomic_write_json(pointer_path, pointer, overwrite=True)


@_single_writer
def materialize_sightings_product(
    release: str | Path,
    *,
    product_root: str | Path,
) -> KillerWhaleSightingsProduct:
    """Publish simple latest files plus an immutable date/release generation."""

    root = Path(product_root).expanduser().resolve()
    manifest_path = resolve_sightings_release_manifest(release)
    report = validate_sightings_release(manifest_path)
    report.require_valid()
    release_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_id = str(release_payload["release_id"])
    identity = dict(release_payload["identity"])
    end_date = str(identity["end_date"])

    observations = resolve_sightings_release_artifact(
        manifest_path, "whale.sightings.observations"
    )
    imputed = resolve_sightings_release_artifact(
        manifest_path, "whale.sightings.imputed_retrospective"
    )
    model = resolve_sightings_release_artifact(
        manifest_path, "whale.sightings.imputation_model"
    )
    metrics_path = model.path / "model_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Imputation model metrics are missing: {metrics_path}")
    model_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    composite_identity, composite_ids = _table_identity(observations.path)
    imputed_identity, imputed_ids = _table_identity(imputed.path)
    if composite_ids != imputed_ids:
        raise ValueError("Imputation changed composite sighting identity or order")

    processed = root / SIGHTINGS_PROCESSED_RELATIVE
    dated_root = processed / "by-date" / end_date / release_id
    if dated_root.exists():
        _validate_existing_generation(dated_root, release_id)
    else:
        dated_root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{release_id[:12]}-", dir=dated_root.parent)
        )
        try:
            dated_composite = staging / "composite-sightings.parquet"
            dated_imputed = staging / "imputed-sightings.parquet"
            shutil.copy2(observations.path, dated_composite)
            shutil.copy2(imputed.path, dated_imputed)
            if (
                checksum_path(dated_composite, logical_name=observations.path.name)
                != observations.checksum
            ):
                raise OSError(
                    "Composite sightings changed during product materialization"
                )
            if (
                checksum_path(dated_imputed, logical_name=imputed.path.name)
                != imputed.checksum
            ):
                raise OSError(
                    "Imputed sightings changed during product materialization"
                )
            composite_identity["source_artifact_checksum"] = composite_identity[
                "checksum"
            ]
            composite_identity["checksum"] = checksum_path(dated_composite)
            imputed_identity["source_artifact_checksum"] = imputed_identity["checksum"]
            imputed_identity["checksum"] = checksum_path(dated_imputed)
            payload = {
                "schema_version": PRODUCT_SCHEMA_VERSION,
                "product": "killer-whale-sightings",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "release_id": release_id,
                "release_manifest": _relative(manifest_path, root),
                "release_profile": identity.get("profile", {}).get("name"),
                "public_eligible": release_payload.get("public_eligible") is True,
                "coverage_gates": identity.get("gates", []),
                "tables": {
                    "composite": {
                        **composite_identity,
                        "dataset_id": observations.dataset_id,
                        "path": _relative(
                            dated_root / "composite-sightings.parquet", root
                        ),
                    },
                    "imputed": {
                        **imputed_identity,
                        "dataset_id": imputed.dataset_id,
                        "path": _relative(
                            dated_root / "imputed-sightings.parquet", root
                        ),
                    },
                },
                "imputation_model": {
                    "dataset_id": model.dataset_id,
                    "artifact_path": _relative(model.path, root),
                    "artifact_checksum": model.checksum,
                    "metrics_path": _relative(metrics_path, root),
                    "fit_at_utc": model_metrics.get("fit_at_utc"),
                    "fit_run_id": model_metrics.get("fit_run_id"),
                    "model_sha256": model_metrics.get("model_sha256"),
                    "training_summary": model_metrics.get("training_summary", {}),
                    "evaluations": model_metrics.get("evaluations", {}),
                    "config": model_metrics.get("config", {}),
                },
                "stable_aliases": {
                    "composite": "processed/sightings/final/composite-sightings.parquet",
                    "imputed": "processed/sightings/final/imputed-sightings.parquet",
                    "model_manifest": "processed/sightings/final/imputation-model-manifest.json",
                    "report": "processed/sightings/final/sightings-report.html",
                },
            }
            atomic_write_json(
                staging / "imputation-model-manifest.json", payload, overwrite=False
            )
            build_sightings_report_html(
                composite_path=dated_composite,
                imputed_path=dated_imputed,
                model_manifest_path=staging / "imputation-model-manifest.json",
                output_path=staging / "sightings-report.html",
                overwrite=False,
                display_root=dated_root,
            )
            payload["report"] = {
                "path": _relative(dated_root / "sightings-report.html", root),
                "checksum": checksum_path(staging / "sightings-report.html"),
            }
            atomic_write_json(
                staging / "imputation-model-manifest.json", payload, overwrite=True
            )
            os.replace(staging, dated_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    composite_path = processed / "composite-sightings.parquet"
    imputed_path = processed / "imputed-sightings.parquet"
    model_manifest_path = processed / "imputation-model-manifest.json"
    report_path = processed / "sightings-report.html"
    # Older generations did not bundle a report. Build a separate immutable
    # sidecar when replaying one, without modifying its tables or manifest.
    dated_report = dated_root / "sightings-report.html"
    if not dated_report.is_file():
        dated_report = (
            processed
            / "reports"
            / f"{release_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.html"
        )
        build_sightings_report_html(
            composite_path=dated_root / composite_path.name,
            imputed_path=dated_root / imputed_path.name,
            model_manifest_path=dated_root / model_manifest_path.name,
            output_path=dated_report,
            overwrite=False,
        )
    pointer = {
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "release_id": release_id,
        "product_date": end_date,
        "generation": _relative(dated_root, root),
        "manifest": _relative(dated_root / model_manifest_path.name, root),
        "manifest_checksum": checksum_path(dated_root / model_manifest_path.name),
        "composite": _relative(dated_root / composite_path.name, root),
        "imputed": _relative(dated_root / imputed_path.name, root),
        "report": _relative(dated_report, root),
        "report_checksum": checksum_path(dated_report),
    }
    # Publish one authoritative pointer only after every file exists. Roll back
    # compatibility copies on ordinary failures; crash/concurrent-reader safety
    # comes from resolving immutable paths through the pointer, not flat aliases.
    aliases = {
        composite_path: dated_root / composite_path.name,
        imputed_path: dated_root / imputed_path.name,
        model_manifest_path: dated_root / model_manifest_path.name,
        report_path: dated_report,
    }
    dated_pointer = dated_root.parent / "latest.json"
    with tempfile.TemporaryDirectory(
        prefix=".product-backup-", dir=processed
    ) as backup_dir:
        backups = {}
        for index, destination in enumerate((*aliases, dated_pointer)):
            backup = Path(backup_dir) / str(index)
            if destination.exists():
                shutil.copy2(destination, backup)
                backups[destination] = backup
            else:
                backups[destination] = None
        try:
            for destination, source in aliases.items():
                _atomic_copy(source, destination)
            atomic_write_json(dated_pointer, pointer, overwrite=True)
            atomic_write_json(processed / "latest.json", pointer, overwrite=True)
        except Exception:
            for destination, backup in backups.items():
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            raise
    return KillerWhaleSightingsProduct(
        release_manifest=manifest_path,
        dated_root=dated_root,
        composite_path=dated_root / composite_path.name,
        imputed_path=dated_root / imputed_path.name,
        model_manifest_path=dated_root / model_manifest_path.name,
        report_path=dated_report,
    )


def cleanup_completed_sightings_run(
    *, product_root: str | Path, completed_run_id: str
) -> None:
    """Remove workspaces owned by one successfully materialized product run.

    Failed or concurrent run directories are retained. Empty directories left by
    older completed stages are pruned so the two temporary roots disappear when
    they no longer contain recoverable state.
    """

    root = Path(product_root).expanduser().resolve()
    if not completed_run_id or Path(completed_run_id).name != completed_run_id:
        raise ValueError(f"Invalid completed sightings run id: {completed_run_id!r}")

    run_root = root / "_sightings_product_runs"
    completed_candidate = run_root / completed_run_id
    if completed_candidate.is_symlink():
        completed_candidate.unlink()
    elif completed_candidate.exists():
        shutil.rmtree(completed_candidate)

    staging_root = root / ".staging"
    if staging_root.is_dir():
        prefix = f"{completed_run_id}-"
        for candidate in staging_root.iterdir():
            if not candidate.name.startswith(prefix):
                continue
            if candidate.is_symlink():
                candidate.unlink()
            elif candidate.is_dir():
                shutil.rmtree(candidate)

    for work_root in (staging_root, run_root):
        if not work_root.is_dir():
            continue
        for candidate in sorted(
            work_root.rglob("*"), key=lambda path: len(path.parts), reverse=True
        ):
            if candidate.is_dir() and not candidate.is_symlink():
                try:
                    candidate.rmdir()
                except OSError:
                    pass
        try:
            work_root.rmdir()
        except OSError:
            pass


__all__ = [
    "DEFAULT_MAX_SIGHTINGS_GROWTH_FRACTION",
    "KillerWhaleSightingsProduct",
    "SIGHTINGS_FINAL_RELATIVE",
    "SIGHTINGS_IMPUTED_RELATIVE",
    "SIGHTINGS_NORMALIZED_RELATIVE",
    "SIGHTINGS_PROCESSED_RELATIVE",
    "SIGHTINGS_RAW_RELATIVE",
    "SIGHTINGS_ROOT_RELATIVE",
    "SightingsArtifactRetention",
    "SightingsProductLayout",
    "cleanup_completed_sightings_run",
    "current_sightings_row_count",
    "materialize_sightings_product",
    "prune_prior_sightings_artifacts",
    "record_sightings_report",
    "resolve_sightings_product",
    "sightings_product_layout",
]
