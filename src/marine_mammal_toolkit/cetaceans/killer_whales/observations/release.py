"""Immutable, manifest-last releases for the sightings pipeline.

The stage implementations still own their run-scoped working products.  This
module turns a validated set of those products into an immutable generation and
advances one release pointer only after the copied generation validates.  A
failed build can therefore leave unreferenced candidate files, but it cannot
partially replace the previously promoted sightings release.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path
from marine_mammal_toolkit.tools._core.data import ValidationReport

RELEASE_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class SightingsReleaseProfile:
    """A bounded build product, rather than an open-ended configuration alias."""

    name: str
    resolutions: tuple[int, ...]
    frequencies: tuple[str, ...]
    include_imputation: bool
    include_counts: bool
    include_model_grid: bool
    include_intensity: bool
    require_verified_cohort: bool
    public_by_default: bool


@dataclass(frozen=True)
class SightingsReleaseArtifact:
    """One checksum-verified artifact resolved from an immutable release."""

    release_id: str
    manifest_path: Path
    dataset_id: str
    path: Path
    checksum: str
    schema_version: str
    producer: str
    row_count: int | None
    file_count: int
    snapshot_id: str | None
    processing_mode: str | None
    sensitivity: str
    config_hash: str
    end_date: str
    created_at_utc: str
    coverage_status: str
    coverage_through: str | None


RELEASE_PROFILES: Mapping[str, SightingsReleaseProfile] = {
    "observations-only": SightingsReleaseProfile(
        name="observations-only",
        resolutions=(),
        frequencies=(),
        include_imputation=False,
        include_counts=False,
        include_model_grid=False,
        include_intensity=False,
        require_verified_cohort=False,
        public_by_default=False,
    ),
    "imputation-only": SightingsReleaseProfile(
        name="imputation-only",
        resolutions=(),
        frequencies=(),
        include_imputation=True,
        include_counts=False,
        include_model_grid=False,
        include_intensity=False,
        require_verified_cohort=False,
        public_by_default=False,
    ),
    "production-retrospective": SightingsReleaseProfile(
        name="production-retrospective",
        resolutions=(4,),
        frequencies=("weekly",),
        include_imputation=True,
        include_counts=True,
        include_model_grid=True,
        include_intensity=True,
        require_verified_cohort=True,
        public_by_default=False,
    ),
    "authoritative-counts": SightingsReleaseProfile(
        name="authoritative-counts",
        resolutions=(4, 5, 6),
        frequencies=("daily", "weekly"),
        include_imputation=True,
        include_counts=True,
        include_model_grid=False,
        include_intensity=False,
        require_verified_cohort=False,
        public_by_default=False,
    ),
    "research-h6": SightingsReleaseProfile(
        name="research-h6",
        resolutions=(6,),
        frequencies=("weekly",),
        include_imputation=True,
        include_counts=True,
        include_model_grid=True,
        include_intensity=True,
        require_verified_cohort=False,
        public_by_default=False,
    ),
    "observed-only": SightingsReleaseProfile(
        name="observed-only",
        resolutions=(4, 5, 6),
        frequencies=("daily", "weekly"),
        include_imputation=False,
        include_counts=True,
        include_model_grid=False,
        include_intensity=False,
        require_verified_cohort=False,
        public_by_default=False,
    ),
}


def release_profile(name: str) -> SightingsReleaseProfile:
    try:
        return RELEASE_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sightings release profile {name!r}; expected one of "
            f"{sorted(RELEASE_PROFILES)}"
        ) from exc


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError(f"Cannot construct a release path from {value!r}")
    return cleaned


def _stage_summary(manifest: RunManifest) -> dict[str, Any]:
    """Remove mutable/absolute paths while retaining causal stage lineage."""

    return {
        "workflow": manifest.workflow,
        "schema_version": manifest.schema_version,
        "status": manifest.status,
        "stage_signature": manifest.stage_signature,
        "config_hash": manifest.config_hash,
        "code_revision": manifest.code_revision,
        "input_checksums": sorted(
            item.checksum for item in manifest.inputs if item.checksum is not None
        ),
        "output_checksums": sorted(
            item.checksum for item in manifest.outputs if item.checksum is not None
        ),
        "snapshot_id": (
            manifest.data_snapshot.snapshot_id
            if manifest.data_snapshot is not None
            else None
        ),
    }


def _artifact_identity(artifact: ArtifactRef) -> dict[str, Any]:
    if not artifact.checksum:
        raise ValueError(
            f"Release artifact {artifact.dataset_id or artifact.kind} has no checksum"
        )
    return {
        "dataset_id": artifact.dataset_id or artifact.kind,
        "checksum": artifact.checksum,
        "schema_version": artifact.schema_version,
        "producer": artifact.producer,
        "processing_mode": artifact.processing_mode,
        "knowledge_cutoff": artifact.knowledge_cutoff,
        "snapshot_id": (
            artifact.data_snapshot.snapshot_id
            if artifact.data_snapshot is not None
            else None
        ),
    }


def _portable_input_reference(value: str) -> str:
    """Retain content identifiers while keeping mutable absolute paths out of releases."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        digest = hashlib.sha256(str(candidate).encode("utf-8")).hexdigest()
        return f"absolute-path-redacted:sha256:{digest}"
    return value


def _copy_artifact(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=shutil.copy2)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _materialize_artifact(
    artifact: ArtifactRef,
    *,
    generation: Path,
    ordinal: int,
) -> dict[str, Any]:
    source = artifact.path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Release artifact does not exist: {source}")
    actual_checksum = checksum_path(source)
    if artifact.checksum != actual_checksum:
        raise ValueError(
            f"Release artifact checksum mismatch for {artifact.dataset_id or artifact.kind}: "
            f"manifest={artifact.checksum}, actual={actual_checksum}"
        )
    dataset_id = artifact.dataset_id or artifact.kind
    relative = (
        Path("artifacts")
        / f"{ordinal:03d}_{_safe_component(dataset_id)}"
        / artifact.checksum[:20]
        / source.name
    )
    destination = generation / relative
    _copy_artifact(source, destination)
    copied_checksum = checksum_path(destination)
    if copied_checksum != artifact.checksum:
        raise ValueError(f"Copied release artifact changed checksum: {dataset_id}")
    return {
        **_artifact_identity(artifact),
        "path": relative.as_posix(),
        "size_bytes": _size_bytes(destination),
        "row_count": artifact.row_count,
        "file_count": (
            sum(1 for item in destination.rglob("*") if item.is_file())
            if destination.is_dir()
            else 1
        ),
        "inputs": sorted(
            _portable_input_reference(str(item)) for item in artifact.inputs
        ),
        "sensitivity": artifact.sensitivity,
    }


def _required_gate_failures(gates: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(gate.get("name") or "unnamed_gate")
        for gate in gates
        if bool(gate.get("required", True)) and gate.get("passed") is not True
    ]


def promote_sightings_release(
    *,
    release_root: str | Path,
    profile: SightingsReleaseProfile,
    end_date: date,
    config_hash: str,
    run_id: str,
    artifacts: Iterable[ArtifactRef],
    stage_manifests: Iterable[RunManifest],
    gates: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Copy a complete validated generation and atomically advance ``latest.json``.

    All required gates are checked before any authoritative pointer is touched.
    Existing generations are immutable and are never overwritten.
    """

    root = Path(release_root).resolve()
    resolved_artifacts = tuple(artifacts)
    if not resolved_artifacts:
        raise ValueError("A sightings release must contain at least one artifact")
    paths = [item.path.resolve() for item in resolved_artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError(
            "A sightings release cannot inventory the same artifact path twice"
        )
    gate_payload = tuple(dict(item) for item in gates)
    failed = _required_gate_failures(gate_payload)
    if failed:
        raise ValueError(
            f"Required sightings release gates failed: {', '.join(failed)}"
        )
    stages = tuple(_stage_summary(item) for item in stage_manifests)
    identity = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "profile": asdict(profile),
        "end_date": end_date.isoformat(),
        "config_hash": config_hash,
        "artifacts": sorted(
            (_artifact_identity(item) for item in resolved_artifacts),
            key=lambda item: (item["dataset_id"], item["checksum"]),
        ),
        "stages": sorted(
            stages,
            key=lambda item: (
                str(item.get("workflow")),
                str(item.get("stage_signature")),
            ),
        ),
        "gates": sorted(gate_payload, key=lambda item: str(item.get("name"))),
    }
    release_id = _canonical_hash(identity)
    final_generation = root / "generations" / release_id
    final_manifest = final_generation / "manifest.json"
    if final_generation.exists():
        report = validate_sightings_release(final_manifest)
        report.require_valid()
        atomic_write_json(
            root / "latest.json",
            {
                "schema_version": RELEASE_SCHEMA_VERSION,
                "release_id": release_id,
                "manifest": f"generations/{release_id}/manifest.json",
            },
            overwrite=True,
        )
        return final_manifest

    root.mkdir(parents=True, exist_ok=True)
    staging_parent = root / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    generation = Path(
        tempfile.mkdtemp(prefix=f"{_safe_component(run_id)}-", dir=staging_parent)
    )
    try:
        inventory = [
            _materialize_artifact(item, generation=generation, ordinal=index)
            for index, item in enumerate(resolved_artifacts)
        ]
        manifest_payload = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "identity": identity,
            "inventory": inventory,
            "metadata": dict(metadata or {}),
            "public_eligible": bool(profile.public_by_default) and not failed,
        }
        atomic_write_json(
            generation / "manifest.json", manifest_payload, overwrite=False
        )
        report = validate_sightings_release(generation / "manifest.json")
        report.require_valid()
        final_generation.parent.mkdir(parents=True, exist_ok=True)
        os.replace(generation, final_generation)
        atomic_write_json(
            root / "latest.json",
            {
                "schema_version": RELEASE_SCHEMA_VERSION,
                "release_id": release_id,
                "manifest": f"generations/{release_id}/manifest.json",
            },
            overwrite=True,
        )
    except Exception:
        shutil.rmtree(generation, ignore_errors=True)
        raise
    finally:
        try:
            staging_parent.rmdir()
        except OSError:
            pass
    return final_manifest


def resolve_sightings_release_manifest(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "latest.json"
    if candidate.name == "latest.json":
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        pointer = json.loads(candidate.read_text(encoding="utf-8"))
        reference = Path(str(pointer.get("manifest", "")))
        if not str(reference):
            raise ValueError(f"Sightings release pointer has no manifest: {candidate}")
        if reference.is_absolute():
            raise ValueError(
                "Sightings release pointers must use relative manifest paths"
            )
        candidate = (candidate.parent / reference).resolve()
    return candidate


def _release_coverage(identity: Mapping[str, Any]) -> tuple[str, str | None]:
    gates = identity.get("gates", ())
    if not isinstance(gates, list):
        return "unverified", None
    coverage = next(
        (
            item
            for item in gates
            if isinstance(item, dict) and item.get("name") == "verified_target_cohort"
        ),
        None,
    )
    if coverage is None:
        return "unverified", None
    status = str(coverage.get("coverage_status") or "unverified")
    through = coverage.get("coverage_through")
    return status, str(through) if through else None


def resolve_sightings_release_artifact(
    path: str | Path,
    dataset_id: str,
    *,
    verify_checksum: bool = True,
) -> SightingsReleaseArtifact:
    """Resolve exactly one release artifact without consulting mutable stage pointers.

    The release identity, inventory membership, path containment, and selected
    artifact checksum are validated on every resolution.  This is intentionally
    narrower than :func:`validate_sightings_release`, which validates every
    artifact and can be expensive for multi-gigabyte source snapshots.
    """

    manifest_path = resolve_sightings_release_manifest(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sightings release manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != RELEASE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported sightings release schema: {payload.get('schema_version')}"
        )
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Sightings release is missing its identity payload")
    release_id = str(payload.get("release_id") or "")
    if release_id != _canonical_hash(identity):
        raise ValueError("Sightings release id does not match its canonical identity")

    inventory = payload.get("inventory")
    if not isinstance(inventory, list):
        raise ValueError("Sightings release inventory is missing")
    matches = [
        item
        for item in inventory
        if isinstance(item, dict) and str(item.get("dataset_id")) == dataset_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Sightings release must inventory exactly one {dataset_id}; found {len(matches)}"
        )
    entry = matches[0]
    identity_matches = [
        item
        for item in identity.get("artifacts", ())
        if isinstance(item, dict) and str(item.get("dataset_id")) == dataset_id
    ]
    if len(identity_matches) != 1 or str(identity_matches[0].get("checksum")) != str(
        entry.get("checksum")
    ):
        raise ValueError(
            f"Release identity does not bind inventory artifact {dataset_id}"
        )

    relative = Path(str(entry.get("path") or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid release inventory path for {dataset_id}: {relative}")
    generation = manifest_path.parent.resolve()
    artifact_path = (generation / relative).resolve()
    try:
        artifact_path.relative_to(generation)
    except ValueError as exc:
        raise ValueError(
            f"Release inventory escapes its generation: {relative}"
        ) from exc
    if not artifact_path.exists():
        raise FileNotFoundError(f"Release artifact is missing: {artifact_path}")
    expected_checksum = str(entry.get("checksum") or "")
    if not expected_checksum:
        raise ValueError(f"Release artifact has no checksum: {dataset_id}")
    if verify_checksum and checksum_path(artifact_path) != expected_checksum:
        raise ValueError(f"Release artifact checksum mismatch: {dataset_id}")
    coverage_status, coverage_through = _release_coverage(identity)
    return SightingsReleaseArtifact(
        release_id=release_id,
        manifest_path=manifest_path,
        dataset_id=dataset_id,
        path=artifact_path,
        checksum=expected_checksum,
        schema_version=str(entry.get("schema_version") or "1"),
        producer=str(entry.get("producer") or "unknown"),
        row_count=(
            int(entry["row_count"]) if entry.get("row_count") is not None else None
        ),
        file_count=int(entry.get("file_count") or 0),
        snapshot_id=(str(entry["snapshot_id"]) if entry.get("snapshot_id") else None),
        processing_mode=(
            str(entry["processing_mode"]) if entry.get("processing_mode") else None
        ),
        sensitivity=str(entry.get("sensitivity") or "internal"),
        config_hash=str(identity.get("config_hash") or ""),
        end_date=str(identity.get("end_date") or ""),
        created_at_utc=str(payload.get("created_at_utc") or ""),
        coverage_status=coverage_status,
        coverage_through=coverage_through,
    )


def _inventory_files(generation: Path, entry: Mapping[str, Any]) -> set[str]:
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid release inventory path: {relative}")
    resolved = (generation / relative).resolve()
    try:
        resolved.relative_to(generation.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Release inventory escapes its generation: {relative}"
        ) from exc
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if resolved.is_file():
        return {relative.as_posix()}
    return {
        item.relative_to(generation).as_posix()
        for item in resolved.rglob("*")
        if item.is_file()
    }


def validate_sightings_release(
    path: str | Path,
    *,
    require_public_eligible: bool = False,
) -> ValidationReport:
    """Validate identity, checksums, inventory completeness, and required gates."""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest_path = resolve_sightings_release_manifest(path)
    except Exception as exc:
        return ValidationReport(False, "whale.sightings.release", errors=(str(exc),))
    if not manifest_path.is_file():
        return ValidationReport(
            False,
            "whale.sightings.release",
            errors=(f"Missing sightings release manifest: {manifest_path}",),
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ValidationReport(False, "whale.sightings.release", errors=(str(exc),))
    if str(payload.get("schema_version")) != RELEASE_SCHEMA_VERSION:
        errors.append(
            f"Unsupported sightings release schema: {payload.get('schema_version')}"
        )
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        errors.append("Sightings release is missing its identity payload")
        identity = {}
    expected_release_id = _canonical_hash(identity)
    release_id = str(payload.get("release_id") or "")
    if release_id != expected_release_id:
        errors.append("Sightings release id does not match its canonical identity")
    if (
        manifest_path.parent.name != release_id
        and ".staging" not in manifest_path.parts
    ):
        errors.append("Sightings generation directory does not match release id")
    inventory = payload.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        errors.append("Sightings release inventory is empty")
        inventory = []
    identity_artifacts = (
        identity.get("artifacts", []) if isinstance(identity, dict) else []
    )
    identity_keys = {
        (str(item.get("dataset_id")), str(item.get("checksum")))
        for item in identity_artifacts
        if isinstance(item, dict)
    }
    inventory_keys = {
        (str(item.get("dataset_id")), str(item.get("checksum")))
        for item in inventory
        if isinstance(item, dict)
    }
    if identity_keys != inventory_keys:
        errors.append(
            "Release identity artifacts do not match the materialized inventory"
        )
    generation = manifest_path.parent
    listed_files: set[str] = {"manifest.json"}
    seen_roots: set[str] = set()
    for entry in inventory:
        if not isinstance(entry, dict):
            errors.append("Release inventory entries must be objects")
            continue
        relative = str(entry.get("path") or "")
        if relative in seen_roots:
            errors.append(f"Duplicate release inventory path: {relative}")
            continue
        seen_roots.add(relative)
        try:
            files = _inventory_files(generation, entry)
            listed_files.update(files)
            artifact_path = generation / relative
            actual_checksum = checksum_path(artifact_path)
            if actual_checksum != entry.get("checksum"):
                errors.append(f"Checksum mismatch: {relative}")
            if _size_bytes(artifact_path) != int(entry.get("size_bytes", -1)):
                errors.append(f"Size mismatch: {relative}")
            actual_file_count = (
                sum(1 for item in artifact_path.rglob("*") if item.is_file())
                if artifact_path.is_dir()
                else 1
            )
            if actual_file_count != int(entry.get("file_count", -1)):
                errors.append(f"File-count mismatch: {relative}")
        except Exception as exc:
            errors.append(str(exc))
    actual_files = {
        item.relative_to(generation).as_posix()
        for item in generation.rglob("*")
        if item.is_file()
    }
    if actual_files != listed_files:
        extras = sorted(actual_files - listed_files)
        missing = sorted(listed_files - actual_files)
        if extras:
            errors.append(f"Release contains unlisted files: {extras[:5]}")
        if missing:
            errors.append(f"Release inventory references missing files: {missing[:5]}")
    gates = identity.get("gates", []) if isinstance(identity, dict) else []
    failures = _required_gate_failures(gates if isinstance(gates, list) else [])
    if failures:
        errors.append(f"Required release gates failed: {', '.join(failures)}")
    if require_public_eligible and payload.get("public_eligible") is not True:
        errors.append("Sightings release is not eligible for public promotion")
    if payload.get("public_eligible") is not True:
        warnings.append("Sightings release is internal and cannot be publicly promoted")
    return ValidationReport(
        valid=not errors,
        dataset_id="whale.sightings.release",
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        metrics={
            "release_id": release_id,
            "artifact_count": len(inventory),
            "file_count": max(0, len(actual_files) - 1),
            "manifest": str(manifest_path),
        },
    )


__all__ = [
    "RELEASE_PROFILES",
    "SightingsReleaseProfile",
    "SightingsReleaseArtifact",
    "promote_sightings_release",
    "release_profile",
    "resolve_sightings_release_artifact",
    "resolve_sightings_release_manifest",
    "validate_sightings_release",
]
