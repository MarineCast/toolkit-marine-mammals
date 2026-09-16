from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from marine_mammal_toolkit.tools.schemas.artifacts import (
    VERIFIED_SOURCE_COVERAGE_STATUSES,
)
from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import DataSnapshotMetadata
from marine_mammal_toolkit.tools.schemas.artifacts import SourceWatermark
from marine_mammal_toolkit.tools._core.data import StageResult
from marine_mammal_toolkit.tools._core.data import ValidationReport
from marine_mammal_toolkit.tools._core.persistence import checksum_path


def build_data_snapshot(
    sources: Iterable[ArtifactRef],
    *,
    default_coverage_start: str,
    duplicate_resolution_as_of: str | None = None,
) -> DataSnapshotMetadata:
    """Build a deterministic coverage envelope from immutable source snapshots."""

    watermarks: list[SourceWatermark] = []
    for artifact in sources:
        metadata_path = artifact.path / "snapshot.json"
        if not metadata_path.exists():
            raise ValueError(f"Source snapshot is missing metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        source = str(
            metadata.get("source")
            or (artifact.dataset_id or "").removeprefix("whale.sightings.source_")
        ).lower()
        coverage_status = str(metadata.get("coverage_status") or "legacy_unverified")
        coverage_start = (
            metadata.get("coverage_start") if "coverage_status" in metadata else None
        )
        coverage_through = (
            metadata.get("coverage_through") if "coverage_status" in metadata else None
        )
        if coverage_status not in VERIFIED_SOURCE_COVERAGE_STATUSES:
            coverage_start = None
            coverage_through = None
        if (coverage_start is None) != (coverage_through is None):
            raise ValueError(
                f"Source snapshot has a partial verified interval: {metadata_path}"
            )
        retrieved_at = metadata.get("retrieved_at") or artifact.created_at
        retrieval_id = metadata.get("retrieval_id") or artifact.run_id
        if not retrieved_at or not retrieval_id:
            raise ValueError(
                f"Source snapshot lacks retrieval identity: {metadata_path}"
            )
        watermarks.append(
            SourceWatermark(
                source=source,
                retrieval_id=str(retrieval_id),
                coverage_start=str(coverage_start)[:10] if coverage_start else None,
                coverage_through=(
                    str(coverage_through)[:10] if coverage_through else None
                ),
                checksum=artifact.checksum or checksum_path(artifact.path),
                retrieved_at=str(retrieved_at),
                observed_start=(
                    str(metadata["observed_start"])[:10]
                    if metadata.get("observed_start")
                    else None
                ),
                observed_through=(
                    str(metadata["observed_through"])[:10]
                    if metadata.get("observed_through")
                    else None
                ),
                requested_start=(
                    str(metadata["requested_start"])[:10]
                    if metadata.get("requested_start")
                    else None
                ),
                requested_through=(
                    str(metadata["requested_through"])[:10]
                    if metadata.get("requested_through")
                    else None
                ),
                coverage_status=coverage_status,
            )
        )
    if not watermarks:
        raise ValueError("At least one source snapshot is required")
    ordered = tuple(sorted(watermarks, key=lambda item: item.source))
    identity_payload = [
        {
            "source": item.source,
            "retrieval_id": item.retrieval_id,
            "checksum": item.checksum,
            "coverage_start": item.coverage_start,
            "coverage_through": item.coverage_through,
            "coverage_status": item.coverage_status,
        }
        for item in ordered
    ]
    snapshot_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    created = max(
        pd_timestamp(item.retrieved_at).astimezone(timezone.utc) for item in ordered
    ).isoformat()
    verified = all(
        item.coverage_status in VERIFIED_SOURCE_COVERAGE_STATUSES
        and item.coverage_start is not None
        and item.coverage_through is not None
        for item in ordered
    )
    verified_starts = [
        item.coverage_start for item in ordered if item.coverage_start is not None
    ]
    verified_ends = [
        item.coverage_through for item in ordered if item.coverage_through is not None
    ]
    coverage_start = max(verified_starts) if verified else None
    coverage_through = min(verified_ends) if verified else None
    if (
        verified
        and coverage_start is not None
        and coverage_through is not None
        and coverage_start > coverage_through
    ):
        raise ValueError("Verified source coverage intervals do not overlap")
    return DataSnapshotMetadata(
        snapshot_id=snapshot_id,
        snapshot_created_at=created,
        coverage_start=coverage_start,
        coverage_through=coverage_through,
        source_watermarks=ordered,
        duplicate_resolution_as_of=duplicate_resolution_as_of,
        coverage_status="verified_intersection" if verified else "unverified",
    )


def pd_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and require a timezone for provenance values."""

    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Provenance timestamp must include a timezone: {value}")
    return parsed


def with_imputation_as_of(
    snapshot: DataSnapshotMetadata, imputation_as_of: str
) -> DataSnapshotMetadata:
    return replace(snapshot, imputation_as_of=imputation_as_of)


def stage_signature(
    *,
    stage: str,
    semantic_version: str,
    config_hash: str,
    inputs: tuple[ArtifactRef, ...],
    parameters: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    payload: dict[str, object] = {
        "stage": stage,
        "semantic_version": semantic_version,
        "code_revision": code_revision(),
        "config_hash": config_hash,
        "inputs": [
            {
                "dataset_id": item.dataset_id,
                "checksum": item.checksum or checksum_path(item.path),
                "schema_version": item.schema_version,
            }
            for item in inputs
        ],
        "parameters": parameters or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest(), payload


def resume_result(
    *,
    enabled: bool,
    manifest_path: Path,
    config_hash: str,
    inputs: tuple[ArtifactRef, ...],
    signature: str | None = None,
) -> StageResult | None:
    if not enabled or not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text())
    if payload.get("config_hash") != config_hash:
        return None
    if signature is not None and payload.get("stage_signature") != signature:
        return None
    expected_inputs = tuple(item.checksum or str(item.path) for item in inputs)
    recorded_inputs = tuple(
        item.get("checksum") or item.get("path") for item in payload.get("inputs", ())
    )
    if expected_inputs != recorded_inputs:
        return None
    outputs = tuple(ArtifactRef.from_dict(item) for item in payload.get("outputs", ()))
    if not outputs or not all(
        item.path.exists() and item.checksum == checksum_path(item.path)
        for item in outputs
    ):
        return None
    reports = tuple(
        ValidationReport(
            True,
            item.dataset_id or item.kind,
            metrics={"resumed": True, "checksum": item.checksum},
        )
        for item in outputs
    )
    return StageResult(outputs, reports, skipped=True)


def code_revision() -> str:
    """Identify installed producer content, including uncommitted source changes."""
    configured = os.getenv("MARINE_MAMMALS_CODE_REVISION")
    if configured:
        return f"marine-mammal-toolkit:{configured}"
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return f"marine-mammal-toolkit:sha256:{digest.hexdigest()}"
