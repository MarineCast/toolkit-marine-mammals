from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import time
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import requests

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools._core.data import StageResult
from marine_mammal_toolkit.tools._core.data import ValidationReport
from marine_mammal_toolkit.tools._core.persistence import checksum_path

from marine_mammal_toolkit.cetaceans.killer_whales.observations.adapters import (
    adapt_snapshot,
)
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    SightingsPipelineConfig,
)
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.schemas.observations import SightingsCollectionRequest
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
    collect_cwr_snapshot,
)
from marine_mammal_toolkit.tools.observations.runtime import build_data_snapshot
from marine_mammal_toolkit.tools.observations.runtime import code_revision
from marine_mammal_toolkit.tools.observations.runtime import stage_signature

LOGGER = logging.getLogger(__name__)


def _raw_response_hash(snapshot: Path) -> str:
    """Hash exact provider/file bytes without collection-side metadata."""

    excluded = {"snapshot.json", "source_policy.json", "gbif_policy.json"}
    files = sorted(
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(snapshot).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _semantic_content_hash(adapted: pa.Table) -> str:
    """Hash canonical provider meaning independently of retrieval timestamps and row order."""

    excluded = {
        "SOURCE_RETRIEVED_AT_UTC",
        "SOURCE_PAYLOAD_CORRECTED",
        "LAST_CORRECTED_AT_UTC",
    }
    columns = [name for name in adapted.column_names if name not in excluded]
    semantic = adapted.select(columns)
    if semantic.num_rows and "SOURCE_RECORD_ID" in semantic.column_names:
        semantic = semantic.sort_by([("SOURCE_RECORD_ID", "ascending")])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, semantic.schema) as writer:
        writer.write_table(semantic)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _latest_snapshot(root: Path) -> Path:
    snapshots = sorted(path for path in root.glob("*/snapshot.json") if path.is_file())
    if not snapshots:
        raise FileNotFoundError(f"No immutable snapshot available in {root}")
    return snapshots[-1].parent


def _latest_cohort_snapshots(raw_root: Path) -> dict[str, Path]:
    """Resolve the last atomically published source cohort, if one exists."""

    pointer_path = raw_root / "manifests/latest.json"
    if not pointer_path.exists():
        return {}
    pointer = json.loads(pointer_path.read_text())
    manifest_path = Path(str(pointer["manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = (pointer_path.parent / manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    snapshots: dict[str, Path] = {}
    for output in manifest.get("outputs", ()):
        dataset_id = str(output.get("dataset_id") or "")
        if not dataset_id.startswith("whale.sightings.source_"):
            continue
        source = dataset_id.removeprefix("whale.sightings.source_")
        snapshots[source] = Path(str(output["path"]))
    return snapshots


def _default_start(
    previous_snapshot: Path | None,
    config: SightingsPipelineConfig,
    source_name: str,
    end: date,
) -> date:
    source_min_date = config.collection.sources[source_name].min_date or config.min_date  # type: ignore[index]
    if previous_snapshot is None:
        return date.fromisoformat(source_min_date)
    watermark = json.loads((previous_snapshot / "snapshot.json").read_text()).get(
        "watermark"
    )
    if not watermark:
        return date.fromisoformat(source_min_date)
    return max(
        date.fromisoformat(source_min_date),
        date.fromisoformat(watermark) - timedelta(days=config.collection.overlap_days),
    )


def _validate_collection_window(
    request: SightingsCollectionRequest, config: SightingsPipelineConfig
) -> date:
    today = datetime.now(ZoneInfo(config.model_timezone)).date()
    end = request.end_date or today
    if end > today:
        raise ValueError(f"Sightings end_date cannot be in the future: {end}>{today}")
    if request.start_date is not None and request.start_date > end:
        raise ValueError(
            f"Sightings start_date cannot follow end_date: {request.start_date}>{end}"
        )
    return end


def _range_snapshot_mode(
    source: str,
    start: date,
    config: SightingsPipelineConfig,
    previous_snapshot: Path | None = None,
) -> str:
    # Provider query completeness does not prove deletion/tombstone semantics.
    # Once state exists, even a complete-range retrieval is applied as a
    # non-destructive upsert unless a source-specific contract later proves it
    # is an authoritative replacement feed.
    if previous_snapshot is not None:
        return "DELTA_UPSERT"
    source_min = date.fromisoformat(
        config.collection.sources[source].min_date or config.min_date  # type: ignore[index]
    )
    return "FULL_REPLACE" if start <= source_min else "DELTA_UPSERT"


def _assert_monotonic_watermark(
    source: str, previous_snapshot: Path | None, end: date
) -> None:
    if previous_snapshot is None:
        return
    metadata = json.loads((previous_snapshot / "snapshot.json").read_text())
    watermark = metadata.get("watermark")
    if watermark and end < date.fromisoformat(str(watermark)):
        raise ValueError(
            f"Refusing out-of-order {source} collection window: "
            f"end={end} precedes published watermark={watermark}"
        )


def _snapshot_validation_errors(
    snapshot: Path,
    source: str,
    *,
    expected_rows: int | None = None,
    expected_semantic_hash: str | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        metadata = json.loads((snapshot / "snapshot.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{source}: invalid snapshot.json: {exc}"]
    if metadata.get("source") != source:
        errors.append(f"{source}: snapshot source does not match directory")
    if str(metadata.get("snapshot_mode", "")).upper() not in {
        "FULL_REPLACE",
        "DELTA_UPSERT",
    }:
        errors.append(f"{source}: invalid snapshot_mode")
    if expected_rows is None:
        try:
            expected_rows = adapt_snapshot(source, snapshot).num_rows
        except (
            Exception
        ) as exc:  # validation reports the source-specific adapter failure
            errors.append(f"{source}: adapter validation failed: {exc}")
    if (
        expected_rows is not None
        and int(metadata.get("row_count", -1)) != expected_rows
    ):
        errors.append(
            f"{source}: row_count mismatch metadata={metadata.get('row_count')} "
            f"actual={expected_rows}"
        )
    raw_response_sha256 = metadata.get("raw_response_sha256")
    if raw_response_sha256 is not None and str(
        raw_response_sha256
    ) != _raw_response_hash(snapshot):
        errors.append(f"{source}: raw_response_sha256 mismatch")
    semantic_content_sha256 = metadata.get("semantic_content_sha256")
    if expected_semantic_hash is not None and str(semantic_content_sha256 or "") != str(
        expected_semantic_hash
    ):
        errors.append(f"{source}: semantic_content_sha256 mismatch")
    if semantic_content_sha256 is not None and (
        len(str(semantic_content_sha256)) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(semantic_content_sha256)
        )
    ):
        errors.append(f"{source}: invalid semantic_content_sha256")
    snapshot_root = snapshot.resolve()
    checksums = metadata.get("file_checksums")
    if not isinstance(checksums, dict) or not checksums:
        errors.append(f"{source}: missing file_checksums")
        return errors
    actual_files = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path.name != "snapshot.json"
    }
    undeclared = actual_files - {str(value) for value in checksums}
    if undeclared:
        errors.append(
            f"{source}: files missing from checksum inventory: {sorted(undeclared)}"
        )
    for relative, expected in checksums.items():
        path = (snapshot / str(relative)).resolve()
        if not path.is_relative_to(snapshot_root) or not path.is_file():
            errors.append(f"{source}: missing or unsafe snapshot file {relative}")
        elif checksum_path(path) != str(expected):
            errors.append(f"{source}: checksum mismatch for {relative}")
    return errors


def _raw_schema_profile(snapshot: Path, source: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if source == "twm" or source == "acartia":
        for path in sorted(snapshot.rglob("*.csv")):
            frame = pd.read_csv(path, dtype=str)
            rows.extend(frame.to_dict("records"))
    if source != "twm" and (snapshot / "payload.json").exists():
        payload = json.loads((snapshot / "payload.json").read_text())
        rows.extend(
            payload if isinstance(payload, list) else payload.get("results", [])
        )
    if source == "cwr":
        archive = json.loads((snapshot / "archive_extracts.json").read_text())
        rows.extend(archive.get("index_entries", []))
        atlist = json.loads((snapshot / "atlist_maps.json").read_text())
        for source_map in atlist.get("maps", {}).values():
            rows.extend(source_map.get("markers_payload", {}).get("markers", []))
    keys = sorted({str(key) for row in rows for key in row})

    def missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float):
            return bool(pd.isna(value))
        return False

    return {
        "row_count": len(rows),
        "fields": {
            key: {
                "types": sorted(
                    {
                        type(row.get(key)).__name__
                        for row in rows
                        if row.get(key) is not None
                    }
                ),
                "null_rate": (
                    sum(missing(row.get(key)) for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
            }
            for key in keys
        },
    }


def _observed_date_bounds(adapted: Any) -> tuple[str | None, str | None]:
    """Return observed source bounds without presenting them as complete coverage."""

    frame = adapted.select(["OBSERVED_AT_RAW", "OBSERVED_DATE_RAW"]).to_pandas()
    observed = frame["OBSERVED_AT_RAW"].combine_first(frame["OBSERVED_DATE_RAW"])
    parsed = pd.to_datetime(
        observed, format="mixed", errors="coerce", utc=True
    ).dropna()
    if parsed.empty:
        return None, None
    return parsed.min().date().isoformat(), parsed.max().date().isoformat()


def collect_sightings(request: SightingsCollectionRequest) -> StageResult:
    document, config = load_sightings_config(request.config)
    retrieval_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_root = request.data_root / "raw/whale/sightings"
    end = _validate_collection_window(request, config)
    published_cohort_exists = (raw_root / "manifests/latest.json").exists()
    previous_cohort = _latest_cohort_snapshots(raw_root)
    outputs: list[ArtifactRef] = []
    source_latest_payloads: dict[str, dict[str, str]] = {}
    snapshot_errors: list[str] = []

    for source_name in ("twm", "acartia", "maplify", "inaturalist", "cwr", "gbif"):
        settings = config.collection.sources[source_name]
        if not settings.enabled:
            continue
        snapshots_root = raw_root / source_name / "snapshots"
        adapted_row_count: int | None = None
        adapted_semantic_hash: str | None = None
        previous_snapshot = previous_cohort.get(source_name)
        if previous_snapshot is None and not published_cohort_exists:
            # One-time compatibility path for repositories that predate the
            # cohort manifest pointer. Once a cohort exists it is authoritative.
            try:
                previous_snapshot = _latest_snapshot(snapshots_root)
            except FileNotFoundError:
                pass
        if request.offline:
            if previous_snapshot is None:
                raise FileNotFoundError(
                    f"No published immutable snapshot available for offline source {source_name}"
                )
            snapshot = previous_snapshot
        else:
            _assert_monotonic_watermark(source_name, previous_snapshot, end)
            snapshot = snapshots_root / retrieval_id
            if snapshot.exists():
                raise FileExistsError(f"Immutable snapshot already exists: {snapshot}")
            snapshot.mkdir(parents=True)
            request_parameters: dict[str, Any]
            source_metrics: dict[str, Any] = {}
            snapshot_mode: str
            if source_name == "twm":
                configured = (
                    document.resolve_path(settings.local_path)
                    if settings.local_path
                    else None
                )
                files = request.twm_files or (
                    tuple(sorted(configured.glob("*.csv")))
                    if configured and configured.exists()
                    else ()
                )
                if not files:
                    raise FileNotFoundError(
                        "TWM collection requires at least one local CSV"
                    )
                from .sources.twm import collect_twm_files

                collect_twm_files(files, snapshot)
                snapshot_mode = (
                    "DELTA_UPSERT" if previous_snapshot is not None else "FULL_REPLACE"
                )
                watermark = end.isoformat()
                request_parameters = {"files": [str(path) for path in files]}
            elif source_name == "acartia":
                rows = _fetch_acartia(config)
                (snapshot / "payload.json").write_text(json.dumps(rows, default=str))
                configured = (
                    document.resolve_path(settings.local_path)
                    if settings.local_path
                    else None
                )
                supplemental_files = (
                    tuple(sorted(configured.rglob("*.csv")))
                    if configured and configured.exists()
                    else ()
                )
                for source_file in supplemental_files:
                    assert configured is not None
                    relative = source_file.relative_to(configured)
                    destination = snapshot / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, destination)
                # The Acartia current endpoint is not a documented historical
                # tombstone feed. After seeding state, updates are therefore
                # non-destructive upserts even when a row disappears upstream.
                snapshot_mode = (
                    "DELTA_UPSERT" if previous_snapshot is not None else "FULL_REPLACE"
                )
                watermark = end.isoformat()
                request_parameters = {
                    "endpoint": settings.url,
                    "supplemental_files": [str(path) for path in supplemental_files],
                }
            elif source_name in {"maplify", "inaturalist"}:
                start = request.start_date or (
                    date.fromisoformat(settings.min_date or config.min_date)
                    if request.full_refresh
                    else _default_start(previous_snapshot, config, source_name, end)
                )
                rows = (
                    _fetch_maplify(config, start, end)
                    if source_name == "maplify"
                    else _fetch_inaturalist(config, start, end)
                )
                (snapshot / "payload.json").write_text(json.dumps(rows, default=str))
                snapshot_mode = _range_snapshot_mode(
                    source_name, start, config, previous_snapshot
                )
                watermark = end.isoformat()
                bbox = settings.bbox or config.full_area
                request_parameters = {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "bbox": bbox.tuple(),
                }
            elif source_name == "gbif":
                start = request.start_date or date.fromisoformat(
                    settings.min_date or config.min_date
                )
                rows, dataset_metadata = _fetch_gbif(config, start, end)
                (snapshot / "payload.json").write_text(json.dumps(rows, default=str))
                (snapshot / "datasets.json").write_text(
                    json.dumps(dataset_metadata, default=str)
                )
                (snapshot / "gbif_policy.json").write_text(
                    json.dumps(
                        {
                            "max_coordinate_uncertainty_m": settings.max_coordinate_uncertainty_m,
                            "max_event_spread_km": settings.max_event_spread_km,
                        }
                    )
                )
                snapshot_mode = _range_snapshot_mode(
                    source_name, start, config, previous_snapshot
                )
                watermark = end.isoformat()
                request_parameters = {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "bbox": config.full_area.tuple(),
                    "taxon_key": settings.taxon_key,
                    "checklist_key": settings.checklist_key,
                    "basis_of_record": settings.basis_of_record,
                    "occurrence_status": settings.occurrence_status,
                    "has_coordinate": True,
                    "has_geospatial_issue": False,
                    "dataset_keys": [item.key for item in settings.dataset_allowlist],
                    "excluded_dataset_keys": list(settings.excluded_dataset_keys),
                    "page_size": settings.page_size,
                    "max_coordinate_uncertainty_m": settings.max_coordinate_uncertainty_m,
                    "max_event_spread_km": settings.max_event_spread_km,
                }
                source_metrics = {
                    "occurrence_row_count": len(rows),
                    "dataset_count": len(dataset_metadata),
                    "occurrences_by_dataset": dict(
                        sorted(
                            Counter(
                                str(item.get("datasetKey")) for item in rows
                            ).items()
                        )
                    ),
                }
            elif source_name == "cwr":
                source_metrics = collect_cwr_snapshot(
                    settings,
                    snapshot,
                    previous_snapshot=previous_snapshot,
                    full_refresh=request.full_refresh,
                )
                snapshot_mode = (
                    "DELTA_UPSERT" if previous_snapshot is not None else "FULL_REPLACE"
                )
                watermark = end.isoformat()
                request_parameters = {
                    "archive_index_url": settings.archive_index_url,
                    "archive_years": list(settings.archive_years),
                    "archive_year_url_template": settings.archive_year_url_template,
                    "archive_fetch_workers": settings.archive_fetch_workers,
                    "atlist_api_root": settings.atlist_api_root,
                    "atlist_maps": {
                        str(year): {
                            "map_id": item.map_id,
                            "page_url": item.page_url,
                        }
                        for year, item in sorted(settings.atlist_maps.items())
                    },
                    "archive_refresh": (
                        "full" if request.full_refresh else "reuse_latest"
                    ),
                }
            else:  # pragma: no cover - source names are configuration-validated
                raise ValueError(f"Unsupported sightings source: {source_name}")
            if snapshot_mode == "DELTA_UPSERT" and previous_snapshot is None:
                raise ValueError(
                    f"Cannot initialize {source_name} from a partial/delta collection; "
                    "run a complete configured-range collection first"
                )
            atomic_write_json(
                snapshot / "source_policy.json",
                {
                    "source_license": settings.source_license,
                    "source_use_class": settings.source_use_class,
                    "source_license_terms_url": settings.source_license_terms_url,
                    "source_attribution": settings.source_attribution,
                    "source_license_reviewed_at": settings.source_license_reviewed_at,
                    "source_license_version": settings.source_license_version,
                    "source_license_jurisdiction": settings.source_license_jurisdiction,
                    "max_coordinate_uncertainty_m": settings.max_coordinate_uncertainty_m,
                },
            )
            adapted = adapt_snapshot(source_name, snapshot)
            adapted_row_count = adapted.num_rows
            if source_name == "gbif":
                adapted_frame = adapted.select(
                    ["SOURCE_QC_STATUS", "SOURCE_QC_DETAIL", "SOURCE_OCCURRENCE_COUNT"]
                ).to_pandas()
                accepted = adapted_frame["SOURCE_QC_STATUS"].eq("ACCEPTED")
                exclusion_reasons: Counter[str] = Counter()
                excluded_occurrences_by_reason: Counter[str] = Counter()
                for excluded_row in adapted_frame.loc[~accepted].itertuples(
                    index=False
                ):
                    detail = json.loads(str(excluded_row.SOURCE_QC_DETAIL))
                    reasons = [str(reason) for reason in detail.get("reasons", [])]
                    exclusion_reasons.update(reasons)
                    for reason in reasons:
                        excluded_occurrences_by_reason[reason] += int(
                            excluded_row.SOURCE_OCCURRENCE_COUNT
                        )
                source_metrics.update(
                    {
                        "source_event_count": int(len(adapted_frame)),
                        "admitted_event_count": int(accepted.sum()),
                        "quarantined_event_count": int((~accepted).sum()),
                        "admitted_occurrence_count": int(
                            adapted_frame.loc[accepted, "SOURCE_OCCURRENCE_COUNT"].sum()
                        ),
                        "quarantined_occurrence_count": int(
                            adapted_frame.loc[
                                ~accepted, "SOURCE_OCCURRENCE_COUNT"
                            ].sum()
                        ),
                        "exclusions_by_reason": dict(sorted(exclusion_reasons.items())),
                        "excluded_occurrences_by_reason": dict(
                            sorted(excluded_occurrences_by_reason.items())
                        ),
                    }
                )
                if source_metrics["occurrence_row_count"] != (
                    source_metrics["admitted_occurrence_count"]
                    + source_metrics["quarantined_occurrence_count"]
                ):
                    raise ValueError(
                        "GBIF occurrence rows do not reconcile to admitted plus quarantined"
                    )
            observed_start, observed_through = _observed_date_bounds(adapted)
            requested_start = request_parameters.get("start")
            requested_through = request_parameters.get("end")
            configured_verified = (
                settings.verified_coverage_start,
                settings.verified_coverage_through,
            )
            if all(configured_verified):
                coverage_start, coverage_through = configured_verified
                coverage_status = "configured_verified"
            elif source_name in {"maplify", "inaturalist", "gbif"}:
                coverage_start = str(requested_start)
                coverage_through = str(requested_through)
                coverage_status = "request_complete"
            else:
                coverage_start = None
                coverage_through = None
                coverage_status = "observed_only"
            schema_fingerprint = hashlib.sha256(
                str(adapted.schema).encode()
            ).hexdigest()
            raw_profile = _raw_schema_profile(snapshot, source_name)
            raw_fingerprint = hashlib.sha256(
                json.dumps(raw_profile, sort_keys=True, default=str).encode()
            ).hexdigest()
            raw_response_sha256 = _raw_response_hash(snapshot)
            semantic_content_sha256 = _semantic_content_hash(adapted)
            adapted_semantic_hash = semantic_content_sha256
            metadata = {
                "source": source_name,
                "snapshot_mode": snapshot_mode,
                "retrieval_id": retrieval_id,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "request": request_parameters,
                "row_count": adapted.num_rows,
                "watermark": watermark,
                "requested_start": requested_start,
                "requested_through": requested_through,
                "observed_start": observed_start,
                "observed_through": observed_through,
                "coverage_start": coverage_start,
                "coverage_through": coverage_through,
                "coverage_status": coverage_status,
                "schema_fingerprint": schema_fingerprint,
                "raw_schema_fingerprint": raw_fingerprint,
                "raw_schema_profile": raw_profile,
                "raw_response_sha256": raw_response_sha256,
                "semantic_content_sha256": semantic_content_sha256,
                "semantic_row_count": adapted.num_rows,
                "source_license": settings.source_license,
                "source_use_class": settings.source_use_class,
                "source_license_terms_url": settings.source_license_terms_url,
                "source_attribution": settings.source_attribution,
                "source_license_reviewed_at": settings.source_license_reviewed_at,
                "source_license_version": settings.source_license_version,
                "source_license_jurisdiction": settings.source_license_jurisdiction,
                "file_checksums": {
                    path.relative_to(snapshot).as_posix(): checksum_path(path)
                    for path in sorted(snapshot.rglob("*"))
                    if path.is_file()
                },
                **source_metrics,
            }
            atomic_write_json(snapshot / "snapshot.json", metadata)
            source_latest_payloads[source_name] = {
                "snapshot": snapshot.relative_to(raw_root / source_name).as_posix(),
                "watermark": watermark,
                "cohort_retrieval_id": retrieval_id,
            }
        snapshot_metadata = json.loads((snapshot / "snapshot.json").read_text())
        snapshot_errors.extend(
            _snapshot_validation_errors(
                snapshot,
                source_name,
                expected_rows=adapted_row_count,
                expected_semantic_hash=adapted_semantic_hash,
            )
        )
        snapshot_schema_version = (
            "5"
            if snapshot_metadata.get("semantic_content_sha256")
            else (
                "4"
                if source_name in {"cwr", "gbif"}
                else "3" if "coverage_status" in snapshot_metadata else "1"
            )
        )
        outputs.append(
            ArtifactRef(
                kind="source",
                dataset_id=f"whale.sightings.source_{source_name}",
                path=snapshot,
                producer="whale.sightings.collect",
                schema_version=snapshot_schema_version,
                run_id=request.run_id,
                config_hash=document.config_hash,
                checksum=checksum_path(snapshot),
                file_count=sum(1 for path in snapshot.rglob("*") if path.is_file()),
                freshness="offline" if request.offline else "current",
            )
        )
    data_snapshot = build_data_snapshot(
        outputs,
        default_coverage_start=config.min_date,
    )
    outputs = [replace(item, data_snapshot=data_snapshot) for item in outputs]
    enabled_sources = {
        name for name, settings in config.collection.sources.items() if settings.enabled
    }
    output_sources = {
        (item.dataset_id or "").removeprefix("whale.sightings.source_")
        for item in outputs
    }
    if output_sources != enabled_sources:
        snapshot_errors.append(
            "Collected sources do not match configured enabled sources"
        )
    report = ValidationReport(
        valid=not snapshot_errors,
        dataset_id="whale.sightings.sources",
        errors=tuple(snapshot_errors),
        metrics={
            "source_count": len(outputs),
            "enabled_sources": sorted(enabled_sources),
        },
    )
    report.require_valid()
    signature, signature_payload = stage_signature(
        stage="whale.sightings.collect",
        semantic_version="9",
        config_hash=document.config_hash,
        inputs=tuple(outputs),
        parameters={
            "start_date": request.start_date,
            "end_date": end,
            "offline": request.offline,
            "full_refresh": request.full_refresh,
            "enabled_sources": sorted(enabled_sources),
        },
    )
    manifest = RunManifest(
        run_id=request.run_id,
        workflow="whale.sightings.collect",
        config_hash=document.config_hash,
        resolved_config=document.redacted_data(),
        outputs=tuple(outputs),
        source_snapshots=tuple(outputs),
        data_snapshot=data_snapshot,
        code_revision=code_revision(),
        schema_version="9",
        stage_signature=signature,
        stages=(
            {
                "name": "collect",
                "semantic_version": "9",
                "signature": signature_payload,
                "source_count": len(outputs),
            },
        ),
    )
    manifest_path = raw_root / "manifests/collect" / f"{signature}.json"
    manifest.write(manifest_path, overwrite=request.force)
    # Per-source convenience pointers are advanced only after the complete
    # enabled cohort validates. The cohort manifest pointer is written last and
    # is the sole authoritative input for subsequent incremental collection.
    if not request.offline:
        for source_name in sorted(source_latest_payloads):
            atomic_write_json(
                raw_root / source_name / "latest.json",
                source_latest_payloads[source_name],
                overwrite=True,
            )
    cohort_pointer = raw_root / "manifests/latest.json"
    atomic_write_json(
        cohort_pointer,
        {
            "manifest": manifest_path.relative_to(cohort_pointer.parent).as_posix(),
            "run_id": request.run_id,
            "stage_signature": signature,
        },
        overwrite=True,
    )
    return StageResult(tuple(outputs), (report,), manifest)


from .http import (
    _request_json,
    _retry_after_seconds,
    _require_unique_ids,
    _require_unique_composite_ids,
)


def _fetch_inaturalist(config, start, end):
    from .sources.inaturalist import fetch_inaturalist

    return fetch_inaturalist(
        config.collection.sources["inaturalist"],
        start,
        end,
        bbox=config.full_area,
        taxon_id=config.collection.sources["inaturalist"].taxon_id,
        request_json=_request_json,
    )


def _fetch_acartia(config):
    from .sources.acartia import fetch_acartia

    return fetch_acartia(
        config.collection.sources["acartia"], request_json=_request_json
    )


def _fetch_maplify(config, start, end):
    from .sources.maplify import fetch_maplify

    return fetch_maplify(
        config.collection.sources["maplify"],
        start,
        end,
        bbox=config.full_area,
        request_json=_request_json,
    )


def _fetch_gbif(config, start, end):
    from .sources.gbif import fetch_gbif

    return fetch_gbif(
        config.collection.sources["gbif"],
        start,
        end,
        bbox=config.full_area,
        request_json=_request_json,
    )
