"""Run-scoped orchestration for the complete sightings release DAG."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import DataSnapshotMetadata
from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path
from marine_mammal_toolkit.tools._core.data import ProcessingMode
from marine_mammal_toolkit.tools._core.data import StageResult

from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.schemas.observations import CountRequest
from marine_mammal_toolkit.tools.schemas.observations import ImputationRequest
from marine_mammal_toolkit.tools.schemas.observations import IntensityRequest
from marine_mammal_toolkit.tools.schemas.observations import ModelGridRequest
from marine_mammal_toolkit.tools.schemas.observations import NormalizationRequest
from marine_mammal_toolkit.tools.schemas.observations import SightingsCollectionRequest
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    SightingsReleaseProfile,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    promote_sightings_release,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    release_profile,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    resolve_sightings_release_artifact,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    resolve_sightings_release_manifest,
)


class SightingsReleaseBlocked(RuntimeError):
    """A safety gate prevented promotion while leaving the prior release intact."""

    def __init__(self, message: str, *, candidate_report: Path):
        super().__init__(message)
        self.candidate_report = candidate_report


@dataclass(frozen=True)
class SightingsPipelineRunRequest:
    config: Path
    data_root: Path
    artifact_root: Path
    output_root: Path
    end_date: date
    profile: str = "production-retrospective"
    start_date: date | None = None
    run_id: str | None = None
    offline: bool = False
    twm_files: tuple[Path, ...] = ()
    force: bool = False
    resume: bool = False


@dataclass(frozen=True)
class SightingsPipelineRunResult:
    release_manifest: Path
    candidate_root: Path
    stage_results: Mapping[str, StageResult]
    gates: tuple[Mapping[str, Any], ...]


def _artifact(
    artifacts: Iterable[ArtifactRef], dataset_id: str, *, required: bool = True
) -> ArtifactRef | None:
    matches = [item for item in artifacts if item.dataset_id == dataset_id]
    if len(matches) > 1:
        raise ValueError(f"Stage produced duplicate {dataset_id} artifacts")
    if not matches:
        if required:
            raise ValueError(f"Stage did not produce required artifact {dataset_id}")
        return None
    return matches[0]


def _load_run_manifest(path: Path) -> RunManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshot_payload = payload.get("data_snapshot")
    snapshot = (
        DataSnapshotMetadata.from_dict(snapshot_payload)
        if snapshot_payload is not None
        else None
    )
    return RunManifest(
        run_id=str(payload["run_id"]),
        workflow=str(payload["workflow"]),
        config_hash=str(payload["config_hash"]),
        resolved_config=dict(payload.get("resolved_config") or {}),
        inputs=tuple(ArtifactRef.from_dict(item) for item in payload.get("inputs", ())),
        outputs=tuple(
            ArtifactRef.from_dict(item) for item in payload.get("outputs", ())
        ),
        source_snapshots=tuple(
            ArtifactRef.from_dict(item) for item in payload.get("source_snapshots", ())
        ),
        stages=tuple(payload.get("stages", ())),
        status=str(payload.get("status") or "complete"),
        failure=payload.get("failure"),
        code_revision=payload.get("code_revision"),
        schema_version=str(payload.get("schema_version") or "1"),
        stage_signature=payload.get("stage_signature"),
        data_snapshot=snapshot,
        created_at=str(
            payload.get("created_at") or datetime.now(timezone.utc).isoformat()
        ),
    )


def _latest_collect(data_root: Path) -> StageResult:
    release_pointer = (
        data_root / "processed/domain/whale_layer/sightings/releases/latest.json"
    )
    if release_pointer.is_file():
        release_manifest_path = resolve_sightings_release_manifest(release_pointer)
        payload = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        identity = payload.get("identity") or {}
        source_dataset_ids = {
            f"whale.sightings.source_{source}"
            for source in ("twm", "acartia", "maplify", "inaturalist", "cwr", "gbif")
        }
        inventory = {
            str(item.get("dataset_id")): item
            for item in payload.get("inventory", ())
            if isinstance(item, dict)
            and str(item.get("dataset_id")) in source_dataset_ids
        }
        if inventory:
            artifacts: list[ArtifactRef] = []
            generation = release_manifest_path.parent.resolve()
            for dataset_id, entry in sorted(inventory.items()):
                binding = resolve_sightings_release_artifact(
                    release_manifest_path, dataset_id, verify_checksum=False
                )
                relative = Path(str(entry.get("path") or ""))
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                ):
                    raise ValueError(f"Invalid released source path: {relative}")
                source_path = (generation / relative).resolve()
                if (
                    not source_path.is_relative_to(generation)
                    or not source_path.is_dir()
                ):
                    raise ValueError(
                        f"Released source snapshot is missing or unsafe: {source_path}"
                    )
                if source_path != binding.path:
                    raise ValueError(f"Released source binding changed: {dataset_id}")
                expected = str(entry.get("checksum") or "")
                if not expected or checksum_path(source_path) != expected:
                    raise ValueError(f"Released source checksum mismatch: {dataset_id}")
                artifacts.append(
                    ArtifactRef(
                        kind="raw",
                        dataset_id=dataset_id,
                        path=source_path,
                        producer=str(
                            entry.get("producer") or "whale.sightings.collect"
                        ),
                        schema_version=str(entry.get("schema_version") or "1"),
                        run_id=str(payload.get("run_id") or "released-source-cohort"),
                        config_hash=str(identity.get("config_hash") or ""),
                        checksum=expected,
                        row_count=(
                            int(entry["row_count"])
                            if entry.get("row_count") is not None
                            else None
                        ),
                        file_count=int(entry.get("file_count") or 0),
                        processing_mode=(
                            str(entry["processing_mode"])
                            if entry.get("processing_mode")
                            else None
                        ),
                        freshness="released",
                        sensitivity=str(entry.get("sensitivity") or "internal"),
                    )
                )
            snapshot_ids = {
                str(entry.get("snapshot_id"))
                for entry in inventory.values()
                if entry.get("snapshot_id")
            }
            if len(snapshot_ids) > 1:
                raise ValueError("Released sources do not share one cohort snapshot id")
            manifest = RunManifest(
                run_id=str(payload.get("run_id") or "released-source-cohort"),
                workflow="whale.sightings.collect",
                config_hash=str(identity.get("config_hash") or ""),
                resolved_config={
                    "release_id": payload.get("release_id"),
                    "release_manifest": str(release_manifest_path),
                },
                outputs=tuple(artifacts),
                source_snapshots=tuple(artifacts),
                status="complete",
                schema_version="release-replay-v1",
                stage_signature=f"released-source-cohort-{payload.get('release_id')}",
                created_at=str(
                    payload.get("created_at_utc")
                    or datetime.now(timezone.utc).isoformat()
                ),
            )
            return StageResult(
                outputs=tuple(artifacts),
                validations=(),
                manifest=manifest,
                skipped=True,
            )

    pointer_path = data_root / "raw/whale/sightings/manifests/latest.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"Offline sightings release requires a complete collect pointer: {pointer_path}"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(pointer.get("manifest") or ""))
    if not manifest_path.is_absolute():
        manifest_path = (pointer_path.parent / manifest_path).resolve()
    manifest = _load_run_manifest(manifest_path)
    return StageResult(
        outputs=manifest.outputs, validations=(), manifest=manifest, skipped=True
    )


def _universe_artifacts(
    data_root: Path, resolutions: tuple[int, ...]
) -> tuple[ArtifactRef, ...]:
    root = data_root / "processed/environment/seascape/full_counting"
    manifest_path = root / "_dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing full counting universe manifest: {manifest_path}. "
            "Build the sightings water universes first."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered = {
        item.dataset_id: item
        for item in (
            ArtifactRef.from_dict(value) for value in payload.get("outputs", ())
        )
    }
    result: list[ArtifactRef] = []
    for resolution in resolutions:
        dataset_id = f"environment.seascape.h3_full_counting_universe_r{resolution}"
        artifact = registered.get(dataset_id)
        if artifact is None or not artifact.path.exists():
            raise FileNotFoundError(
                f"Missing registered full counting universe H{resolution}"
            )
        if artifact.checksum and checksum_path(artifact.path) != artifact.checksum:
            raise ValueError(f"Full counting universe checksum mismatch: {dataset_id}")
        result.append(artifact)
    return tuple(result)


def estimate_sightings_build(
    *,
    data_root: Path,
    profile: SightingsReleaseProfile,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Return an intentionally conservative dense-build resource estimate."""

    if start_date > end_date:
        raise ValueError("start_date cannot follow end_date")
    periods = {
        "daily": (end_date - start_date).days + 1,
        "weekly": math.ceil(((end_date - start_date).days + 1) / 7),
    }
    cells_by_resolution: dict[int, int | None] = {}
    manifest_path = (
        data_root
        / "processed/environment/seascape/full_counting/_dataset_manifest.json"
    )
    outputs: dict[str, ArtifactRef] = {}
    if manifest_path.is_file():
        outputs = {
            item.dataset_id or "": item
            for item in (
                ArtifactRef.from_dict(value)
                for value in json.loads(manifest_path.read_text()).get("outputs", ())
            )
        }
    for resolution in profile.resolutions:
        artifact = outputs.get(
            f"environment.seascape.h3_full_counting_universe_r{resolution}"
        )
        cells_by_resolution[resolution] = (
            artifact.row_count if artifact is not None else None
        )
    estimated_rows: int | None = 0
    for resolution, cells in cells_by_resolution.items():
        del resolution
        if cells is None:
            estimated_rows = None
            break
        estimated_rows += (
            int(cells) * sum(periods[item] for item in profile.frequencies) * 3
        )
    # Observed Parquet partitions in this pipeline are normally 45-130 bytes/row.
    # Use the high end and a deliberately modest validation throughput so a dry
    # run does not understate operational cost.
    estimated_disk_bytes = estimated_rows * 130 if estimated_rows is not None else None
    estimated_runtime_seconds = (
        math.ceil(estimated_rows / 100_000) if estimated_rows is not None else None
    )
    return {
        "profile": profile.name,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "periods": {key: periods[key] for key in profile.frequencies},
        "cells_by_resolution": cells_by_resolution,
        "estimated_dense_rows": estimated_rows,
        "estimated_disk_bytes_upper": estimated_disk_bytes,
        "estimated_validation_runtime_seconds_lower_bound": estimated_runtime_seconds,
        "estimate_semantics": "planning bound; excludes collection, normalization, and fitting",
    }


def _coverage_gate(
    observations: ArtifactRef,
    *,
    profile: SightingsReleaseProfile,
    end_date: date,
) -> dict[str, Any]:
    snapshot = observations.data_snapshot
    status = snapshot.coverage_status if snapshot is not None else "missing"
    coverage_through = snapshot.coverage_through if snapshot is not None else None
    passed = (
        snapshot is not None
        and status == "verified_intersection"
        and coverage_through is not None
        and date.fromisoformat(coverage_through) >= end_date
    )
    return {
        "name": "verified_target_cohort",
        "required": profile.require_verified_cohort,
        "passed": passed,
        "coverage_status": status,
        "coverage_through": coverage_through,
        "end_date": end_date.isoformat(),
        "failure_semantics": (
            "Dense zero/no-report rows are prohibited; sparse reported facts remain valid"
        ),
    }


def _license_gate(
    outputs: Iterable[ArtifactRef],
    source_snapshots: Iterable[ArtifactRef],
    *,
    required: bool,
) -> dict[str, Any]:
    source = _artifact(outputs, "whale.sightings.source_records", required=False)
    observations = _artifact(outputs, "whale.sightings.observations", required=False)
    if source is None or observations is None:
        return {
            "name": "public_license_evidence",
            "required": required,
            "passed": False,
            "reason": "normalization outputs are missing source/license lineage",
        }
    records = pd.read_parquet(
        source.path,
        columns=["SOURCE", "SOURCE_RECORD_ID", "SOURCE_LICENSE", "SOURCE_USE_CLASS"],
    )
    public = pd.read_parquet(
        observations.path,
        columns=["PUBLIC_RELEASE_ELIGIBLE", "SOURCE_RECORD_IDS"],
    )
    contributor_ids = {
        str(record_id)
        for values in public["SOURCE_RECORD_IDS"]
        if values is not None
        for record_id in list(values)
    }
    selected = records[
        records.SOURCE_RECORD_ID.astype(str).isin(contributor_ids)
    ].copy()
    license_value = (
        selected.SOURCE_LICENSE.fillna("").astype(str).str.strip().str.upper()
    )
    invalid = selected[
        selected.SOURCE_USE_CLASS.ne("REDISTRIBUTABLE")
        | license_value.isin({"", "UNKNOWN", "UNREVIEWED"})
    ]
    missing_ids = contributor_ids - set(selected.SOURCE_RECORD_ID.astype(str))
    evidence_fields = (
        "source_license_terms_url",
        "source_attribution",
        "source_license_reviewed_at",
        "source_license_version",
        "source_license_jurisdiction",
    )
    snapshot_evidence: dict[str, dict[str, Any]] = {}
    for artifact in source_snapshots:
        source_name = (
            str(artifact.dataset_id or "")
            .removeprefix("whale.sightings.source_")
            .upper()
        )
        metadata_path = artifact.path / "snapshot.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        )
        snapshot_evidence[source_name] = {
            field: metadata.get(field) for field in evidence_fields
        }
    contributing_sources = sorted(set(selected.SOURCE.astype(str).str.upper()))
    sources_without_evidence = [
        source_name
        for source_name in contributing_sources
        if any(
            not snapshot_evidence.get(source_name, {}).get(field)
            for field in evidence_fields
        )
    ]
    return {
        "name": "public_license_evidence",
        "required": required,
        "passed": invalid.empty and not missing_ids and not sources_without_evidence,
        "contributor_records": len(contributor_ids),
        "invalid_or_unreviewed_records": int(len(invalid) + len(missing_ids)),
        "sources_without_complete_license_evidence": sources_without_evidence,
        "default_for_missing_evidence": "INTERNAL_ONLY",
    }


def _imputation_gates(artifact: ArtifactRef | None) -> tuple[dict[str, Any], ...]:
    if artifact is None:
        return (
            {
                "name": "imputation_mass_conservation",
                "required": False,
                "passed": True,
                "reason": "observed-only profile",
            },
        )
    columns = pd.read_parquet(artifact.path)
    hard = columns["IMPUTATION_APPLIED"].fillna(False).astype(bool)
    observed_detail = columns.get("ECOTYPE_DETAIL_OBSERVED")
    unknown = (
        observed_detail.fillna("UNKNOWN").astype(str).str.upper().eq("UNKNOWN")
        if isinstance(observed_detail, pd.Series)
        else pd.Series(True, index=columns.index)
    )
    soft = (
        columns["USE_FOR_PROBABILISTIC_COUNTS"].fillna(False).astype(bool)
        & unknown
        & ~hard
    )
    if "SOFT_COUNT_CERTIFIED" in columns:
        soft_certified = columns["SOFT_COUNT_CERTIFIED"].fillna(False).astype(bool)
    else:
        soft_certified = pd.Series(False, index=columns.index)
    if "STABILITY_EVALUATED" in columns:
        stability_evaluated = columns["STABILITY_EVALUATED"].fillna(False).astype(bool)
    else:
        stability_evaluated = pd.Series(False, index=columns.index)
    stable = (
        columns.get("PREDICTION_STABLE", pd.Series(False, index=columns.index))
        .fillna(False)
        .astype(bool)
    )
    unsafe_soft = soft & (~soft_certified | ~stability_evaluated | ~stable)
    class_certified = (
        columns["CLASS_CERTIFIED_FOR_HARD_LABEL"].fillna(False).astype(bool)
    )
    hard_certified = (
        columns.get("HARD_LABEL_CERTIFIED", pd.Series(False, index=columns.index))
        .fillna(False)
        .astype(bool)
    )
    unsafe_hard = hard & (
        ~class_certified | ~hard_certified | ~stability_evaluated | ~stable
    )
    expected_names = [
        name
        for name in (
            "EXPECTED_SRKW_COUNT",
            "EXPECTED_TRANSIENT_COUNT",
            "EXPECTED_OTHER_COUNT",
            "EXPECTED_UNKNOWN_COUNT",
        )
        if name in columns
    ]
    expected = columns[expected_names].apply(pd.to_numeric, errors="coerce")
    finite = bool(np.isfinite(expected.to_numpy(dtype=float)).all())
    bounded = bool(expected.ge(0).all().all() and expected.le(1).all().all())
    unit_mass = bool(np.allclose(expected.sum(axis=1), 1.0, atol=1e-9))
    return (
        {
            "name": "certified_imputation_use",
            "required": True,
            "passed": not unsafe_soft.any() and not unsafe_hard.any(),
            "soft_count_rows": int(soft.sum()),
            "hard_imputed_rows": int(hard.sum()),
            "unsafe_soft_rows": int(unsafe_soft.sum()),
            "unsafe_hard_rows": int(unsafe_hard.sum()),
        },
        {
            "name": "imputation_mass_conservation",
            "required": True,
            "passed": finite and bounded and unit_mass,
            "components": expected_names,
            "finite": finite,
            "bounded": bounded,
            "unit_mass": unit_mass,
        },
    )


def _write_blocked_candidate(
    candidate_root: Path,
    *,
    request: SightingsPipelineRunRequest,
    gates: Iterable[Mapping[str, Any]],
    completed_stages: Iterable[str],
) -> Path:
    path = candidate_root / "BLOCKED_RELEASE.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1",
            "status": "blocked",
            "run_id": request.run_id,
            "profile": request.profile,
            "end_date": request.end_date.isoformat(),
            "completed_stages": list(completed_stages),
            "gates": [dict(item) for item in gates],
            "prior_release_unchanged": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        overwrite=True,
    )
    return path


def _required_failures(gates: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(item.get("name") or "unnamed_gate")
        for item in gates
        if bool(item.get("required", True)) and item.get("passed") is not True
    ]


def run_sightings_pipeline(
    request: SightingsPipelineRunRequest,
) -> SightingsPipelineRunResult:
    """Execute the manifest-linked DAG in isolation, then promote one generation."""

    from marine_mammal_toolkit.tools.observations.post_process.aggregation import (
        build_intensity,
    )
    from marine_mammal_toolkit.tools.observations.post_process.aggregation import (
        build_model_grid,
    )
    from marine_mammal_toolkit.tools.observations.collect.pipeline import (
        collect_sightings,
    )
    from marine_mammal_toolkit.tools.observations.post_process.counts import (
        build_counts,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.observations.imputation import (
        fit_imputation_model,
    )
    from marine_mammal_toolkit.tools.observations.impute.service import impute_sightings
    from marine_mammal_toolkit.tools.observations.impute.settings import (
        load_imputation_settings,
    )
    from marine_mammal_toolkit.tools.observations.process.pipeline import (
        normalize_sightings,
    )

    document, settings = load_sightings_config(request.config)
    profile = release_profile(request.profile)
    run_id = request.run_id or (
        f"sightings-{request.end_date:%Y%m%d}-{document.config_hash[:10]}-"
        f"{datetime.now(timezone.utc):%H%M%S}"
    )
    candidate_root = (
        request.data_root.resolve() / "_sightings_release_candidates" / run_id
    )
    candidate_data = candidate_root / "data"
    candidate_artifacts = candidate_root / "artifacts"
    candidate_outputs = candidate_root / "outputs"
    if candidate_root.exists() and any(candidate_root.iterdir()) and not request.resume:
        raise FileExistsError(
            f"Sightings release candidate already exists: {candidate_root}"
        )
    candidate_root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, StageResult] = {}
    if request.offline:
        stages["collect"] = _latest_collect(request.data_root.resolve())
    else:
        stages["collect"] = collect_sightings(
            SightingsCollectionRequest(
                config=document,
                data_root=candidate_data,
                artifact_root=candidate_artifacts,
                output_root=candidate_outputs,
                run_id=f"{run_id}-collect",
                force=request.force,
                resume=request.resume,
                offline=False,
                start_date=request.start_date,
                end_date=request.end_date,
                twm_files=request.twm_files,
                full_refresh=True,
            )
        )
    stages["normalize"] = normalize_sightings(
        NormalizationRequest(
            config=document,
            inputs=stages["collect"].outputs,
            data_root=candidate_data,
            artifact_root=candidate_artifacts,
            output_root=candidate_outputs,
            run_id=f"{run_id}-normalize",
            force=request.force,
            resume=request.resume,
        )
    )
    observations = _artifact(
        stages["normalize"].outputs, "whale.sightings.observations"
    )
    assert observations is not None
    associations = _artifact(
        stages["normalize"].outputs, "whale.sightings.associations", required=False
    )
    gates: list[Mapping[str, Any]] = [
        _coverage_gate(observations, profile=profile, end_date=request.end_date),
        _license_gate(
            stages["normalize"].outputs,
            stages["collect"].outputs,
            required=profile.public_by_default,
        ),
    ]
    failures = _required_failures(gates)
    if failures:
        report = _write_blocked_candidate(
            candidate_root,
            request=SightingsPipelineRunRequest(
                **{**request.__dict__, "run_id": run_id}
            ),
            gates=gates,
            completed_stages=stages,
        )
        raise SightingsReleaseBlocked(
            f"Sightings release blocked before dense processing: {', '.join(failures)}",
            candidate_report=report,
        )

    imputed: ArtifactRef | None = None
    model_manifest: RunManifest | None = None
    if profile.include_imputation:
        workflow = load_imputation_settings(request.config)
        fit = fit_imputation_model(
            observations_path=observations.path,
            associations_path=associations.path if associations is not None else None,
            models_dir=candidate_artifacts / "models/sighting_imputation",
            config=workflow.config,
            source_config_path=request.config,
            run_id=f"{run_id}-imputation-model",
            evaluate_strategies=workflow.evaluate_strategies,
        )
        model_artifact = ArtifactRef(
            kind="model",
            dataset_id="whale.sightings.imputation_model",
            path=fit.run_dir,
            producer="whale.sightings.imputation.fit",
            schema_version="1",
            run_id=run_id,
            config_hash=document.config_hash,
            checksum=checksum_path(fit.run_dir),
            file_count=sum(1 for item in fit.run_dir.rglob("*") if item.is_file()),
            inputs=tuple(
                item.checksum or str(item.path)
                for item in (observations, associations)
                if item is not None
            ),
            processing_mode=ProcessingMode.RETROSPECTIVE.value,
            data_snapshot=observations.data_snapshot,
        )
        model_manifest = RunManifest(
            run_id=f"{run_id}-imputation-model",
            workflow="whale.sightings.imputation.fit",
            config_hash=document.config_hash,
            resolved_config=document.redacted_data(),
            inputs=tuple(
                item for item in (observations, associations) if item is not None
            ),
            outputs=(model_artifact,),
            data_snapshot=observations.data_snapshot,
            schema_version="1",
            stage_signature=model_artifact.checksum,
        )
        stages["impute"] = impute_sightings(
            ImputationRequest(
                config=document,
                observations=observations,
                associations=associations,
                model_path=fit.model_path,
                data_root=candidate_data,
                artifact_root=candidate_artifacts,
                output_root=candidate_outputs,
                run_id=f"{run_id}-impute",
                mode=ProcessingMode.RETROSPECTIVE,
                force=request.force,
                resume=request.resume,
            )
        )
        imputed = _artifact(
            stages["impute"].outputs, "whale.sightings.imputed_retrospective"
        )
        assert imputed is not None
        gates.extend(_imputation_gates(imputed))
        failures = _required_failures(gates)
        if failures:
            report = _write_blocked_candidate(
                candidate_root,
                request=SightingsPipelineRunRequest(
                    **{**request.__dict__, "run_id": run_id}
                ),
                gates=gates,
                completed_stages=stages,
            )
            raise SightingsReleaseBlocked(
                f"Sightings release blocked after imputation: {', '.join(failures)}",
                candidate_report=report,
            )
    else:
        gates.extend(_imputation_gates(None))

    if profile.include_counts:
        universes = _universe_artifacts(
            request.data_root.resolve(), profile.resolutions
        )
        count_input = imputed or observations
        stages["counts"] = build_counts(
            CountRequest(
                config=document,
                observations=count_input,
                associations=associations,
                water_universes=universes,
                data_root=candidate_data,
                artifact_root=candidate_artifacts,
                output_root=candidate_outputs,
                run_id=f"{run_id}-counts",
                start_date=request.start_date or date.fromisoformat(settings.min_date),
                end_date=request.end_date,
                resolutions=profile.resolutions,
                mode=ProcessingMode.RETROSPECTIVE,
                force=request.force,
                resume=request.resume,
            )
        )
        count_validations = tuple(stages["counts"].validations)
        gates.append(
            {
                "name": "sparse_count_artifacts_validated",
                "required": True,
                "passed": bool(count_validations)
                and all(report.valid for report in count_validations),
                "validated_artifacts": sorted(
                    artifact.dataset_id or artifact.kind
                    for artifact in stages["counts"].outputs
                ),
                "validation_errors": sorted(
                    {error for report in count_validations for error in report.errors}
                ),
            }
        )
    if profile.include_model_grid:
        counts_artifact = _artifact(
            stages["counts"].outputs, "whale.sightings.ecotype_counts"
        )
        assert counts_artifact is not None
        universes = _universe_artifacts(
            request.data_root.resolve(), profile.resolutions
        )
        stages["model_grid"] = build_model_grid(
            ModelGridRequest(
                config=document,
                ecotype_counts=counts_artifact,
                water_universes=universes,
                data_root=candidate_data,
                artifact_root=candidate_artifacts,
                output_root=candidate_outputs,
                run_id=f"{run_id}-model-grid",
                start_date=request.start_date or date.fromisoformat(settings.min_date),
                end_date=request.end_date,
                resolutions=profile.resolutions,
                frequencies=profile.frequencies,
                mode=ProcessingMode.RETROSPECTIVE,
                force=request.force,
                resume=request.resume,
            )
        )
    if profile.include_intensity:
        grid = _artifact(
            stages["model_grid"].outputs, "whale.sightings.reported_sighting_grid"
        )
        assert grid is not None
        stages["intensity"] = build_intensity(
            IntensityRequest(
                config=document,
                model_grid=grid,
                data_root=candidate_data,
                artifact_root=candidate_artifacts,
                output_root=candidate_outputs,
                run_id=f"{run_id}-intensity",
                mode=ProcessingMode.RETROSPECTIVE,
                force=request.force,
                resume=request.resume,
            )
        )

    release_artifacts = [
        output for result in stages.values() for output in result.outputs
    ]
    stage_manifests = [
        result.manifest for result in stages.values() if result.manifest is not None
    ]
    if model_manifest is not None:
        release_artifacts.extend(model_manifest.outputs)
        stage_manifests.append(model_manifest)
    release_manifest = promote_sightings_release(
        release_root=(
            request.data_root.resolve()
            / "processed/domain/whale_layer/sightings/releases"
        ),
        profile=profile,
        end_date=request.end_date,
        config_hash=document.config_hash,
        run_id=run_id,
        artifacts=release_artifacts,
        stage_manifests=stage_manifests,
        gates=gates,
        metadata={
            "target_semantics": "reported-sighting activity, not whale absence or presence",
            "effort_adjusted": False,
            "candidate_run_id": run_id,
        },
    )
    return SightingsPipelineRunResult(
        release_manifest=release_manifest,
        candidate_root=candidate_root,
        stage_results=stages,
        gates=tuple(gates),
    )


__all__ = [
    "SightingsPipelineRunRequest",
    "SightingsPipelineRunResult",
    "SightingsReleaseBlocked",
    "estimate_sightings_build",
    "run_sightings_pipeline",
]
