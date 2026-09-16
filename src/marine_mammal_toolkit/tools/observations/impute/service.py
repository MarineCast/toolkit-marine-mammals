from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path
from marine_mammal_toolkit.tools._core.config.paths import project_root
from marine_mammal_toolkit.tools._core.data import DATASETS
from marine_mammal_toolkit.tools._core.data import ArtifactStore
from marine_mammal_toolkit.tools._core.data import ProcessingMode
from marine_mammal_toolkit.tools._core.data import StageResult
from marine_mammal_toolkit.tools._core.data import ValidationReport

from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.schemas.observations import IMPUTED_OBSERVATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import ImputationRequest
from marine_mammal_toolkit.tools.observations.runtime import code_revision
from marine_mammal_toolkit.tools.observations.runtime import resume_result
from marine_mammal_toolkit.tools.observations.runtime import stage_signature
from marine_mammal_toolkit.tools.observations.runtime import with_imputation_as_of
from marine_mammal_toolkit.tools.observations.impute.model import (
    DIAGNOSTIC_META_COLUMNS,
)
from marine_mammal_toolkit.tools.observations.impute.model import (
    SelectiveDateContextImputer,
)
from marine_mammal_toolkit.tools.observations.impute.pipeline import resolve_model_path
from marine_mammal_toolkit.tools.observations.impute.pipeline import (
    validate_training_input_provenance,
)
from marine_mammal_toolkit.tools.observations.impute.report import (
    build_imputation_report_payload,
)
from marine_mammal_toolkit.tools.observations.impute.report import write_imputation_html


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_for_data_root(configured_output: Path, data_root: Path) -> Path:
    """Rebase the canonical configured data path into a run-scoped data root.

    The normal CLI still resolves to the same destination.  Transactional
    release runs use an isolated data root, so writing the project-relative
    ``data/...`` path directly would otherwise mutate the currently published
    artifact before the release gates complete.
    """

    canonical_data_root = (project_root() / "data").resolve()
    resolved_output = configured_output.resolve()
    try:
        relative = resolved_output.relative_to(canonical_data_root)
    except ValueError:
        if data_root.resolve() != canonical_data_root:
            raise ValueError(
                "Run-scoped imputation requires artifacts.output beneath the canonical data root"
            )
        return resolved_output
    return data_root.resolve() / relative


def _write_diagnostics(frame: pd.DataFrame, destination: Path) -> Path:
    """Persist row-level diagnostics atomically outside the canonical table."""

    selected = [
        name
        for name in (
            "OBSERVATION_ID",
            "ENCOUNTER_ID",
            "ECOTYPE_DETAIL_OBSERVED",
            "P_SRKW_RAW",
            "P_SRKW",
            "P_TRANSIENT",
            "P_OTHER",
            "PREDICTED_CLASS",
            "CONFORMAL_SET",
            "CONFORMAL_SET_SIZE",
            "CONFORMAL_P_TRANSIENT",
            "CONFORMAL_P_SRKW",
            "OOD_SCORE",
            "OOD_THRESHOLD",
            "OOD_MARGIN",
            "OOD_INLIER",
            "OTHER_SUPPORT_VETO",
            "EVIDENCE_REGIME",
            "ABSTENTION_REASON",
            "ACCEPTANCE_THRESHOLD",
            "ACCEPTANCE_POLICY_SCOPE",
            "ACCEPTANCE_POLICY_ERROR_UPPER",
            "POLICY_ACCEPTED",
            "CLASS_CERTIFIED_FOR_HARD_LABEL",
            "HARD_LABEL_CERTIFIED",
            "SOFT_COUNT_CERTIFIED",
            "SOFT_CERTIFICATION_STRATUM_SUPPORTED",
            "ENCOUNTER_LABEL_CONFLICT",
            "SRKW_DOMAIN_SUPPORTED",
            "TRANSIENT_DOMAIN_SUPPORTED",
            "IMPUTATION_DOMAIN_SUPPORTED",
            "BINARY_MODEL_DOMAIN_SUPPORTED",
            "MODEL_DOMAIN_STATUS",
            "P_SRKW_NO_SAME_DAY",
            "P_SRKW_NO_FUTURE",
            "P_SRKW_NO_NEAREST_ENCOUNTER",
            "MAX_STABILITY_PROBABILITY_SHIFT",
            "STABILITY_CLASS_AGREEMENT",
            "PREDICTION_STABLE",
            "STABILITY_EVALUATED",
            "IMPUTATION_MODEL_VERSION",
            "IMPUTATION_MODEL_RUN_ID",
            "IMPUTATION_DATA_SNAPSHOT_ID",
            *DIAGNOSTIC_META_COLUMNS,
        )
        if name in frame
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.loc[:, selected].to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def impute_sightings(request: ImputationRequest) -> StageResult:
    if request.observations is None:
        raise ValueError("Imputation requires an explicit observations artifact")
    if request.model_path is None:
        raise ValueError("Imputation requires an explicit fitted model path")
    if request.mode is ProcessingMode.AS_OF:
        raise NotImplementedError(
            "AS_OF selective imputation requires a time-frozen model and is not yet enabled"
        )
    if request.knowledge_cutoff is not None:
        raise ValueError(
            "knowledge_cutoff is not supported for retrospective imputation; "
            "AS_OF remains disabled until the full model state can be replayed"
        )
    model_path = resolve_model_path(request.model_path)
    model_checksum = _sha256(model_path)
    document, settings = load_sightings_config(request.config)
    configured_output = _output_for_data_root(
        document.resolve_path(settings.imputation.artifacts.output), request.data_root
    )
    input_artifacts = tuple(
        item
        for item in (request.observations, request.associations)
        if item is not None
    )
    signature, signature_payload = stage_signature(
        stage="whale.sightings.impute",
        semantic_version="10",
        config_hash=document.config_hash,
        inputs=input_artifacts,
        parameters={
            "mode": request.mode.value,
            "knowledge_cutoff": request.knowledge_cutoff,
            "input_schema_version": request.observations.schema_version,
            "model_path": str(model_path),
            "model_sha256": model_checksum,
            "configured_output": str(configured_output),
        },
    )
    manifest_path = request.data_root / (
        f"processed/domain/whale_layer/sightings/manifests/impute/{request.mode.value}/{signature}.json"
    )
    resumed = resume_result(
        enabled=request.resume,
        manifest_path=manifest_path,
        config_hash=document.config_hash,
        inputs=input_artifacts,
        signature=signature,
    )
    if resumed is not None:
        return resumed
    if request.observations.data_snapshot is None:
        raise ValueError("Imputation input is missing required data snapshot metadata")
    imputation_as_of = datetime.now(timezone.utc).isoformat()
    data_snapshot = with_imputation_as_of(
        request.observations.data_snapshot, imputation_as_of
    )
    frame = pd.read_parquet(request.observations.path)
    required = {
        "OBSERVATION_ID",
        "SIGHTING_DATE_UTC",
        "SOURCE_EVENT_AT_UTC",
        "AVAILABLE_AT_UTC",
        "LATITUDE",
        "LONGITUDE",
        "ECOTYPE_DETAIL",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Imputation input is missing columns: {missing}")
    input_count = len(frame)
    available_at = pd.to_datetime(
        frame["AVAILABLE_AT_UTC"],
        utc=True,
    )
    input_max = available_at.max()
    original_ids = frame["OBSERVATION_ID"].astype(str).tolist()
    imputer = SelectiveDateContextImputer.load(model_path)
    if imputer.config.regime != "retrospective":
        raise ValueError(
            "Retrospective pipeline imputation requires a retrospectively fitted model; "
            f"found regime={imputer.config.regime!r}"
        )
    validate_training_input_provenance(
        imputer,
        observations_path=request.observations.path,
        associations_path=(
            request.associations.path if request.associations is not None else None
        ),
        data_snapshot_id=request.observations.data_snapshot.snapshot_id,
    )
    result = imputer.apply_to_all(frame)
    result["IMPUTATION_MODEL_VERSION"] = imputer.training_summary_.get(
        "MODEL_VERSION", "UNKNOWN"
    )
    result["IMPUTATION_MODEL_FIT_AT_UTC"] = pd.to_datetime(
        imputer.training_summary_.get("FIT_AT_UTC"), utc=True, errors="coerce"
    )
    result["IMPUTATION_MODEL_RUN_ID"] = imputer.training_summary_.get("FIT_RUN_ID")
    result["IMPUTATION_DATA_SNAPSHOT_ID"] = (
        request.observations.data_snapshot.snapshot_id
    )
    unknown = result["ECOTYPE_DETAIL_OBSERVED"].eq("UNKNOWN")
    domains_configured = bool(getattr(imputer, "model_domain_geometries_", {}))
    result["MODEL_DOMAIN_STATUS"] = "NOT_APPLICABLE"
    if not domains_configured:
        result.loc[unknown, "MODEL_DOMAIN_STATUS"] = "MODEL_DOMAIN_NOT_CONFIGURED"
    else:
        binary_supported = (
            result.get(
                "BINARY_MODEL_DOMAIN_SUPPORTED", pd.Series(False, index=result.index)
            )
            .fillna(False)
            .astype(bool)
        )
        partial_supported = result.get(
            "SRKW_DOMAIN_SUPPORTED", pd.Series(False, index=result.index)
        ).fillna(False).astype(bool) | result.get(
            "TRANSIENT_DOMAIN_SUPPORTED", pd.Series(False, index=result.index)
        ).fillna(
            False
        ).astype(
            bool
        )
        result.loc[unknown, "MODEL_DOMAIN_STATUS"] = "OUTSIDE_MODEL_SUPPORT"
        result.loc[unknown & partial_supported, "MODEL_DOMAIN_STATUS"] = (
            "PARTIAL_MODEL_SUPPORT"
        )
        result.loc[unknown & binary_supported, "MODEL_DOMAIN_STATUS"] = "SUPPORTED"
    for boolean_name in (
        "SOFT_COUNT_CERTIFIED",
        "HARD_LABEL_CERTIFIED",
        "STABILITY_EVALUATED",
        "ENCOUNTER_LABEL_CONFLICT",
        "SRKW_DOMAIN_SUPPORTED",
        "TRANSIENT_DOMAIN_SUPPORTED",
        "IMPUTATION_DOMAIN_SUPPORTED",
        "BINARY_MODEL_DOMAIN_SUPPORTED",
    ):
        if boolean_name not in result:
            result[boolean_name] = False
        result[boolean_name] = result[boolean_name].fillna(False).astype(bool)
    probabilistically_labeled = result["ECOTYPE_DETAIL_OBSERVED"].eq(
        "UNKNOWN"
    ) & result[["P_SRKW", "P_TRANSIENT"]].notna().any(axis=1)
    imputed_or_reclassified = (
        result["IMPUTATION_APPLIED"].fillna(False).astype(bool)
        | result["ECOTYPE_DETAIL_EFFECTIVE"].ne(result["ECOTYPE_DETAIL_OBSERVED"])
        | probabilistically_labeled
    )
    result.loc[imputed_or_reclassified, "LABEL_AVAILABLE_AT_UTC"] = pd.Timestamp(
        imputation_as_of
    )
    for field in IMPUTED_OBSERVATION_SCHEMA:
        if field.name not in result:
            result[field.name] = pd.NA
    if result["OBSERVATION_ID"].astype(str).tolist() != original_ids:
        raise ValueError("Imputation changed observation identity or order")
    table = pa.Table.from_pandas(
        result[IMPUTED_OBSERVATION_SCHEMA.names], preserve_index=False
    ).cast(IMPUTED_OBSERVATION_SCHEMA)
    spec = replace(
        DATASETS.get(f"whale.sightings.imputed_{request.mode.value}"),
        schema=IMPUTED_OBSERVATION_SCHEMA,
        schema_version="9",
        path_template=str(configured_output),
    )
    store = ArtifactStore(
        data_root=request.data_root,
        artifact_root=request.artifact_root,
        output_root=request.output_root,
    )
    artifact, report = store.write_table(
        table,
        spec,
        run_id=request.run_id,
        producer="whale.sightings.impute",
        config_hash=document.config_hash,
        inputs=input_artifacts,
        mode=request.mode,
        knowledge_cutoff=request.knowledge_cutoff,
        data_snapshot=data_snapshot,
        force=request.force,
    )
    diagnostics_path = configured_output.with_name(
        f"imputation_diagnostics_{request.mode.value}.parquet"
    )
    _write_diagnostics(result, diagnostics_path)
    diagnostics_artifact = ArtifactRef(
        kind="diagnostic",
        dataset_id=f"whale.sightings.imputation_diagnostics_{request.mode.value}",
        path=diagnostics_path,
        producer="whale.sightings.impute.diagnostics.v1",
        schema_version="1",
        run_id=request.run_id,
        config_hash=document.config_hash,
        checksum=checksum_path(diagnostics_path),
        row_count=len(result),
        file_count=1,
        inputs=tuple(item.checksum or str(item.path) for item in input_artifacts)
        + (model_checksum, artifact.checksum or str(artifact.path)),
        processing_mode=request.mode.value,
        knowledge_cutoff=request.knowledge_cutoff,
        data_snapshot=data_snapshot,
        sensitivity="internal",
    )
    report_root = configured_output.parent
    report_stem = f"imputation_report_{request.mode.value}"
    report_json_path = report_root / f"{report_stem}.json"
    report_html_path = report_root / f"{report_stem}.html"
    report_payload = build_imputation_report_payload(
        imputer=imputer,
        predictions=result,
        metadata={
            "run_id": request.run_id,
            "processing_mode": request.mode.value,
            "imputation_as_of": imputation_as_of,
            "model_path": str(model_path),
            "model_sha256": model_checksum,
            "fit_report_path": str(model_path.parent / "fit_report.html"),
            "predictions_path": str(artifact.path),
            "predictions_sha256": artifact.checksum,
            "normalization_snapshot_id": request.observations.data_snapshot.snapshot_id,
        },
    )
    atomic_write_json(report_json_path, report_payload, overwrite=request.force)
    write_imputation_html(
        report_html_path,
        payload=report_payload,
        overwrite=request.force,
    )
    report_inputs = tuple(
        item.checksum or str(item.path) for item in input_artifacts
    ) + (
        model_checksum,
        artifact.checksum or str(artifact.path),
    )
    report_artifacts = tuple(
        ArtifactRef(
            kind="report",
            dataset_id=f"whale.sightings.imputation_report_{request.mode.value}_{format_name}",
            path=path,
            producer="whale.sightings.impute.report.v1",
            schema_version="1",
            run_id=request.run_id,
            config_hash=document.config_hash,
            inputs=report_inputs,
            checksum=checksum_path(path),
            file_count=1,
            processing_mode=request.mode.value,
            knowledge_cutoff=request.knowledge_cutoff,
            data_snapshot=data_snapshot,
        )
        for format_name, path in (
            ("html", report_html_path),
            ("json", report_json_path),
        )
    )
    outputs = (artifact, diagnostics_artifact, *report_artifacts)
    excluded_count = input_count - len(result)
    output_max = pd.to_datetime(
        result["AVAILABLE_AT_UTC"],
        utc=True,
    ).max()
    encounter_sizes = result.drop_duplicates("ENCOUNTER_ID")[
        "ENCOUNTER_SIZE"
    ].value_counts()
    identity_report = ValidationReport(
        valid=artifact.row_count == len(original_ids),
        dataset_id=str(spec.dataset_id),
        errors=(
            ()
            if artifact.row_count == len(original_ids)
            else ("Imputation row count changed",)
        ),
        metrics={
            "input_count": input_count,
            "output_count": len(result),
            "excluded_after_cutoff": excluded_count,
            "anchor_count": int(
                result["ECOTYPE_DETAIL_OBSERVED"].isin(["SRKW", "TRANSIENT"]).sum()
            ),
            "query_count": int(result["ECOTYPE_DETAIL_OBSERVED"].eq("UNKNOWN").sum()),
            "imputed_count": int(result["IMPUTATION_APPLIED"].sum()),
            "imputed_by_ecotype": report_payload["imputation"][
                "hard_imputed_points_by_ecotype"
            ],
            "probabilistic_count": int(result["USE_FOR_PROBABILISTIC_COUNTS"].sum()),
            "imputation_report_html": str(report_html_path),
            "imputation_report_json": str(report_json_path),
            "model_path": str(model_path),
            "model_sha256": model_checksum,
            "model_version": imputer.training_summary_.get("MODEL_VERSION"),
            "model_fit_at_utc": imputer.training_summary_.get("FIT_AT_UTC"),
            "training_observations_sha256": imputer.training_summary_.get(
                "OBSERVATIONS_SHA256"
            ),
            "training_associations_sha256": imputer.training_summary_.get(
                "ASSOCIATIONS_SHA256"
            ),
            "training_data_snapshot_id": imputer.training_summary_.get(
                "TRAINING_DATA_SNAPSHOT_ID"
            ),
            "class_certification": imputer.training_summary_.get(
                "CLASS_CERTIFICATION", {}
            ),
            "observed_ecotype_distribution": {
                str(label): int(count)
                for label, count in result["ECOTYPE_DETAIL_OBSERVED"]
                .value_counts(dropna=False)
                .items()
            },
            "effective_ecotype_distribution": {
                str(label): int(count)
                for label, count in result["ECOTYPE_DETAIL_EFFECTIVE"]
                .value_counts(dropna=False)
                .items()
            },
            "abstention_reason_distribution": {
                str(reason): int(count)
                for reason, count in result["ABSTENTION_REASON"]
                .fillna("NOT_APPLICABLE")
                .value_counts()
                .items()
            },
            "encounter_size_distribution": {
                str(int(size)): int(count) for size, count in encounter_sizes.items()
            },
            "input_max_timestamp": (
                input_max.isoformat() if pd.notna(input_max) else None
            ),
            "output_max_timestamp": (
                output_max.isoformat() if pd.notna(output_max) else None
            ),
        },
    )
    identity_report.require_valid()
    manifest = RunManifest(
        run_id=request.run_id,
        workflow="whale.sightings.impute",
        config_hash=document.config_hash,
        resolved_config=document.redacted_data(),
        inputs=input_artifacts,
        outputs=outputs,
        data_snapshot=data_snapshot,
        code_revision=code_revision(),
        schema_version="9",
        stage_signature=signature,
        stages=(
            {
                "name": "impute",
                "semantic_version": "10",
                "signature": signature_payload,
                "method": imputer.training_summary_.get("MODEL_VERSION"),
                "mode": request.mode.value,
                "knowledge_cutoff": request.knowledge_cutoff,
                "imputation_as_of": imputation_as_of,
                **identity_report.metrics,
            },
        ),
    )
    manifest.write(manifest_path, overwrite=request.force)
    latest_pointer = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/manifests/impute/{request.mode.value}/latest.json"
    )
    atomic_write_json(
        latest_pointer,
        {
            "manifest": manifest_path.relative_to(latest_pointer.parent).as_posix(),
            "stage_signature": signature,
        },
        overwrite=True,
    )
    report_validation = ValidationReport(
        True,
        "whale.sightings.imputation_report",
        metrics={
            "html_path": str(report_html_path),
            "json_path": str(report_json_path),
            "hard_imputed_points": report_payload["imputation"]["hard_imputed_points"],
            "hard_imputed_points_by_ecotype": report_payload["imputation"][
                "hard_imputed_points_by_ecotype"
            ],
        },
    )
    return StageResult(
        manifest.outputs, (report, identity_report, report_validation), manifest
    )
