from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path

from marine_mammal_toolkit.tools.schemas.observations import ASSOCIATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import OBSERVATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import OBSERVATION_SCHEMA_V8
from marine_mammal_toolkit.tools.observations.impute.config import ImputationConfig
from marine_mammal_toolkit.tools.observations.impute.io import (
    load_preprocessed_sightings,
)
from marine_mammal_toolkit.tools.observations.impute.io import write_json
from marine_mammal_toolkit.tools.observations.impute.io import write_predictions
from marine_mammal_toolkit.tools.observations.impute.model import (
    SelectiveDateContextImputer,
)


@dataclass(frozen=True)
class WorkflowResult:
    output_dir: Path
    model_path: Path
    predictions_path: Path
    predictions_csv_path: Path
    metrics_path: Path
    map_path: Path | None
    imputer: SelectiveDateContextImputer
    predictions: pd.DataFrame


@dataclass(frozen=True)
class FitResult:
    run_dir: Path
    model_path: Path
    metrics_path: Path
    report_html_path: Path
    report_pdf_path: Path
    latest_path: Path
    imputer: SelectiveDateContextImputer


@dataclass(frozen=True)
class ImputationResult:
    model_path: Path
    predictions_path: Path
    metadata_path: Path
    predictions: pd.DataFrame


def _evaluation_payload(imputer: SelectiveDateContextImputer) -> dict[str, Any]:
    return {
        name: {
            "metrics": result.metrics,
            "policy": result.policy.to_dict(),
        }
        for name, result in imputer.evaluations_.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_normalization_output(
    *,
    outputs: list[dict[str, Any]],
    dataset_id: str,
    configured_path: Path,
    expected_schema: pa.Schema,
    manifest_schema_version: str,
) -> dict[str, Any]:
    matching = [
        item
        for item in outputs
        if item.get("dataset_id") == dataset_id
        and Path(item.get("path", "")).expanduser().resolve() == configured_path
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Latest normalization manifest does not identify configured {dataset_id} input"
        )
    artifact = matching[0]
    expected_checksum = artifact.get("checksum")
    if not expected_checksum:
        raise ValueError(f"Normalization manifest lacks a checksum for {dataset_id}")
    if checksum_path(configured_path) != str(expected_checksum):
        raise ValueError(
            f"Normalized {dataset_id} checksum does not match its manifest"
        )
    if str(artifact.get("schema_version") or "") != manifest_schema_version:
        raise ValueError(
            f"Normalized {dataset_id} schema version is inconsistent with manifest"
        )

    files = (
        sorted(configured_path.rglob("*.parquet"))
        if configured_path.is_dir()
        else [configured_path]
    )
    if not files:
        raise ValueError(f"Normalized {dataset_id} has no Parquet files")
    actual_file_count = len(files)
    if int(artifact.get("file_count") or 0) != actual_file_count:
        raise ValueError(
            f"Normalized {dataset_id} file count does not match its manifest"
        )
    actual_row_count = 0
    for file_path in files:
        parquet = pq.ParquetFile(file_path)
        actual_row_count += int(parquet.metadata.num_rows)
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
            raise ValueError(
                f"Normalized {dataset_id} does not match its canonical Arrow schema"
            )
    if (
        artifact.get("row_count") is None
        or int(artifact["row_count"]) != actual_row_count
    ):
        raise ValueError(
            f"Normalized {dataset_id} row count does not match its manifest"
        )
    return artifact


def _normalization_snapshot_id(
    observations_path: str | Path,
    associations_path: str | Path | None = None,
) -> str | None:
    observations = Path(observations_path).expanduser().resolve()
    pointer = observations.parent / "manifests/normalize/latest.json"
    if not pointer.exists():
        return None
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    manifest_path = Path(pointer_payload["manifest"]).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (pointer.parent / manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Latest normalization manifest is not complete")
    manifest_schema_version = str(manifest.get("schema_version") or "")
    if not manifest_schema_version:
        raise ValueError("Latest normalization manifest lacks a schema version")
    observation_schemas = {
        "8": OBSERVATION_SCHEMA_V8,
        "9": OBSERVATION_SCHEMA,
    }
    if manifest_schema_version not in observation_schemas:
        raise ValueError(
            f"Unsupported normalized observation schema version: {manifest_schema_version}"
        )
    outputs = manifest.get("outputs") or []
    observation_artifact = _validate_normalization_output(
        outputs=outputs,
        dataset_id="whale.sightings.observations",
        configured_path=observations,
        expected_schema=observation_schemas[manifest_schema_version],
        manifest_schema_version=manifest_schema_version,
    )
    association_artifact = None
    if associations_path is not None:
        association_artifact = _validate_normalization_output(
            outputs=outputs,
            dataset_id="whale.sightings.associations",
            configured_path=Path(associations_path).expanduser().resolve(),
            expected_schema=ASSOCIATION_SCHEMA,
            manifest_schema_version=manifest_schema_version,
        )
    data_snapshot = manifest.get("data_snapshot") or {}
    snapshot_id = (
        str(data_snapshot["snapshot_id"]) if data_snapshot.get("snapshot_id") else None
    )
    for artifact in (observation_artifact, association_artifact):
        if artifact is None:
            continue
        artifact_snapshot = (artifact.get("data_snapshot") or {}).get("snapshot_id")
        if snapshot_id and str(artifact_snapshot or "") != snapshot_id:
            raise ValueError(
                "Normalized input snapshot id is inconsistent with its manifest"
            )
    return snapshot_id


def record_training_input_provenance(
    imputer: SelectiveDateContextImputer,
    *,
    observations_path: str | Path,
    associations_path: str | Path | None,
) -> None:
    observations = Path(observations_path).expanduser().resolve()
    imputer.training_summary_["OBSERVATIONS_SHA256"] = _sha256(observations)
    imputer.training_summary_["ASSOCIATIONS_SHA256"] = (
        _sha256(Path(associations_path).expanduser().resolve())
        if associations_path is not None
        else None
    )
    imputer.training_summary_["TRAINING_DATA_SNAPSHOT_ID"] = _normalization_snapshot_id(
        observations,
        associations_path,
    )


def validate_training_input_provenance(
    imputer: SelectiveDateContextImputer,
    *,
    observations_path: str | Path,
    associations_path: str | Path | None,
    data_snapshot_id: str | None = None,
) -> None:
    expected_observations = imputer.training_summary_.get("OBSERVATIONS_SHA256")
    if not expected_observations:
        raise ValueError(
            "Imputation model lacks normalized observation input provenance; refit it"
        )
    actual_observations = _sha256(Path(observations_path).expanduser().resolve())
    if actual_observations != expected_observations:
        raise ValueError(
            "Imputation observations do not match the fitted model input checksum"
        )
    if "ASSOCIATIONS_SHA256" not in imputer.training_summary_:
        raise ValueError(
            "Imputation model lacks association input provenance; refit it"
        )
    expected_associations = imputer.training_summary_["ASSOCIATIONS_SHA256"]
    if expected_associations is not None:
        if associations_path is None:
            raise ValueError(
                "Imputation model was fitted with associations but none were supplied"
            )
        actual_associations = _sha256(Path(associations_path).expanduser().resolve())
        if actual_associations != expected_associations:
            raise ValueError(
                "Imputation associations do not match the fitted model input checksum"
            )
    elif associations_path is not None:
        raise ValueError(
            "Imputation model was fitted without associations but associations supplied"
        )
    expected_snapshot = imputer.training_summary_.get("TRAINING_DATA_SNAPSHOT_ID")
    if data_snapshot_id is not None:
        if not expected_snapshot:
            raise ValueError(
                "Imputation model lacks a training data snapshot id; refit it"
            )
        if str(expected_snapshot) != str(data_snapshot_id):
            raise ValueError(
                "Imputation data snapshot does not match the fitted model snapshot"
            )


def _stratified_oof_summary(oof: pd.DataFrame) -> pd.DataFrame:
    """Emit independent encounter diagnostics across required deployment strata."""

    if oof.empty:
        return pd.DataFrame()
    independent = oof[oof["IS_ENCOUNTER_REPRESENTATIVE"].fillna(False)].copy()
    dimensions = [
        columns
        for columns in (
            ("SOURCE",),
            ("ERA",),
            ("SPATIAL_REGION",),
            ("EVIDENCE_REGIME",),
            ("SOURCE", "ERA"),
            ("SOURCE", "ERA", "EVIDENCE_REGIME"),
        )
        if set(columns).issubset(independent.columns)
    ]
    records: list[dict[str, Any]] = []
    for columns in dimensions:
        grouped = independent.groupby(list(columns), dropna=False, sort=True)
        for values, group in grouped:
            values = values if isinstance(values, tuple) else (values,)
            y = pd.to_numeric(group["Y_TRUE"], errors="coerce").to_numpy(dtype=int)
            probability = pd.to_numeric(group["P_SRKW"], errors="coerce").to_numpy(
                dtype=float
            )
            accepted = group["ACCEPTED"].fillna(False).to_numpy(dtype=bool)
            predicted = probability >= 0.5
            record: dict[str, Any] = {
                "STRATIFICATION": " x ".join(columns),
                "INDEPENDENT_ENCOUNTER_N": int(len(group)),
                "SRKW_N": int(y.sum()),
                "BRIER": float(brier_score_loss(y, probability)),
                "LOG_LOSS": float(log_loss(y, probability, labels=[0, 1])),
                "AUC": (
                    float(roc_auc_score(y, probability))
                    if len(set(y.tolist())) == 2
                    else None
                ),
                "ACCEPTED_N": int(accepted.sum()),
                "COVERAGE": float(accepted.mean()),
                "ACCEPTED_ERROR": (
                    float((predicted[accepted] != y[accepted]).mean())
                    if accepted.any()
                    else None
                ),
            }
            record.update(
                {column: str(value) for column, value in zip(columns, values)}
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _export_evaluations(imputer: SelectiveDateContextImputer, output: Path) -> None:
    if not imputer.tuning_results_.empty:
        tuning_dir = output / "metrics" / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        imputer.tuning_results_.to_csv(
            tuning_dir / "candidate_results.csv", index=False
        )
    for name, evaluation in imputer.evaluations_.items():
        prefix = output / "metrics" / name
        prefix.mkdir(parents=True, exist_ok=True)
        evaluation.oof_predictions.to_csv(prefix / "oof_predictions.csv", index=False)
        evaluation.reliability.to_csv(prefix / "reliability.csv", index=False)
        evaluation.risk_coverage.to_csv(prefix / "risk_coverage.csv", index=False)
        evaluation.metrics_by_regime.to_csv(
            prefix / "metrics_by_regime.csv", index=False
        )
        _stratified_oof_summary(evaluation.oof_predictions).to_csv(
            prefix / "metrics_by_source_era_region.csv", index=False
        )
        write_json(prefix / "summary.json", evaluation.metrics)
        write_json(prefix / "acceptance_policy.json", evaluation.policy.to_dict())


def fit_imputation_model(
    *,
    imputer_factory,
    model_domains=None,
    observations_path: str | Path,
    models_dir: str | Path,
    associations_path: str | Path | None = None,
    config: ImputationConfig | None = None,
    source_config_path: str | Path | None = None,
    run_id: str | None = None,
    evaluate_strategies: tuple[str, ...] = (
        "reconstruction",
        "encounter",
        "blocked",
        "purged_blocked",
    ),
) -> FitResult:
    """Fit and report a reusable model without scoring the production table."""

    from marine_mammal_toolkit.tools.observations.impute.report import (
        write_training_html,
    )
    from marine_mammal_toolkit.tools.observations.impute.report import (
        write_training_pdf,
    )

    cfg = config or ImputationConfig()
    cfg.validate()
    fit_at = datetime.now(timezone.utc)
    resolved_run_id = run_id or fit_at.strftime("%Y%m%dT%H%M%SZ")
    resolved_source_config = (
        Path(source_config_path).expanduser().resolve()
        if source_config_path is not None
        else None
    )
    if resolved_source_config is not None and not resolved_source_config.is_file():
        raise FileNotFoundError(
            f"Sightings source configuration does not exist: {resolved_source_config}"
        )
    source_config_sha256 = (
        _sha256(resolved_source_config) if resolved_source_config is not None else None
    )
    root = Path(models_dir).expanduser().resolve()
    run_dir = root / resolved_run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Model run already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    observations, associations = load_preprocessed_sightings(
        observations_path, associations_path
    )
    imputer = imputer_factory(cfg).fit(
        observations,
        associations,
        evaluate_strategies=evaluate_strategies,
    )
    if model_domains is not None:
        imputer.set_model_domains(
            model_domains.geometries,
            provenance={
                str(artifact.dataset_id): {
                    "path": str(artifact.path),
                    "checksum": artifact.checksum,
                    "producer": artifact.producer,
                    "schema_version": artifact.schema_version,
                    "spatial_coverage": dict(artifact.spatial_coverage or {}),
                }
                for artifact in model_domains.artifacts
            },
        )
    record_training_input_provenance(
        imputer,
        observations_path=observations_path,
        associations_path=associations_path,
    )
    imputer.training_summary_["FIT_AT_UTC"] = fit_at.isoformat()
    imputer.training_summary_["FIT_RUN_ID"] = resolved_run_id
    imputer.training_summary_["IMPUTATION_CONFIG_SHA256"] = _canonical_sha256(
        cfg.to_dict()
    )
    imputer.training_summary_["SOURCE_CONFIG_SHA256"] = (
        f"sha256:{source_config_sha256}" if source_config_sha256 is not None else None
    )
    domain_checksums = sorted(
        (
            str(dataset_id),
            str(details.get("checksum")),
        )
        for dataset_id, details in imputer.model_domain_provenance_.items()
        if isinstance(details, dict) and details.get("checksum")
    )
    imputer.training_summary_["MODEL_DOMAIN_SHA256"] = (
        _canonical_sha256(domain_checksums) if domain_checksums else None
    )
    imputer.training_summary_["MODEL_IDENTITY_SHA256"] = _canonical_sha256(
        {
            "model_version": imputer.training_summary_.get("MODEL_VERSION"),
            "fit_run_id": resolved_run_id,
            "training_data_snapshot_id": imputer.training_summary_.get(
                "TRAINING_DATA_SNAPSHOT_ID"
            ),
            "observations_sha256": imputer.training_summary_.get("OBSERVATIONS_SHA256"),
            "associations_sha256": imputer.training_summary_.get("ASSOCIATIONS_SHA256"),
            "imputation_config_sha256": imputer.training_summary_.get(
                "IMPUTATION_CONFIG_SHA256"
            ),
            "source_config_sha256": imputer.training_summary_.get(
                "SOURCE_CONFIG_SHA256"
            ),
            "model_domain_sha256": imputer.training_summary_.get("MODEL_DOMAIN_SHA256"),
            "tuned_parameters": imputer.tuned_params_,
        }
    )
    model_path = imputer.save(run_dir / "ecotype_imputer.joblib")
    model_sha256 = _sha256(model_path)
    metrics_payload = {
        "fit_at_utc": fit_at.isoformat(),
        "fit_run_id": resolved_run_id,
        "model_sha256": model_sha256,
        "source_config_path": (
            str(resolved_source_config) if resolved_source_config else None
        ),
        "source_config_sha256": source_config_sha256,
        "config": cfg.to_dict(),
        "training_summary": imputer.training_summary_,
        "evaluations": _evaluation_payload(imputer),
    }
    metrics_path = write_json(run_dir / "model_metrics.json", metrics_payload)
    write_json(run_dir / "model_config.json", cfg.to_dict())
    _export_evaluations(imputer, run_dir)
    metadata = {
        "fit_at_utc": fit_at.isoformat(),
        "fit_run_id": resolved_run_id,
        "source_config": (
            str(resolved_source_config)
            if resolved_source_config
            else "Direct Python configuration"
        ),
        "config_sha256": source_config_sha256[:16] if source_config_sha256 else "-",
        "model_sha256": model_sha256[:16],
        "tuned_parameters": json.dumps(imputer.tuned_params_, sort_keys=True),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "observations": str(Path(observations_path).expanduser().resolve()),
        "associations": (
            str(Path(associations_path).expanduser().resolve())
            if associations_path is not None
            else "Not provided"
        ),
    }
    report_html_path = write_training_html(
        run_dir / "fit_report.html", imputer=imputer, metadata=metadata
    )
    report_pdf_path = write_training_pdf(
        run_dir / "fit_report.pdf", imputer=imputer, metadata=metadata
    )
    latest_path = write_json(
        root / "latest.json",
        {
            "fit_run_id": resolved_run_id,
            "fit_at_utc": fit_at.isoformat(),
            "model_path": model_path.relative_to(root).as_posix(),
            "model_sha256": model_sha256,
            "report_html_path": report_html_path.relative_to(root).as_posix(),
            "report_pdf_path": report_pdf_path.relative_to(root).as_posix(),
        },
    )
    return FitResult(
        run_dir=run_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        report_html_path=report_html_path,
        report_pdf_path=report_pdf_path,
        latest_path=latest_path,
        imputer=imputer,
    )


def resolve_model_path(path: str | Path) -> Path:
    """Resolve a joblib file, fit run directory, or models directory latest pointer."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file() and candidate.suffix == ".joblib":
        return candidate
    pointer = candidate / "latest.json" if candidate.is_dir() else candidate
    if pointer.is_file() and pointer.name == "latest.json":
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        model = Path(payload["model_path"]).expanduser()
        if not model.is_absolute():
            model = (pointer.parent / model).resolve()
        if not model.exists():
            raise FileNotFoundError(
                f"Model referenced by {pointer} does not exist: {model}"
            )
        expected_checksum = payload.get("model_sha256")
        if not expected_checksum:
            raise ValueError(f"Model pointer lacks model_sha256: {pointer}")
        if _sha256(model) != str(expected_checksum):
            raise ValueError(f"Model checksum does not match latest pointer: {pointer}")
        return model
    run_model = candidate / "ecotype_imputer.joblib"
    if run_model.exists():
        return run_model
    raise FileNotFoundError(f"Could not resolve an imputation model from {candidate}")


def apply_imputation_model(
    *,
    observations_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    associations_path: str | Path | None = None,
    source_config_path: str | Path | None = None,
) -> ImputationResult:
    """Apply a saved fit to normalized sightings and write the post-processing handoff."""

    resolved_model = resolve_model_path(model_path)
    observations, _ = load_preprocessed_sightings(observations_path, associations_path)
    imputer = SelectiveDateContextImputer.load(resolved_model)
    validate_training_input_provenance(
        imputer,
        observations_path=observations_path,
        associations_path=associations_path,
        data_snapshot_id=_normalization_snapshot_id(
            observations_path, associations_path
        ),
    )
    predictions = imputer.apply_to_all(observations)
    predictions["IMPUTATION_MODEL_VERSION"] = imputer.training_summary_.get(
        "MODEL_VERSION", "UNKNOWN"
    )
    predictions["IMPUTATION_MODEL_FIT_AT_UTC"] = pd.to_datetime(
        imputer.training_summary_.get("FIT_AT_UTC"), utc=True, errors="coerce"
    )
    predictions["IMPUTATION_MODEL_RUN_ID"] = imputer.training_summary_.get("FIT_RUN_ID")
    output = write_predictions(predictions, Path(output_path).expanduser().resolve())
    metadata_path = write_json(
        output.with_suffix(output.suffix + ".metadata.json"),
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_config_path": (
                str(Path(source_config_path).expanduser().resolve())
                if source_config_path is not None
                else None
            ),
            "model_path": str(resolved_model),
            "model_sha256": _sha256(resolved_model),
            "model_version": imputer.training_summary_.get("MODEL_VERSION"),
            "fit_at_utc": imputer.training_summary_.get("FIT_AT_UTC"),
            "fit_run_id": imputer.training_summary_.get("FIT_RUN_ID"),
            "observations_path": str(Path(observations_path).expanduser().resolve()),
            "row_count": int(len(predictions)),
            "hard_imputation_count": int(predictions["IMPUTATION_APPLIED"].sum()),
            "label_tier_counts": predictions["ECOTYPE_LABEL_TIER"]
            .value_counts()
            .to_dict(),
        },
    )
    return ImputationResult(
        model_path=resolved_model,
        predictions_path=output,
        metadata_path=metadata_path,
        predictions=predictions,
    )


def run_imputation_workflow(
    *,
    imputer_factory,
    observations_path: str | Path,
    output_dir: str | Path,
    associations_path: str | Path | None = None,
    config: ImputationConfig | None = None,
    make_map: bool = True,
    evaluate_strategies: tuple[str, ...] = (
        "reconstruction",
        "encounter",
        "blocked",
        "purged_blocked",
    ),
) -> WorkflowResult:
    """Run loading, cross-fitted evaluation, fitting, imputation, and exports."""

    cfg = config or ImputationConfig()
    cfg.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    observations, associations = load_preprocessed_sightings(
        observations_path, associations_path
    )
    imputer = imputer_factory(cfg).fit(
        observations,
        associations,
        evaluate_strategies=evaluate_strategies,
    )
    record_training_input_provenance(
        imputer,
        observations_path=observations_path,
        associations_path=associations_path,
    )
    predictions = imputer.apply_to_all(observations)

    model_path = imputer.save(output / "ecotype_imputer.joblib")
    predictions_csv_path = output / "sightings_with_imputation.csv"
    write_predictions(predictions, predictions_csv_path)
    predictions_path = output / "sightings_with_imputation.parquet"
    try:
        write_predictions(predictions, predictions_path)
    except (ImportError, ModuleNotFoundError):
        predictions_path = predictions_csv_path

    metrics_payload = {
        "config": cfg.to_dict(),
        "training_summary": imputer.training_summary_,
        "evaluations": _evaluation_payload(imputer),
    }
    metrics_path = write_json(output / "model_metrics.json", metrics_payload)
    write_json(output / "model_config.json", cfg.to_dict())
    _export_evaluations(imputer, output)

    map_path: Path | None = None
    if make_map:
        try:
            from marine_mammal_toolkit.tools.observations.impute.visualization import (
                make_animated_map,
            )
            from marine_mammal_toolkit.tools.observations.impute.visualization import (
                save_map_html,
            )

            figure = make_animated_map(
                predictions,
                year=cfg.map_year,
                frame_unit=cfg.map_frame,
            )
            map_path = save_map_html(
                figure,
                output / f"ecotype_map_{cfg.map_year}_{cfg.map_frame}.html",
            )
        except ValueError as exc:
            write_json(output / "map_not_created.json", {"reason": str(exc)})

    return WorkflowResult(
        output_dir=output,
        model_path=model_path,
        predictions_path=predictions_path,
        predictions_csv_path=predictions_csv_path,
        metrics_path=metrics_path,
        map_path=map_path,
        imputer=imputer,
        predictions=predictions,
    )
