from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h3  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools._core.data import DATASETS
from marine_mammal_toolkit.tools._core.data import ValidationReport
from marine_mammal_toolkit.tools._core.persistence import checksum_path
from marine_mammal_toolkit.tools.quality.tables import validate_path
from marine_mammal_toolkit.tools.quality.tables import validate_table

from marine_mammal_toolkit.tools.schemas.observations import ASSOCIATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import AUDIT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import COUNT_EXCLUSION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import COUNT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import COUNT_SCHEMA_V6
from marine_mammal_toolkit.tools.schemas.observations import DETAIL_COUNT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import DETAIL_COUNT_SCHEMA_V6
from marine_mammal_toolkit.tools.schemas.observations import IDENTITY_ALIAS_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import IDENTITY_LINEAGE_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import IDENTITY_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import IMPUTED_OBSERVATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import (
    IMPUTED_OBSERVATION_SCHEMA_V8,
)
from marine_mammal_toolkit.tools.schemas.observations import MODEL_GRID_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import MODEL_GRID_SCHEMA_V6
from marine_mammal_toolkit.tools.schemas.observations import MODEL_INTENSITY_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import MODEL_INTENSITY_SCHEMA_V6
from marine_mammal_toolkit.tools.schemas.observations import OBSERVATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import OBSERVATION_SCHEMA_V8
from marine_mammal_toolkit.tools.schemas.observations import PERIOD_TOTAL_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import PERIOD_TOTAL_SCHEMA_V6
from marine_mammal_toolkit.tools.schemas.observations import POD_COUNT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import POD_COUNT_SCHEMA_V6
from marine_mammal_toolkit.tools.schemas.observations import SOURCE_HISTORY_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import SOURCE_RECORD_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import TOTAL_COUNT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import TOTAL_COUNT_SCHEMA_V6

SCHEMAS = {
    "whale.sightings.source_record_history": SOURCE_HISTORY_SCHEMA,
    "whale.sightings.source_records": SOURCE_RECORD_SCHEMA,
    "whale.sightings.observations": OBSERVATION_SCHEMA,
    "whale.sightings.associations": ASSOCIATION_SCHEMA,
    "whale.sightings.normalization_audit": AUDIT_SCHEMA,
    "whale.sightings.identity_resolution": IDENTITY_SCHEMA,
    "whale.sightings.identity_aliases": IDENTITY_ALIAS_SCHEMA,
    "whale.sightings.identity_lineage": IDENTITY_LINEAGE_SCHEMA,
    "whale.sightings.imputed_retrospective": IMPUTED_OBSERVATION_SCHEMA,
    "whale.sightings.imputed_as_of": IMPUTED_OBSERVATION_SCHEMA,
    "whale.sightings.ecotype_counts": COUNT_SCHEMA,
    "whale.sightings.ecotype_detail_counts": DETAIL_COUNT_SCHEMA,
    "whale.sightings.orca_total_counts": TOTAL_COUNT_SCHEMA,
    "whale.sightings.pod_counts": POD_COUNT_SCHEMA,
    "whale.sightings.period_totals": PERIOD_TOTAL_SCHEMA,
    "whale.sightings.count_exclusions": COUNT_EXCLUSION_SCHEMA,
    "whale.sightings.reported_sighting_grid": MODEL_GRID_SCHEMA,
    "whale.sightings.relative_reported_activity": MODEL_INTENSITY_SCHEMA,
    "whale.sightings.relative_intensity": MODEL_INTENSITY_SCHEMA,
}


def _read_directory(path: Path) -> pa.Table:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise ValueError("Dataset directory contains no Parquet partitions")
    # Read each physical file without Hive partition inference. Partition keys are
    # already durable columns in every sightings table.
    return pa.concat_tables([pq.ParquetFile(file).read() for file in files])


def _domain_errors(frame: pd.DataFrame, dataset_id: str) -> list[str]:
    errors = []
    if "LATITUDE" in frame and not frame.LATITUDE.between(-90, 90).all():
        errors.append("LATITUDE contains values outside [-90, 90]")
    if "LONGITUDE" in frame and not frame.LONGITUDE.between(-180, 180).all():
        errors.append("LONGITUDE contains values outside [-180, 180]")
    observation_datasets = {
        "whale.sightings.observations",
        "whale.sightings.imputed_retrospective",
        "whale.sightings.imputed_as_of",
    }
    if "OBSERVATION_ID" in frame and dataset_id in observation_datasets:
        if frame.OBSERVATION_ID.duplicated().any():
            errors.append("OBSERVATION_ID must be unique")
        stamp = pd.to_datetime(frame.SIGHTING_DATE_UTC, utc=True)
        if not (
            stamp.dt.hour.eq(12)
            & stamp.dt.minute.eq(0)
            & stamp.dt.second.eq(0)
            & stamp.dt.microsecond.eq(0)
            & stamp.dt.date.eq(frame.SIGHTING_DATE)
        ).all():
            errors.append("SIGHTING_DATE_UTC must equal SIGHTING_DATE at 12:00:00Z")
        if not frame.CANONICAL_TIME_SYNTHETIC.eq(True).all():
            errors.append("CANONICAL_TIME_SYNTHETIC must be true")
        if frame.AVAILABLE_AT_UTC.isna().any():
            errors.append("AVAILABLE_AT_UTC is required")
        provenance_columns = {
            "EVENT_DATE",
            "RECORD_AVAILABLE_AT_UTC",
            "LABEL_AVAILABLE_AT_UTC",
            "LAST_CORRECTED_AT_UTC",
            "SOURCE",
            "SOURCE_RECORD_ID",
        }
        missing_provenance = sorted(provenance_columns - set(frame))
        if missing_provenance:
            errors.append(f"Missing availability metadata: {missing_provenance}")
        else:
            if not frame.EVENT_DATE.eq(frame.SIGHTING_DATE).all():
                errors.append("EVENT_DATE must equal SIGHTING_DATE")
            record_available = pd.to_datetime(frame.RECORD_AVAILABLE_AT_UTC, utc=True)
            compatibility_available = pd.to_datetime(frame.AVAILABLE_AT_UTC, utc=True)
            label_available = pd.to_datetime(frame.LABEL_AVAILABLE_AT_UTC, utc=True)
            if record_available.isna().any() or label_available.isna().any():
                errors.append("Record and label availability timestamps are required")
            if not record_available.eq(compatibility_available).all():
                errors.append("RECORD_AVAILABLE_AT_UTC must equal AVAILABLE_AT_UTC")
            if not label_available.ge(record_available).all():
                errors.append(
                    "LABEL_AVAILABLE_AT_UTC cannot precede record availability"
                )
            if frame.LAST_CORRECTED_AT_UTC.isna().any():
                errors.append("LAST_CORRECTED_AT_UTC is required")
            if not frame.apply(
                lambda row: row.SOURCE_RECORD_ID in set(row.SOURCE_RECORD_IDS), axis=1
            ).all():
                errors.append("SOURCE_RECORD_ID must be one of SOURCE_RECORD_IDS")
        if not frame.apply(
            lambda row: row.SOURCE_REPORT_COUNT == len(set(row.SOURCE_RECORD_IDS)),
            axis=1,
        ).all():
            errors.append(
                "SOURCE_REPORT_COUNT must equal unique SOURCE_RECORD_IDS length"
            )
        if "SOURCE_OCCURRENCE_COUNT" in frame:
            if not frame.SOURCE_OCCURRENCE_COUNT.ge(frame.SOURCE_REPORT_COUNT).all():
                errors.append("SOURCE_OCCURRENCE_COUNT must be >= SOURCE_REPORT_COUNT")
        if "PUBLIC_RELEASE_ELIGIBLE" in frame:
            if frame.PUBLIC_RELEASE_ELIGIBLE.isna().any():
                errors.append("PUBLIC_RELEASE_ELIGIBLE cannot be null")
        if "COORDINATE_UNCERTAINTY_M" in frame:
            uncertainty = pd.to_numeric(frame.COORDINATE_UNCERTAINTY_M, errors="coerce")
            supplied = frame.COORDINATE_UNCERTAINTY_M.notna()
            if (
                not np.isfinite(uncertainty.loc[supplied]).all()
                or not uncertainty.loc[supplied].ge(0).all()
            ):
                errors.append("Coordinate uncertainty must be finite and nonnegative")
            if frame.COORDINATE_SELECTION_METHOD.astype(str).str.strip().eq("").any():
                errors.append("Coordinate selection method is required")
            if frame.OBSERVATION_QUALITY_TIER.astype(str).str.strip().eq("").any():
                errors.append("Observation quality tier is required")
            for column in (
                "CONTRIBUTOR_LICENSE_SUMMARY",
                "FIELD_PROVENANCE",
                "SEMANTIC_CORRECTION_LINEAGE",
            ):
                try:
                    frame[column].map(json.loads)
                except (TypeError, ValueError, json.JSONDecodeError):
                    errors.append(f"{column} must contain valid JSON")
    if "ECOTYPE_DETAIL" in frame:
        allowed = {"SRKW", "TRANSIENT", "NRKW", "OFFSHORE", "UNKNOWN", "MIXED"}
        invalid = set(frame.ECOTYPE_DETAIL.dropna()) - allowed
        if invalid:
            errors.append(f"Invalid ECOTYPE_DETAIL values: {sorted(invalid)}")
    if {"ECOTYPE_DETAIL", "ECOTYPE_BUCKET"} <= set(frame):
        mapping = {
            "SRKW": "SRKW",
            "TRANSIENT": "TRANSIENT",
            "NRKW": "OTHER",
            "OFFSHORE": "OTHER",
            "UNKNOWN": "OTHER",
            "MIXED": "OTHER",
        }
        expected = frame.ECOTYPE_DETAIL.map(mapping)
        if not expected.eq(frame.ECOTYPE_BUCKET).all():
            errors.append("ECOTYPE_DETAIL does not map to the expected ECOTYPE_BUCKET")
    if "ECOTYPE_BUCKET" in frame:
        invalid = set(frame.ECOTYPE_BUCKET.dropna()) - {"SRKW", "TRANSIENT", "OTHER"}
        if invalid:
            errors.append(f"Invalid ECOTYPE_BUCKET values: {sorted(invalid)}")
        if (
            dataset_id == "whale.sightings.observations"
            and frame.ECOTYPE_BUCKET.isna().any()
        ):
            errors.append("Every observation requires exactly one ECOTYPE_BUCKET")
    if dataset_id in {
        "whale.sightings.imputed_retrospective",
        "whale.sightings.imputed_as_of",
    }:
        if (
            frame.ENCOUNTER_ID.isna().any()
            or frame.ENCOUNTER_ID.astype(str).str.strip().eq("").any()
        ):
            errors.append("Every imputed observation requires ENCOUNTER_ID")
        if frame.ENCOUNTER_SIZE.isna().any() or not frame.ENCOUNTER_SIZE.ge(1).all():
            errors.append("Every imputed observation requires ENCOUNTER_SIZE >= 1")
        else:
            actual_sizes = frame.groupby("ENCOUNTER_ID")["OBSERVATION_ID"].transform(
                "nunique"
            )
            if not frame.ENCOUNTER_SIZE.eq(actual_sizes).all():
                errors.append(
                    "ENCOUNTER_SIZE must equal unique observation membership for ENCOUNTER_ID"
                )
        modeled = frame[["P_SRKW", "P_TRANSIENT"]].notna().any(axis=1)
        complete_probabilities = frame[["P_SRKW", "P_TRANSIENT"]].notna().all(axis=1)
        if not modeled.eq(complete_probabilities).all():
            errors.append("P_SRKW and P_TRANSIENT must be present or null together")
        probabilities = frame.loc[complete_probabilities, ["P_SRKW", "P_TRANSIENT"]]
        if not probabilities.apply(lambda values: values.between(0, 1)).all().all():
            errors.append(
                "Imputation probabilities must be finite values within [0, 1]"
            )
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-9):
            errors.append("P_SRKW plus P_TRANSIENT must equal one")
        if "P_OTHER" in frame:
            other_probability = pd.to_numeric(frame.P_OTHER, errors="coerce")
            present_other = frame.P_OTHER.notna()
            if not other_probability.loc[present_other].between(0, 1).all():
                errors.append("P_OTHER must be finite and within [0, 1] when present")
            if not np.allclose(
                other_probability.loc[complete_probabilities & present_other],
                0.0,
                atol=1e-9,
            ):
                errors.append(
                    "Binary SRKW/Transient scores must reserve zero P_OTHER mass"
                )
            known_other = ~frame.ECOTYPE_DETAIL_OBSERVED.isin(
                ["SRKW", "TRANSIENT", "UNKNOWN"]
            )
            if not present_other.loc[known_other].all() or not np.allclose(
                other_probability.loc[known_other], 1.0, atol=1e-9
            ):
                errors.append("Known Other observations must carry unit P_OTHER")
        expected_columns = [
            "EXPECTED_SRKW_COUNT",
            "EXPECTED_TRANSIENT_COUNT",
            *(["EXPECTED_OTHER_COUNT"] if "EXPECTED_OTHER_COUNT" in frame else []),
            "EXPECTED_UNKNOWN_COUNT",
        ]
        expected = frame[expected_columns]
        if not expected.apply(lambda values: values.between(0, 1)).all().all():
            errors.append("Expected ecotype contributions must be within [0, 1]")
        modeled_or_unknown = frame.ECOTYPE_DETAIL_OBSERVED.isin(
            ["SRKW", "TRANSIENT", "UNKNOWN"]
        )
        mass_rows = (
            frame.index
            if "EXPECTED_OTHER_COUNT" in frame
            else frame.index[modeled_or_unknown]
        )
        if not np.allclose(expected.loc[mass_rows].sum(axis=1), 1.0, atol=1e-9):
            errors.append(
                "Every represented ecotype row must have unit expected contribution"
            )
        if "EXPECTED_OTHER_COUNT" in frame:
            known_other = ~modeled_or_unknown
            if not np.allclose(
                frame.loc[known_other, "EXPECTED_OTHER_COUNT"], 1.0, atol=1e-9
            ):
                errors.append(
                    "Known Other observations must retain unit expected Other mass"
                )
            query = frame.ECOTYPE_DETAIL_OBSERVED.eq("UNKNOWN")
            probabilistic_query = query & frame.USE_FOR_PROBABILISTIC_COUNTS.astype(
                bool
            )
            hard_query = query & frame.IMPUTATION_APPLIED.astype(bool)
            if (
                "SOFT_COUNT_CERTIFIED" not in frame
                or not frame.loc[probabilistic_query, "SOFT_COUNT_CERTIFIED"]
                .fillna(False)
                .all()
            ):
                errors.append(
                    "Probability-count-eligible unknowns require soft certification"
                )
            if (
                "STABILITY_EVALUATED" not in frame
                or not frame.loc[
                    probabilistic_query | hard_query, "STABILITY_EVALUATED"
                ]
                .fillna(False)
                .all()
            ):
                errors.append("Count-eligible unknowns require evaluated stability")
            if (
                not frame.loc[probabilistic_query | hard_query, "PREDICTION_STABLE"]
                .fillna(False)
                .all()
            ):
                errors.append("Count-eligible unknowns require stable predictions")
            if (
                "BINARY_MODEL_DOMAIN_SUPPORTED" not in frame
                or not frame.loc[
                    probabilistic_query | hard_query, "BINARY_MODEL_DOMAIN_SUPPORTED"
                ]
                .fillna(False)
                .all()
            ):
                errors.append(
                    "Count-eligible unknowns must be inside binary model support"
                )
            if "ENCOUNTER_LABEL_CONFLICT" in frame:
                conflict = frame.ENCOUNTER_LABEL_CONFLICT.fillna(False).astype(bool)
                if (
                    frame.loc[
                        conflict,
                        [
                            "ELIGIBLE_FOR_TRAINING",
                            "ELIGIBLE_FOR_EVALUATION",
                            "USE_FOR_HARD_COUNTS",
                            "USE_FOR_PROBABILISTIC_COUNTS",
                        ],
                    ]
                    .fillna(False)
                    .any(axis=None)
                ):
                    errors.append(
                        "Mixed-ecotype encounters must be quarantined from model use"
                    )
        applied = frame.IMPUTATION_APPLIED.astype(bool)
        not_applied = ~applied
        if not frame.loc[applied, "ECOTYPE_DETAIL_OBSERVED"].eq("UNKNOWN").all():
            errors.append(
                "Only originally unknown observations may receive hard imputations"
            )
        if frame.loc[applied, "ECOTYPE_DETAIL_IMPUTED"].isna().any():
            errors.append("Applied imputations require an imputed ecotype")
        if (
            not frame.loc[applied, "ECOTYPE_DETAIL_IMPUTED"]
            .isin(["SRKW", "TRANSIENT"])
            .all()
        ):
            errors.append("Applied imputations must resolve to SRKW or TRANSIENT")
        if (
            not frame.loc[applied, "ECOTYPE_DETAIL_EFFECTIVE"]
            .eq(frame.loc[applied, "ECOTYPE_DETAIL_IMPUTED"])
            .all()
        ):
            errors.append("Applied imputation and effective ecotype disagree")
        if not frame.loc[applied, "USE_FOR_HARD_COUNTS"].astype(bool).all():
            errors.append("Applied imputations must be enabled for hard counts")
        if not frame.loc[applied, "CLASS_CERTIFIED_FOR_HARD_LABEL"].fillna(False).all():
            errors.append("Applied imputations require certified classes")
        if "HARD_LABEL_CERTIFIED" in frame:
            if (
                not frame["HARD_LABEL_CERTIFIED"]
                .fillna(False)
                .eq(frame["CLASS_CERTIFIED_FOR_HARD_LABEL"].fillna(False))
                .all()
            ):
                errors.append("Hard-label certification aliases disagree")
            if not frame.loc[applied, "HARD_LABEL_CERTIFIED"].fillna(False).all():
                errors.append("Applied imputations require hard-label certification")
        if not frame.loc[applied, "PREDICTION_STABLE"].fillna(False).all():
            errors.append("Applied imputations require stable predictions")
        if frame.loc[applied, "ELIGIBLE_FOR_TRAINING"].astype(bool).any():
            errors.append(
                "Imputed labels cannot be eligible as observed training truth"
            )
        if frame.loc[not_applied, "ECOTYPE_DETAIL_IMPUTED"].notna().any():
            errors.append("Abstained rows cannot retain a hard imputed ecotype")
        if (
            not frame.loc[not_applied, "ECOTYPE_DETAIL_EFFECTIVE"]
            .eq(frame.loc[not_applied, "ECOTYPE_DETAIL_OBSERVED"])
            .all()
        ):
            errors.append("Abstained rows must retain their observed ecotype")
    if "H3_INDEX" in frame:
        valid = frame.H3_INDEX.astype(str).map(h3.is_valid_cell)
        if not valid.all():
            errors.append(f"Invalid H3 cells: {int((~valid).sum())}")
        if "H3_RESOLUTION" in frame and valid.all():
            if (
                not frame.H3_INDEX.astype(str)
                .map(h3.get_resolution)
                .eq(pd.to_numeric(frame.H3_RESOLUTION))
                .all()
            ):
                errors.append("H3_RESOLUTION does not match H3_INDEX")
    if "SIGHTING_COUNT" in frame:
        sparse = dataset_id in {
            "whale.sightings.ecotype_counts",
            "whale.sightings.ecotype_detail_counts",
            "whale.sightings.orca_total_counts",
            "whale.sightings.pod_counts",
        }
        if sparse:
            has_expected = frame.get(
                "EXPECTED_SIGHTING_COUNT", pd.Series(0.0, index=frame.index)
            ).gt(0)
            if not (frame.SIGHTING_COUNT.gt(0) | has_expected).all():
                errors.append("Sparse rows require a hard or expected sighting count")
        if not frame.SOURCE_REPORT_COUNT.ge(frame.SIGHTING_COUNT).all():
            errors.append("SOURCE_REPORT_COUNT must be >= SIGHTING_COUNT")
    if "SIGHTING_COUNT" in frame and not frame.SIGHTING_COUNT.ge(0).all():
        errors.append("SIGHTING_COUNT must be nonnegative")
    if "SOURCE_REPORT_COUNT" in frame and not frame.SOURCE_REPORT_COUNT.ge(0).all():
        errors.append("SOURCE_REPORT_COUNT must be nonnegative")
    if dataset_id in {
        "whale.sightings.source_record_history",
        "whale.sightings.source_records",
    }:
        if "SOURCE_OCCURRENCE_COUNT" in frame:
            populated = frame.SOURCE_OCCURRENCE_COUNT.dropna()
            if not populated.ge(1).all():
                errors.append("SOURCE_OCCURRENCE_COUNT must be positive when populated")
        if "SOURCE_USE_CLASS" in frame:
            invalid = set(frame.SOURCE_USE_CLASS.dropna()) - {
                "REDISTRIBUTABLE",
                "INTERNAL_ONLY",
            }
            if invalid:
                errors.append(f"Invalid SOURCE_USE_CLASS values: {sorted(invalid)}")
        if "SOURCE_QC_STATUS" in frame:
            invalid = set(frame.SOURCE_QC_STATUS.dropna()) - {"ACCEPTED", "QUARANTINED"}
            if invalid:
                errors.append(f"Invalid SOURCE_QC_STATUS values: {sorted(invalid)}")
    for column in (
        "EXPECTED_SIGHTING_COUNT",
        "MATURE_EXPECTED_SIGHTING_COUNT",
        "PROVISIONAL_EXPECTED_COUNT",
        "EXPECTED_UNKNOWN_COUNT",
    ):
        if column in frame and not frame[column].ge(0).all():
            errors.append(f"{column} must be nonnegative")
    if {
        "EXPECTED_SIGHTING_COUNT",
        "MATURE_EXPECTED_SIGHTING_COUNT",
        "PROVISIONAL_EXPECTED_COUNT",
    } <= set(frame):
        reconstructed = (
            frame.MATURE_EXPECTED_SIGHTING_COUNT + frame.PROVISIONAL_EXPECTED_COUNT
        )
        if not np.allclose(frame.EXPECTED_SIGHTING_COUNT, reconstructed, atol=1e-9):
            errors.append(
                "Expected counts must equal mature plus provisional expected counts"
            )
    if "REPORTED_SIGHTING" in frame:
        if not frame.REPORTED_SIGHTING.isin([0, 1]).all():
            errors.append("REPORTED_SIGHTING must be 0 or 1")
        if (
            "SIGHTING_COUNT" in frame
            and not frame.REPORTED_SIGHTING.eq(
                frame.SIGHTING_COUNT.gt(0).astype(int)
            ).all()
        ):
            errors.append("REPORTED_SIGHTING must equal int(SIGHTING_COUNT > 0)")
    cohort_fields = {"TARGET_COHORT_ID", "TARGET_COHORT_STATUS", "REPORTING_STATE"}
    present_cohort_fields = cohort_fields.intersection(frame.columns)
    if present_cohort_fields and present_cohort_fields != cohort_fields:
        errors.append(
            "Target cohort fields must be present together; missing "
            f"{sorted(cohort_fields - present_cohort_fields)}"
        )
    elif present_cohort_fields:
        if (
            frame.TARGET_COHORT_ID.isna().any()
            or frame.TARGET_COHORT_ID.astype(str).str.strip().eq("").any()
        ):
            errors.append("TARGET_COHORT_ID must be non-empty")
        invalid_status = set(frame.TARGET_COHORT_STATUS.dropna()) - {
            "VERIFIED_COMPLETE",
            "UNVERIFIED",
        }
        if invalid_status:
            errors.append(
                f"Invalid TARGET_COHORT_STATUS values: {sorted(invalid_status)}"
            )
        invalid_reporting = set(frame.REPORTING_STATE.dropna()) - {
            "REPORTED",
            "NO_REPORT",
            "UNAVAILABLE",
        }
        if invalid_reporting:
            errors.append(
                f"Invalid REPORTING_STATE values: {sorted(invalid_reporting)}"
            )
        unverified = frame.TARGET_COHORT_STATUS.eq("UNVERIFIED")
        if frame.loc[unverified, "REPORTING_STATE"].eq("NO_REPORT").any():
            errors.append("Unverified target cohorts cannot emit NO_REPORT")
        verified = frame.TARGET_COHORT_STATUS.eq("VERIFIED_COMPLETE")
        if frame.loc[verified, "REPORTING_STATE"].eq("UNAVAILABLE").any():
            errors.append("Verified target cohorts cannot emit UNAVAILABLE")
        represented = pd.Series(False, index=frame.index)
        if "SIGHTING_COUNT" in frame:
            represented = represented | pd.to_numeric(
                frame.SIGHTING_COUNT, errors="coerce"
            ).fillna(0).gt(0)
        if "EXPECTED_SIGHTING_COUNT" in frame:
            represented = represented | pd.to_numeric(
                frame.EXPECTED_SIGHTING_COUNT, errors="coerce"
            ).fillna(0).gt(0)
        if not frame.loc[represented, "REPORTING_STATE"].eq("REPORTED").all():
            errors.append("Rows with represented sighting mass must be REPORTED")
        if "REPORTED_SIGHTING" in frame:
            positive = frame.REPORTED_SIGHTING.eq(1)
            if not frame.loc[positive, "REPORTING_STATE"].eq("REPORTED").all():
                errors.append(
                    "Positive dense sightings must have REPORTING_STATE=REPORTED"
                )
    if {
        "FREQUENCY",
        "PERIOD_START",
        "PERIOD_END",
        "YEAR",
        "PERIOD_STATUS",
        "IS_COMPLETE_PERIOD",
    } <= set(frame):
        start = pd.to_datetime(frame.PERIOD_START)
        end = pd.to_datetime(frame.PERIOD_END)
        if not frame.YEAR.eq(start.dt.year).all():
            errors.append("YEAR must equal the calendar year of PERIOD_START")
        daily = frame.FREQUENCY.eq("daily")
        weekly = frame.FREQUENCY.eq("weekly")
        if daily.any() and not end[daily].eq(start[daily]).all():
            errors.append("Daily PERIOD_END must equal PERIOD_START")
        if (
            weekly.any()
            and not end[weekly].eq(start[weekly] + pd.Timedelta(days=6)).all()
        ):
            errors.append("Weekly PERIOD_END must equal PERIOD_START + 6 days")
        if not frame.IS_COMPLETE_PERIOD.eq(frame.PERIOD_STATUS.eq("COMPLETE")).all():
            errors.append("PERIOD_STATUS and IS_COMPLETE_PERIOD disagree")
    if (
        "RELATIVE_SIGHTING_INTENSITY" in frame
        and not frame.RELATIVE_SIGHTING_INTENSITY.between(0, 1).all()
    ):
        errors.append("RELATIVE_SIGHTING_INTENSITY must be within [0, 1]")
    if "RELATIVE_REPORTED_ACTIVITY" in frame:
        if not frame.RELATIVE_REPORTED_ACTIVITY.between(0, 1).all():
            errors.append("RELATIVE_REPORTED_ACTIVITY must be within [0, 1]")
        if "RELATIVE_SIGHTING_INTENSITY" in frame and not np.allclose(
            frame.RELATIVE_REPORTED_ACTIVITY,
            frame.RELATIVE_SIGHTING_INTENSITY,
            atol=1e-7,
        ):
            errors.append(
                "Compatibility intensity alias must equal relative reported activity"
            )
    return errors


def _validate_manifest(path: Path, dataset_id: str, errors: list[str]) -> None:
    manifest_path = path / "_dataset_manifest.json"
    if not manifest_path.exists():
        errors.append("Partitioned dataset is missing _dataset_manifest.json")
        return
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("partitions", [])
    listed = {
        str(entry["path"] if isinstance(entry, dict) else entry) for entry in entries
    }
    actual = {str(file.relative_to(path)) for file in path.rglob("*.parquet")}
    if listed != actual:
        errors.append("Dataset manifest partition list does not match files")
    if dataset_id == "whale.sightings.reported_sighting_grid":
        for entry in entries:
            if not isinstance(entry, dict):
                errors.append("Dense manifest requires expected partition metadata")
                break
            table = pq.ParquetFile(path / entry["path"]).read()
            frame = table.to_pandas()
            if table.num_rows != int(entry["expected_rows"]):
                errors.append(f"Dense row mismatch: {entry['path']}")
            if frame.H3_INDEX.nunique() != int(entry["expected_cells"]):
                errors.append(f"Dense cell mismatch: {entry['path']}")
            if frame.PERIOD_START.nunique() != int(entry["expected_periods"]):
                errors.append(f"Dense period mismatch: {entry['path']}")
            if checksum_path(path / entry["path"]) != entry["checksum"]:
                errors.append(f"Dense checksum mismatch: {entry['path']}")
        for expected in manifest.get("expected_groups", []):
            matching = [
                entry
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("bucket") == expected.get("bucket")
                and entry.get("frequency") == expected.get("frequency")
                and entry.get("resolution") == expected.get("resolution")
            ]
            if sum(int(entry["expected_rows"]) for entry in matching) != int(
                expected["expected_rows"]
            ):
                errors.append(
                    "Dense group row mismatch: "
                    f"{expected['bucket']}/{expected['frequency']}/H{expected['resolution']}"
                )


def _validate_weekly_reconciliation(
    frame: pd.DataFrame,
    errors: list[str],
    requested_start: object,
    requested_end: object,
) -> None:
    if frame.empty or not {"daily", "weekly"} <= set(frame.FREQUENCY):
        return
    daily = frame[frame.FREQUENCY.eq("daily")].copy()
    day = pd.to_datetime(daily.PERIOD_START)
    daily["WEEK_START"] = (day - pd.to_timedelta(day.dt.weekday, unit="D")).dt.date
    minimum_day = pd.Timestamp(requested_start).date()
    maximum_day = pd.Timestamp(requested_end).date()
    complete_week = daily.WEEK_START.map(
        lambda value: value >= minimum_day
        and value + pd.Timedelta(days=6) <= maximum_day
    )
    daily = daily.loc[complete_week]
    keys = ["H3_INDEX", "H3_RESOLUTION", "WEEK_START", "ECOTYPE_BUCKET"]
    measures = [
        "SIGHTING_COUNT",
        "SOURCE_REPORT_COUNT",
        "OBSERVED_SIGHTING_COUNT",
        "HARD_IMPUTED_COUNT",
        "EXPECTED_SIGHTING_COUNT",
        "MATURE_EXPECTED_SIGHTING_COUNT",
        "PROVISIONAL_EXPECTED_COUNT",
        "EXPECTED_UNKNOWN_COUNT",
    ]
    measures = [column for column in measures if column in daily]
    summed = daily.groupby(keys, as_index=False)[measures].sum()
    weekly_rows = frame[frame.FREQUENCY.eq("weekly")].copy()
    weekly_start = pd.to_datetime(weekly_rows.PERIOD_START).dt.date
    calendar_complete = weekly_start.map(
        lambda value: value >= minimum_day
        and value + pd.Timedelta(days=6) <= maximum_day
    )
    # IS_COMPLETE_PERIOD describes reporting-cohort completeness.  Sparse
    # all-source facts correctly set it false when feed coverage is unverified,
    # but their closed calendar weeks must still reconcile to the daily facts.
    weekly = weekly_rows.loc[calendar_complete][
        [
            "H3_INDEX",
            "H3_RESOLUTION",
            "PERIOD_START",
            "ECOTYPE_BUCKET",
            *measures,
        ]
    ]
    weekly = weekly.rename(columns={"PERIOD_START": "WEEK_START"})
    comparison = weekly.merge(
        summed, on=keys, how="outer", suffixes=("_WEEKLY", "_DAILY"), indicator=True
    )
    reconciled = comparison._merge.eq("both").all()
    for column in measures:
        left = comparison[f"{column}_WEEKLY"]
        right = comparison[f"{column}_DAILY"]
        reconciled = reconciled and np.allclose(left, right, atol=1e-9)
    if not reconciled:
        errors.append("Weekly counts do not equal summed daily counts")


def validate_sightings_artifact(artifact: ArtifactRef) -> ValidationReport:
    dataset_id = artifact.dataset_id or artifact.kind
    path = artifact.path
    errors: list[str] = []
    if not path.exists():
        return ValidationReport(
            False, dataset_id, errors=(f"Missing artifact: {path}",)
        )
    if artifact.checksum and checksum_path(path) != artifact.checksum:
        errors.append("Artifact checksum does not match content")
    if (
        dataset_id.startswith("whale.sightings.source_")
        and dataset_id not in SCHEMAS
        and path.is_dir()
    ):
        snapshots = (
            list(path.glob("*/snapshot.json"))
            if path.name == "snapshots"
            else [path / "snapshot.json"]
        )
        if not any(candidate.exists() for candidate in snapshots):
            errors.append("Source artifact has no snapshot.json")
        return ValidationReport(not errors, dataset_id, errors=tuple(errors))
    schema = SCHEMAS.get(dataset_id)
    if schema is None:
        return ValidationReport(
            False, dataset_id, errors=(f"No schema for {dataset_id}",)
        )
    if dataset_id in {
        "whale.sightings.source_record_history",
        "whale.sightings.source_records",
    }:
        spec = replace(
            DATASETS.get(dataset_id),
            schema=schema,
            schema_version=artifact.schema_version or "8",
        )
        base = validate_path(path, spec)
        errors.extend(base.errors)
        try:
            files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
            columns = [
                "SOURCE_OCCURRENCE_COUNT",
                "SOURCE_USE_CLASS",
                "SOURCE_QC_STATUS",
            ]
            slim = pa.concat_tables(
                [pq.ParquetFile(item).read(columns=columns) for item in files],
                promote_options="permissive",
            ).to_pandas()
            errors.extend(_domain_errors(slim, dataset_id))
        except Exception as exc:
            errors.append(str(exc))
        return ValidationReport(
            not errors,
            dataset_id,
            schema_valid=base.schema_valid,
            key_unique=base.key_unique,
            errors=tuple(errors),
            metrics=base.metrics,
        )
    try:
        table = _read_directory(path) if path.is_dir() else pq.read_table(path)
    except Exception as exc:
        return ValidationReport(False, dataset_id, errors=tuple(errors + [str(exc)]))
    if (
        dataset_id
        in {
            "whale.sightings.imputed_retrospective",
            "whale.sightings.imputed_as_of",
        }
        and "EXPECTED_OTHER_COUNT" not in table.column_names
    ):
        schema = IMPUTED_OBSERVATION_SCHEMA_V8
    elif (
        dataset_id == "whale.sightings.observations"
        and "COORDINATE_SELECTION_METHOD" not in table.column_names
    ):
        schema = OBSERVATION_SCHEMA_V8
    elif "TARGET_COHORT_ID" not in table.column_names:
        legacy_count_schemas = {
            "whale.sightings.ecotype_counts": COUNT_SCHEMA_V6,
            "whale.sightings.ecotype_detail_counts": DETAIL_COUNT_SCHEMA_V6,
            "whale.sightings.orca_total_counts": TOTAL_COUNT_SCHEMA_V6,
            "whale.sightings.pod_counts": POD_COUNT_SCHEMA_V6,
            "whale.sightings.period_totals": PERIOD_TOTAL_SCHEMA_V6,
            "whale.sightings.reported_sighting_grid": MODEL_GRID_SCHEMA_V6,
            "whale.sightings.relative_reported_activity": MODEL_INTENSITY_SCHEMA_V6,
            "whale.sightings.relative_intensity": MODEL_INTENSITY_SCHEMA_V6,
        }
        schema = legacy_count_schemas.get(dataset_id, schema)
    base = validate_table(
        table,
        replace(
            DATASETS.get(dataset_id),
            schema=schema,
            schema_version=artifact.schema_version or "4",
        ),
    )
    errors.extend(base.errors)
    frame = table.to_pandas()
    errors.extend(_domain_errors(frame, dataset_id))
    if dataset_id == "whale.sightings.normalization_audit":
        source_path = path.parent / "state/source_current.parquet"
        if source_path.exists():
            source_ids = set(
                pd.read_parquet(
                    source_path, columns=["SOURCE_RECORD_ID"]
                ).SOURCE_RECORD_ID
            )
            missing = source_ids - set(frame.SOURCE_RECORD_ID)
            if missing:
                errors.append(f"Audit does not cover {len(missing)} source records")
    if dataset_id == "whale.sightings.ecotype_counts":
        dataset_manifest = json.loads((path / "_dataset_manifest.json").read_text())
        _validate_weekly_reconciliation(
            frame,
            errors,
            dataset_manifest["start_date"],
            dataset_manifest["end_date"],
        )
        total_path = path.parent / "orca_total_counts"
        if total_path.exists():
            total = _read_directory(total_path).to_pandas()
            keys = ["H3_INDEX", "H3_RESOLUTION", "FREQUENCY", "PERIOD_START"]
            measures = [
                "SIGHTING_COUNT",
                "SOURCE_REPORT_COUNT",
                "OBSERVED_SIGHTING_COUNT",
                "HARD_IMPUTED_COUNT",
                "EXPECTED_SIGHTING_COUNT",
                "MATURE_EXPECTED_SIGHTING_COUNT",
                "PROVISIONAL_EXPECTED_COUNT",
                "EXPECTED_UNKNOWN_COUNT",
            ]
            measures = [
                column for column in measures if column in frame and column in total
            ]
            bucket_sum = frame.groupby(keys, as_index=False)[measures].sum()
            comparison = total.merge(
                bucket_sum,
                on=keys,
                how="outer",
                suffixes=("_TOTAL", "_BUCKET"),
                indicator=True,
            )
            reconciled = comparison._merge.eq("both").all()
            for column in measures:
                reconciled = reconciled and np.allclose(
                    comparison[f"{column}_TOTAL"],
                    comparison[f"{column}_BUCKET"],
                    atol=1e-9,
                )
            if not reconciled:
                errors.append("Three-bucket counts do not reconcile with total counts")
    if path.is_dir():
        _validate_manifest(path, dataset_id, errors)
    return ValidationReport(
        valid=not errors,
        dataset_id=dataset_id,
        schema_valid=base.schema_valid,
        key_unique=base.key_unique,
        errors=tuple(dict.fromkeys(errors)),
        metrics={**base.metrics, "checksum": checksum_path(path)},
    )
