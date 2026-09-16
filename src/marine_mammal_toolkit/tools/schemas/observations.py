from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools._core.data import CollectionRequest
from marine_mammal_toolkit.tools._core.data import StageRequest

SOURCE_RECORD_SCHEMA = pa.schema(
    [
        # Polars uses Arrow large-string/list buffers natively. Keeping that
        # representation at the source-state boundary avoids a full copy of
        # every historical text column on each normalization run.
        pa.field("SOURCE_RECORD_ID", pa.large_string(), nullable=False),
        pa.field("SOURCE", pa.large_string(), nullable=False),
        pa.field("SOURCE_NATIVE_ID", pa.large_string()),
        pa.field("OBSERVED_AT_RAW", pa.large_string()),
        pa.field("OBSERVED_DATE_RAW", pa.large_string()),
        pa.field("CREATED_AT_RAW", pa.large_string()),
        pa.field("LATITUDE_RAW", pa.large_string()),
        pa.field("LONGITUDE_RAW", pa.large_string()),
        pa.field("SPECIES_RAW", pa.large_string()),
        pa.field("DESCRIPTION_RAW", pa.large_string()),
        pa.field("POD_ECOTYPE_RAW", pa.large_string()),
        pa.field("SOURCE_DATASET_ID", pa.large_string()),
        pa.field("SOURCE_EVENT_ID", pa.large_string()),
        pa.field("SOURCE_OCCURRENCE_IDS", pa.large_list(pa.large_string())),
        pa.field("SOURCE_OCCURRENCE_COUNT", pa.int32()),
        pa.field("SOURCE_LICENSE", pa.large_string()),
        pa.field("SOURCE_USE_CLASS", pa.large_string()),
        pa.field("COORDINATE_UNCERTAINTY_M", pa.float64()),
        pa.field("SOURCE_QC_STATUS", pa.large_string()),
        pa.field("SOURCE_QC_DETAIL", pa.large_string()),
        # Full-refresh state can exceed Arrow string's 2 GiB total offset limit
        # once several verbose source payloads are retained together.
        pa.field("SOURCE_PAYLOAD", pa.large_string(), nullable=False),
        pa.field("SOURCE_RETRIEVED_AT_UTC", pa.timestamp("us", tz="UTC")),
        pa.field("SOURCE_PAYLOAD_CORRECTED", pa.bool_(), nullable=False),
        pa.field("LAST_CORRECTED_AT_UTC", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

SOURCE_HISTORY_SCHEMA = pa.schema(
    [
        *SOURCE_RECORD_SCHEMA,
        pa.field("RETRIEVAL_ID", pa.large_string(), nullable=False),
        pa.field("RETRIEVED_AT", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("PAYLOAD_CHECKSUM", pa.large_string(), nullable=False),
        pa.field("RAW_SCHEMA_FINGERPRINT", pa.large_string(), nullable=False),
        pa.field("SNAPSHOT_MODE", pa.large_string(), nullable=False),
    ]
)

OBSERVATION_SCHEMA_V8 = pa.schema(
    [
        pa.field("OBSERVATION_ID", pa.string(), nullable=False),
        pa.field("EVENT_DATE", pa.date32(), nullable=False),
        pa.field("SIGHTING_DATE", pa.date32(), nullable=False),
        pa.field("SIGHTING_DATE_UTC", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("SOURCE_EVENT_AT_UTC", pa.timestamp("us", tz="UTC")),
        pa.field("SOURCE_CREATED_AT_UTC", pa.timestamp("us", tz="UTC")),
        pa.field("AVAILABLE_AT_UTC", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field(
            "RECORD_AVAILABLE_AT_UTC", pa.timestamp("us", tz="UTC"), nullable=False
        ),
        pa.field(
            "LABEL_AVAILABLE_AT_UTC", pa.timestamp("us", tz="UTC"), nullable=False
        ),
        pa.field("LAST_CORRECTED_AT_UTC", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("DATE_BASIS", pa.string(), nullable=False),
        pa.field("SOURCE_TIMEZONE", pa.string(), nullable=False),
        pa.field("MODEL_TIMEZONE", pa.string(), nullable=False),
        pa.field("SOURCE_TIME_PRECISION", pa.string(), nullable=False),
        pa.field("CANONICAL_TIME_SYNTHETIC", pa.bool_(), nullable=False),
        pa.field("LATITUDE", pa.float64(), nullable=False),
        pa.field("LONGITUDE", pa.float64(), nullable=False),
        pa.field("COORDINATE_TRANSFORM", pa.string(), nullable=False),
        pa.field("SPECIES_COMMON", pa.string(), nullable=False),
        pa.field("SPECIES_SCIENTIFIC", pa.string(), nullable=False),
        pa.field("ECOTYPE_DETAIL", pa.string(), nullable=False),
        pa.field("ECOTYPE_BUCKET", pa.string(), nullable=False),
        pa.field("SOURCE", pa.string(), nullable=False),
        pa.field("SOURCE_RECORD_ID", pa.string(), nullable=False),
        pa.field("SOURCE_REPORT_COUNT", pa.int16(), nullable=False),
        pa.field("SOURCE_OCCURRENCE_COUNT", pa.int32(), nullable=False),
        pa.field("SOURCE_RECORD_IDS", pa.list_(pa.string()), nullable=False),
        pa.field("PUBLIC_RELEASE_ELIGIBLE", pa.bool_(), nullable=False),
        pa.field("NORMALIZATION_VERSION", pa.string(), nullable=False),
    ]
)

OBSERVATION_SCHEMA = pa.schema(
    [
        *OBSERVATION_SCHEMA_V8,
        pa.field("COORDINATE_UNCERTAINTY_M", pa.float64()),
        pa.field("COORDINATE_SELECTION_METHOD", pa.string(), nullable=False),
        pa.field("OBSERVATION_QUALITY_TIER", pa.string(), nullable=False),
        pa.field("CONTRIBUTOR_LICENSE_SUMMARY", pa.string(), nullable=False),
        pa.field("FIELD_PROVENANCE", pa.string(), nullable=False),
        pa.field("SEMANTIC_CORRECTION_LINEAGE", pa.string(), nullable=False),
    ]
)

ASSOCIATION_SCHEMA = pa.schema(
    [
        pa.field("OBSERVATION_ID", pa.string(), nullable=False),
        pa.field("SOURCE_RECORD_ID", pa.string(), nullable=False),
        pa.field("SOURCE", pa.string(), nullable=False),
        pa.field("ASSOCIATION_KIND", pa.string(), nullable=False),
        pa.field("ASSOCIATION_VALUE", pa.string(), nullable=False),
        pa.field("EVIDENCE_TEXT", pa.string()),
        pa.field("EVIDENCE_FIELD", pa.string(), nullable=False),
        pa.field("RULE_ID", pa.string(), nullable=False),
        pa.field("CONFIDENCE", pa.string(), nullable=False),
        pa.field("CONFLICTING", pa.bool_(), nullable=False),
    ]
)

AUDIT_SCHEMA = pa.schema(
    [
        pa.field("SOURCE_RECORD_ID", pa.string(), nullable=False),
        pa.field("OBSERVATION_ID", pa.string()),
        pa.field("STATUS", pa.string(), nullable=False),
        pa.field("REASON", pa.string(), nullable=False),
        pa.field("DETAIL", pa.string()),
        pa.field("RULE_ID", pa.string(), nullable=False),
    ]
)

IDENTITY_SCHEMA = pa.schema(
    [
        pa.field("SOURCE_RECORD_ID", pa.string(), nullable=False),
        pa.field("OBSERVATION_ID", pa.string(), nullable=False),
        pa.field("FIRST_SEEN_RUN_ID", pa.string(), nullable=False),
    ]
)

IDENTITY_ALIAS_SCHEMA = pa.schema(
    [
        pa.field("ALIAS_OBSERVATION_ID", pa.string(), nullable=False),
        pa.field("CANONICAL_OBSERVATION_ID", pa.string(), nullable=False),
        pa.field("RESOLVED_RUN_ID", pa.string(), nullable=False),
    ]
)

IDENTITY_LINEAGE_SCHEMA = pa.schema(
    [
        pa.field("OLD_OBSERVATION_ID", pa.string()),
        pa.field("NEW_OBSERVATION_ID", pa.string(), nullable=False),
        pa.field("RESOLUTION_TYPE", pa.string(), nullable=False),
        pa.field("RESOLVED_RUN_ID", pa.string(), nullable=False),
    ]
)

COUNT_MEASURE_FIELDS = (
    pa.field("SIGHTING_COUNT", pa.int32(), nullable=False),
    pa.field("SOURCE_REPORT_COUNT", pa.int32(), nullable=False),
    pa.field("OBSERVED_SIGHTING_COUNT", pa.int32(), nullable=False),
    pa.field("HARD_IMPUTED_COUNT", pa.int32(), nullable=False),
    pa.field("EXPECTED_SIGHTING_COUNT", pa.float64(), nullable=False),
    pa.field("MATURE_EXPECTED_SIGHTING_COUNT", pa.float64(), nullable=False),
    pa.field("PROVISIONAL_EXPECTED_COUNT", pa.float64(), nullable=False),
    pa.field("EXPECTED_UNKNOWN_COUNT", pa.float64(), nullable=False),
)

COUNT_SCHEMA_V6 = pa.schema(
    [
        pa.field("H3_INDEX", pa.string(), nullable=False),
        pa.field("H3_RESOLUTION", pa.int8(), nullable=False),
        pa.field("FREQUENCY", pa.string(), nullable=False),
        pa.field("PERIOD_START", pa.date32(), nullable=False),
        pa.field("PERIOD_END", pa.date32(), nullable=False),
        pa.field("YEAR", pa.int16(), nullable=False),
        pa.field("DAY_OF_YEAR", pa.int16()),
        pa.field("ISO_YEAR", pa.int16()),
        pa.field("ISO_WEEK", pa.int8()),
        pa.field("PERIOD_STATUS", pa.string(), nullable=False),
        pa.field("IS_COMPLETE_PERIOD", pa.bool_(), nullable=False),
        pa.field("ECOTYPE_BUCKET", pa.string(), nullable=False),
        *COUNT_MEASURE_FIELDS,
    ]
)

TARGET_COHORT_FIELDS = (
    pa.field("TARGET_COHORT_ID", pa.string(), nullable=False),
    pa.field("TARGET_COHORT_STATUS", pa.string(), nullable=False),
    pa.field("REPORTING_STATE", pa.string(), nullable=False),
)

COUNT_SCHEMA = pa.schema([*COUNT_SCHEMA_V6, *TARGET_COHORT_FIELDS])

DETAIL_COUNT_SCHEMA_V6 = COUNT_SCHEMA_V6.set(
    COUNT_SCHEMA_V6.get_field_index("ECOTYPE_BUCKET"),
    pa.field("ECOTYPE_DETAIL", pa.string(), nullable=False),
)
DETAIL_COUNT_SCHEMA = COUNT_SCHEMA.set(
    COUNT_SCHEMA.get_field_index("ECOTYPE_BUCKET"),
    pa.field("ECOTYPE_DETAIL", pa.string(), nullable=False),
)

TOTAL_COUNT_SCHEMA_V6 = pa.schema(
    [field for field in COUNT_SCHEMA_V6 if field.name != "ECOTYPE_BUCKET"]
)
TOTAL_COUNT_SCHEMA = pa.schema(
    [field for field in COUNT_SCHEMA if field.name != "ECOTYPE_BUCKET"]
)

POD_COUNT_SCHEMA_V6 = COUNT_SCHEMA_V6.set(
    COUNT_SCHEMA_V6.get_field_index("ECOTYPE_BUCKET"),
    pa.field("POD", pa.string(), nullable=False),
)
POD_COUNT_SCHEMA = COUNT_SCHEMA.set(
    COUNT_SCHEMA.get_field_index("ECOTYPE_BUCKET"),
    pa.field("POD", pa.string(), nullable=False),
)


PERIOD_TOTAL_SCHEMA_V6 = pa.schema(
    [
        field
        for field in TOTAL_COUNT_SCHEMA_V6
        if field.name not in {"H3_INDEX", "H3_RESOLUTION"}
    ]
)
PERIOD_TOTAL_SCHEMA = pa.schema(
    [
        field
        for field in TOTAL_COUNT_SCHEMA
        if field.name not in {"H3_INDEX", "H3_RESOLUTION"}
    ]
)

COUNT_EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("OBSERVATION_ID", pa.string(), nullable=False),
        pa.field("SIGHTING_DATE", pa.date32(), nullable=False),
        pa.field("LATITUDE", pa.float64(), nullable=False),
        pa.field("LONGITUDE", pa.float64(), nullable=False),
        pa.field("H3_INDEX", pa.string(), nullable=False),
        pa.field("H3_RESOLUTION", pa.int8(), nullable=False),
        pa.field("REASON", pa.string(), nullable=False),
    ]
)

MODEL_GRID_SCHEMA_V6 = COUNT_SCHEMA_V6.append(
    pa.field("REPORTED_SIGHTING", pa.int8(), nullable=False)
)
MODEL_GRID_SCHEMA = COUNT_SCHEMA.append(
    pa.field("REPORTED_SIGHTING", pa.int8(), nullable=False)
)

MODEL_INTENSITY_SCHEMA_V6 = MODEL_GRID_SCHEMA_V6.append(
    pa.field("RELATIVE_SIGHTING_INTENSITY", pa.float32(), nullable=False)
)
MODEL_INTENSITY_SCHEMA = MODEL_GRID_SCHEMA.append(
    pa.field("RELATIVE_REPORTED_ACTIVITY", pa.float32(), nullable=False)
).append(pa.field("RELATIVE_SIGHTING_INTENSITY", pa.float32(), nullable=False))

IMPUTED_EXTRA_FIELDS_V8 = (
    pa.field("ENCOUNTER_ID", pa.string(), nullable=False),
    pa.field("ENCOUNTER_SIZE", pa.int32(), nullable=False),
    pa.field("ECOTYPE_DETAIL_OBSERVED", pa.string(), nullable=False),
    pa.field("ECOTYPE_DETAIL_IMPUTED", pa.string()),
    pa.field("ECOTYPE_DETAIL_EFFECTIVE", pa.string(), nullable=False),
    pa.field("ECOTYPE_BUCKET_EFFECTIVE", pa.string(), nullable=False),
    pa.field("IMPUTATION_APPLIED", pa.bool_(), nullable=False),
    pa.field("IMPUTATION_METHOD", pa.string()),
    pa.field("IMPUTATION_CONFIDENCE", pa.float32()),
    pa.field("IMPUTATION_TIME_DIRECTION", pa.string(), nullable=False),
    pa.field("ELIGIBLE_FOR_TRAINING", pa.bool_(), nullable=False),
    pa.field("ELIGIBLE_FOR_EVALUATION", pa.bool_(), nullable=False),
    pa.field("P_SRKW", pa.float64()),
    pa.field("P_TRANSIENT", pa.float64()),
    pa.field("EVIDENCE_REGIME", pa.string()),
    pa.field("ABSTENTION_REASON", pa.string()),
    pa.field("ECOTYPE_LABEL_TIER", pa.string(), nullable=False),
    pa.field("EXPECTED_SRKW_COUNT", pa.float64(), nullable=False),
    pa.field("EXPECTED_TRANSIENT_COUNT", pa.float64(), nullable=False),
    pa.field("EXPECTED_UNKNOWN_COUNT", pa.float64(), nullable=False),
    pa.field("USE_FOR_HARD_COUNTS", pa.bool_(), nullable=False),
    pa.field("USE_FOR_PROBABILISTIC_COUNTS", pa.bool_(), nullable=False),
    pa.field("RETROSPECTIVE_CONTEXT_COMPLETE", pa.bool_(), nullable=False),
    pa.field("IMPUTATION_CONTEXT_STATUS", pa.string(), nullable=False),
    pa.field("CONTEXT_COMPLETE_THROUGH_DATE", pa.date32()),
    pa.field("CLASS_CERTIFIED_FOR_HARD_LABEL", pa.bool_()),
    pa.field("ACCEPTANCE_POLICY_SCOPE", pa.string()),
    pa.field("ACCEPTANCE_POLICY_ERROR_UPPER", pa.float64()),
    pa.field("PREDICTION_STABLE", pa.bool_()),
    pa.field("IMPUTATION_MODEL_VERSION", pa.string(), nullable=False),
    pa.field("IMPUTATION_MODEL_FIT_AT_UTC", pa.timestamp("us", tz="UTC")),
    pa.field("IMPUTATION_MODEL_RUN_ID", pa.string()),
)
IMPUTED_OBSERVATION_SCHEMA_V8 = pa.schema(
    [*OBSERVATION_SCHEMA_V8, *IMPUTED_EXTRA_FIELDS_V8]
)

IMPUTED_EXTRA_FIELDS = (
    *IMPUTED_EXTRA_FIELDS_V8[:13],
    pa.field("P_OTHER", pa.float64()),
    *IMPUTED_EXTRA_FIELDS_V8[13:18],
    pa.field("EXPECTED_OTHER_COUNT", pa.float64(), nullable=False),
    *IMPUTED_EXTRA_FIELDS_V8[18:25],
    pa.field("SOFT_COUNT_CERTIFIED", pa.bool_(), nullable=False),
    pa.field("HARD_LABEL_CERTIFIED", pa.bool_(), nullable=False),
    pa.field("STABILITY_EVALUATED", pa.bool_(), nullable=False),
    pa.field("ENCOUNTER_LABEL_CONFLICT", pa.bool_(), nullable=False),
    pa.field("SRKW_DOMAIN_SUPPORTED", pa.bool_(), nullable=False),
    pa.field("TRANSIENT_DOMAIN_SUPPORTED", pa.bool_(), nullable=False),
    pa.field("IMPUTATION_DOMAIN_SUPPORTED", pa.bool_(), nullable=False),
    pa.field("BINARY_MODEL_DOMAIN_SUPPORTED", pa.bool_(), nullable=False),
    pa.field("MODEL_DOMAIN_STATUS", pa.string()),
    *IMPUTED_EXTRA_FIELDS_V8[25:],
    pa.field("IMPUTATION_DATA_SNAPSHOT_ID", pa.string(), nullable=False),
)
IMPUTED_OBSERVATION_SCHEMA = pa.schema([*OBSERVATION_SCHEMA, *IMPUTED_EXTRA_FIELDS])


@dataclass(frozen=True)
class NormalizationRequest(StageRequest):
    pass


@dataclass(frozen=True)
class SightingsCollectionRequest(CollectionRequest):
    start_date: date | None = None
    end_date: date | None = None
    twm_files: tuple[Path, ...] = ()
    full_refresh: bool = False


@dataclass(frozen=True)
class ImputationRequest(StageRequest):
    observations: ArtifactRef | None = None
    associations: ArtifactRef | None = None
    model_path: Path | None = None


@dataclass(frozen=True)
class CountRequest(StageRequest):
    observations: ArtifactRef | None = None
    associations: ArtifactRef | None = None
    water_universes: tuple[ArtifactRef, ...] = ()
    start_date: date | None = None
    end_date: date | None = None
    resolutions: tuple[int, ...] = (4, 5, 6)


@dataclass(frozen=True)
class ModelGridRequest(StageRequest):
    ecotype_counts: ArtifactRef | None = None
    water_universes: tuple[ArtifactRef, ...] = ()
    start_date: date | None = None
    end_date: date | None = None
    resolutions: tuple[int, ...] = (4, 5, 6)
    frequencies: tuple[str, ...] = ("daily", "weekly")


@dataclass(frozen=True)
class IntensityRequest(StageRequest):
    model_grid: ArtifactRef | None = None
