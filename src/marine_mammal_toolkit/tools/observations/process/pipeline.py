from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pyarrow as pa

from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools._core.data import DATASETS
from marine_mammal_toolkit.tools._core.data import ArtifactStore
from marine_mammal_toolkit.tools._core.data import StageResult
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
from marine_mammal_toolkit.tools.schemas.observations import ASSOCIATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import AUDIT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import IDENTITY_ALIAS_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import IDENTITY_LINEAGE_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import IDENTITY_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import OBSERVATION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import SOURCE_HISTORY_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import SOURCE_RECORD_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import NormalizationRequest
from marine_mammal_toolkit.tools.observations.runtime import build_data_snapshot
from marine_mammal_toolkit.tools.observations.runtime import code_revision
from marine_mammal_toolkit.tools.observations.runtime import resume_result
from marine_mammal_toolkit.tools.observations.runtime import stage_signature

NORMALIZATION_VERSION = "9"
LOGGER = logging.getLogger(__name__)
SourceKey = Literal["twm", "acartia", "maplify", "inaturalist", "cwr", "gbif"]
SOURCE_KEYS: dict[str, SourceKey] = {
    "TWM": "twm",
    "ACARTIA": "acartia",
    "MAPLIFY": "maplify",
    "INATURALIST": "inaturalist",
    "CWR": "cwr",
    "GBIF": "gbif",
}


def _source_key(source: str) -> SourceKey:
    return SOURCE_KEYS[source]


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except TypeError:
        pass
    except ValueError:
        pass
    return bool(str(value).strip())


def _parse_explicit_date(value: Any, formats: tuple[str, ...]) -> date | None:
    if not _present(value):
        return None
    text = str(value).strip()
    for date_format in formats:
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    return None


def _parse_timestamp(value: Any, timezone_name: str) -> pd.Timestamp | None:
    if not _present(value):
        return None
    try:
        parsed = pd.to_datetime(
            str(value).strip(), format="ISO8601", errors="raise", utc=False
        )
    except (TypeError, ValueError):
        return None
    stamp = pd.Timestamp(parsed)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(
            ZoneInfo(timezone_name), ambiguous="NaT", nonexistent="NaT"
        )
    return None if pd.isna(stamp) else stamp.tz_convert("UTC")


def _contains_clock(value: Any) -> bool:
    if not _present(value):
        return False
    text = str(value).strip()
    return bool(re.search(r"(?:T|\s)\d{1,2}:\d{2}", text))


def _temporal_fields(
    row: Mapping[str, Any], config: SightingsPipelineConfig
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source = str(row["SOURCE"])
    settings = config.collection.sources[_source_key(source)]
    source_timezone = settings.timezone
    raw_event_is_timestamp = _contains_clock(row["OBSERVED_AT_RAW"])
    event_at = (
        _parse_timestamp(row["OBSERVED_AT_RAW"], source_timezone)
        if raw_event_is_timestamp
        else None
    )
    created_at = _parse_timestamp(row["CREATED_AT_RAW"], source_timezone)
    retrieved_at = (
        pd.Timestamp(row["SOURCE_RETRIEVED_AT_UTC"]).tz_convert("UTC")
        if _present(row["SOURCE_RETRIEVED_AT_UTC"])
        else None
    )
    available_at = created_at or retrieved_at
    if available_at is None:
        raise ValueError(
            f"Missing availability timestamp for {row['SOURCE_RECORD_ID']}"
        )
    explicit_date = _parse_explicit_date(
        row["OBSERVED_DATE_RAW"], settings.date_formats
    )
    event_date = (
        event_at.tz_convert(ZoneInfo(config.model_timezone)).date()
        if event_at is not None
        else _parse_explicit_date(row["OBSERVED_AT_RAW"], settings.date_formats)
    )
    conflict = None
    if explicit_date is not None:
        sighting_date = explicit_date
        basis = "OBSERVED_DATE_RAW"
        if event_date is not None and event_date != explicit_date:
            conflict = {
                "explicit_date": explicit_date.isoformat(),
                "timestamp_date": event_date.isoformat(),
            }
    elif event_date is not None:
        sighting_date = event_date
        basis = (
            "ACARTIA_CREATED_EVENT"
            if source == "ACARTIA"
            and settings.created_is_event_time
            and _present(row["CREATED_AT_RAW"])
            and str(row["CREATED_AT_RAW"]).strip()
            == str(row["OBSERVED_AT_RAW"]).strip()
            else "SOURCE_EVENT_AT_RAW"
        )
    else:
        return None, None
    canonical = pd.Timestamp(datetime.combine(sighting_date, time(12), timezone.utc))
    return (
        {
            "SIGHTING_DATE": sighting_date,
            "SIGHTING_DATE_UTC": canonical,
            "SOURCE_EVENT_AT_UTC": event_at,
            "SOURCE_CREATED_AT_UTC": created_at,
            "AVAILABLE_AT_UTC": available_at,
            "DATE_BASIS": basis,
            "SOURCE_TIMEZONE": source_timezone,
            "MODEL_TIMEZONE": config.model_timezone,
            "SOURCE_TIME_PRECISION": "TIMESTAMP" if raw_event_is_timestamp else "DATE",
            "CANONICAL_TIME_SYNTHETIC": True,
        },
        conflict,
    )


def _coordinates(
    row: Mapping[str, Any], config: SightingsPipelineConfig
) -> tuple[float, float, str] | None:
    try:
        latitude = float(row["LATITUDE_RAW"])
        longitude = float(row["LONGITUDE_RAW"])
    except TypeError:
        return None
    except ValueError:
        return None
    bbox = config.full_area

    def inside(lat: float, lon: float) -> bool:
        return (
            bbox.min_lat <= lat <= bbox.max_lat and bbox.min_lon <= lon <= bbox.max_lon
        )

    if inside(latitude, longitude):
        return latitude, longitude, "NONE"
    candidates: list[tuple[float, float, str]] = []
    if inside(longitude, latitude):
        candidates.append((longitude, latitude, "SWAPPED_LAT_LON"))
    if str(row.get("SOURCE") or "").upper() == "TWM":
        if inside(latitude, -longitude):
            candidates.append((latitude, -longitude, "WEST_SIGN_INFERRED"))
        if inside(longitude, -latitude):
            candidates.append(
                (longitude, -latitude, "SWAPPED_LAT_LON_WEST_SIGN_INFERRED")
            )
    unique = {(lat, lon): transform for lat, lon, transform in candidates}
    if len(unique) == 1:
        (lat, lon), transform = next(iter(unique.items()))
        return lat, lon, transform
    return None


def _haversine_m(left: dict[str, Any], right: dict[str, Any]) -> float:
    radius = 6_371_008.8
    lat1, lat2 = math.radians(left["LATITUDE"]), math.radians(right["LATITUDE"])
    dlat = lat2 - lat1
    dlon = math.radians(right["LONGITUDE"] - left["LONGITUDE"])
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(value))


def _compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_type, right_type = left["ECOTYPE_DETAIL"], right["ECOTYPE_DETAIL"]
    return (
        left_type in {"UNKNOWN", "MIXED"}
        or right_type in {"UNKNOWN", "MIXED"}
        or left_type == right_type
    )


def _strong_ids(record: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (item["ASSOCIATION_KIND"], item["ASSOCIATION_VALUE"])
        for item in record["EVIDENCE"]
        if item["ASSOCIATION_KIND"] in {"MEMBER", "SOCIAL_GROUP"}
    }


def _match_metrics(
    left: dict[str, Any], right: dict[str, Any], config: SightingsPipelineConfig
) -> tuple[bool, dict[str, float | None]]:
    if left["SOURCE"] == right["SOURCE"] or not _compatible(left, right):
        return False, {"distance_m": None, "minutes": None}
    if left["SIGHTING_DATE"] != right["SIGHTING_DATE"]:
        return False, {"distance_m": None, "minutes": None}
    distance = _haversine_m(left, right)
    if distance > config.deduplication.distance_tolerance_m:
        return False, {"distance_m": distance, "minutes": None}
    if (
        left["SOURCE_EVENT_AT_UTC"] is not None
        and right["SOURCE_EVENT_AT_UTC"] is not None
    ):
        minutes = (
            abs(
                (
                    left["SOURCE_EVENT_AT_UTC"] - right["SOURCE_EVENT_AT_UTC"]
                ).total_seconds()
            )
            / 60
        )
        return minutes <= config.deduplication.timestamp_tolerance_minutes, {
            "distance_m": distance,
            "minutes": minutes,
        }
    return bool(_strong_ids(left) & _strong_ids(right)), {
        "distance_m": distance,
        "minutes": None,
    }


def _cluster(
    records: list[dict[str, Any]],
    config: SightingsPipelineConfig,
    audit: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    policy = config.observation_policy
    groups_by_date: dict[date, list[list[dict[str, Any]]]] = {}
    ordered = sorted(
        records,
        key=lambda item: (
            item["SIGHTING_DATE"],
            item["SOURCE_EVENT_AT_UTC"] or item["SIGHTING_DATE_UTC"],
            policy.source_priority[item["SOURCE"]],
            item["SOURCE_RECORD_ID"],
        ),
    )
    for record in ordered:
        groups = groups_by_date.setdefault(record["SIGHTING_DATE"], [])
        compatible_groups: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        partial_groups: list[list[dict[str, Any]]] = []
        for group in groups:
            comparisons = [_match_metrics(record, member, config) for member in group]
            if comparisons and all(matched for matched, _ in comparisons):
                comparison_details = [
                    {"with": member["SOURCE_RECORD_ID"], **metrics}
                    for member, (_, metrics) in zip(group, comparisons, strict=True)
                ]
                compatible_groups.append((group, comparison_details))
            elif any(matched for matched, _metrics in comparisons):
                partial_groups.append(group)
        if len(compatible_groups) != 1:
            groups.append([record])
            if compatible_groups or partial_groups:
                audit.append(
                    _audit(
                        record["SOURCE_RECORD_ID"],
                        None,
                        "REVIEW_REQUIRED",
                        "UNCERTAIN_MATCH",
                        json.dumps(
                            {
                                "complete_candidate_groups": [
                                    sorted(item["SOURCE_RECORD_ID"] for item in group)
                                    for group, _details in compatible_groups
                                ],
                                "partial_chain_groups": [
                                    sorted(item["SOURCE_RECORD_ID"] for item in group)
                                    for group in partial_groups
                                ],
                            },
                            sort_keys=True,
                        ),
                        "dedup.v5.ambiguous_review",
                    )
                )
        else:
            destination, comparison_details = compatible_groups[0]
            destination.append(record)
            audit.append(
                _audit(
                    record["SOURCE_RECORD_ID"],
                    None,
                    "MERGED",
                    "CROSS_SOURCE_CLUSTER",
                    json.dumps({"complete_linkage": comparison_details}),
                    "dedup.v4.complete_linkage",
                )
            )
    return [group for groups in groups_by_date.values() for group in groups]


def _audit(
    source_id: str,
    observation_id: str | None,
    status: str,
    reason: str,
    detail: str | None,
    rule: str,
) -> dict[str, Any]:
    return {
        "SOURCE_RECORD_ID": source_id,
        "OBSERVATION_ID": observation_id,
        "STATUS": status,
        "REASON": reason,
        "DETAIL": detail,
        "RULE_ID": rule,
    }


def _normalize_records(
    source_frame: pd.DataFrame | pl.DataFrame, config: SightingsPipelineConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = config.observation_policy
    if isinstance(source_frame, pd.DataFrame):
        source_frame = pl.from_pandas(source_frame)
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    min_date = datetime.strptime(config.min_date, "%Y-%m-%d").date()
    for row in source_frame.iter_rows(named=True):
        qc_status = (
            str(row.get("SOURCE_QC_STATUS"))
            if _present(row.get("SOURCE_QC_STATUS"))
            else "ACCEPTED"
        )
        if qc_status.upper() != "ACCEPTED":
            detail = (
                str(row.get("SOURCE_QC_DETAIL"))
                if _present(row.get("SOURCE_QC_DETAIL"))
                else None
            )
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "QUARANTINED",
                    "SOURCE_QC_QUARANTINED",
                    detail,
                    "source.qc.v1",
                )
            )
            continue
        if not policy.accepts(row):
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "REJECTED",
                    "NON_ORCA",
                    None,
                    "species.orca",
                )
            )
            continue
        temporal, conflict = _temporal_fields(row, config)
        if temporal is None:
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "REJECTED",
                    "MISSING_OBSERVATION_DATE",
                    None,
                    "time.required",
                )
            )
            continue
        if temporal["SIGHTING_DATE"] < min_date:
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "REJECTED",
                    "BEFORE_MIN_DATE",
                    None,
                    "time.min_date",
                )
            )
            continue
        if conflict:
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "WARNING",
                    "DATE_TIMESTAMP_CONFLICT",
                    json.dumps(conflict),
                    "time.explicit_date_precedence",
                )
            )
        coordinates = _coordinates(row, config)
        if coordinates is None:
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "REJECTED",
                    "INVALID_OR_OUTSIDE_AOI_COORDINATES",
                    None,
                    "geo.full_area",
                )
            )
            continue
        if "WEST_SIGN_INFERRED" in coordinates[2]:
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "WARNING",
                    "WEST_SIGN_INFERRED",
                    json.dumps(
                        {
                            "latitude_raw": row.get("LATITUDE_RAW"),
                            "longitude_raw": row.get("LONGITUDE_RAW"),
                            "transform": coordinates[2],
                        },
                        sort_keys=True,
                    ),
                    "geo.twm_west_sign.v1",
                )
            )
        evidence = policy.evidence(row)
        detail = policy.classify(evidence)
        last_corrected = row.get("LAST_CORRECTED_AT_UTC")
        if not _present(last_corrected):
            last_corrected = pd.Timestamp.now(tz="UTC")
        else:
            last_corrected = pd.Timestamp(last_corrected)
            last_corrected = (
                last_corrected.tz_localize("UTC")
                if last_corrected.tzinfo is None
                else last_corrected.tz_convert("UTC")
            )
        payload_corrected = bool(row.get("SOURCE_PAYLOAD_CORRECTED", False))
        occurrence_count = (
            int(row.get("SOURCE_OCCURRENCE_COUNT"))
            if _present(row.get("SOURCE_OCCURRENCE_COUNT"))
            else 1
        )
        use_class = (
            str(row.get("SOURCE_USE_CLASS"))
            if _present(row.get("SOURCE_USE_CLASS"))
            else "INTERNAL_ONLY"
        )
        uncertainty_raw = row.get("COORDINATE_UNCERTAINTY_M")
        try:
            coordinate_uncertainty = (
                float(uncertainty_raw) if _present(uncertainty_raw) else None
            )
        except (TypeError, ValueError):
            coordinate_uncertainty = None
        if coordinate_uncertainty is not None and (
            not math.isfinite(coordinate_uncertainty) or coordinate_uncertainty < 0
        ):
            coordinate_uncertainty = None
        records.append(
            {
                "SOURCE_RECORD_ID": str(row["SOURCE_RECORD_ID"]),
                "SOURCE": str(row["SOURCE"]),
                **temporal,
                "LABEL_SOURCE_AVAILABLE_AT_UTC": (
                    max(temporal["AVAILABLE_AT_UTC"], last_corrected)
                    if payload_corrected
                    else temporal["AVAILABLE_AT_UTC"]
                ),
                "LAST_CORRECTED_AT_UTC": last_corrected,
                "LATITUDE": coordinates[0],
                "LONGITUDE": coordinates[1],
                "COORDINATE_TRANSFORM": coordinates[2],
                "COORDINATE_UNCERTAINTY_M": coordinate_uncertainty,
                "EVIDENCE": evidence,
                "ECOTYPE_DETAIL": detail,
                "ECOTYPE_BUCKET": policy.detail_to_bucket[detail],
                "SOURCE_OCCURRENCE_COUNT": occurrence_count,
                "SOURCE_USE_CLASS": use_class,
                "SOURCE_LICENSE": (
                    str(row.get("SOURCE_LICENSE"))
                    if _present(row.get("SOURCE_LICENSE"))
                    else "UNKNOWN"
                ),
                "SOURCE_QC_DETAIL": (
                    str(row.get("SOURCE_QC_DETAIL"))
                    if _present(row.get("SOURCE_QC_DETAIL"))
                    else None
                ),
                "SOURCE_PAYLOAD_CORRECTED": payload_corrected,
            }
        )
        if str(row["SOURCE"]).upper() == "GBIF":
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "GROUPED",
                    "GBIF_SOURCE_EVENT_GROUPING",
                    (
                        str(row.get("SOURCE_QC_DETAIL"))
                        if _present(row.get("SOURCE_QC_DETAIL"))
                        else None
                    ),
                    "source.gbif.group.v1",
                )
            )
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    (
                        "RETAINED_PUBLIC"
                        if use_class == "REDISTRIBUTABLE"
                        else "RETAINED_INTERNAL"
                    ),
                    (
                        "SOURCE_LICENSE_REDISTRIBUTABLE"
                        if use_class == "REDISTRIBUTABLE"
                        else "SOURCE_LICENSE_INTERNAL_ONLY"
                    ),
                    (
                        str(row.get("SOURCE_LICENSE"))
                        if _present(row.get("SOURCE_LICENSE"))
                        else "UNKNOWN_LICENSE"
                    ),
                    "license.public_release.v1",
                )
            )
        elif use_class != "REDISTRIBUTABLE":
            audit.append(
                _audit(
                    str(row["SOURCE_RECORD_ID"]),
                    None,
                    "RETAINED_INTERNAL",
                    "SOURCE_LICENSE_INTERNAL_ONLY",
                    (
                        str(row.get("SOURCE_LICENSE"))
                        if _present(row.get("SOURCE_LICENSE"))
                        else "UNKNOWN_LICENSE"
                    ),
                    "license.public_release.v1",
                )
            )
    return records, audit


def _coordinate_selection(
    group: list[dict[str, Any]], *, policy
) -> tuple[dict[str, Any], str]:
    def rank(record: dict[str, Any]) -> tuple[Any, ...]:
        uncertainty = record.get("COORDINATE_UNCERTAINTY_M")
        has_uncertainty = uncertainty is not None and math.isfinite(float(uncertainty))
        return (
            not has_uncertainty,
            float(uncertainty) if has_uncertainty else math.inf,
            record["COORDINATE_TRANSFORM"] != "NONE",
            policy.source_priority[record["SOURCE"]],
            record["SOURCE_RECORD_ID"],
        )

    selected = min(group, key=rank)
    method = (
        "MINIMUM_REPORTED_UNCERTAINTY"
        if selected.get("COORDINATE_UNCERTAINTY_M") is not None
        else "REVIEWED_SOURCE_PRIORITY"
    )
    return selected, method


def _quality_tier(record: dict[str, Any]) -> str:
    detail: dict[str, Any] = {}
    raw_detail = record.get("SOURCE_QC_DETAIL")
    if raw_detail:
        try:
            parsed = json.loads(str(raw_detail))
            detail = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            detail = {}
    quality_grade = str(detail.get("quality_grade") or "").strip().lower()
    if quality_grade in {"casual", "needs_id"}:
        return "LOW_QUALITY_NON_ANCHOR"
    uncertainty = record.get("COORDINATE_UNCERTAINTY_M")
    if uncertainty is None:
        return "UNKNOWN_COORDINATE_UNCERTAINTY"
    if record["COORDINATE_TRANSFORM"] != "NONE":
        return "COORDINATE_CORRECTED"
    if float(uncertainty) <= 100 and record["SOURCE_TIME_PRECISION"] == "TIMESTAMP":
        return "HIGH"
    if float(uncertainty) <= 1_000:
        return "STANDARD"
    return "LOW_SPATIAL_PRECISION"


def _materialize(
    groups: list[list[dict[str, Any]]],
    audit: list[dict[str, Any]],
    observation_ids: list[str],
    *,
    policy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        group = sorted(
            group,
            key=lambda item: (
                policy.source_priority[item["SOURCE"]],
                item["SOURCE_RECORD_ID"],
            ),
        )
        observation_id = observation_ids[group_index]
        preferred = sorted(
            group,
            key=lambda item: (
                item["DATE_BASIS"] != "OBSERVED_DATE_RAW",
                item["SOURCE_EVENT_AT_UTC"] is None,
                policy.source_priority[item["SOURCE"]],
            ),
        )[0]
        coordinate_record, coordinate_method = _coordinate_selection(
            group, policy=policy
        )
        evidence = [item for record in group for item in record["EVIDENCE"]]
        detail = policy.classify(evidence)
        conflicting = detail == "MIXED"
        label_support = [
            record
            for record in group
            if any(
                item["ASSOCIATION_KIND"] == "ECOTYPE"
                and (detail == "MIXED" or item["ASSOCIATION_VALUE"] == detail)
                for item in record["EVIDENCE"]
            )
        ] or group
        record_available_at = min(item["AVAILABLE_AT_UTC"] for item in group)
        observations.append(
            {
                "OBSERVATION_ID": observation_id,
                "EVENT_DATE": preferred["SIGHTING_DATE"],
                **{
                    key: preferred[key]
                    for key in (
                        "SIGHTING_DATE",
                        "SIGHTING_DATE_UTC",
                        "SOURCE_EVENT_AT_UTC",
                        "SOURCE_CREATED_AT_UTC",
                        "DATE_BASIS",
                        "SOURCE_TIMEZONE",
                        "MODEL_TIMEZONE",
                        "SOURCE_TIME_PRECISION",
                        "CANONICAL_TIME_SYNTHETIC",
                    )
                },
                "AVAILABLE_AT_UTC": record_available_at,
                "RECORD_AVAILABLE_AT_UTC": record_available_at,
                "LABEL_AVAILABLE_AT_UTC": max(
                    item["LABEL_SOURCE_AVAILABLE_AT_UTC"] for item in label_support
                ),
                "LAST_CORRECTED_AT_UTC": max(
                    item["LAST_CORRECTED_AT_UTC"] for item in group
                ),
                "LATITUDE": coordinate_record["LATITUDE"],
                "LONGITUDE": coordinate_record["LONGITUDE"],
                "COORDINATE_TRANSFORM": coordinate_record["COORDINATE_TRANSFORM"],
                "SPECIES_COMMON": policy.common_name,
                "SPECIES_SCIENTIFIC": policy.scientific_name,
                "ECOTYPE_DETAIL": detail,
                "ECOTYPE_BUCKET": policy.detail_to_bucket[detail],
                "SOURCE": preferred["SOURCE"],
                "SOURCE_RECORD_ID": preferred["SOURCE_RECORD_ID"],
                "SOURCE_REPORT_COUNT": len(group),
                "SOURCE_OCCURRENCE_COUNT": sum(
                    int(item.get("SOURCE_OCCURRENCE_COUNT") or 1) for item in group
                ),
                "SOURCE_RECORD_IDS": sorted(item["SOURCE_RECORD_ID"] for item in group),
                "PUBLIC_RELEASE_ELIGIBLE": all(
                    item.get("SOURCE_USE_CLASS") == "REDISTRIBUTABLE" for item in group
                ),
                "NORMALIZATION_VERSION": NORMALIZATION_VERSION,
                "COORDINATE_UNCERTAINTY_M": coordinate_record.get(
                    "COORDINATE_UNCERTAINTY_M"
                ),
                "COORDINATE_SELECTION_METHOD": coordinate_method,
                "OBSERVATION_QUALITY_TIER": _quality_tier(coordinate_record),
                "CONTRIBUTOR_LICENSE_SUMMARY": json.dumps(
                    [
                        {
                            "source_record_id": item["SOURCE_RECORD_ID"],
                            "source": item["SOURCE"],
                            "license": item.get("SOURCE_LICENSE") or "UNKNOWN",
                            "use_class": item.get("SOURCE_USE_CLASS")
                            or "INTERNAL_ONLY",
                        }
                        for item in group
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "FIELD_PROVENANCE": json.dumps(
                    {
                        "coordinate_source_record_id": coordinate_record[
                            "SOURCE_RECORD_ID"
                        ],
                        "time_source_record_id": preferred["SOURCE_RECORD_ID"],
                        "label_source_record_ids": sorted(
                            item["SOURCE_RECORD_ID"] for item in label_support
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "SEMANTIC_CORRECTION_LINEAGE": json.dumps(
                    [
                        {
                            "source_record_id": item["SOURCE_RECORD_ID"],
                            "last_corrected_at_utc": item[
                                "LAST_CORRECTED_AT_UTC"
                            ].isoformat(),
                            "coordinate_transform": item["COORDINATE_TRANSFORM"],
                        }
                        for item in group
                        if item.get("SOURCE_PAYLOAD_CORRECTED")
                        or item["COORDINATE_TRANSFORM"] != "NONE"
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        unique = {}
        for item in evidence:
            output = {key: value for key, value in item.items() if key != "_PRIORITY"}
            output["OBSERVATION_ID"] = observation_id
            output["CONFLICTING"] = (
                conflicting and output["ASSOCIATION_KIND"] == "ECOTYPE"
            )
            key = (
                output["OBSERVATION_ID"],
                output["SOURCE_RECORD_ID"],
                output["ASSOCIATION_KIND"],
                output["ASSOCIATION_VALUE"],
                output["RULE_ID"],
                output["EVIDENCE_FIELD"],
            )
            unique[key] = output
        associations.extend(unique.values())
        for record in group:
            audit.append(
                _audit(
                    record["SOURCE_RECORD_ID"],
                    observation_id,
                    "ACCEPTED",
                    "NORMALIZED",
                    None,
                    "normalize.v5",
                )
            )
    return observations, associations


def _resolve_identities(
    groups: list[list[dict[str, Any]]],
    identity_path: Path,
    alias_path: Path,
    lineage_path: Path,
    run_id: str,
    *,
    policy,
) -> tuple[list[str], pa.Table, pa.Table, pa.Table]:
    existing = (
        pd.read_parquet(identity_path)
        if identity_path.exists()
        else pd.DataFrame(columns=IDENTITY_SCHEMA.names)
    )
    aliases = (
        pd.read_parquet(alias_path)
        if alias_path.exists()
        else pd.DataFrame(columns=IDENTITY_ALIAS_SCHEMA.names)
    )
    lineage = (
        pd.read_parquet(lineage_path)
        if lineage_path.exists()
        else pd.DataFrame(columns=IDENTITY_LINEAGE_SCHEMA.names)
    )
    alias_map = (
        dict(
            zip(
                aliases.ALIAS_OBSERVATION_ID,
                aliases.CANONICAL_OBSERVATION_ID,
                strict=True,
            )
        )
        if not aliases.empty
        else {}
    )

    def canonical_alias(value: str) -> str:
        seen: set[str] = set()
        current = value
        while current in alias_map:
            if current in seen:
                raise ValueError(f"Identity alias cycle detected at {current}")
            seen.add(current)
            current = alias_map[current]
        return current

    mapping = (
        {
            source_id: canonical_alias(obs_id)
            for source_id, obs_id in zip(
                existing.SOURCE_RECORD_ID, existing.OBSERVATION_ID, strict=True
            )
        }
        if not existing.empty
        else {}
    )
    first_seen = (
        dict(zip(existing.SOURCE_RECORD_ID, existing.FIRST_SEEN_RUN_ID, strict=True))
        if not existing.empty
        else {}
    )
    rank = {
        value: index
        for index, value in enumerate(existing.OBSERVATION_ID.drop_duplicates())
    }
    known_by_group = [
        {
            mapping[record["SOURCE_RECORD_ID"]]
            for record in group
            if record["SOURCE_RECORD_ID"] in mapping
        }
        for group in groups
    ]
    old_to_groups: dict[str, list[int]] = {}
    for group_index, known_ids in enumerate(known_by_group):
        for old_id in known_ids:
            old_to_groups.setdefault(old_id, []).append(group_index)
    split_ids = {
        old_id for old_id, indices in old_to_groups.items() if len(indices) > 1
    }
    alias_rows = aliases.to_dict("records")
    lineage_rows = lineage.to_dict("records")
    resolved: list[str] = []
    new_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        source_ids = sorted(record["SOURCE_RECORD_ID"] for record in group)
        known_candidates = sorted(
            known_by_group[group_index] - split_ids,
            key=lambda value: (rank.get(value, len(rank)), value),
        )
        split_parents = sorted(known_by_group[group_index] & split_ids)
        if split_parents:
            canonical = (
                policy.identity_prefix
                + hashlib.sha256(
                    ("split|" + "|".join(source_ids)).encode()
                ).hexdigest()[:24]
            )
            for old_id in sorted(known_by_group[group_index]):
                resolution_type = (
                    "SPLIT" if old_id in split_parents else "MERGE_INTO_SPLIT"
                )
                lineage_rows.append(
                    {
                        "OLD_OBSERVATION_ID": old_id,
                        "NEW_OBSERVATION_ID": canonical,
                        "RESOLUTION_TYPE": resolution_type,
                        "RESOLVED_RUN_ID": run_id,
                    }
                )
                if old_id not in split_parents:
                    alias_rows.append(
                        {
                            "ALIAS_OBSERVATION_ID": old_id,
                            "CANONICAL_OBSERVATION_ID": canonical,
                            "RESOLVED_RUN_ID": run_id,
                        }
                    )
        elif known_candidates:
            canonical = known_candidates[0]
            for alias in known_candidates[1:]:
                alias_rows.append(
                    {
                        "ALIAS_OBSERVATION_ID": alias,
                        "CANONICAL_OBSERVATION_ID": canonical,
                        "RESOLVED_RUN_ID": run_id,
                    }
                )
                lineage_rows.append(
                    {
                        "OLD_OBSERVATION_ID": alias,
                        "NEW_OBSERVATION_ID": canonical,
                        "RESOLUTION_TYPE": "MERGE",
                        "RESOLVED_RUN_ID": run_id,
                    }
                )
        else:
            canonical = (
                policy.identity_prefix
                + hashlib.sha256(source_ids[0].encode()).hexdigest()[:24]
            )
        resolved.append(canonical)
        new_rows.extend(
            {
                "SOURCE_RECORD_ID": source_id,
                "OBSERVATION_ID": canonical,
                "FIRST_SEEN_RUN_ID": first_seen.get(source_id, run_id),
            }
            for source_id in source_ids
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError("Identity resolution produced duplicate canonical IDs")
    replaced = {row["SOURCE_RECORD_ID"] for row in new_rows}
    identity_frame = pd.concat(
        [existing[~existing.SOURCE_RECORD_ID.isin(replaced)], pd.DataFrame(new_rows)],
        ignore_index=True,
    )
    alias_frame = pd.DataFrame(
        alias_rows, columns=IDENTITY_ALIAS_SCHEMA.names
    ).drop_duplicates("ALIAS_OBSERVATION_ID", keep="last")
    if not alias_frame.empty:
        collapsed = dict(
            zip(
                alias_frame.ALIAS_OBSERVATION_ID,
                alias_frame.CANONICAL_OBSERVATION_ID,
                strict=True,
            )
        )

        def resolve_new(value: str) -> str:
            seen: set[str] = set()
            current = value
            while current in collapsed:
                if current in seen:
                    raise ValueError(f"Identity alias cycle detected at {current}")
                seen.add(current)
                current = collapsed[current]
            return current

        alias_frame["CANONICAL_OBSERVATION_ID"] = (
            alias_frame.CANONICAL_OBSERVATION_ID.map(resolve_new)
        )
        if alias_frame.ALIAS_OBSERVATION_ID.eq(
            alias_frame.CANONICAL_OBSERVATION_ID
        ).any():
            raise ValueError("Identity alias cannot point to itself")
    lineage_frame = pd.DataFrame(
        lineage_rows, columns=IDENTITY_LINEAGE_SCHEMA.names
    ).drop_duplicates()
    return (
        resolved,
        pa.Table.from_pandas(
            identity_frame, schema=IDENTITY_SCHEMA, preserve_index=False
        ),
        pa.Table.from_pandas(
            alias_frame, schema=IDENTITY_ALIAS_SCHEMA, preserve_index=False
        ),
        pa.Table.from_pandas(
            lineage_frame, schema=IDENTITY_LINEAGE_SCHEMA, preserve_index=False
        ),
    )


def _assemble_source_state(
    snapshots: dict[str, Path],
    processed_root: Path,
    normalization_as_of: pd.Timestamp,
    *,
    expected_config_hash: str | None = None,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    list[dict[str, Any]],
    dict[str, bool],
    dict[str, int],
]:
    phase_started = perf_counter()

    def log_phase(name: str) -> None:
        nonlocal phase_started
        finished = perf_counter()
        LOGGER.info(
            "Sightings source-state phase %s completed in %.3fs",
            name,
            finished - phase_started,
        )
        phase_started = finished

    history_path = processed_root / "_state/source_history.parquet"

    latest_pointer = processed_root / "manifests/normalize/latest.json"
    snapshots_already_applied = False
    if latest_pointer.exists() and history_path.exists():
        try:
            pointer = json.loads(latest_pointer.read_text())
            latest_manifest_path = Path(str(pointer["manifest"]))
            if not latest_manifest_path.is_absolute():
                latest_manifest_path = (
                    latest_pointer.parent / latest_manifest_path
                ).resolve()
            latest_manifest = json.loads(latest_manifest_path.read_text())
            applied = {
                str(item.get("dataset_id")): Path(str(item["path"])).resolve()
                for item in latest_manifest.get("inputs", ())
                if str(item.get("dataset_id", "")).startswith("whale.sightings.source_")
            }
            expected = {
                f"whale.sightings.source_{source}": snapshot.resolve()
                for source, snapshot in snapshots.items()
            }
            compatible_workflows = {
                "whale.sightings.normalize.v8",
                f"whale.sightings.normalize.v{NORMALIZATION_VERSION}",
            }
            version_matches = expected_config_hash is None or (
                latest_manifest.get("workflow") in compatible_workflows
                and latest_manifest.get("config_hash") == expected_config_hash
            )
            snapshots_already_applied = applied == expected and version_matches
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            snapshots_already_applied = False

    if snapshots_already_applied:
        history_files = (
            sorted(history_path.rglob("*.parquet"))
            if history_path.is_dir()
            else [history_path]
        )
        history_index = pl.read_parquet(
            [str(item) for item in history_files], columns=["SOURCE_RECORD_ID"]
        )
        prior_state_path = processed_root / "_state/source_current.parquet"
        prior_state = pl.read_parquet(prior_state_path)
        audit_path = processed_root / "audit.parquet"
        updates = (
            pl.read_parquet(audit_path)
            .filter(pl.col("RULE_ID") == "source_state.upsert")
            .to_dicts()
            if audit_path.exists()
            else []
        )
        LOGGER.info(
            "Sightings source state reused for %d already-applied immutable snapshots",
            len(snapshots),
        )
        log_phase("reuse_applied_source_state")
        return (
            history_index,
            prior_state,
            updates,
            {
                "whale.sightings.source_record_history": False,
                "whale.sightings.source_records": False,
            },
            {
                "whale.sightings.source_record_history": history_index.height,
                "whale.sightings.source_records": prior_state.height,
            },
        )

    empty_history = pl.from_arrow(
        pa.Table.from_pylist([], schema=SOURCE_HISTORY_SCHEMA)
    )
    empty_state = pl.from_arrow(pa.Table.from_pylist([], schema=SOURCE_RECORD_SCHEMA))
    history_index_columns = [
        "SOURCE_RECORD_ID",
        "SOURCE",
        "RETRIEVAL_ID",
        "RETRIEVED_AT",
        "PAYLOAD_CHECKSUM",
        "LAST_CORRECTED_AT_UTC",
    ]
    history_files = (
        sorted(history_path.rglob("*.parquet"))
        if history_path.exists() and history_path.is_dir()
        else ([history_path] if history_path.exists() else [])
    )
    prior_history = (
        pl.read_parquet(
            [str(item) for item in history_files], columns=history_index_columns
        )
        if history_files
        else empty_history.select(history_index_columns)
    )
    prior_state_path = processed_root / "_state/source_current.parquet"
    prior_state = (
        pl.read_parquet(prior_state_path) if prior_state_path.exists() else empty_state
    )
    state_types = empty_state.schema
    for field in SOURCE_RECORD_SCHEMA:
        if field.name not in prior_state:
            prior_state = prior_state.with_columns(
                pl.lit(None, dtype=state_types[field.name]).alias(field.name)
            )
    if not prior_state.is_empty():
        prior_state = prior_state.with_columns(
            pl.col("SOURCE_OCCURRENCE_COUNT").cast(pl.Int32, strict=False).fill_null(1),
            pl.col("SOURCE_USE_CLASS").fill_null("INTERNAL_ONLY"),
            pl.col("SOURCE_QC_STATUS").fill_null("ACCEPTED"),
        )
    log_phase("read_prior_state")
    incoming_parts: list[pl.DataFrame] = []
    normalization_datetime = normalization_as_of.to_pydatetime()
    for source, snapshot in sorted(snapshots.items()):
        metadata = json.loads((snapshot / "snapshot.json").read_text())
        retrieval_id = str(metadata["retrieval_id"])
        retrieved_at = pd.Timestamp(metadata["retrieved_at"])
        retrieved_at = (
            retrieved_at.tz_localize("UTC")
            if retrieved_at.tzinfo is None
            else retrieved_at.tz_convert("UTC")
        )
        fingerprint = str(
            metadata.get("raw_schema_fingerprint")
            or metadata.get("schema_fingerprint")
            or "unknown"
        )
        snapshot_mode = str(metadata.get("snapshot_mode") or "FULL_REPLACE").upper()
        if snapshot_mode not in {"FULL_REPLACE", "DELTA_UPSERT"}:
            raise ValueError(f"Unsupported snapshot_mode={snapshot_mode} for {source}")
        prior_source = prior_state.filter(
            pl.col("SOURCE").str.to_uppercase() == source.upper()
        )
        if not prior_source.is_empty():
            prior_latest = prior_source.get_column("SOURCE_RETRIEVED_AT_UTC").max()
            if prior_latest is not None and retrieved_at.to_pydatetime() < prior_latest:
                raise ValueError(
                    f"Refusing out-of-order {source} snapshot {retrieval_id}: "
                    f"retrieved_at={retrieved_at.isoformat()} precedes current "
                    f"state={prior_latest}"
                )
        adapted = pl.from_arrow(adapt_snapshot(source, snapshot))
        payload_checksums = [
            hashlib.sha256(payload.encode()).hexdigest()
            for payload in adapted.get_column("SOURCE_PAYLOAD")
        ]
        incoming_parts.append(
            adapted.with_columns(
                pl.lit(retrieved_at.to_pydatetime()).alias("SOURCE_RETRIEVED_AT_UTC"),
                pl.lit(retrieval_id).alias("RETRIEVAL_ID"),
                pl.lit(retrieved_at.to_pydatetime()).alias("RETRIEVED_AT"),
                pl.Series("PAYLOAD_CHECKSUM", payload_checksums, dtype=pl.String),
                pl.lit(fingerprint).alias("RAW_SCHEMA_FINGERPRINT"),
                pl.lit(snapshot_mode).alias("SNAPSHOT_MODE"),
                pl.lit(False).alias("SOURCE_PAYLOAD_CORRECTED"),
                pl.lit(normalization_datetime).alias("LAST_CORRECTED_AT_UTC"),
            ).select(SOURCE_HISTORY_SCHEMA.names)
        )
    log_phase("adapt_snapshots")
    incoming = (
        pl.concat(incoming_parts, how="vertical_relaxed")
        if incoming_parts
        else empty_history.clone()
    )
    prior_last_checksum = (
        prior_history.sort(["SOURCE_RECORD_ID", "RETRIEVED_AT", "RETRIEVAL_ID"])
        .group_by("SOURCE_RECORD_ID", maintain_order=True)
        .agg(pl.col("PAYLOAD_CHECKSUM").last().alias("_PRIOR_PAYLOAD_CHECKSUM"))
    )
    incoming_ordered = (
        incoming.with_row_index("_INCOMING_ORDER")
        .sort(
            ["SOURCE_RECORD_ID", "RETRIEVED_AT", "RETRIEVAL_ID", "_INCOMING_ORDER"],
            maintain_order=True,
        )
        .join(
            prior_last_checksum,
            on="SOURCE_RECORD_ID",
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.col("PAYLOAD_CHECKSUM")
            .shift(1)
            .over("SOURCE_RECORD_ID")
            .alias("_WITHIN_SNAPSHOT_PREVIOUS_CHECKSUM")
        )
        .with_columns(
            pl.coalesce(
                "_WITHIN_SNAPSHOT_PREVIOUS_CHECKSUM", "_PRIOR_PAYLOAD_CHECKSUM"
            ).alias("_PREVIOUS_PAYLOAD_CHECKSUM")
        )
    )
    history_delta = (
        incoming_ordered.filter(
            pl.col("_PREVIOUS_PAYLOAD_CHECKSUM").is_null()
            | (pl.col("PAYLOAD_CHECKSUM") != pl.col("_PREVIOUS_PAYLOAD_CHECKSUM"))
        )
        .with_columns(
            pl.col("_PREVIOUS_PAYLOAD_CHECKSUM")
            .is_not_null()
            .alias("SOURCE_PAYLOAD_CORRECTED"),
            pl.when(pl.col("_PREVIOUS_PAYLOAD_CHECKSUM").is_not_null())
            .then(pl.col("RETRIEVED_AT"))
            .otherwise(pl.col("LAST_CORRECTED_AT_UTC"))
            .alias("LAST_CORRECTED_AT_UTC"),
        )
        .select(SOURCE_HISTORY_SCHEMA.names)
        .join(
            prior_history.select(
                "SOURCE_RECORD_ID", "RETRIEVAL_ID", "PAYLOAD_CHECKSUM"
            ),
            on=["SOURCE_RECORD_ID", "RETRIEVAL_ID", "PAYLOAD_CHECKSUM"],
            how="anti",
        )
    )
    history_index = pl.concat(
        [prior_history, history_delta.select(history_index_columns)],
        how="vertical_relaxed",
    )
    log_phase("deduplicate_history")
    ordered_history = history_index.sort(
        ["SOURCE_RECORD_ID", "RETRIEVED_AT", "RETRIEVAL_ID"]
    )
    correction_state = (
        ordered_history.with_columns(
            (
                pl.col("PAYLOAD_CHECKSUM")
                != pl.col("PAYLOAD_CHECKSUM").shift(1).over("SOURCE_RECORD_ID")
            )
            .fill_null(False)
            .alias("_PAYLOAD_CHANGED")
        )
        .group_by("SOURCE_RECORD_ID", maintain_order=True)
        .agg(
            pl.col("_PAYLOAD_CHANGED").any().alias("SOURCE_PAYLOAD_CORRECTED"),
            pl.col("RETRIEVED_AT")
            .filter(pl.col("_PAYLOAD_CHANGED"))
            .max()
            .alias("_LAST_CHANGE_AT_UTC"),
            pl.col("LAST_CORRECTED_AT_UTC").min().alias("_FIRST_NORMALIZED_AT_UTC"),
        )
        .with_columns(
            pl.when(pl.col("SOURCE_PAYLOAD_CORRECTED"))
            .then(pl.col("_LAST_CHANGE_AT_UTC"))
            .otherwise(
                pl.col("_FIRST_NORMALIZED_AT_UTC").fill_null(normalization_datetime)
            )
            .alias("LAST_CORRECTED_AT_UTC")
        )
        .drop("_LAST_CHANGE_AT_UTC", "_FIRST_NORMALIZED_AT_UTC")
    )
    log_phase("derive_correction_state")
    current_parts: list[pl.DataFrame] = []
    source_names = sorted(
        set(prior_state.get_column("SOURCE").drop_nulls().to_list())
        | set(incoming.get_column("SOURCE").drop_nulls().to_list())
    )
    for source in source_names:
        incoming_source = incoming.filter(pl.col("SOURCE") == source)
        if incoming_source.is_empty():
            current_parts.append(prior_state.filter(pl.col("SOURCE") == source))
            continue
        latest_retrieval = incoming_source.sort(["RETRIEVED_AT", "RETRIEVAL_ID"]).row(
            -1, named=True
        )
        mode = str(latest_retrieval["SNAPSHOT_MODE"])
        latest_id = str(latest_retrieval["RETRIEVAL_ID"])
        prior_source = prior_state.filter(pl.col("SOURCE") == source)
        if mode == "DELTA_UPSERT" and prior_source.is_empty():
            raise ValueError(
                f"Cannot initialize {source} state from DELTA_UPSERT snapshot; run a full refresh"
            )
        if mode == "FULL_REPLACE":
            candidates = incoming_source.filter(pl.col("RETRIEVAL_ID") == latest_id)
        else:
            candidates = pl.concat(
                [prior_source, incoming_source.select(SOURCE_RECORD_SCHEMA.names)],
                how="vertical_relaxed",
            )
        selected = (
            candidates.with_columns(
                (
                    ~pl.col("SOURCE_PAYLOAD").str.contains(
                        '"_source_file":', literal=True
                    )
                )
                .cast(pl.Int8)
                .alias("_STATE_PRIORITY")
            )
            .sort(["SOURCE_RETRIEVED_AT_UTC", "_STATE_PRIORITY", "SOURCE_RECORD_ID"])
            .unique(subset="SOURCE_RECORD_ID", keep="last", maintain_order=True)
            .select(SOURCE_RECORD_SCHEMA.names)
        )
        if not prior_source.is_empty() and not selected.is_empty():
            # Keep first-seen availability stable when the raw payload is
            # unchanged, but do not retain the whole prior row: source policy
            # and QC contracts can legitimately change independently of the
            # provider payload and must propagate into current state.
            selected = (
                selected.join(
                    prior_source.select(
                        "SOURCE_RECORD_ID",
                        pl.col("SOURCE_PAYLOAD").alias("_PRIOR_SOURCE_PAYLOAD"),
                        pl.col("SOURCE_RETRIEVED_AT_UTC").alias(
                            "_PRIOR_SOURCE_RETRIEVED_AT_UTC"
                        ),
                    ),
                    on="SOURCE_RECORD_ID",
                    how="left",
                    maintain_order="left",
                )
                .with_columns(
                    pl.when(pl.col("SOURCE_PAYLOAD") == pl.col("_PRIOR_SOURCE_PAYLOAD"))
                    .then(pl.col("_PRIOR_SOURCE_RETRIEVED_AT_UTC"))
                    .otherwise(pl.col("SOURCE_RETRIEVED_AT_UTC"))
                    .alias("SOURCE_RETRIEVED_AT_UTC")
                )
                .drop("_PRIOR_SOURCE_PAYLOAD", "_PRIOR_SOURCE_RETRIEVED_AT_UTC")
                .select(SOURCE_RECORD_SCHEMA.names)
            )
        current_parts.append(selected)
    current = (
        pl.concat(current_parts, how="vertical_relaxed")
        if current_parts
        else empty_state.clone()
    )
    current = current.drop("SOURCE_PAYLOAD_CORRECTED", "LAST_CORRECTED_AT_UTC").join(
        correction_state, on="SOURCE_RECORD_ID", how="left", maintain_order="left"
    )
    log_phase("assemble_current_state")
    updates = []
    incoming_updates = (
        incoming.group_by("SOURCE_RECORD_ID")
        .agg(
            pl.len().alias("_ROW_COUNT"),
            pl.col("PAYLOAD_CHECKSUM").n_unique().alias("_PAYLOAD_VERSIONS"),
            pl.col("RETRIEVAL_ID").unique().sort().alias("_RETRIEVAL_IDS"),
        )
        .filter(pl.col("_ROW_COUNT") >= 2)
        .sort("SOURCE_RECORD_ID")
    )
    for update in incoming_updates.iter_rows(named=True):
        changed = int(update["_PAYLOAD_VERSIONS"]) > 1
        updates.append(
            _audit(
                str(update["SOURCE_RECORD_ID"]),
                None,
                "UPDATED" if changed else "NO_OP",
                "SOURCE_RECORD_UPDATE" if changed else "DUPLICATE_SOURCE_PAYLOAD",
                json.dumps(
                    {
                        "retrieval_ids": update["_RETRIEVAL_IDS"],
                        "payload_versions": int(update["_PAYLOAD_VERSIONS"]),
                    }
                ),
                "source_state.upsert",
            )
        )
    if not prior_state.is_empty() and not incoming.is_empty():
        changed_current = (
            current.select("SOURCE_RECORD_ID", "SOURCE_PAYLOAD")
            .join(
                prior_state.select(
                    "SOURCE_RECORD_ID",
                    pl.col("SOURCE_PAYLOAD").alias("_PRIOR_SOURCE_PAYLOAD"),
                ),
                on="SOURCE_RECORD_ID",
                how="inner",
                maintain_order="left",
            )
            .filter(pl.col("SOURCE_PAYLOAD") != pl.col("_PRIOR_SOURCE_PAYLOAD"))
        )
        updates.extend(
            _audit(
                str(source_id),
                None,
                "UPDATED",
                "SOURCE_RECORD_UPDATE",
                None,
                "source_state.upsert",
            )
            for source_id in changed_current.get_column("SOURCE_RECORD_ID")
        )
    log_phase("derive_update_audit")
    current = current.select(SOURCE_RECORD_SCHEMA.names)
    state_changed = {
        "whale.sightings.source_record_history": not history_delta.is_empty(),
        "whale.sightings.source_records": not (
            prior_state.height == current.height
            and prior_state.columns == current.columns
            and prior_state.hash_rows(19, 37, 59, 83).sum()
            == current.hash_rows(19, 37, 59, 83).sum()
            and prior_state.hash_rows(23, 41, 61, 89).sum()
            == current.hash_rows(23, 41, 61, 89).sum()
        ),
    }
    log_phase("finalize_frames")
    return (
        history_delta if not history_delta.is_empty() else prior_history,
        current,
        updates,
        state_changed,
        {
            "whale.sightings.source_record_history": history_index.height,
            "whale.sightings.source_records": current.height,
        },
    )


def _add_legacy_v3_lineage(
    groups: list[list[dict[str, Any]]],
    resolved_ids: list[str],
    processed_root: Path,
    lineage_table: pa.Table,
    run_id: str,
) -> pa.Table:
    candidates = sorted(
        (processed_root / "legacy/v3").glob("*/identity_resolution.parquet")
    )
    if not candidates:
        return lineage_table
    legacy = pd.read_parquet(candidates[-1])
    mapping = dict(zip(legacy.SOURCE_RECORD_ID, legacy.OBSERVATION_ID, strict=True))
    old_to_new: dict[str, set[str]] = {}
    for group, new_id in zip(groups, resolved_ids, strict=True):
        for record in group:
            old_id = mapping.get(record["SOURCE_RECORD_ID"])
            if old_id:
                old_to_new.setdefault(old_id, set()).add(new_id)
    rows = lineage_table.to_pandas().to_dict("records")
    for old_id, new_ids in old_to_new.items():
        resolution_type = "SPLIT" if len(new_ids) > 1 else "MIGRATED"
        rows.extend(
            {
                "OLD_OBSERVATION_ID": old_id,
                "NEW_OBSERVATION_ID": new_id,
                "RESOLUTION_TYPE": resolution_type,
                "RESOLVED_RUN_ID": run_id,
            }
            for new_id in sorted(new_ids)
        )
    return pa.Table.from_pylist(rows, schema=IDENTITY_LINEAGE_SCHEMA)


def _spec(dataset_id: str, schema: pa.Schema):
    return replace(DATASETS.get(dataset_id), schema=schema, schema_version="9")


def _normalization_report_metrics(
    source_frame: pd.DataFrame | pl.DataFrame,
    records: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
    observations: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    source_values = (
        source_frame.get_column("SOURCE").to_list()
        if isinstance(source_frame, pl.DataFrame)
        else source_frame["SOURCE"].astype(str).tolist()
    )
    source_event_counts = Counter(str(value) for value in source_values)
    admitted_event_counts = Counter(str(item["SOURCE"]) for item in records)
    admitted_occurrences = Counter()
    for item in records:
        admitted_occurrences[str(item["SOURCE"])] += int(
            item.get("SOURCE_OCCURRENCE_COUNT") or 1
        )
    qc_exclusions: Counter[str] = Counter()
    gbif_quarantined_occurrences = 0
    if isinstance(source_frame, pl.DataFrame):
        gbif_rows = source_frame.filter(
            pl.col("SOURCE").str.to_uppercase() == "GBIF"
        ).iter_rows(named=True)
    else:
        gbif_source = source_frame.loc[
            source_frame["SOURCE"].astype(str).str.upper().eq("GBIF")
        ]
        gbif_rows = gbif_source.to_dict("records")
    for row in gbif_rows:
        if str(row["SOURCE_QC_STATUS"]).upper() == "ACCEPTED":
            continue
        occurrence_count = row["SOURCE_OCCURRENCE_COUNT"]
        gbif_quarantined_occurrences += (
            int(occurrence_count) if pd.notna(occurrence_count) else 1
        )
        try:
            detail = json.loads(str(row["SOURCE_QC_DETAIL"]))
        except (TypeError, json.JSONDecodeError):
            detail = {}
        qc_exclusions.update(str(reason) for reason in detail.get("reasons", []))
    cross_source_groups = [
        group for group in groups if len({str(item["SOURCE"]) for item in group}) > 1
    ]
    gbif_cross_source_groups = [
        group
        for group in cross_source_groups
        if any(str(item["SOURCE"]).upper() == "GBIF" for item in group)
    ]
    audit_reasons = Counter(str(item["REASON"]) for item in audit)
    return {
        "source_event_counts": dict(sorted(source_event_counts.items())),
        "admitted_source_event_counts": dict(sorted(admitted_event_counts.items())),
        "admitted_source_occurrence_counts": dict(sorted(admitted_occurrences.items())),
        "gbif_quarantined_occurrence_count": gbif_quarantined_occurrences,
        "gbif_exclusions_by_reason": dict(sorted(qc_exclusions.items())),
        "canonical_observations_by_preferred_source": dict(
            sorted(Counter(str(item["SOURCE"]) for item in observations).items())
        ),
        "observed_ecotype_distribution": dict(
            sorted(
                Counter(str(item["ECOTYPE_DETAIL"]) for item in observations).items()
            )
        ),
        "public_release_eligibility": dict(
            sorted(
                Counter(
                    "ELIGIBLE" if item["PUBLIC_RELEASE_ELIGIBLE"] else "INTERNAL_ONLY"
                    for item in observations
                ).items()
            )
        ),
        "cross_source_merge_count": len(cross_source_groups),
        "gbif_cross_source_merge_count": len(gbif_cross_source_groups),
        "suppressed_source_event_count": sum(len(group) - 1 for group in groups),
        "audit_reason_counts": dict(sorted(audit_reasons.items())),
    }


def normalize_sightings(request: NormalizationRequest) -> StageResult:
    phase_started = perf_counter()

    def log_stage_phase(name: str) -> None:
        nonlocal phase_started
        finished = perf_counter()
        LOGGER.info(
            "Sightings normalization phase %s completed in %.3fs",
            name,
            finished - phase_started,
        )
        phase_started = finished

    document, config = load_sightings_config(request.config)
    processed_root = request.data_root / "processed/domain/whale_layer/sightings"
    state_paths = [
        processed_root / "_state/source_history.parquet",
        processed_root / "_state/source_current.parquet",
        processed_root / "_state/identity/assignments.parquet",
        processed_root / "_state/identity/aliases.parquet",
        processed_root / "_state/identity/lineage.parquet",
    ]
    signature, signature_payload = stage_signature(
        stage="whale.sightings.normalize",
        semantic_version="9",
        config_hash=document.config_hash,
        inputs=request.inputs,
        parameters={
            "state_checksums": {
                path.name: checksum_path(path) if path.exists() else None
                for path in state_paths
            },
            "normalization_version": NORMALIZATION_VERSION,
        },
    )
    manifest_path = processed_root / f"manifests/normalize/{signature}.json"
    resumed = resume_result(
        enabled=request.resume,
        manifest_path=manifest_path,
        config_hash=document.config_hash,
        inputs=request.inputs,
        signature=signature,
    )
    if resumed is not None:
        return resumed
    log_stage_phase("prepare")
    snapshots = {
        (item.dataset_id or "").removeprefix("whale.sightings.source_"): item.path
        for item in request.inputs
        if (item.dataset_id or "").startswith("whale.sightings.source_")
    }
    enabled = {
        name for name, settings in config.collection.sources.items() if settings.enabled
    }
    if set(snapshots) != enabled:
        raise ValueError(
            f"Normalization sources mismatch: expected={sorted(enabled)}, received={sorted(snapshots)}"
        )
    normalization_as_of = pd.Timestamp.now(tz="UTC")
    data_snapshot = build_data_snapshot(
        request.inputs,
        default_coverage_start=config.min_date,
        duplicate_resolution_as_of=normalization_as_of.isoformat(),
    )
    log_stage_phase("data_snapshot")
    history_table, source_table, audit, state_changed, state_row_counts = (
        _assemble_source_state(
            snapshots,
            processed_root,
            normalization_as_of,
            expected_config_hash=document.config_hash,
        )
    )
    log_stage_phase("source_state")
    source_frame = source_table
    records, normalization_audit = _normalize_records(source_frame, config)
    audit.extend(normalization_audit)
    log_stage_phase("normalize_records")
    groups = _cluster(records, config, audit)
    log_stage_phase("cluster")
    observation_ids, identity_table, alias_table, lineage_table = _resolve_identities(
        groups,
        processed_root / "_state/identity/assignments.parquet",
        processed_root / "_state/identity/aliases.parquet",
        processed_root / "_state/identity/lineage.parquet",
        request.run_id,
        policy=config.observation_policy,
    )
    log_stage_phase("resolve_identities")
    lineage_table = _add_legacy_v3_lineage(
        groups, observation_ids, processed_root, lineage_table, request.run_id
    )
    observations, associations = _materialize(
        groups, audit, observation_ids, policy=config.observation_policy
    )
    log_stage_phase("materialize")
    report_metrics = _normalization_report_metrics(
        source_frame, records, groups, observations, audit
    )
    log_stage_phase("report_metrics")
    tables = (
        ("whale.sightings.source_record_history", history_table, SOURCE_HISTORY_SCHEMA),
        ("whale.sightings.source_records", source_table, SOURCE_RECORD_SCHEMA),
        (
            "whale.sightings.observations",
            pa.Table.from_pylist(observations, schema=OBSERVATION_SCHEMA),
            OBSERVATION_SCHEMA,
        ),
        (
            "whale.sightings.associations",
            pa.Table.from_pylist(associations, schema=ASSOCIATION_SCHEMA),
            ASSOCIATION_SCHEMA,
        ),
        (
            "whale.sightings.normalization_audit",
            pa.Table.from_pylist(audit, schema=AUDIT_SCHEMA),
            AUDIT_SCHEMA,
        ),
        ("whale.sightings.identity_resolution", identity_table, IDENTITY_SCHEMA),
        ("whale.sightings.identity_aliases", alias_table, IDENTITY_ALIAS_SCHEMA),
        ("whale.sightings.identity_lineage", lineage_table, IDENTITY_LINEAGE_SCHEMA),
    )
    log_stage_phase("construct_tables")
    store = ArtifactStore(
        data_root=request.data_root,
        artifact_root=request.artifact_root,
        output_root=request.output_root,
    )
    outputs, reports = [], []
    for dataset_id, table, schema in tables:
        write_started = perf_counter()
        spec = _spec(dataset_id, schema)
        if (
            dataset_id == "whale.sightings.source_record_history"
            and state_changed[dataset_id]
        ):
            artifact, report = store.append_table(
                table,
                spec,
                run_id=request.run_id,
                producer="whale.sightings.normalize.v9",
                config_hash=document.config_hash,
                row_count=state_row_counts[dataset_id],
                inputs=request.inputs,
                data_snapshot=data_snapshot,
            )
            LOGGER.info(
                "Sightings artifact %s appended %d payload transitions",
                dataset_id,
                table.height,
            )
        elif (
            dataset_id in state_changed
            and not state_changed[dataset_id]
            and spec.path(
                data_root=store.data_root,
                artifact_root=store.artifact_root,
                output_root=store.output_root,
            ).exists()
        ):
            artifact, report = store.reference_existing(
                spec,
                run_id=request.run_id,
                producer="whale.sightings.normalize.v9",
                config_hash=document.config_hash,
                row_count=state_row_counts[dataset_id],
                inputs=request.inputs,
                data_snapshot=data_snapshot,
            )
            LOGGER.info("Sightings artifact %s reused without rewrite", dataset_id)
        else:
            artifact, report = store.write_table(
                table,
                spec,
                run_id=request.run_id,
                producer="whale.sightings.normalize.v9",
                config_hash=document.config_hash,
                inputs=request.inputs,
                data_snapshot=data_snapshot,
                force=request.force,
            )
        outputs.append(artifact)
        reports.append(report)
        LOGGER.info(
            "Sightings artifact %s persisted and validated in %.3fs",
            dataset_id,
            perf_counter() - write_started,
        )
        phase_started = perf_counter()
    conflict_count = sum(item["REASON"] == "DATE_TIMESTAMP_CONFLICT" for item in audit)
    manifest = RunManifest(
        run_id=request.run_id,
        workflow="whale.sightings.normalize.v9",
        config_hash=document.config_hash,
        resolved_config=document.redacted_data(),
        inputs=request.inputs,
        outputs=tuple(outputs),
        source_snapshots=request.inputs,
        data_snapshot=data_snapshot,
        schema_version="9",
        code_revision=code_revision(),
        stage_signature=signature,
        stages=(
            {
                "name": "normalize",
                "semantic_version": "9",
                "signature": signature_payload,
                "accepted": len(observations),
                "audit_rows": len(audit),
                "date_timestamp_conflicts": conflict_count,
                **report_metrics,
            },
        ),
    )
    manifest.write(manifest_path, overwrite=request.force)
    latest_pointer = processed_root / "manifests/normalize/latest.json"
    atomic_write_json(
        latest_pointer,
        {
            "manifest": manifest_path.relative_to(latest_pointer.parent).as_posix(),
            "stage_signature": signature,
        },
        overwrite=True,
    )
    return StageResult(tuple(outputs), tuple(reports), manifest)
