"""Source-record identity and scalar normalization shared by adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def source_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text.lower() in {"nan", "na", "n/a", "none", "null"}:
        return None
    return text or None


def source_record(
    source: str, native_id: Any, payload: dict[str, Any], **values: Any
) -> dict:
    native = source_text(native_id)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    record_id = (
        f"{source}:{native}"
        if native
        else f"{source}:sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:24]}"
    )
    return {
        "SOURCE_RECORD_ID": record_id,
        "SOURCE": source,
        "SOURCE_NATIVE_ID": native,
        "OBSERVED_AT_RAW": source_text(values.get("observed_at")),
        "OBSERVED_DATE_RAW": source_text(values.get("observed_date")),
        "CREATED_AT_RAW": source_text(values.get("created_at")),
        "LATITUDE_RAW": source_text(values.get("latitude")),
        "LONGITUDE_RAW": source_text(values.get("longitude")),
        "SPECIES_RAW": source_text(values.get("species")),
        "DESCRIPTION_RAW": source_text(values.get("description")),
        "POD_ECOTYPE_RAW": source_text(values.get("pod_ecotype")),
        "SOURCE_DATASET_ID": source_text(values.get("source_dataset_id")),
        "SOURCE_EVENT_ID": source_text(values.get("source_event_id")),
        "SOURCE_OCCURRENCE_IDS": values.get("source_occurrence_ids"),
        "SOURCE_OCCURRENCE_COUNT": int(values.get("source_occurrence_count") or 1),
        "SOURCE_LICENSE": source_text(values.get("source_license")),
        "SOURCE_USE_CLASS": source_text(values.get("source_use_class"))
        or "INTERNAL_ONLY",
        "COORDINATE_UNCERTAINTY_M": values.get("coordinate_uncertainty_m"),
        "SOURCE_QC_STATUS": source_text(values.get("source_qc_status")) or "ACCEPTED",
        "SOURCE_QC_DETAIL": source_text(values.get("source_qc_detail")),
        "SOURCE_PAYLOAD": canonical,
    }
