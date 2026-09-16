"""Provider-specific acquisition with explicit settings and injectable HTTP."""

from __future__ import annotations
import os
from collections import Counter
from datetime import date
from typing import Any
from ..http import _request_json, _require_unique_ids, _require_unique_composite_ids


def _gbif_geometry(bbox) -> str:
    return (
        "POLYGON(("
        f"{bbox.min_lon} {bbox.min_lat},"
        f"{bbox.max_lon} {bbox.min_lat},"
        f"{bbox.max_lon} {bbox.max_lat},"
        f"{bbox.min_lon} {bbox.max_lat},"
        f"{bbox.min_lon} {bbox.min_lat}"
        "))"
    )


def fetch_gbif(
    source, start: date, end: date, *, bbox, request_json=None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies = {item.key: item for item in source.dataset_allowlist}
    allowed_keys = sorted(policies)
    if set(allowed_keys) & set(source.excluded_dataset_keys):
        raise ValueError("GBIF allowlist intersects the configured exclusion list")
    common_params: dict[str, Any] = {
        "taxonKey": source.taxon_key,
        "checklistKey": source.checklist_key,
        "basisOfRecord": source.basis_of_record,
        "occurrenceStatus": source.occurrence_status,
        "hasCoordinate": "true",
        "geometry": _gbif_geometry(source.bbox or bbox),
        "eventDate": f"{start.isoformat()},{end.isoformat()}",
        "datasetKey": allowed_keys,
        "limit": source.page_size,
    }
    if source.require_no_geospatial_issue:
        common_params["hasGeospatialIssue"] = "false"

    rows: list[dict[str, Any]] = []
    offset = 0
    declared_total: int | None = None
    while True:
        payload = (request_json or _request_json)(
            str(source.url),
            params={**common_params, "offset": offset},
            token=None,
            timeout=source.timeout_seconds,
            retries=source.max_retries,
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("results"), list
        ):
            raise ValueError("Unexpected GBIF occurrence response shape")
        if payload.get("count") is None:
            raise ValueError("GBIF response lacks count completeness metadata")
        current_total = int(payload["count"])
        if current_total > source.max_search_results:
            raise ValueError(
                f"GBIF query exceeds search limit: {current_total}>{source.max_search_results}"
            )
        if declared_total is None:
            declared_total = current_total
        elif current_total != declared_total:
            raise ValueError("GBIF count changed during pagination")
        page_rows = payload["results"]
        if len(page_rows) > source.page_size:
            raise ValueError(
                f"GBIF returned {len(page_rows)} rows for page size {source.page_size}"
            )
        rows.extend(page_rows)
        offset += len(page_rows)
        if bool(payload.get("endOfRecords")) or not page_rows:
            break
        if offset >= source.max_search_results:
            raise ValueError("GBIF pagination reached the hard search offset limit")
    if declared_total is None or len(rows) != declared_total:
        raise ValueError(
            f"Incomplete GBIF response: declared={declared_total}, rows={len(rows)}"
        )
    gbif_ids = [str(item.get("gbifID") or item.get("key") or "") for item in rows]
    if any(not value for value in gbif_ids):
        raise ValueError("GBIF occurrence response contains a row without gbifID")
    duplicate_ids = [value for value, count in Counter(gbif_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(
            f"GBIF pagination returned duplicate gbifID values: {duplicate_ids[:10]}"
        )
    returned_datasets = {str(item.get("datasetKey") or "") for item in rows}
    unexpected = returned_datasets - set(allowed_keys)
    if unexpected:
        raise ValueError(
            f"GBIF returned non-allowlisted datasets: {sorted(unexpected)}"
        )

    dataset_metadata: list[dict[str, Any]] = []
    for key in allowed_keys:
        metadata = (request_json or _request_json)(
            f"{str(source.dataset_metadata_url).rstrip('/')}/{key}",
            params={},
            token=None,
            timeout=source.timeout_seconds,
            retries=source.max_retries,
        )
        if not isinstance(metadata, dict) or str(metadata.get("key")) != key:
            raise ValueError(f"Unexpected GBIF dataset metadata for {key}")
        policy = policies[key]
        metadata = {
            **metadata,
            "_orcacast_policy": {
                "title": policy.title,
                "event_id_is_encounter": policy.event_id_is_encounter,
                "happywhale_exact_fallback": policy.happywhale_exact_fallback,
                "use_class": policy.use_class,
                "license": metadata.get("license"),
            },
        }
        dataset_metadata.append(metadata)
    return rows, dataset_metadata
