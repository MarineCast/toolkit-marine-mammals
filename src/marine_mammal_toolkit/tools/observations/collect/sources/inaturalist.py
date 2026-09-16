"""Provider-specific acquisition with explicit settings and injectable HTTP."""

from __future__ import annotations
import os
from collections import Counter
from datetime import date
from typing import Any
from ..http import _request_json, _require_unique_ids, _require_unique_composite_ids


def fetch_inaturalist(
    source, start: date, end: date, *, bbox, taxon_id, request_json=None
) -> list[dict[str, Any]]:
    token = os.getenv(source.credential_env) if source.credential_env else None
    rows: list[dict[str, Any]] = []
    page = 1
    declared_total: int | None = None
    while True:
        payload = (request_json or _request_json)(
            str(source.url),
            params={
                "taxon_id": taxon_id,
                "d1": start.isoformat(),
                "d2": end.isoformat(),
                "swlat": bbox.min_lat,
                "swlng": bbox.min_lon,
                "nelat": bbox.max_lat,
                "nelng": bbox.max_lon,
                # ID ordering gives pagination a deterministic, unique sort key.
                "order_by": "id",
                "order": "asc",
                "per_page": 200,
                "page": page,
            },
            token=token,
            timeout=source.timeout_seconds,
            retries=source.max_retries,
        )
        results = payload.get("results", []) if isinstance(payload, dict) else []
        total = payload.get("total_results") if isinstance(payload, dict) else None
        if total is None:
            raise ValueError(
                "iNaturalist response lacks total_results completeness metadata"
            )
        current_total = int(total)
        if declared_total is None:
            declared_total = current_total
        elif current_total != declared_total:
            raise ValueError("iNaturalist total_results changed during pagination")
        rows.extend(results)
        if len(results) < 200:
            break
        page += 1
    if declared_total is None or len(rows) != declared_total:
        raise ValueError(
            f"Incomplete iNaturalist response: declared={declared_total}, rows={len(rows)}"
        )
    _require_unique_ids(rows, "iNaturalist", ("id",))
    return rows
