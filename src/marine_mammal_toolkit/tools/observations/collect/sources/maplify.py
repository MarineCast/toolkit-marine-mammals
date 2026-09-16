"""Provider-specific acquisition with explicit settings and injectable HTTP."""

from __future__ import annotations
import os
from collections import Counter
from datetime import date
from typing import Any
from ..http import _request_json, _require_unique_ids, _require_unique_composite_ids


def fetch_maplify(
    source, start: date, end: date, *, bbox, request_json=None
) -> list[dict[str, Any]]:
    """Fetch one complete WASEAK date window from the Maplify stream."""
    bbox = source.bbox or bbox
    payload = (request_json or _request_json)(
        str(source.url),
        params={
            "start": start.isoformat(),
            "end": end.isoformat(),
            "BBOX": ",".join(str(value) for value in bbox.tuple()),
        },
        token=None,
        timeout=source.timeout_seconds,
        retries=source.max_retries,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Unexpected Maplify response shape")
    rows = payload["results"]
    declared_count = payload.get("count")
    if declared_count is None:
        raise ValueError("Maplify response lacks count completeness metadata")
    try:
        count = int(declared_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Maplify count: {declared_count!r}") from exc
    if count != len(rows):
        raise ValueError(
            f"Incomplete Maplify response: declared count={count}, rows={len(rows)}"
        )
    # Maplify IDs are scoped to the contributing feed.  Mirror feeds can use
    # the same numeric ID for different records, so validating the numeric ID
    # alone incorrectly rejects a complete response.
    _require_unique_composite_ids(rows, "Maplify", ("source", "id"))
    return rows
