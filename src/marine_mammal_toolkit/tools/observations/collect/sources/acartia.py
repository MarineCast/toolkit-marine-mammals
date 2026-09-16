"""Provider-specific acquisition with explicit settings and injectable HTTP."""

from __future__ import annotations
import os
from collections import Counter
from datetime import date
from typing import Any
from ..http import _request_json, _require_unique_ids, _require_unique_composite_ids


def fetch_acartia(source, *, request_json=None) -> list[dict[str, Any]]:
    token = os.getenv(source.credential_env) if source.credential_env else None
    payload = (request_json or _request_json)(
        str(source.url),
        params={"access_token": token} if token else {},
        token=None,
        timeout=source.timeout_seconds,
        retries=source.max_retries,
    )
    if isinstance(payload, list):
        return payload
    for key in ("list", "sightings", "data", "results"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("Unexpected Acartia response shape")
