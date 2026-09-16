"""HTTP retry and complete-response validation utilities."""

from __future__ import annotations
import logging, random, time
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
import requests

LOGGER = logging.getLogger(__name__)


class _TransientHttpError(RuntimeError):
    def __init__(self, status_code: int, retry_after: float | None = None):
        super().__init__(f"Transient HTTP status {status_code}")
        self.retry_after = retry_after


def _retry_after_seconds(
    value: str | None, *, now: datetime | None = None
) -> float | None:
    """Parse Retry-After seconds or an RFC-compliant HTTP date."""

    if value is None or not value.strip():
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - current.astimezone(timezone.utc)).total_seconds())


def _require_unique_ids(
    rows: list[dict[str, Any]], source: str, id_fields: tuple[str, ...]
) -> None:
    identifiers: list[str] = []
    for row in rows:
        value = next(
            (
                str(row[field]).strip()
                for field in id_fields
                if row.get(field) is not None
            ),
            "",
        )
        if not value:
            raise ValueError(f"{source} response contains a row without a source ID")
        identifiers.append(value)
    duplicate_ids = [
        value for value, count in Counter(identifiers).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(
            f"{source} response contains duplicate source IDs: {duplicate_ids[:10]}"
        )


def _require_unique_composite_ids(
    rows: list[dict[str, Any]], source: str, id_fields: tuple[str, ...]
) -> None:
    """Validate an upstream identity whose components are jointly scoped."""

    identifiers: list[tuple[str, ...]] = []
    for row in rows:
        value = tuple(str(row.get(field) or "").strip().lower() for field in id_fields)
        if any(not component for component in value):
            raise ValueError(
                f"{source} response contains a row without composite source ID fields "
                f"{id_fields}"
            )
        identifiers.append(value)
    duplicate_ids = [
        value for value, count in Counter(identifiers).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(
            f"{source} response contains duplicate composite source IDs: "
            f"{duplicate_ids[:10]}"
        )


def _request_json(
    url: str,
    *,
    params: dict[str, Any] | list[tuple[str, Any]],
    token: str | None,
    timeout: int,
    retries: int,
    transport=None,
) -> Any:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = (transport or requests.get)(
                url, params=params, headers=headers, timeout=timeout
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_header = response.headers.get("Retry-After")
                retry_after = _retry_after_seconds(retry_header)
                raise _TransientHttpError(response.status_code, retry_after)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError, _TransientHttpError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            retry_after = (
                exc.retry_after if isinstance(exc, _TransientHttpError) else None
            )
            delay = (
                retry_after
                if retry_after is not None
                else (0.5 * (2**attempt) + random.uniform(0, 0.25))
            )
            LOGGER.warning(
                "Transient collection failure url=%s attempt=%d/%d retry_in=%.2fs error=%s",
                url,
                attempt + 1,
                retries + 1,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error
