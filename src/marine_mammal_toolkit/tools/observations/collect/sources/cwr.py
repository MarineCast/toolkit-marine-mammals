from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
import logging
import random
import re
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from marine_mammal_toolkit.tools.schemas.sources import BBox
from marine_mammal_toolkit.tools.schemas.sources import SourceSettings
from marine_mammal_toolkit.tools.observations.process.records import (
    source_record as _source_record,
)

LOGGER = logging.getLogger(__name__)
USER_AGENT = "OrcaCast-CWR-collector/1.0 (bounded research request)"

CWR_SEVERE_QC_FLAGS = {
    "ARCHIVE_INDEX_PAGE_IDENTITY_MISMATCH",
    "ARCHIVE_INDEX_PAGE_DATE_MISMATCH",
    "ARCHIVE_DATE_YEAR_MISMATCH",
    "ATLIST_MAP_YEAR_DATE_MISMATCH",
    "MISSING_ARCHIVE_DATE",
    "MISSING_ATLIST_DATE",
    "UNPARSED_ARCHIVE_DATE",
    "UNPARSED_ATLIST_DATE",
    "SEQUENCE_DATE_DISAGREEMENT",
    "MISSING_OR_INVALID_ATLIST_COORDINATE",
    "ARCHIVE_START_COORDINATE_INVALID_OR_OUTSIDE_CWR_BOUNDS",
    "ARCHIVE_END_COORDINATE_INVALID_OR_OUTSIDE_CWR_BOUNDS",
}


class _TransientCwrHttpError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


_WIX_REQUEST_LOCK = threading.Lock()
_WIX_NEXT_REQUEST_AT = 0.0
_WIX_BLOCKED_UNTIL = 0.0
_WIX_MIN_REQUEST_INTERVAL_SECONDS = 0.75


def _wait_for_wix_request_slot(url: str) -> None:
    if "wixsite.com" not in url:
        return
    global _WIX_NEXT_REQUEST_AT
    with _WIX_REQUEST_LOCK:
        now = time.monotonic()
        ready_at = max(now, _WIX_NEXT_REQUEST_AT, _WIX_BLOCKED_UNTIL)
        _WIX_NEXT_REQUEST_AT = ready_at + _WIX_MIN_REQUEST_INTERVAL_SECONDS
    if ready_at > now:
        time.sleep(ready_at - now)


def _block_wix_requests(url: str, delay_seconds: float) -> None:
    if "wixsite.com" not in url:
        return
    global _WIX_BLOCKED_UNTIL
    with _WIX_REQUEST_LOCK:
        _WIX_BLOCKED_UNTIL = max(_WIX_BLOCKED_UNTIL, time.monotonic() + delay_seconds)


FIELD_ALIASES = {
    "date": "encounter_date",
    "encdate": "encounter_date",
    "sequence": "encounter_sequence",
    "encseq": "encounter_sequence",
    "encounternumber": "encounter_number",
    "enc": "encounter_number",
    "encstarttime": "start_time",
    "observbegin": "start_time",
    "encendtime": "end_time",
    "observend": "end_time",
    "vessel": "vessel",
    "observers": "observers",
    "staff": "staff",
    "otherobservers": "other_observers",
    "podsorecotype": "pods_or_ecotype",
    "pods": "pods_or_ecotype",
    "location": "location_description",
    "locationdescr": "location_description",
    "beginlatlong": "begin_lat_long",
    "endlatlong": "end_lat_long",
    "startlatitude": "start_latitude",
    "startlongitude": "start_longitude",
    "endlatitude": "end_latitude",
    "endlongitude": "end_longitude",
    "idsencountered": "individuals",
    "folderid": "folder_id",
}

ENCOUNTER_NAME_PATTERN = re.compile(
    r"^Encounter\s+#(?P<number>[^-]+?)\s+-\s+(?P<date>.+?)\s*$", re.IGNORECASE
)
NOTE_LABEL_PATTERN = re.compile(
    r"(?im)(?:^[\s\u200b]*|,\s*)"
    r"(EncSummary|Encounter\s*Summary|ObservBegin|ObservEnd|Start\s+Time|End\s+Time|"
    r"Vessel|Staff|Other\s+Observers|Pods|IDs\s*Encountered|LocationDescr|Location|"
    r"Start\s+Latitude|Start\s+Longitude|End\s+Latitude|End\s+Longitude)\s*[:=]\s*"
)
NOTE_LABEL_CANONICAL = {
    "encsummary": "summary",
    "encountersummary": "summary",
    "observbegin": "start_time",
    "observend": "end_time",
    "starttime": "start_time",
    "endtime": "end_time",
    "vessel": "vessel",
    "staff": "staff",
    "otherobservers": "other_observers",
    "pods": "pods",
    "idsencountered": "individuals",
    "locationdescr": "location_description",
    "location": "location_description",
    "startlatitude": "start_latitude",
    "startlongitude": "start_longitude",
    "endlatitude": "end_latitude",
    "endlongitude": "end_longitude",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _inline_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _retry_after_seconds(value: str | None) -> float | None:
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
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _request_bytes(
    url: str, *, timeout: int, retries: int, transport=None
) -> tuple[bytes, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            _wait_for_wix_request_slot(url)
            response = (transport or requests.get)(
                url,
                timeout=timeout,
                headers={
                    "Accept": "application/json, text/html",
                    "User-Agent": USER_AGENT,
                },
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_header = response.headers.get("Retry-After")
                retry_after = _retry_after_seconds(retry_header)
                if response.status_code == 429:
                    cooldown = (
                        retry_after if retry_after is not None else 15.0 * (attempt + 1)
                    )
                    _block_wix_requests(url, cooldown)
                raise _TransientCwrHttpError(
                    f"Transient HTTP status {response.status_code}", retry_after
                )
            response.raise_for_status()
            return response.content, {
                "url": url,
                "status": response.status_code,
                "content_type": response.headers.get("content-type"),
                "response_bytes": len(response.content),
                "response_sha256": _sha256(response.content),
                "_raw_body_b64": base64.b64encode(response.content).decode("ascii"),
            }
        except (
            requests.Timeout,
            requests.ConnectionError,
            _TransientCwrHttpError,
        ) as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = (
                exc.retry_after
                if isinstance(exc, _TransientCwrHttpError)
                and exc.retry_after is not None
                else 0.5 * (2**attempt) + random.uniform(0, 0.25)
            )
            LOGGER.warning(
                "Transient CWR fetch failure url=%s attempt=%d/%d retry_in=%.2fs error=%s",
                url,
                attempt + 1,
                retries + 1,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(
        f"Failed to fetch CWR source {url}: {last_error}"
    ) from last_error


def _request_json(
    url: str, *, timeout: int, retries: int, transport=None
) -> tuple[Any, dict[str, Any]]:
    kwargs = {"transport": transport} if transport is not None else {}
    content, evidence = _request_bytes(url, timeout=timeout, retries=retries, **kwargs)
    try:
        return json.loads(content), evidence
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed CWR JSON from {url}") from exc


def _persist_raw_bodies(payload: Any, root: Path, snapshot: Path) -> int:
    """Persist exact HTTP bodies as deterministic gzip files and remove inline transport bytes."""

    written = 0

    def visit(value: Any) -> None:
        nonlocal written
        if isinstance(value, dict):
            encoded = value.pop("_raw_body_b64", None)
            if encoded is not None:
                body = base64.b64decode(str(encoded), validate=True)
                url = str(value.get("url") or f"response-{written}")
                content_type = str(value.get("content_type") or "").lower()
                extension = "json" if "json" in content_type else "html"
                filename = (
                    f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.{extension}.gz"
                )
                destination = root / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as raw_handle:
                    with gzip.GzipFile(
                        fileobj=raw_handle, mode="wb", mtime=0
                    ) as compressed:
                        compressed.write(body)
                value["raw_response_path"] = destination.relative_to(
                    snapshot
                ).as_posix()
                value["raw_response_gzip_sha256"] = _sha256(destination.read_bytes())
                written += 1
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return written


def _cwr_qc(
    flags: list[str], latitude: float | None, longitude: float | None
) -> tuple[str, list[str]]:
    reasons = sorted(set(flags).intersection(CWR_SEVERE_QC_FLAGS))
    if latitude is None or longitude is None:
        reasons.append("SOURCE_COORDINATE_UNAVAILABLE")
    reasons = sorted(set(reasons))
    return ("QUARANTINED" if reasons else "ACCEPTED"), reasons


def parse_archive_index(year: int, content: bytes) -> list[dict[str, Any]]:
    soup = BeautifulSoup(content, "html.parser")
    prefix = f"https://whaleresearch.wixsite.com/{year}encounters/"
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for paragraph in soup.find_all("p"):
        hrefs: list[str] = []
        for anchor in paragraph.find_all("a", href=True):
            href = str(anchor["href"]).split("?", 1)[0].split("#", 1)[0]
            if href.startswith(prefix) and href != prefix and href not in hrefs:
                hrefs.append(href)
        if not hrefs:
            continue
        index_title = _inline_text(paragraph.get_text("", strip=True))
        if not re.search(r"#\s*\d+", index_title) or "•" not in index_title:
            continue
        dedupe_key = hrefs[0], index_title
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        parts = [part.strip() for part in index_title.lstrip("« ").split("•", 2)]
        number_match = re.search(r"#\s*(\d+)", parts[0])
        sequence_match = re.search(r"(?:Seq|Sequence)\s*#?\s*(\d+)", parts[0], re.I)
        if number_match is None:
            continue
        entries.append(
            {
                "source_year": year,
                "record_series": (
                    "uav_encounter" if "uav" in hrefs[0].casefold() else "encounter"
                ),
                "encounter_number": number_match.group(1),
                "index_sequence": sequence_match.group(1) if sequence_match else None,
                "index_date_text": parts[1] if len(parts) > 1 else None,
                "index_descriptor": parts[2] if len(parts) > 2 else None,
                "index_title": index_title,
                "record_page_url": hrefs[0],
            }
        )
    return entries


def parse_page_fields(content: bytes) -> dict[str, str]:
    soup = BeautifulSoup(content, "html.parser")
    fields: dict[str, str] = {}
    for paragraph in soup.find_all("p"):
        text = _inline_text(paragraph.get_text(" ", strip=True))
        match = re.match(r"^([^:]{1,50}):\s*(.*)$", text)
        if not match:
            continue
        label = re.sub(r"[^a-z0-9]", "", match.group(1).casefold())
        canonical = FIELD_ALIASES.get(label)
        if canonical and canonical not in fields:
            fields[canonical] = match.group(2).strip()
    return fields


def _fetch_archive(settings: SourceSettings, *, transport=None) -> dict[str, Any]:
    assert settings.archive_index_url is not None
    assert settings.archive_year_url_template is not None
    landing_content, landing_evidence = _request_bytes(
        settings.archive_index_url,
        timeout=settings.timeout_seconds,
        retries=settings.max_retries,
        **({"transport": transport} if transport is not None else {}),
    )
    del landing_content
    index_results: dict[str, Any] = {}
    index_entries: list[dict[str, Any]] = []
    for year in settings.archive_years:
        url = settings.archive_year_url_template.format(year=year)
        content, evidence = _request_bytes(
            url,
            timeout=settings.timeout_seconds,
            retries=settings.max_retries,
            **({"transport": transport} if transport is not None else {}),
        )
        entries = parse_archive_index(year, content)
        if not entries:
            raise ValueError(f"CWR archive year {year} returned no encounter entries")
        index_results[str(year)] = {**evidence, "entry_count": len(entries)}
        index_entries.extend(entries)

    unique_urls = sorted({str(item["record_page_url"]) for item in index_entries})
    record_pages: dict[str, Any] = {}

    def fetch_page(url: str) -> dict[str, Any]:
        content, evidence = _request_bytes(
            url,
            timeout=settings.timeout_seconds,
            retries=settings.max_retries,
            **({"transport": transport} if transport is not None else {}),
        )
        return {**evidence, "fields": parse_page_fields(content), "error": None}

    with ThreadPoolExecutor(max_workers=settings.archive_fetch_workers) as executor:
        future_urls = {executor.submit(fetch_page, url): url for url in unique_urls}
        for future in as_completed(future_urls):
            url = future_urls[future]
            try:
                record_pages[url] = future.result()
            except Exception as exc:
                raise RuntimeError(f"CWR archive page fetch failed: {url}") from exc

    if set(record_pages) != set(unique_urls):
        raise ValueError("CWR archive page inventory did not reconcile")
    return {
        "schema_version": 2,
        "archive_index_url": settings.archive_index_url,
        "archive_landing_response": landing_evidence,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_results": index_results,
        "index_entries": index_entries,
        "record_pages": record_pages,
    }


def _fetch_atlist(settings: SourceSettings, *, transport=None) -> dict[str, Any]:
    assert settings.atlist_api_root is not None
    maps: dict[str, Any] = {}
    for year, map_settings in sorted(settings.atlist_maps.items()):
        api_root = f"{settings.atlist_api_root.rstrip('/')}/{map_settings.map_id}"
        fields_url = f"{api_root}/fields"
        markers_url = f"{api_root}/markers"
        fields, fields_response = _request_json(
            fields_url,
            timeout=settings.timeout_seconds,
            retries=settings.max_retries,
            **({"transport": transport} if transport is not None else {}),
        )
        markers_payload, markers_response = _request_json(
            markers_url,
            timeout=settings.timeout_seconds,
            retries=settings.max_retries,
            **({"transport": transport} if transport is not None else {}),
        )
        if not isinstance(fields, dict):
            raise ValueError(f"CWR Atlist fields response for {year} is not an object")
        if not isinstance(markers_payload, dict) or not isinstance(
            markers_payload.get("markers"), list
        ):
            raise ValueError(f"CWR Atlist markers response for {year} is malformed")
        maps[str(year)] = {
            "year": year,
            "map_id": map_settings.map_id,
            "page_url": map_settings.page_url,
            "fields_url": fields_url,
            "markers_url": markers_url,
            "fields_response": fields_response,
            "markers_response": markers_response,
            "fields": fields,
            "markers_payload": markers_payload,
        }
    return {
        "schema_version": 1,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "maps": maps,
    }


def collect_cwr_snapshot(
    settings: SourceSettings,
    snapshot: Path,
    *,
    previous_snapshot: Path | None,
    full_refresh: bool,
    transport=None,
) -> dict[str, Any]:
    (snapshot / "cwr_policy.json").write_text(
        settings.model_dump_json(exclude_none=True), encoding="utf-8"
    )
    archive_path = snapshot / "archive_extracts.json"
    archive_reused = False
    if not full_refresh and previous_snapshot is not None:
        previous_archive = previous_snapshot / "archive_extracts.json"
        if previous_archive.is_file():
            shutil.copy2(previous_archive, archive_path)
            previous_raw = previous_snapshot / "raw_responses/archive"
            if previous_raw.is_dir():
                shutil.copytree(
                    previous_raw,
                    snapshot / "raw_responses/archive",
                    copy_function=shutil.copy2,
                )
            archive_reused = True
    if not archive_reused:
        archive_payload = _fetch_archive(
            settings, **({"transport": transport} if transport is not None else {})
        )
        archive_raw_count = _persist_raw_bodies(
            archive_payload, snapshot / "raw_responses/archive", snapshot
        )
        archive_path.write_text(
            json.dumps(archive_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    else:
        archive_raw_count = len(list((snapshot / "raw_responses/archive").glob("*.gz")))
    atlist_path = snapshot / "atlist_maps.json"
    atlist_payload = _fetch_atlist(
        settings, **({"transport": transport} if transport is not None else {})
    )
    atlist_raw_count = _persist_raw_bodies(
        atlist_payload, snapshot / "raw_responses/atlist", snapshot
    )
    atlist_path.write_text(
        json.dumps(atlist_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    rows, metrics = adapt_cwr_snapshot(snapshot, settings)
    native_ids = [str(row["SOURCE_NATIVE_ID"]) for row in rows]
    duplicates = [value for value, count in Counter(native_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"CWR produced duplicate native IDs: {duplicates[:10]}")
    expected_years = set(settings.archive_years) | set(settings.atlist_maps)
    actual_years = {
        int(str(row["SOURCE_DATASET_ID"]).split(":")[-1])
        for row in rows
        if row.get("SOURCE_DATASET_ID")
    }
    if actual_years != expected_years:
        raise ValueError(
            f"CWR configured years do not reconcile: expected={sorted(expected_years)}, actual={sorted(actual_years)}"
        )
    return {
        **metrics,
        "archive_reused": archive_reused,
        "raw_response_body_count": archive_raw_count + atlist_raw_count,
    }


def _parse_archive_date(value: str | None, year: int) -> tuple[str | None, list[str]]:
    if not value:
        return None, ["MISSING_ARCHIVE_DATE"]
    normalized = re.sub(r"(?i)\bSept\b", "Sep", _inline_text(value)).replace(" ", "")
    flags: list[str] = []
    if re.search(r"[-/]\d{3}$", normalized):
        normalized = re.sub(r"([-/])\d{3}$", rf"\g<1>{year}", normalized)
        flags.append("ARCHIVE_DATE_YEAR_REPAIRED")
    for date_format in (
        "%d-%b-%y",
        "%d-%B-%y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d-%b",
        "%d-%B",
    ):
        try:
            if "%Y" not in date_format and "%y" not in date_format:
                parsed = datetime.strptime(
                    f"{normalized}-{year}", f"{date_format}-%Y"
                ).date()
            else:
                parsed = datetime.strptime(normalized, date_format).date()
            if parsed.year != year:
                flags.append("ARCHIVE_DATE_YEAR_MISMATCH")
            return parsed.isoformat(), flags
        except ValueError:
            pass
    return None, [*flags, "UNPARSED_ARCHIVE_DATE"]


def _parse_atlist_date(value: str | None, year: int) -> tuple[str | None, list[str]]:
    if not value:
        return None, ["MISSING_ATLIST_DATE"]
    normalized = re.sub(r"\bSept\b", "Sep", value.strip(), flags=re.I)
    flags: list[str] = []
    malformed = ",," in normalized or bool(re.search(r",(?=\d{4}$)", normalized))
    normalized = normalized.replace(",,", ",")
    normalized = re.sub(r",(?=\d{4}$)", ", ", normalized)
    normalized = re.sub(r"^([A-Za-z]{3})\.", r"\1", normalized)
    if malformed:
        flags.append("ATLIST_DATE_TEXT_NORMALIZED")
    if re.search(r",\s*\d{3}$", normalized):
        normalized = re.sub(r",\s*\d{3}$", f", {year}", normalized)
        flags.append("ATLIST_DATE_YEAR_REPAIRED")
    elif not re.search(r"(?:,\s*(?:\d{2}|\d{4})|\s+\d{4})$", normalized):
        normalized = f"{normalized}, {year}"
        flags.append("ATLIST_DATE_YEAR_INFERRED")
    for date_format in ("%b %d, %Y", "%B %d, %Y", "%b %d, %y", "%B %d, %y"):
        try:
            parsed = datetime.strptime(normalized, date_format).date()
            if parsed.year != year:
                flags.append("ATLIST_MAP_YEAR_DATE_MISMATCH")
            return parsed.isoformat(), flags
        except ValueError:
            pass
    return None, [*flags, "UNPARSED_ATLIST_DATE"]


def normalize_time(value: str | None) -> str | None:
    if not value:
        return None
    compact = re.sub(r"\s+", " ", value.strip().rstrip(".,;")).upper()
    for date_format in ("%I:%M %p", "%I %p", "%H:%M", "%H%M"):
        try:
            return datetime.strptime(compact, date_format).strftime("%H:%M")
        except ValueError:
            pass
    return None


def precise_local_timestamp(
    observed_date: str | None, start_raw: str | None, end_raw: str | None
) -> tuple[str | None, list[str]]:
    flags: list[str] = []
    start = normalize_time(start_raw)
    end = normalize_time(end_raw)
    if not observed_date or not start:
        if start_raw and not start:
            flags.append("UNPARSED_START_TIME")
        return None, flags
    start_minutes = int(start[:2]) * 60 + int(start[3:])
    if not 4 * 60 <= start_minutes <= 21 * 60:
        return None, ["START_TIME_OUTSIDE_LOCAL_PLAUSIBILITY_GATE"]
    if end_raw:
        if not end:
            return None, ["UNPARSED_END_TIME"]
        end_minutes = int(end[:2]) * 60 + int(end[3:])
        duration = end_minutes - start_minutes
        if duration < 0 or duration > 16 * 60:
            return None, ["SOURCE_TIME_RANGE_ANOMALY"]
    return f"{observed_date}T{start}:00", flags


def parse_coordinate_component(
    value: str | None, axis: str
) -> tuple[float | None, str | None]:
    if not value:
        return None, None
    text = (
        _inline_text(value)
        .upper()
        .replace("°", " ")
        .replace("′", " ")
        .replace("'", " ")
    )
    hemisphere = next(
        (token for token in ("N", "S", "E", "W") if re.search(rf"\b{token}\b", text)),
        None,
    )
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None, None
    first = float(numbers[0])
    if len(numbers) >= 2:
        result = abs(first) + float(numbers[1]) / 60.0
        method = "DEGREES_DECIMAL_MINUTES"
    else:
        result = abs(first)
        method = "DECIMAL_DEGREES"
    negative = first < 0 or hemisphere in {"S", "W"}
    if axis == "longitude" and first >= 0 and hemisphere not in {"E", "W"}:
        negative = True
        method += "_WEST_INFERRED_FROM_CWR_DOMAIN"
    result = -result if negative else result
    limit = 90 if axis == "latitude" else 180
    return (
        (None, "INVALID_COORDINATE")
        if not -limit <= result <= limit
        else (result, method)
    )


def _page_coordinates(fields: dict[str, str], bounds: BBox) -> dict[str, Any]:
    start_lat_text = fields.get("start_latitude")
    start_lon_text = fields.get("start_longitude")
    end_lat_text = fields.get("end_latitude")
    end_lon_text = fields.get("end_longitude")
    if (not start_lat_text or not start_lon_text) and "/" in fields.get(
        "begin_lat_long", ""
    ):
        start_lat_text, start_lon_text = [
            item.strip() for item in fields["begin_lat_long"].split("/", 1)
        ]
    if (not end_lat_text or not end_lon_text) and "/" in fields.get("end_lat_long", ""):
        end_lat_text, end_lon_text = [
            item.strip() for item in fields["end_lat_long"].split("/", 1)
        ]
    start_lat, start_lat_method = parse_coordinate_component(start_lat_text, "latitude")
    start_lon, start_lon_method = parse_coordinate_component(
        start_lon_text, "longitude"
    )
    end_lat, end_lat_method = parse_coordinate_component(end_lat_text, "latitude")
    end_lon, end_lon_method = parse_coordinate_component(end_lon_text, "longitude")
    raw_candidates = {
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
    }
    flags: list[str] = []

    def plausible(latitude: float | None, longitude: float | None) -> bool:
        return (
            latitude is not None
            and longitude is not None
            and bounds.min_lat <= latitude <= bounds.max_lat
            and bounds.min_lon <= longitude <= bounds.max_lon
        )

    if start_lat is not None or start_lon is not None:
        if not plausible(start_lat, start_lon):
            start_lat = start_lon = None
            flags.append("ARCHIVE_START_COORDINATE_INVALID_OR_OUTSIDE_CWR_BOUNDS")
    if end_lat is not None or end_lon is not None:
        if not plausible(end_lat, end_lon):
            end_lat = end_lon = None
            flags.append("ARCHIVE_END_COORDINATE_INVALID_OR_OUTSIDE_CWR_BOUNDS")
    methods = sorted(
        {
            item
            for item in (
                start_lat_method,
                start_lon_method,
                end_lat_method,
                end_lon_method,
            )
            if item
        }
    )
    return {
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
        "methods": methods,
        "coordinate_flags": flags,
        "source_text": {
            "start_latitude": start_lat_text,
            "start_longitude": start_lon_text,
            "end_latitude": end_lat_text,
            "end_longitude": end_lon_text,
            "parsed_candidates": raw_candidates,
        },
    }


def _html_to_lines(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"(?i)<br\s*/?>", "\n", value)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def parse_labeled_notes(notes_html: str | None) -> dict[str, str]:
    text = _html_to_lines(notes_html)
    matches = list(NOTE_LABEL_PATTERN.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label = re.sub(r"\s+", "", match.group(1)).casefold()
        fields[NOTE_LABEL_CANONICAL[label]] = text[match.end() : end].strip()
    return fields


def _marker_tags(marker: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for tag in marker.get("tags") or []:
        if isinstance(tag, str):
            result.append(tag)
        elif isinstance(tag, dict) and tag.get("name"):
            result.append(str(tag["name"]))
    return result


def _archive_rows(
    payload: dict[str, Any], settings: SourceSettings
) -> list[dict[str, Any]]:
    assert settings.coordinate_bounds is not None
    entries = payload.get("index_entries")
    pages_by_url = payload.get("record_pages")
    if not isinstance(entries, list) or not isinstance(pages_by_url, dict):
        raise ValueError("Malformed CWR archive extract")
    sequence_rows: list[dict[str, Any]] = []
    for entry in entries:
        page = pages_by_url.get(entry.get("record_page_url"))
        if not isinstance(page, dict) or page.get("status") != 200 or page.get("error"):
            raise ValueError(
                f"Missing successful CWR archive page: {entry.get('record_page_url')}"
            )
        fields = page.get("fields") or {}
        flags: list[str] = []
        page_number = re.search(r"\d+", str(fields.get("encounter_number") or ""))
        page_number_value = str(int(page_number.group())) if page_number else None
        if page_number_value not in {None, str(entry["encounter_number"])}:
            flags.append("ARCHIVE_INDEX_PAGE_IDENTITY_MISMATCH")
            usable_fields: dict[str, str] = {}
        else:
            usable_fields = fields
        page_date, page_flags = _parse_archive_date(
            usable_fields.get("encounter_date"), int(entry["source_year"])
        )
        index_date, index_flags = _parse_archive_date(
            entry.get("index_date_text"), int(entry["source_year"])
        )
        observed_date: str | None
        if page_date and index_date and page_date != index_date:
            flags.append("ARCHIVE_INDEX_PAGE_DATE_MISMATCH")
            observed_date = index_date
        else:
            observed_date = page_date or index_date
        flags.extend(page_flags if usable_fields.get("encounter_date") else index_flags)
        coordinates = _page_coordinates(usable_fields, settings.coordinate_bounds)
        flags.extend(coordinates["coordinate_flags"])
        sequence_text = usable_fields.get("encounter_sequence") or entry.get(
            "index_sequence"
        )
        sequence_match = re.search(r"\d+", sequence_text or "")
        sequence_rows.append(
            {
                **entry,
                "page": page,
                "raw_fields": fields,
                "fields": usable_fields,
                "observed_date": observed_date,
                "sequence_sort": (
                    int(sequence_match.group()) if sequence_match else 9999
                ),
                "flags": flags,
                **coordinates,
            }
        )

    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sequence_rows:
        groups[
            (
                int(row["source_year"]),
                str(row["record_series"]),
                str(row["encounter_number"]),
            )
        ].append(row)
    rows: list[dict[str, Any]] = []
    for (year, series, number), pages in sorted(groups.items()):
        pages.sort(key=lambda item: (item["sequence_sort"], item["record_page_url"]))
        flags = sorted({flag for page in pages for flag in page["flags"]})
        if len(pages) > 1:
            flags.append("MULTIPLE_SEQUENCE_PAGES_AGGREGATED")
        dates = list(
            dict.fromkeys(
                page["observed_date"] for page in pages if page["observed_date"]
            )
        )
        if len(dates) > 1:
            flags.append("SEQUENCE_DATE_DISAGREEMENT")
        first_start = next(
            (
                page
                for page in pages
                if page["start_lat"] is not None and page["start_lon"] is not None
            ),
            None,
        )
        last_end = next(
            (
                page
                for page in reversed(pages)
                if page["end_lat"] is not None and page["end_lon"] is not None
            ),
            None,
        )
        if first_start:
            latitude, longitude = first_start["start_lat"], first_start["start_lon"]
            coordinate_role = "ARCHIVE_FIRST_VALID_SEQUENCE_START"
        elif last_end:
            latitude, longitude = last_end["end_lat"], last_end["end_lon"]
            coordinate_role = "ARCHIVE_FINAL_SEQUENCE_END_FALLBACK"
        else:
            latitude = longitude = None
            coordinate_role = "SOURCE_COORDINATE_UNAVAILABLE"
        group_date = str(dates[0]) if dates else None
        start_raw = pages[0]["fields"].get("start_time")
        end_raw = pages[-1]["fields"].get("end_time")
        observed_at, time_flags = precise_local_timestamp(
            group_date, start_raw, end_raw
        )
        flags.extend(time_flags)
        pod_values = list(
            dict.fromkeys(
                str(value)
                for page in pages
                for value in (
                    page["fields"].get("pods_or_ecotype"),
                    page["fields"].get("individuals"),
                    page.get("index_descriptor"),
                )
                if value
            )
        )
        locations = list(
            dict.fromkeys(
                page["fields"].get("location_description")
                for page in pages
                if page["fields"].get("location_description")
            )
        )
        urls = [str(page["record_page_url"]) for page in pages]
        native_id = f"wix:{year}:{series}:{number}"
        payload_row = {
            "source_system": "wix",
            "source_year": year,
            "record_series": series,
            "encounter_number": number,
            "index_titles": [page["index_title"] for page in pages],
            "component_pages": [
                {
                    "url": page["record_page_url"],
                    "status": page["page"]["status"],
                    "content_type": page["page"].get("content_type"),
                    "response_bytes": page["page"].get("response_bytes"),
                    "response_sha256": page["page"].get("response_sha256"),
                    "fields": page["raw_fields"],
                    "coordinate_source_text": page["source_text"],
                }
                for page in pages
            ],
            "archive_index": payload["index_results"][str(year)],
            "coordinate_role": coordinate_role,
            "start_time_raw": start_raw,
            "end_time_raw": end_raw,
            "qc_flags": sorted(set(flags)),
        }
        qc_status, quarantine_reasons = _cwr_qc(flags, latitude, longitude)
        rows.append(
            _source_record(
                "CWR",
                native_id,
                payload_row,
                observed_at=observed_at,
                observed_date=group_date,
                latitude=latitude,
                longitude=longitude,
                species="Orcinus orca",
                description=" → ".join(locations),
                pod_ecotype=" | ".join(pod_values),
                source_dataset_id=f"cwr:wix:{year}",
                source_event_id=native_id,
                source_occurrence_ids=urls,
                source_occurrence_count=len(pages),
                source_license=settings.source_license,
                source_use_class=settings.source_use_class,
                source_qc_status=qc_status,
                source_qc_detail=json.dumps(
                    {
                        "flags": sorted(set(flags)),
                        "quarantine_reasons": quarantine_reasons,
                        "coordinate_role": coordinate_role,
                        "component_count": len(pages),
                        "raw_start_time": start_raw,
                        "raw_end_time": end_raw,
                    },
                    sort_keys=True,
                ),
            )
        )
    return rows


def _atlist_rows(
    payload: dict[str, Any], settings: SourceSettings
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maps = payload.get("maps")
    if not isinstance(maps, dict):
        raise ValueError("Malformed CWR Atlist extract")
    rows: list[dict[str, Any]] = []
    raw_count = 0
    excluded: list[dict[str, str]] = []
    by_year: dict[str, Any] = {}
    for year_text, source in sorted(maps.items()):
        year = int(year_text)
        marker_payload = source.get("markers_payload")
        markers = (
            marker_payload.get("markers") if isinstance(marker_payload, dict) else None
        )
        if not isinstance(markers, list):
            raise ValueError(f"Malformed CWR Atlist marker list for {year}")
        raw_count += len(markers)
        accepted_count = 0
        for marker in markers:
            if not isinstance(marker, dict):
                raise ValueError(f"Non-object CWR Atlist marker for {year}")
            marker_id = marker.get("id")
            if not marker_id:
                raise ValueError(f"CWR Atlist marker without UUID for {year}")
            title = str(marker.get("name") or "")
            match = ENCOUNTER_NAME_PATTERN.match(title)
            if match is None:
                excluded.append(
                    {"year": str(year), "marker_id": str(marker_id), "title": title}
                )
                continue
            accepted_count += 1
            observed_date, flags = _parse_atlist_date(match.group("date").strip(), year)
            fields = parse_labeled_notes(marker.get("notes"))
            start_raw = fields.get("start_time")
            end_raw = fields.get("end_time")
            observed_at, time_flags = precise_local_timestamp(
                observed_date, start_raw, end_raw
            )
            flags.extend(time_flags)
            latitude: float | None
            longitude: float | None
            try:
                latitude = float(marker["lat"])
                longitude = float(marker["long"])
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                latitude = longitude = None
                flags.append("MISSING_OR_INVALID_ATLIST_COORDINATE")
            tags = _marker_tags(marker)
            pod_values = list(
                dict.fromkeys(
                    [
                        *tags,
                        *re.split(r"[,;]\s*", fields.get("pods", "")),
                        *re.split(r"[,;]\s*", fields.get("individuals", "")),
                    ]
                )
            )
            pod_values = [value.strip() for value in pod_values if value.strip()]
            native_id = f"atlist:{source['map_id']}:{marker_id}"
            payload_row = {
                "source_system": "atlist",
                "source_year": year,
                "map_id": source["map_id"],
                "page_url": source["page_url"],
                "fields_url": source["fields_url"],
                "markers_url": source["markers_url"],
                "fields_response_sha256": source["fields_response"]["response_sha256"],
                "markers_response_sha256": source["markers_response"][
                    "response_sha256"
                ],
                "marker": marker,
                "parsed_fields": fields,
                "coordinate_role": "ATLIST_MARKER_UNSPECIFIED",
                "start_time_raw": start_raw,
                "end_time_raw": end_raw,
                "qc_flags": sorted(set(flags)),
            }
            qc_status, quarantine_reasons = _cwr_qc(flags, latitude, longitude)
            rows.append(
                _source_record(
                    "CWR",
                    native_id,
                    payload_row,
                    observed_at=observed_at,
                    observed_date=observed_date,
                    created_at=marker.get("createdAt"),
                    latitude=latitude,
                    longitude=longitude,
                    species="Orcinus orca",
                    description=fields.get("location_description"),
                    pod_ecotype=" | ".join(pod_values),
                    source_dataset_id=f"cwr:atlist:{year}",
                    source_event_id=native_id,
                    source_occurrence_ids=[str(marker_id)],
                    source_occurrence_count=1,
                    source_license=settings.source_license,
                    source_use_class=settings.source_use_class,
                    source_qc_status=qc_status,
                    source_qc_detail=json.dumps(
                        {
                            "flags": sorted(set(flags)),
                            "quarantine_reasons": quarantine_reasons,
                            "coordinate_role": "ATLIST_MARKER_UNSPECIFIED",
                            "raw_start_time": start_raw,
                            "raw_end_time": end_raw,
                        },
                        sort_keys=True,
                    ),
                )
            )
        by_year[str(year)] = {
            "raw_marker_count": len(markers),
            "encounter_marker_count": accepted_count,
            "excluded_marker_count": len(markers) - accepted_count,
        }
    if raw_count != len(rows) + len(excluded):
        raise ValueError("CWR Atlist raw markers do not reconcile")
    return rows, {
        "raw_marker_count": raw_count,
        "encounter_marker_count": len(rows),
        "excluded_marker_count": len(excluded),
        "excluded_markers": excluded,
        "atlist_by_year": by_year,
    }


def adapt_cwr_snapshot(
    snapshot: Path, settings: SourceSettings
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_payload = json.loads(
        (snapshot / "archive_extracts.json").read_text(encoding="utf-8")
    )
    atlist_payload = json.loads(
        (snapshot / "atlist_maps.json").read_text(encoding="utf-8")
    )
    archive_rows = _archive_rows(archive_payload, settings)
    atlist_rows, atlist_metrics = _atlist_rows(atlist_payload, settings)
    rows = [*archive_rows, *atlist_rows]
    mapped_count = sum(
        row["LATITUDE_RAW"] is not None and row["LONGITUDE_RAW"] is not None
        for row in rows
    )
    quarantined_count = sum(row["SOURCE_QC_STATUS"] == "QUARANTINED" for row in rows)
    metrics = {
        "source_event_count": len(rows),
        "archive_event_count": len(archive_rows),
        "atlist_event_count": len(atlist_rows),
        "map_ready_event_count": mapped_count,
        "coordinate_unavailable_event_count": len(rows) - mapped_count,
        "quarantined_event_count": quarantined_count,
        **atlist_metrics,
    }
    if metrics["atlist_event_count"] != metrics["encounter_marker_count"]:
        raise ValueError("CWR Atlist adapted records do not reconcile")
    return rows, metrics


def settings_from_snapshot(snapshot: Path) -> SourceSettings:
    policy_path = snapshot / "cwr_policy.json"
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"CWR snapshot lacks its immutable policy: {policy_path}"
        )
    return SourceSettings.model_validate_json(policy_path.read_text(encoding="utf-8"))
