from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pyarrow as pa

from marine_mammal_toolkit.tools.schemas.observations import SOURCE_RECORD_SCHEMA
from marine_mammal_toolkit.tools.observations.process.records import (
    source_record as _source_record,
)
from marine_mammal_toolkit.tools.observations.process.records import (
    source_text as _text,
)


def first_present(*values: Any) -> Any | None:
    """Return the first scalar value that is not null, NaN, or blank text."""
    for value in values:
        if value is None:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except TypeError:
            pass
        except ValueError:
            pass
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "na", "n/a", "none", "null"}:
                continue
        return value
    return None


def _gbif_day(item: dict[str, Any]) -> str | None:
    try:
        year = int(item["year"])
        month = int(item["month"])
        day = int(item["day"])
        return pd.Timestamp(year=year, month=month, day=day).date().isoformat()
    except (KeyError, TypeError, ValueError):
        return None


def _gbif_precise_timestamp(item: dict[str, Any]) -> str | None:
    value = _text(item.get("eventDate"))
    if value is None or "/" in value:
        return None
    if "T" not in value and not any(char.isdigit() for char in value[10:]):
        return None
    try:
        # This field is only used to retain an already-ISO source value.  A
        # scalar pandas parser here dominated GBIF adaptation because it
        # constructed a Series-like parsing path once per occurrence.
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return str(value)


def _gbif_coordinate(item: dict[str, Any]) -> tuple[float, float] | None:
    try:
        latitude = float(item["decimalLatitude"])
        longitude = float(item["decimalLongitude"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return latitude, longitude


def _distance_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    radius_km = 6371.0088
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(value))


def _gbif_group_key(
    item: dict[str, Any], policy: dict[str, Any] | None
) -> tuple[str, str]:
    dataset_key = str(item.get("datasetKey") or "missing-dataset")
    event_id = _text(item.get("eventID"))
    if policy and policy.get("event_id_is_encounter") and event_id:
        return dataset_key, f"event:{event_id}"
    if policy and policy.get("happywhale_exact_fallback"):
        timestamp = _gbif_precise_timestamp(item)
        coordinate = _gbif_coordinate(item)
        if timestamp and coordinate:
            identity = json.dumps(
                [
                    dataset_key,
                    timestamp,
                    round(coordinate[0], 7),
                    round(coordinate[1], 7),
                ],
                separators=(",", ":"),
            )
            digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
            return dataset_key, f"exact:{digest}"
    occurrence_id = first_present(item.get("occurrenceID"), item.get("gbifID"))
    if occurrence_id is None:
        canonical = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
        occurrence_id = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:24]}"
    return dataset_key, f"occurrence:{occurrence_id}"


def _gbif_use_class(policy: dict[str, Any] | None, licenses: list[str]) -> str:
    if not policy or policy.get("use_class") != "REDISTRIBUTABLE" or not licenses:
        return "INTERNAL_ONLY"
    normalized = [value.lower().replace("-", "_") for value in licenses]
    redistribution_safe = all(
        (
            "publicdomain/zero" in value
            or "cc0" in value
            or "licenses/by/" in value
            or "cc_by_" in value
        )
        and "by_nc" not in value
        and "by_nd" not in value
        and "/by-nc" not in value
        and "/by-nd" not in value
        for value in normalized
    )
    return "REDISTRIBUTABLE" if redistribution_safe else "INTERNAL_ONLY"


def adapt_gbif(
    items: Iterable[dict[str, Any]],
    *,
    scientific_name: str,
    dataset_policies: dict[str, dict[str, Any]],
    max_coordinate_uncertainty_m: float,
    max_event_spread_km: float,
) -> list[dict]:
    """Collapse reviewed GBIF occurrence rows to source encounter records."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        policy = dataset_policies.get(str(item.get("datasetKey") or ""))
        groups.setdefault(_gbif_group_key(item, policy), []).append(item)

    rows: list[dict] = []
    for (dataset_key, grouping_key), occurrences in sorted(groups.items()):
        policy = dataset_policies.get(dataset_key)
        reasons: list[str] = []
        if policy is None:
            reasons.append("DATASET_NOT_ALLOWLISTED")
        scientific_names = {
            str(
                first_present(item.get("scientificName"), item.get("species")) or ""
            ).lower()
            for item in occurrences
        }
        if not any("orcinus orca" in value for value in scientific_names):
            reasons.append("NON_ORCA")

        occurrence_days = [_gbif_day(item) for item in occurrences]
        days = sorted({value for value in occurrence_days if value is not None})
        if any(value is None for value in occurrence_days) or len(days) != 1:
            reasons.append("MISSING_OR_CONFLICTING_DAY_PRECISION")
        coordinates = [
            value for item in occurrences if (value := _gbif_coordinate(item))
        ]
        if len(coordinates) != len(occurrences):
            reasons.append("MISSING_OR_INVALID_COORDINATES")
        spread_km = 0.0
        for index, left in enumerate(coordinates):
            for right in coordinates[index + 1 :]:
                spread_km = max(spread_km, _distance_km(left, right))
        if spread_km > max_event_spread_km:
            reasons.append("SOURCE_EVENT_SPREAD_EXCEEDS_LIMIT")

        uncertainties: list[float] = []
        for item in occurrences:
            raw_uncertainty = item.get("coordinateUncertaintyInMeters")
            if raw_uncertainty is None:
                continue
            try:
                uncertainty = float(raw_uncertainty)
            except (TypeError, ValueError):
                reasons.append("INVALID_COORDINATE_UNCERTAINTY")
                continue
            if not math.isfinite(uncertainty) or uncertainty < 0:
                reasons.append("INVALID_COORDINATE_UNCERTAINTY")
                continue
            uncertainties.append(uncertainty)
        coordinate_uncertainty = max(uncertainties) if uncertainties else None
        if (
            coordinate_uncertainty is not None
            and coordinate_uncertainty > max_coordinate_uncertainty_m
        ):
            reasons.append("COORDINATE_UNCERTAINTY_EXCEEDS_LIMIT")

        occurrence_ids = sorted(
            {
                str(value)
                for item in occurrences
                if (
                    value := first_present(item.get("occurrenceID"), item.get("gbifID"))
                )
            }
        )
        exact_timestamps = sorted(
            {value for item in occurrences if (value := _gbif_precise_timestamp(item))}
        )
        description_values: list[str] = []
        structured_values: list[str] = []
        for item in occurrences:
            for field in (
                "occurrenceRemarks",
                "eventRemarks",
                "identificationRemarks",
                "organismRemarks",
                "behavior",
            ):
                if (
                    value := _text(item.get(field))
                ) and value not in description_values:
                    description_values.append(value)
            for field in ("individualID", "organismID"):
                if (value := _text(item.get(field))) and value not in structured_values:
                    structured_values.append(value)

        licenses = sorted(
            {
                str(value)
                for item in occurrences
                if (value := first_present(item.get("license")))
            }
        )
        if not licenses and policy and policy.get("license"):
            licenses = [str(policy["license"])]
        use_class = _gbif_use_class(policy, licenses)
        payload = {
            "_orcacast_grouping_key": grouping_key,
            "_orcacast_grouping_rule": grouping_key.split(":", 1)[0],
            "_orcacast_event_spread_km": spread_km,
            "_orcacast_qc_reasons": sorted(set(reasons)),
            "occurrences": occurrences,
        }
        event_ids = sorted(
            {str(item["eventID"]) for item in occurrences if _text(item.get("eventID"))}
        )
        native_id = f"{dataset_key}:{grouping_key}"
        rows.append(
            _source_record(
                "GBIF",
                native_id,
                payload,
                observed_at=exact_timestamps[0] if len(exact_timestamps) == 1 else None,
                observed_date=days[0] if len(days) == 1 else None,
                created_at=None,
                latitude=(
                    median(item[0] for item in coordinates) if coordinates else None
                ),
                longitude=(
                    median(item[1] for item in coordinates) if coordinates else None
                ),
                species=scientific_name,
                description=" | ".join(description_values),
                pod_ecotype=" | ".join(structured_values),
                source_dataset_id=dataset_key,
                source_event_id=event_ids[0] if len(event_ids) == 1 else None,
                source_occurrence_ids=occurrence_ids,
                source_occurrence_count=len(occurrences),
                source_license=" | ".join(licenses),
                source_use_class=use_class,
                coordinate_uncertainty_m=coordinate_uncertainty,
                source_qc_status="QUARANTINED" if reasons else "ACCEPTED",
                source_qc_detail=json.dumps(
                    {
                        "reasons": sorted(set(reasons)),
                        "grouping_rule": grouping_key.split(":", 1)[0],
                        "occurrence_count": len(occurrences),
                        "event_spread_km": spread_km,
                    },
                    sort_keys=True,
                ),
            )
        )
    return rows


def adapt_inaturalist(
    items: Iterable[dict[str, Any]],
    *,
    source_license: str | None = None,
    source_use_class: str | None = None,
    max_coordinate_uncertainty_m: float = 5000.0,
) -> list[dict]:
    rows = []
    for item in items:
        geojson = item.get("geojson") or {}
        coords = geojson.get("coordinates") if isinstance(geojson, dict) else None
        latitude = first_present(
            coords[1] if isinstance(coords, list) and len(coords) == 2 else None,
            item.get("latitude"),
        )
        longitude = first_present(
            coords[0] if isinstance(coords, list) and len(coords) == 2 else None,
            item.get("longitude"),
        )
        taxon = item.get("taxon") or {}
        reasons: list[str] = []
        captive_raw = first_present(item.get("captive"), item.get("captive_cultivated"))
        captive = (
            captive_raw
            if isinstance(captive_raw, bool)
            else str(captive_raw or "").strip().lower() in {"1", "true", "yes"}
        )
        if captive:
            reasons.append("CAPTIVE_OR_CULTIVATED")
        uncertainty_raw = first_present(
            item.get("positional_accuracy"), item.get("coordinate_uncertainty_m")
        )
        coordinate_uncertainty: float | None = None
        if uncertainty_raw is not None:
            try:
                candidate = float(uncertainty_raw)
                if not math.isfinite(candidate) or candidate < 0:
                    raise ValueError
                coordinate_uncertainty = candidate
            except (TypeError, ValueError):
                reasons.append("INVALID_COORDINATE_UNCERTAINTY")
        if (
            coordinate_uncertainty is not None
            and coordinate_uncertainty > max_coordinate_uncertainty_m
        ):
            reasons.append("COORDINATE_UNCERTAINTY_EXCEEDS_LIMIT")
        qc_detail = {
            "reasons": sorted(set(reasons)),
            "quality_grade": _text(item.get("quality_grade")),
            "captive": bool(captive),
            "positional_accuracy_m": coordinate_uncertainty,
            "geoprivacy": _text(item.get("geoprivacy")),
            "obscured": bool(item.get("obscured", False)),
        }
        rows.append(
            _source_record(
                "INATURALIST",
                item.get("id"),
                item,
                observed_at=item.get("time_observed_at"),
                observed_date=item.get("observed_on"),
                created_at=item.get("created_at"),
                latitude=latitude,
                longitude=longitude,
                species=first_present(taxon.get("name"), item.get("species_guess")),
                description=" | ".join(
                    str(value)
                    for value in (item.get("description"), item.get("place_guess"))
                    if first_present(value) is not None
                ),
                pod_ecotype=item.get("tag_list"),
                source_license=source_license,
                source_use_class=source_use_class,
                coordinate_uncertainty_m=coordinate_uncertainty,
                source_qc_status="QUARANTINED" if reasons else "ACCEPTED",
                source_qc_detail=json.dumps(qc_detail, sort_keys=True),
            )
        )
    return rows


def adapt_acartia(
    items: Iterable[dict[str, Any]],
    *,
    source_license: str | None = None,
    source_use_class: str | None = None,
) -> list[dict]:
    rows = []
    for item in items:
        rows.append(
            _source_record(
                "ACARTIA",
                first_present(
                    item.get("ssemmi_id"), item.get("entry_id"), item.get("id")
                ),
                item,
                observed_at=first_present(
                    item.get("observed_at"),
                    item.get("time_observed_at"),
                    item.get("created"),
                ),
                observed_date=first_present(
                    item.get("observed_on"), item.get("sightdate")
                ),
                # Acartia's `created` field is the occurrence time, not a
                # publication timestamp. Availability is the snapshot retrieval.
                created_at=None,
                latitude=first_present(item.get("latitude"), item.get("lat")),
                longitude=first_present(
                    item.get("longitude"), item.get("lon"), item.get("lng")
                ),
                species=first_present(
                    item.get("type"), item.get("species"), item.get("name")
                ),
                description=first_present(
                    item.get("data_source_comments"),
                    item.get("comments"),
                    item.get("notes"),
                ),
                pod_ecotype=first_present(item.get("pod"), item.get("ecotype")),
                source_license=source_license,
                source_use_class=source_use_class,
            )
        )
    return rows


def adapt_acartia_files(
    files: Iterable[Path],
    *,
    source_root: Path | None = None,
    source_license: str | None = None,
    source_use_class: str | None = None,
) -> list[dict]:
    """Adapt bulk exports through the same Acartia source contract as API rows."""
    items: list[dict[str, Any]] = []
    for path in sorted(files):
        frame = pl.read_csv(
            path,
            infer_schema=False,
            missing_utf8_is_empty_string=True,
        )
        frame = frame.rename(
            {column: str(column).strip().lower() for column in frame.columns}
        )
        for index, payload in enumerate(frame.iter_rows(named=True), start=2):
            payload["_source_file"] = (
                path.relative_to(source_root).as_posix()
                if source_root is not None
                else path.name
            )
            payload["_source_row"] = index
            items.append(payload)
    return adapt_acartia(
        items,
        source_license=source_license,
        source_use_class=source_use_class,
    )


def adapt_maplify(
    items: Iterable[dict[str, Any]],
    *,
    source_license: str | None = None,
    source_use_class: str | None = None,
) -> list[dict]:
    """Adapt Maplify/WASEAK rows without conflating them with Acartia exports."""
    rows = []
    for item in items:
        source_code = str(item.get("source") or "").strip().lower()
        is_test = str(item.get("is_test") or "").strip().lower() in {"1", "true", "yes"}
        # These upstream mirror branches are excluded by the maintained
        # SalishSea.io ingest as duplicate/invalid records. In particular,
        # rwsas duplicates Whale Alert IDs with corrupt taxonomy.
        if source_code in {"rwsas", "wras"} or is_test:
            continue
        species_values = [
            str(value).strip()
            for value in (item.get("scientific_name"), item.get("name"))
            if first_present(value) is not None
        ]
        species = " | ".join(dict.fromkeys(species_values))
        rows.append(
            _source_record(
                "MAPLIFY",
                item.get("id"),
                item,
                # WASEAK's `created` value is the occurrence time. Availability is
                # intentionally the snapshot retrieval time, not this event time.
                observed_at=item.get("created"),
                created_at=None,
                latitude=item.get("latitude"),
                longitude=item.get("longitude"),
                species=species,
                description=item.get("comments"),
                pod_ecotype=item.get("name"),
                source_license=source_license,
                source_use_class=source_use_class,
            )
        )
    return rows


def adapt_twm(
    files: Iterable[Path],
    *,
    scientific_name: str,
    structured_evidence,
    identity_fields: tuple[str, ...],
    source_license: str | None = None,
    source_use_class: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(files):
        frame = pl.read_csv(
            path,
            infer_schema=False,
            missing_utf8_is_empty_string=True,
        )
        frame = frame.rename({column: str(column).lower() for column in frame.columns})
        for index, item in enumerate(frame.iter_rows(named=True)):
            payload = dict(item)
            observed_date = first_present(item.get("sightdate"), item.get("date"))
            observed_time = first_present(item.get("time1"), item.get("time"))
            observed_at = (
                f"{observed_date}T{observed_time}"
                if observed_date and observed_time
                else None
            )
            native_id = first_present(item.get("id"), item.get("observation_id"))
            if native_id is None:
                identity_values = {
                    key: first_present(item.get(key)) for key in identity_fields
                }
                fingerprint = hashlib.sha256(
                    json.dumps(identity_values, sort_keys=True, default=str).encode()
                ).hexdigest()[:24]
                native_id = f"content:{fingerprint}"
            pod_ecotype = structured_evidence(item)
            rows.append(
                _source_record(
                    "TWM",
                    native_id,
                    payload,
                    observed_at=observed_at,
                    observed_date=observed_date,
                    created_at=None,
                    latitude=first_present(item.get("latitude"), item.get("lat")),
                    longitude=first_present(item.get("longitude"), item.get("lon")),
                    species=scientific_name,
                    description=first_present(item.get("notes"), item.get("comments")),
                    pod_ecotype=pod_ecotype,
                    source_license=source_license,
                    source_use_class=source_use_class,
                )
            )
    return rows


def adapt_snapshot(
    source: str,
    snapshot: Path,
    *,
    scientific_name: str,
    twm_evidence,
    twm_identity_fields,
) -> pa.Table:
    policy_path = snapshot / "source_policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    source_license = _text(policy.get("source_license"))
    source_use_class = _text(policy.get("source_use_class"))
    if source == "twm":
        rows = adapt_twm(
            (path for path in snapshot.glob("*.csv") if path.is_file()),
            scientific_name=scientific_name,
            structured_evidence=twm_evidence,
            identity_fields=twm_identity_fields,
            source_license=source_license,
            source_use_class=source_use_class,
        )
    elif source == "cwr":
        from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
            adapt_cwr_snapshot,
        )
        from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
            settings_from_snapshot,
        )

        rows, _metrics = adapt_cwr_snapshot(snapshot, settings_from_snapshot(snapshot))
    else:
        payload = json.loads((snapshot / "payload.json").read_text())
        items = payload if isinstance(payload, list) else payload.get("results", [])
        if source == "inaturalist":
            rows = adapt_inaturalist(
                items,
                source_license=source_license,
                source_use_class=source_use_class,
                max_coordinate_uncertainty_m=float(
                    policy.get("max_coordinate_uncertainty_m") or 5000.0
                ),
            )
        elif source == "maplify":
            rows = adapt_maplify(
                items,
                source_license=source_license,
                source_use_class=source_use_class,
            )
        elif source == "gbif":
            dataset_metadata_path = snapshot / "datasets.json"
            dataset_metadata = (
                json.loads(dataset_metadata_path.read_text())
                if dataset_metadata_path.exists()
                else []
            )
            policies = {
                str(item["key"]): dict(item.get("_orcacast_policy") or {})
                for item in dataset_metadata
                if item.get("key")
            }
            policy_path = snapshot / "gbif_policy.json"
            request = (
                json.loads(policy_path.read_text()) if policy_path.exists() else {}
            )
            rows = adapt_gbif(
                items,
                dataset_policies=policies,
                scientific_name=scientific_name,
                max_coordinate_uncertainty_m=float(
                    request.get("max_coordinate_uncertainty_m", 5000.0)
                ),
                max_event_spread_km=float(request.get("max_event_spread_km", 5.0)),
            )
        else:
            rows = adapt_acartia(
                items,
                source_license=source_license,
                source_use_class=source_use_class,
            )
            rows.extend(
                adapt_acartia_files(
                    (path for path in snapshot.rglob("*.csv") if path.is_file()),
                    source_root=snapshot,
                    source_license=source_license,
                    source_use_class=source_use_class,
                )
            )
    return pa.Table.from_pylist(rows, schema=SOURCE_RECORD_SCHEMA)
