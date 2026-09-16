"""Reviewed killer-whale species, ecotype, pod and member interpretation."""

from __future__ import annotations
import json, re
from typing import Any, Mapping
import pandas as pd


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


def _is_killer_whale(row: Mapping[str, Any]) -> bool:
    species = (
        str(row["SPECIES_RAW"]).strip().lower() if _present(row["SPECIES_RAW"]) else ""
    )
    structured = (
        str(row["POD_ECOTYPE_RAW"]).strip().lower()
        if _present(row["POD_ECOTYPE_RAW"])
        else ""
    )
    if "false killer whale" in species or "pseudorca" in species:
        return False
    species_match = bool(
        re.search(r"\b(orcinus orca|killer whales?|orcas?)\b", species)
    )
    structured_match = bool(
        re.search(
            r"\b(srkw|southern residents?|bigg(?:'s)?|transients?|j\s*pod|k\s*pod|l\s*pod)\b",
            structured,
        )
    )
    return species_match or structured_match


def _negated(text: str, match_start: int) -> bool:
    prefix = text[max(0, match_start - 24) : match_start].lower()
    return bool(re.search(r"\b(not|no|unlikely|isn'?t|wasn'?t)\s+(an?\s+)?$", prefix))


def _association(
    row: Mapping[str, Any],
    kind: str,
    value: str,
    text: str,
    field: str,
    rule: str,
    confidence: str,
    priority: int,
) -> dict[str, Any]:
    return {
        "SOURCE_RECORD_ID": str(row["SOURCE_RECORD_ID"]),
        "SOURCE": str(row["SOURCE"]),
        "ASSOCIATION_KIND": kind,
        "ASSOCIATION_VALUE": value,
        "EVIDENCE_TEXT": text[:1000],
        "EVIDENCE_FIELD": field,
        "RULE_ID": rule,
        "CONFIDENCE": confidence,
        "CONFLICTING": False,
        "_PRIORITY": priority,
    }


def _evidence(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    fields = (
        ("POD_ECOTYPE_RAW", row["POD_ECOTYPE_RAW"], True),
        ("DESCRIPTION_RAW", row["DESCRIPTION_RAW"], False),
        ("SPECIES_RAW", row["SPECIES_RAW"], False),
    )
    ecotype_rules = (
        ("SRKW", r"\b(srkw|southern residents?|southern resident killer whales?)\b"),
        ("NRKW", r"\b(nrkw|northern residents?)\b"),
        ("TRANSIENT", r"\b(transients?|bigg'?s?)\b"),
        ("OFFSHORE", r"\boffshores?\b"),
    )
    for field, raw, structured in fields:
        text = str(raw) if _present(raw) else ""
        for value, pattern in ecotype_rules:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if not structured and _negated(text, match.start()):
                    continue
                evidence.append(
                    _association(
                        row,
                        "ECOTYPE",
                        value,
                        text,
                        field,
                        f"ecotype.{value.lower()}.{'structured' if structured else 'text'}",
                        "EXPLICIT",
                        1 if structured else 3,
                    )
                )
        for member in sorted(set(re.findall(r"\b([JKL]\d+[A-Z]?)\b", text.upper()))):
            evidence.extend(
                [
                    _association(
                        row,
                        "MEMBER",
                        member,
                        text,
                        field,
                        "member.srkw_id",
                        "STRONG",
                        2,
                    ),
                    _association(
                        row,
                        "POD",
                        member[0],
                        text,
                        field,
                        "pod.from_member",
                        "STRONG",
                        2,
                    ),
                    _association(
                        row,
                        "ECOTYPE",
                        "SRKW",
                        text,
                        field,
                        "ecotype.from_member",
                        "STRONG",
                        2,
                    ),
                ]
            )
        pod_pattern = (
            r"\b([JKL])(?:\s*[- ]?PODS?)?\b"
            if structured
            else r"\b([JKL])\s*[- ]?PODS?\b"
        )
        for pod in sorted(set(re.findall(pod_pattern, text.upper()))):
            evidence.extend(
                [
                    _association(
                        row,
                        "POD",
                        pod,
                        text,
                        field,
                        "pod.explicit",
                        "STRONG",
                        1 if structured else 2,
                    ),
                    _association(
                        row,
                        "ECOTYPE",
                        "SRKW",
                        text,
                        field,
                        "ecotype.from_pod",
                        "STRONG",
                        1 if structured else 2,
                    ),
                ]
            )
        for group in sorted(set(re.findall(r"\b(T\d+[A-Z]?)\b", text.upper()))):
            evidence.extend(
                [
                    _association(
                        row,
                        "SOCIAL_GROUP",
                        group,
                        text,
                        field,
                        "group.transient_id",
                        "STRONG",
                        2,
                    ),
                    _association(
                        row,
                        "ECOTYPE",
                        "TRANSIENT",
                        text,
                        field,
                        "ecotype.from_group",
                        "STRONG",
                        2,
                    ),
                ]
            )
    if str(row["SOURCE"]).upper() == "GBIF" and _present(row["SOURCE_PAYLOAD"]):
        try:
            payload = json.loads(str(row["SOURCE_PAYLOAD"]))
        except json.JSONDecodeError:
            payload = {}
        occurrences = (
            payload.get("occurrences", []) if isinstance(payload, dict) else []
        )
        dataset_id = (
            str(row["SOURCE_DATASET_ID"])
            if _present(row["SOURCE_DATASET_ID"])
            else "UNKNOWN_DATASET"
        )
        for occurrence in occurrences if isinstance(occurrences, list) else []:
            if not isinstance(occurrence, dict):
                continue
            for field in ("individualID", "organismID"):
                if not _present(occurrence.get(field)):
                    continue
                raw_value = str(occurrence[field]).strip()
                evidence.append(
                    _association(
                        row,
                        "INDIVIDUAL",
                        f"{dataset_id}:{raw_value}",
                        raw_value,
                        f"SOURCE_PAYLOAD.{field}",
                        "individual.gbif_namespaced",
                        "EXPLICIT",
                        4,
                    )
                )
    unique = {
        (
            item["SOURCE_RECORD_ID"],
            item["ASSOCIATION_KIND"],
            item["ASSOCIATION_VALUE"],
            item["RULE_ID"],
            item["EVIDENCE_FIELD"],
        ): item
        for item in evidence
    }
    return list(unique.values())


def _ecotype_detail(evidence: list[dict[str, Any]]) -> str:
    ecotype = [item for item in evidence if item["ASSOCIATION_KIND"] == "ECOTYPE"]
    if not ecotype:
        return "UNKNOWN"
    gbif_values = {
        item["ASSOCIATION_VALUE"]
        for item in ecotype
        if str(item.get("SOURCE", "")).upper() == "GBIF"
    }
    if len(gbif_values) > 1:
        return "MIXED"
    priority = min(int(item["_PRIORITY"]) for item in ecotype)
    values = {
        item["ASSOCIATION_VALUE"] for item in ecotype if item["_PRIORITY"] == priority
    }
    return next(iter(values)) if len(values) == 1 else "MIXED"


SOURCE_PRIORITY = {
    "TWM": 0,
    "ACARTIA": 1,
    "MAPLIFY": 2,
    "INATURALIST": 3,
    "CWR": 4,
    "GBIF": 5,
}

DETAIL_TO_BUCKET = {
    "SRKW": "SRKW",
    "TRANSIENT": "TRANSIENT",
    "NRKW": "OTHER",
    "OFFSHORE": "OTHER",
    "UNKNOWN": "OTHER",
    "MIXED": "OTHER",
}


from marine_mammal_toolkit.tools.schemas.species import ObservationPolicy

OBSERVATION_POLICY = ObservationPolicy(
    accepts=_is_killer_whale,
    evidence=_evidence,
    classify=_ecotype_detail,
    source_priority=SOURCE_PRIORITY,
    detail_to_bucket=DETAIL_TO_BUCKET,
    common_name="Killer Whale",
    scientific_name="Orcinus orca",
    identity_prefix="orca:v4:",
)

from marine_mammal_toolkit.tools.schemas.species import CountPolicy

COUNT_POLICY = CountPolicy(
    expected_columns={
        "SRKW": "EXPECTED_SRKW_COUNT",
        "TRANSIENT": "EXPECTED_TRANSIENT_COUNT",
        "UNKNOWN": "EXPECTED_UNKNOWN_COUNT",
    },
    other_column="EXPECTED_OTHER_COUNT",
    unknown_label="UNKNOWN",
    other_bucket="OTHER",
    buckets=("SRKW", "TRANSIENT", "OTHER"),
)
