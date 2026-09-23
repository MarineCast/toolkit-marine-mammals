from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import h3
import pandas as pd
import pytest
import requests
import yaml

from marine_mammal_toolkit.tools._core.config.paths import project_root
from marine_mammal_toolkit.tools.observations.collect import pipeline as collection
from marine_mammal_toolkit.tools.observations.process.adapters import _source_record
from marine_mammal_toolkit.tools.observations.process.adapters import adapt_gbif
from marine_mammal_toolkit.tools.observations.collect.pipeline import collect_sightings
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.schemas.observations import SightingsCollectionRequest
from marine_mammal_toolkit.tools.observations.post_process.counts import _daily_counts
from marine_mammal_toolkit.tools.observations.impute.encounters import (
    attach_encounter_ids,
)
from marine_mammal_toolkit.tools.observations.impute.pipeline import (
    record_training_input_provenance,
)
from marine_mammal_toolkit.tools.observations.impute.pipeline import (
    validate_training_input_provenance,
)
from marine_mammal_toolkit.tools.observations.process.pipeline import _cluster
from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    _ecotype_detail,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    _evidence,
)
from marine_mammal_toolkit.tools.observations.process.pipeline import _materialize
from marine_mammal_toolkit.tools.observations.process.pipeline import _normalize_records
from marine_mammal_toolkit.tools.observations.process.pipeline import (
    _resolve_identities,
)

HAPPYWHALE = "e0da2d53-86f0-440c-a11a-42ffb0b3fd3e"
INATURALIST = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"


def _config():
    return load_sightings_config(project_root() / "config/data/sightings.yaml")[1]


def _policy(*, use_class: str = "INTERNAL_ONLY") -> dict[str, object]:
    return {
        "title": "Happywhale - Killer whale in North Pacific Ocean",
        "event_id_is_encounter": True,
        "happywhale_exact_fallback": True,
        "use_class": use_class,
        "license": "CC_BY_NC_4_0" if use_class == "INTERNAL_ONLY" else "CC_BY_4_0",
    }


def _row(
    gbif_id: str,
    *,
    event_id: str | None = "encounter-1",
    event_date: str = "2025-06-03T18:30:00Z",
    day: int | None = 3,
    individual_id: str | None = None,
    uncertainty: float | None = 25.0,
    dataset_key: str = HAPPYWHALE,
) -> dict[str, object]:
    row: dict[str, object] = {
        "gbifID": gbif_id,
        "occurrenceID": f"occurrence-{gbif_id}",
        "datasetKey": dataset_key,
        "scientificName": "Orcinus orca (Linnaeus, 1758)",
        "year": 2025,
        "month": 6,
        "eventDate": event_date,
        "decimalLatitude": 48.5,
        "decimalLongitude": -123.2,
        "license": "CC_BY_NC_4_0",
    }
    if event_id is not None:
        row["eventID"] = event_id
    if day is not None:
        row["day"] = day
    if individual_id is not None:
        row["individualID"] = individual_id
    if uncertainty is not None:
        row["coordinateUncertaintyInMeters"] = uncertainty
    return row


def _normalization_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["SOURCE_RETRIEVED_AT_UTC"] = pd.Timestamp("2025-06-10T00:00:00Z")
    frame["SOURCE_PAYLOAD_CORRECTED"] = False
    frame["LAST_CORRECTED_AT_UTC"] = pd.Timestamp("2025-06-10T00:00:00Z")
    return frame


def test_gbif_configuration_is_curated_and_excludes_inaturalist():
    source = _config().collection.sources["gbif"]
    allowed = {dataset.key for dataset in source.dataset_allowlist}
    assert source.taxon_key == "74SZC"
    assert source.checklist_key == "7ddf754f-d193-4cc9-b351-99906754a03b"
    assert source.basis_of_record == "HUMAN_OBSERVATION"
    assert source.occurrence_status == "PRESENT"
    assert source.require_no_geospatial_issue
    assert HAPPYWHALE in allowed
    assert INATURALIST not in allowed
    assert INATURALIST in source.excluded_dataset_keys


def test_gbif_pagination_reconciles_total_and_unique_ids(monkeypatch):
    config = _config()
    allowed = {item.key for item in config.collection.sources["gbif"].dataset_allowlist}

    def response(url, **kwargs):
        if url.endswith("/occurrence/search"):
            offset = kwargs["params"]["offset"]
            results = (
                [
                    {"gbifID": str(index), "datasetKey": HAPPYWHALE}
                    for index in range(300)
                ]
                if offset == 0
                else [{"gbifID": "300", "datasetKey": HAPPYWHALE}]
            )
            return {"count": 301, "endOfRecords": offset > 0, "results": results}
        key = url.rsplit("/", 1)[-1]
        assert key in allowed
        return {"key": key, "title": key, "license": "CC_BY_4_0"}

    monkeypatch.setattr(collection, "_request_json", response)
    rows, metadata = collection._fetch_gbif(
        config, date(2025, 1, 1), date(2025, 12, 31)
    )
    assert len(rows) == 301
    assert len({str(row["gbifID"]) for row in rows}) == 301
    assert {str(item["key"]) for item in metadata} == allowed


def test_gbif_pagination_rejects_changed_total(monkeypatch):
    config = _config()

    def response(url, **kwargs):
        offset = kwargs["params"]["offset"]
        results = (
            [{"gbifID": str(index), "datasetKey": HAPPYWHALE} for index in range(300)]
            if offset == 0
            else [{"gbifID": "300", "datasetKey": HAPPYWHALE}]
        )
        return {
            "count": 301 if offset == 0 else 302,
            "endOfRecords": offset > 0,
            "results": results,
        }

    monkeypatch.setattr(collection, "_request_json", response)
    with pytest.raises(ValueError, match="changed during pagination"):
        collection._fetch_gbif(config, date(2025, 1, 1), date(2025, 12, 31))


def test_gbif_rejects_an_inaturalist_mirror_from_the_response(monkeypatch):
    config = _config()
    monkeypatch.setattr(
        collection,
        "_request_json",
        lambda *args, **kwargs: {
            "count": 1,
            "endOfRecords": True,
            "results": [{"gbifID": "1", "datasetKey": INATURALIST}],
        },
    )
    with pytest.raises(ValueError, match="non-allowlisted"):
        collection._fetch_gbif(config, date(2025, 1, 1), date(2025, 12, 31))


def test_request_json_retries_transient_failures(monkeypatch):
    attempts = 0

    class Response:
        status_code = 200
        headers: dict[str, str] = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def get(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise requests.Timeout("temporary")
        return Response()

    monkeypatch.setattr(collection.requests, "get", get)
    monkeypatch.setattr(collection.time, "sleep", lambda *_: None)
    assert collection._request_json(
        "https://example.test",
        params={},
        token=None,
        timeout=1,
        retries=1,
    ) == {"ok": True}
    assert attempts == 2


def test_happywhale_event_becomes_one_internal_source_report():
    rows = adapt_gbif(
        [_row("1", individual_id="J35"), _row("2", individual_id="orca-123")],
        dataset_policies={HAPPYWHALE: _policy()},
        max_coordinate_uncertainty_m=5000.0,
        max_event_spread_km=5.0,
        scientific_name="Orcinus orca",
    )
    assert len(rows) == 1
    assert rows[0]["SOURCE_OCCURRENCE_COUNT"] == 2
    assert rows[0]["SOURCE_OCCURRENCE_IDS"] == ["occurrence-1", "occurrence-2"]
    assert rows[0]["SOURCE_EVENT_ID"] == "encounter-1"
    assert rows[0]["SOURCE_USE_CLASS"] == "INTERNAL_ONLY"
    assert rows[0]["SOURCE_QC_STATUS"] == "ACCEPTED"
    evidence = _evidence(pd.Series(rows[0]))
    assert _ecotype_detail(evidence) == "SRKW"
    assert any(
        item["ASSOCIATION_KIND"] == "INDIVIDUAL"
        and item["ASSOCIATION_VALUE"] == f"{HAPPYWHALE}:orca-123"
        for item in evidence
    )


def test_gbif_conflicting_biological_evidence_is_mixed():
    rows = adapt_gbif(
        [_row("1", individual_id="J35"), _row("2", individual_id="T65A")],
        dataset_policies={HAPPYWHALE: _policy()},
        max_coordinate_uncertainty_m=5000.0,
        max_event_spread_km=5.0,
        scientific_name="Orcinus orca",
    )
    assert _ecotype_detail(_evidence(pd.Series(rows[0]))) == "MIXED"


def test_date_only_happywhale_rows_are_not_heuristically_collapsed():
    rows = adapt_gbif(
        [
            _row("1", event_id=None, event_date="2025-06-03"),
            _row("2", event_id=None, event_date="2025-06-03"),
        ],
        dataset_policies={HAPPYWHALE: _policy()},
        max_coordinate_uncertainty_m=5000.0,
        max_event_spread_km=5.0,
        scientific_name="Orcinus orca",
    )
    assert len(rows) == 2
    assert all(row["SOURCE_QC_STATUS"] == "ACCEPTED" for row in rows)


@pytest.mark.parametrize(
    ("items", "reason"),
    [
        ([_row("1", uncertainty=5001.0)], "COORDINATE_UNCERTAINTY_EXCEEDS_LIMIT"),
        (
            [_row("1"), _row("2", day=None)],
            "MISSING_OR_CONFLICTING_DAY_PRECISION",
        ),
    ],
)
def test_gbif_invalid_events_are_quarantined(items, reason):
    row = adapt_gbif(
        items,
        dataset_policies={HAPPYWHALE: _policy()},
        max_coordinate_uncertainty_m=5000.0,
        max_event_spread_km=5.0,
        scientific_name="Orcinus orca",
    )[0]
    assert row["SOURCE_QC_STATUS"] == "QUARANTINED"
    assert reason in str(row["SOURCE_QC_DETAIL"])


def test_gbif_public_class_requires_safe_actual_license():
    unsafe = adapt_gbif(
        [_row("1")],
        dataset_policies={HAPPYWHALE: _policy(use_class="REDISTRIBUTABLE")},
        max_coordinate_uncertainty_m=5000.0,
        max_event_spread_km=5.0,
        scientific_name="Orcinus orca",
    )[0]
    safe_row = _row("2")
    safe_row["license"] = "CC_BY_4_0"
    safe = adapt_gbif(
        [safe_row],
        dataset_policies={HAPPYWHALE: _policy(use_class="REDISTRIBUTABLE")},
        max_coordinate_uncertainty_m=5000.0,
        max_event_spread_km=5.0,
        scientific_name="Orcinus orca",
    )[0]
    assert unsafe["SOURCE_USE_CLASS"] == "INTERNAL_ONLY"
    assert safe["SOURCE_USE_CLASS"] == "REDISTRIBUTABLE"


@pytest.mark.parametrize("source", ["TWM", "ACARTIA", "MAPLIFY", "INATURALIST"])
def test_gbif_cross_source_merge_preserves_original_source_priority(source, tmp_path):
    gbif = adapt_gbif(
        [_row("1")],
        dataset_policies={HAPPYWHALE: _policy()},
        max_coordinate_uncertainty_m=5000.0,
        max_event_spread_km=5.0,
        scientific_name="Orcinus orca",
    )[0]
    original = _source_record(
        source,
        "original-1",
        {"id": "original-1"},
        observed_at="2025-06-03T18:35:00Z",
        observed_date="2025-06-03",
        latitude=48.5005,
        longitude=-123.2005,
        species="Orcinus orca",
    )
    records, audit = _normalize_records(
        _normalization_frame([gbif, original]), _config()
    )
    groups = _cluster(records, _config(), audit)
    assert len(groups) == 1
    assert len(groups[0]) == 2
    assignment_path = tmp_path / "assignments.parquet"
    pd.DataFrame(
        {
            "SOURCE_RECORD_ID": [original["SOURCE_RECORD_ID"]],
            "OBSERVATION_ID": ["existing-observation"],
            "FIRST_SEEN_RUN_ID": ["prior-run"],
        }
    ).to_parquet(assignment_path)
    identifiers, *_ = _resolve_identities(
        groups,
        assignment_path,
        tmp_path / "aliases.parquet",
        tmp_path / "lineage.parquet",
        "test-run",
        policy=OBSERVATION_POLICY,
    )
    assert identifiers == ["existing-observation"]
    observations, _ = _materialize(
        groups, audit, identifiers, policy=OBSERVATION_POLICY
    )
    assert observations[0]["SOURCE"] == source
    assert observations[0]["SOURCE_REPORT_COUNT"] == 2
    assert observations[0]["SOURCE_OCCURRENCE_COUNT"] == 2
    assert not observations[0]["PUBLIC_RELEASE_ELIGIBLE"]
    assert any(item["REASON"] == "CROSS_SOURCE_CLUSTER" for item in audit)


def test_one_happywhale_event_counts_once_not_per_occurrence():
    gbif = adapt_gbif(
        [_row("1"), _row("2")],
        dataset_policies={HAPPYWHALE: _policy()},
        max_coordinate_uncertainty_m=5000.0,
        max_event_spread_km=5.0,
        scientific_name="Orcinus orca",
    )[0]
    records, audit = _normalize_records(_normalization_frame([gbif]), _config())
    groups = _cluster(records, _config(), audit)
    observations, associations = _materialize(
        groups, audit, ["observation-1"], policy=OBSERVATION_POLICY
    )
    observation_frame = pd.DataFrame(observations)
    association_frame = pd.DataFrame(
        associations,
        columns=["OBSERVATION_ID", "ASSOCIATION_KIND", "ASSOCIATION_VALUE"],
    )
    cell = h3.latlng_to_cell(48.5, -123.2, 4)
    _bucket, _detail, total, _pod, _excluded = _daily_counts(
        observation_frame, association_frame, {cell}, 4, policy=COUNT_POLICY
    )
    assert observation_frame.iloc[0].SOURCE_REPORT_COUNT == 1
    assert observation_frame.iloc[0].SOURCE_OCCURRENCE_COUNT == 2
    assert total.SIGHTING_COUNT.sum() == 1
    assert total.SOURCE_REPORT_COUNT.sum() == 1


def test_gbif_collection_is_full_replace_and_reuses_offline_snapshot(
    monkeypatch, tmp_path
):
    payload = yaml.safe_load(
        (project_root() / "config/data/sightings.yaml").read_text(encoding="utf-8")
    )
    for name, source in payload["collection"]["sources"].items():
        source["enabled"] = name == "gbif"
    payload["collection"]["sources"]["gbif"]["dataset_allowlist"] = [
        {
            "key": HAPPYWHALE,
            "title": "Happywhale - Killer whale in North Pacific Ocean",
            "event_id_is_encounter": True,
            "happywhale_exact_fallback": True,
            "use_class": "INTERNAL_ONLY",
        }
    ]
    config_path = tmp_path / "sightings.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    metadata = {
        "key": HAPPYWHALE,
        "title": "Happywhale - Killer whale in North Pacific Ocean",
        "license": "CC_BY_NC_4_0",
        "citation": {"text": "Fixture citation"},
        "_orcacast_policy": _policy(),
    }
    requested_windows = []

    def fetch(_config, start, end):
        requested_windows.append((start, end))
        return [_row("1")], [metadata]

    monkeypatch.setattr(collection, "_fetch_gbif", fetch)
    online = collect_sightings(
        SightingsCollectionRequest(
            config=config_path,
            data_root=tmp_path / "data",
            artifact_root=tmp_path / "artifacts",
            output_root=tmp_path / "outputs",
            run_id="online",
            end_date=date(2025, 12, 31),
            force=True,
        )
    )
    snapshot = online.outputs[0].path
    snapshot_metadata = yaml.safe_load((snapshot / "snapshot.json").read_text())
    assert snapshot_metadata["snapshot_mode"] == "FULL_REPLACE"
    assert snapshot_metadata["occurrence_row_count"] == 1
    assert snapshot_metadata["admitted_occurrence_count"] == 1
    assert snapshot_metadata["quarantined_occurrence_count"] == 0
    assert {"payload.json", "datasets.json", "gbif_policy.json"} <= set(
        snapshot_metadata["file_checksums"]
    )
    assert requested_windows == [(date(1980, 1, 1), date(2025, 12, 31))]

    updated = collect_sightings(
        SightingsCollectionRequest(
            config=config_path,
            data_root=tmp_path / "data",
            artifact_root=tmp_path / "artifacts",
            output_root=tmp_path / "outputs",
            run_id="update",
            end_date=date(2026, 1, 2),
            force=True,
        )
    )
    updated_snapshot = updated.outputs[0].path
    updated_metadata = yaml.safe_load((updated_snapshot / "snapshot.json").read_text())
    assert updated_metadata["snapshot_mode"] == "DELTA_UPSERT"
    assert requested_windows[-1] == (date(2025, 12, 29), date(2026, 1, 2))

    offline = collect_sightings(
        SightingsCollectionRequest(
            config=config_path,
            data_root=tmp_path / "data",
            artifact_root=tmp_path / "artifacts",
            output_root=tmp_path / "outputs",
            run_id="offline",
            end_date=date(2026, 1, 2),
            offline=True,
            force=True,
        )
    )
    assert offline.outputs[0].path == updated_snapshot
    assert offline.outputs[0].freshness == "offline"


def test_encounter_ids_are_deterministic_under_row_reordering():
    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": ["a", "b", "c"],
            "SIGHTING_DATE": ["2025-06-03"] * 3,
            "SOURCE_EVENT_AT_UTC": [pd.NaT] * 3,
            "SOURCE_TIME_PRECISION": ["DATE"] * 3,
            "LATITUDE": [48.5, 48.501, 50.0],
            "LONGITUDE": [-123.2, -123.201, -125.0],
        }
    )
    first = attach_encounter_ids(frame).set_index("OBSERVATION_ID")
    second = attach_encounter_ids(frame.iloc[::-1]).set_index("OBSERVATION_ID")
    pd.testing.assert_series_equal(
        first["ENCOUNTER_ID"].sort_index(), second["ENCOUNTER_ID"].sort_index()
    )
    pd.testing.assert_series_equal(
        first["ENCOUNTER_SIZE"].sort_index(), second["ENCOUNTER_SIZE"].sort_index()
    )


def test_model_input_provenance_fails_after_artifact_change(tmp_path):
    observations = tmp_path / "observations.parquet"
    associations = tmp_path / "associations.parquet"
    observations.write_bytes(b"observations-v1")
    associations.write_bytes(b"associations-v1")
    imputer = SimpleNamespace(training_summary_={})
    record_training_input_provenance(
        imputer,
        observations_path=observations,
        associations_path=associations,
    )
    validate_training_input_provenance(
        imputer,
        observations_path=observations,
        associations_path=associations,
    )
    observations.write_bytes(b"observations-v2")
    with pytest.raises(ValueError, match="checksum"):
        validate_training_input_provenance(
            imputer,
            observations_path=observations,
            associations_path=associations,
        )


from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    OBSERVATION_POLICY,
)

from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    COUNT_POLICY,
)
