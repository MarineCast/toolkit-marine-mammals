from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from marine_mammal_toolkit.tools._core.config.paths import project_root
from marine_mammal_toolkit.tools.observations.collect import pipeline as collection
from marine_mammal_toolkit.tools.observations.process.adapters import _source_record
from marine_mammal_toolkit.tools.observations.process.adapters import adapt_acartia
from marine_mammal_toolkit.tools.observations.process.adapters import adapt_inaturalist
from marine_mammal_toolkit.tools.observations.collect.pipeline import collect_sightings
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.schemas.observations import SOURCE_HISTORY_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import SOURCE_RECORD_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import SightingsCollectionRequest
from marine_mammal_toolkit.tools.observations.process.pipeline import (
    _assemble_source_state,
)
from marine_mammal_toolkit.tools.observations.process.pipeline import _normalize_records


def _config_payload(*enabled: str) -> dict:
    payload = yaml.safe_load(
        (project_root() / "config/data/sightings.yaml").read_text(encoding="utf-8")
    )
    for name, source in payload["collection"]["sources"].items():
        source["enabled"] = name in enabled
    return payload


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "sightings.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_missing_license_policy_fails_closed(tmp_path: Path) -> None:
    payload = _config_payload("twm")
    payload["collection"]["sources"]["twm"].pop("source_license")
    path = _write_config(tmp_path, payload)
    with pytest.raises(ValueError, match="explicit source_license"):
        load_sightings_config(path)

    row = _source_record("TEST", "1", {"id": "1"})
    assert row["SOURCE_USE_CLASS"] == "INTERNAL_ONLY"


def test_redistributable_source_requires_complete_reviewed_license_evidence(
    tmp_path: Path,
) -> None:
    payload = _config_payload("twm")
    source = payload["collection"]["sources"]["twm"]
    source["source_license"] = "CC-BY-4.0"
    source["source_use_class"] = "REDISTRIBUTABLE"
    with pytest.raises(ValueError, match="lacks license evidence"):
        load_sightings_config(_write_config(tmp_path, payload))

    source.update(
        {
            "source_license_terms_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_attribution": "Fixture contributor",
            "source_license_reviewed_at": "2026-08-19",
            "source_license_version": "4.0",
            "source_license_jurisdiction": "international",
        }
    )
    _document, config = load_sightings_config(_write_config(tmp_path, payload))
    assert config.collection.sources["twm"].source_use_class == "REDISTRIBUTABLE"


def test_inaturalist_quarantines_captive_and_imprecise_records() -> None:
    rows = adapt_inaturalist(
        [
            {
                "id": 1,
                "observed_on": "2025-01-01",
                "geojson": {"coordinates": [-123.0, 48.0]},
                "taxon": {"name": "Orcinus orca"},
                "quality_grade": "research",
                "captive": True,
                "positional_accuracy": 6001,
                "geoprivacy": "open",
            }
        ],
        source_license="UNKNOWN",
        source_use_class="INTERNAL_ONLY",
        max_coordinate_uncertainty_m=5000.0,
    )
    row = rows[0]
    detail = json.loads(row["SOURCE_QC_DETAIL"])
    assert row["SOURCE_QC_STATUS"] == "QUARANTINED"
    assert row["COORDINATE_UNCERTAINTY_M"] == 6001
    assert detail["quality_grade"] == "research"
    assert detail["reasons"] == [
        "CAPTIVE_OR_CULTIVATED",
        "COORDINATE_UNCERTAINTY_EXCEEDS_LIMIT",
    ]
    assert json.loads(row["SOURCE_PAYLOAD"])["positional_accuracy"] == 6001


def test_twm_uniquely_valid_west_sign_is_recovered_and_audited() -> None:
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    row = _source_record(
        "TWM",
        "west-1",
        {"id": "west-1"},
        observed_date="2025-01-02",
        latitude="48.5",
        longitude="123.2",
        species="Orcinus orca",
    )
    row.update(
        {
            "SOURCE_RETRIEVED_AT_UTC": pd.Timestamp("2025-01-03T00:00:00Z"),
            "SOURCE_PAYLOAD_CORRECTED": False,
            "LAST_CORRECTED_AT_UTC": pd.Timestamp("2025-01-03T00:00:00Z"),
        }
    )

    records, audit = _normalize_records(pd.DataFrame([row]), config)

    assert len(records) == 1
    assert records[0]["LONGITUDE"] == pytest.approx(-123.2)
    assert records[0]["COORDINATE_TRANSFORM"] == "WEST_SIGN_INFERRED"
    assert any(item["REASON"] == "WEST_SIGN_INFERRED" for item in audit)


def test_acartia_event_time_is_not_used_as_record_availability() -> None:
    row = adapt_acartia(
        [
            {
                "id": "1",
                "created": "2020-06-01T12:00:00-07:00",
                "type": "Orcinus orca",
                "latitude": 48.5,
                "longitude": -123.2,
            }
        ]
    )[0]
    assert row["OBSERVED_AT_RAW"] == "2020-06-01T12:00:00-07:00"
    assert row["CREATED_AT_RAW"] is None


def test_retry_after_supports_http_dates() -> None:
    now = datetime(2015, 10, 21, 7, 26, tzinfo=timezone.utc)
    assert collection._retry_after_seconds(
        "Wed, 21 Oct 2015 07:28:00 GMT", now=now
    ) == pytest.approx(120.0)
    assert collection._retry_after_seconds("2.5", now=now) == pytest.approx(2.5)


def test_raw_and_semantic_revision_hashes_have_distinct_contracts(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    payload_path = snapshot / "payload.json"
    payload_path.write_text('{"results":[{"id":1}]}', encoding="utf-8")
    raw_first = collection._raw_response_hash(snapshot)
    payload_path.write_text('{ "results": [ { "id": 1 } ] }\n', encoding="utf-8")
    raw_second = collection._raw_response_hash(snapshot)
    assert raw_first != raw_second

    left = _source_record("TEST", "2", {"id": 2})
    right = _source_record("TEST", "1", {"id": 1})
    first = pa.Table.from_pylist([left, right], schema=SOURCE_RECORD_SCHEMA)
    reordered = pa.Table.from_pylist([right, left], schema=SOURCE_RECORD_SCHEMA)
    assert collection._semantic_content_hash(
        first
    ) == collection._semantic_content_hash(reordered)


def test_api_completeness_checks_reject_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    monkeypatch.setattr(
        collection,
        "_request_json",
        lambda *args, **kwargs: {"total_results": 2, "results": [{"id": 1}, {"id": 1}]},
    )
    with pytest.raises(ValueError, match="duplicate source IDs"):
        collection._fetch_inaturalist(config, date(2025, 1, 1), date(2025, 1, 2))

    monkeypatch.setattr(
        collection,
        "_request_json",
        lambda *args, **kwargs: {
            "count": 2,
            "results": [
                {"source": "whale_alert", "id": 1},
                {"source": "whale_alert", "id": 1},
            ],
        },
    )
    with pytest.raises(ValueError, match="duplicate composite source IDs"):
        collection._fetch_maplify(config, date(2025, 1, 1), date(2025, 1, 2))


def test_maplify_ids_are_scoped_to_the_contributing_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    rows = [
        {"source": "whale_alert", "id": 1},
        {"source": "wras", "id": 1},
    ]
    monkeypatch.setattr(
        collection,
        "_request_json",
        lambda *args, **kwargs: {"count": len(rows), "results": rows},
    )

    assert collection._fetch_maplify(config, date(2025, 1, 1), date(2025, 1, 2)) == rows


def test_collection_window_rejects_reverse_and_future_ranges() -> None:
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    with pytest.raises(ValueError, match="start_date cannot follow"):
        collection._validate_collection_window(
            SimpleNamespace(start_date=date(2025, 1, 2), end_date=date(2025, 1, 1)),
            config,
        )
    with pytest.raises(ValueError, match="cannot be in the future"):
        collection._validate_collection_window(
            SimpleNamespace(start_date=None, end_date=date(2999, 1, 1)),
            config,
        )


def test_first_bounded_window_initializes_and_existing_windows_upsert() -> None:
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    assert (
        collection._range_snapshot_mode("gbif", date(1980, 1, 1), config)
        == "FULL_REPLACE"
    )
    assert (
        collection._range_snapshot_mode(
            "gbif", date(1980, 1, 1), config, Path("prior-snapshot")
        )
        == "DELTA_UPSERT"
    )
    assert (
        collection._range_snapshot_mode("gbif", date(2020, 1, 1), config)
        == "FULL_REPLACE"
    )
    assert (
        collection._range_snapshot_mode("inaturalist", date(2020, 1, 1), config)
        == "FULL_REPLACE"
    )


def test_acartia_archive_is_recursive_and_later_snapshots_are_non_destructive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "acartia"
    archive = source_root / "Archive"
    archive.mkdir(parents=True)
    columns = "id,type,sightdate,latitude,longitude\n"
    (source_root / "current.csv").write_text(
        columns + "current,Orcinus orca,2025-01-01,48.5,-123.2\n",
        encoding="utf-8",
    )
    (archive / "historical.csv").write_text(
        columns + "historic,Orcinus orca,2020-01-01,48.4,-123.1\n",
        encoding="utf-8",
    )
    payload = _config_payload("acartia")
    payload["collection"]["sources"]["acartia"]["local_path"] = str(source_root)
    config_path = _write_config(tmp_path, payload)
    monkeypatch.setattr(collection, "_fetch_acartia", lambda _config: [])
    request = SightingsCollectionRequest(
        config=config_path,
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
        end_date=date(2025, 1, 2),
        force=True,
    )

    first = collect_sightings(request)
    first_metadata = json.loads((first.outputs[0].path / "snapshot.json").read_text())
    assert first_metadata["snapshot_mode"] == "FULL_REPLACE"
    assert first_metadata["row_count"] == 2
    assert "Archive/historical.csv" in first_metadata["file_checksums"]
    raw_root = request.data_root / "raw/whale/sightings"
    cohort_pointer = json.loads((raw_root / "manifests/latest.json").read_text())
    source_pointer = json.loads((raw_root / "acartia/latest.json").read_text())
    assert not Path(cohort_pointer["manifest"]).is_absolute()
    assert not Path(source_pointer["snapshot"]).is_absolute()

    second = collect_sightings(request)
    second_metadata = json.loads((second.outputs[0].path / "snapshot.json").read_text())
    assert second_metadata["snapshot_mode"] == "DELTA_UPSERT"


def test_failed_later_source_does_not_advance_any_latest_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    twm_root = tmp_path / "twm"
    twm_root.mkdir()
    (twm_root / "twm.csv").write_text(
        "sightdate,latitude,longitude\n2025-01-01,48.5,-123.2\n",
        encoding="utf-8",
    )
    payload = _config_payload("twm", "acartia")
    payload["collection"]["sources"]["twm"]["local_path"] = str(twm_root)
    payload["collection"]["sources"]["acartia"]["local_path"] = str(tmp_path / "empty")
    config_path = _write_config(tmp_path, payload)

    def fail_acartia(_config):
        raise RuntimeError("simulated later-source failure")

    monkeypatch.setattr(collection, "_fetch_acartia", fail_acartia)
    with pytest.raises(RuntimeError, match="later-source failure"):
        collect_sightings(
            SightingsCollectionRequest(
                config=config_path,
                data_root=tmp_path / "data",
                artifact_root=tmp_path / "artifacts",
                output_root=tmp_path / "outputs",
                end_date=date(2025, 1, 2),
            )
        )

    raw_root = tmp_path / "data/raw/whale/sightings"
    assert not (raw_root / "twm/latest.json").exists()
    assert not (raw_root / "manifests/latest.json").exists()


def test_policy_change_propagates_without_moving_first_seen_availability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    processed_root = tmp_path / "processed/sightings/normalized"
    state_root = processed_root / "state"
    state_root.mkdir(parents=True)
    first_seen = datetime(2025, 1, 2, tzinfo=timezone.utc)
    refreshed = datetime(2025, 1, 3, tzinfo=timezone.utc)
    payload = {"id": "1"}
    prior = _source_record(
        "TWM",
        "1",
        payload,
        source_use_class="REDISTRIBUTABLE",
        observed_date="2025-01-01",
        latitude=48.5,
        longitude=-123.2,
        species="Orcinus orca",
    )
    prior.update(
        {
            "SOURCE_RETRIEVED_AT_UTC": first_seen,
            "SOURCE_PAYLOAD_CORRECTED": False,
            "LAST_CORRECTED_AT_UTC": first_seen,
        }
    )
    pq.write_table(
        pa.Table.from_pylist([prior], schema=SOURCE_RECORD_SCHEMA),
        state_root / "source_current.parquet",
    )
    history = {
        **prior,
        "RETRIEVAL_ID": "r1",
        "RETRIEVED_AT": first_seen,
        "PAYLOAD_CHECKSUM": hashlib.sha256(
            str(prior["SOURCE_PAYLOAD"]).encode()
        ).hexdigest(),
        "RAW_SCHEMA_FINGERPRINT": "fixture",
        "SNAPSHOT_MODE": "FULL_REPLACE",
    }
    pq.write_table(
        pa.Table.from_pylist([history], schema=SOURCE_HISTORY_SCHEMA),
        state_root / "source_history.parquet",
    )
    incoming = _source_record(
        "TWM",
        "1",
        payload,
        source_license="UNKNOWN",
        source_use_class="INTERNAL_ONLY",
        observed_date="2025-01-01",
        latitude=48.5,
        longitude=-123.2,
        species="Orcinus orca",
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "snapshot.json").write_text(
        json.dumps(
            {
                "retrieval_id": "r2",
                "retrieved_at": refreshed.isoformat(),
                "snapshot_mode": "FULL_REPLACE",
                "raw_schema_fingerprint": "fixture",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.process.pipeline.adapt_snapshot",
        lambda *_args, **_kwargs: pa.Table.from_pylist(
            [incoming], schema=SOURCE_RECORD_SCHEMA
        ),
    )

    _history, current, _audit, changed, _counts = _assemble_source_state(
        {"twm": snapshot}, processed_root, pd.Timestamp("2025-01-04T00:00:00Z")
    )
    result = current.row(0, named=True)
    assert result["SOURCE_LICENSE"] == "UNKNOWN"
    assert result["SOURCE_USE_CLASS"] == "INTERNAL_ONLY"
    assert result["SOURCE_RETRIEVED_AT_UTC"] == first_seen
    assert changed["whale.sightings.source_records"] is True

    (snapshot / "snapshot.json").write_text(
        json.dumps(
            {
                "retrieval_id": "r0",
                "retrieved_at": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
                "snapshot_mode": "FULL_REPLACE",
                "raw_schema_fingerprint": "fixture",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Refusing out-of-order twm snapshot"):
        _assemble_source_state(
            {"twm": snapshot}, processed_root, pd.Timestamp("2025-01-04T00:00:00Z")
        )
