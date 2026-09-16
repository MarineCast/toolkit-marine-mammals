from __future__ import annotations

import base64
import gzip
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests
import yaml

from marine_mammal_toolkit.tools._core.config.paths import project_root
from marine_mammal_toolkit.tools._core.data import DATASETS
from marine_mammal_toolkit.tools.observations.process.adapters import _source_record
from marine_mammal_toolkit.tools.observations.collect.pipeline import collect_sightings
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.schemas.observations import SightingsCollectionRequest
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import _archive_rows
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import _atlist_rows
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
    _persist_raw_bodies,
)
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import _request_bytes
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import _request_json
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
    collect_cwr_snapshot,
)
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
    parse_archive_index,
)
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
    parse_coordinate_component,
)
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
    parse_labeled_notes,
)
from marine_mammal_toolkit.tools.observations.collect.sources.cwr import (
    precise_local_timestamp,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    SOURCE_PRIORITY,
)
from marine_mammal_toolkit.tools.observations.process.pipeline import _cluster
from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    _evidence,
)
from marine_mammal_toolkit.tools.observations.process.pipeline import _materialize
from marine_mammal_toolkit.tools.observations.process.pipeline import _normalize_records


def _config():
    return load_sightings_config(project_root() / "config/data/sightings.yaml")[1]


def _cwr_settings():
    return _config().collection.sources["cwr"]


def _archive_page(
    url: str,
    *,
    number: str,
    sequence: str = "1",
    start: str | None = "48 30.0/123 15.0",
    end: str | None = "48 31.0/123 14.0",
) -> dict[str, object]:
    fields = {
        "encounter_date": "03-Jun-2020",
        "encounter_sequence": sequence,
        "encounter_number": number,
        "start_time": "09:30",
        "end_time": "10:30",
        "pods_or_ecotype": "Bigg's killer whales T65A",
        "location_description": "Haro Strait",
    }
    if start is not None:
        fields["begin_lat_long"] = start
    if end is not None:
        fields["end_lat_long"] = end
    return {
        "url": url,
        "status": 200,
        "content_type": "text/html",
        "response_bytes": 100,
        "response_sha256": "a" * 64,
        "fields": fields,
        "error": None,
    }


def _archive_payload(entries: list[dict[str, object]], pages: dict[str, object]):
    years = sorted({int(item["source_year"]) for item in entries})
    return {
        "schema_version": 2,
        "archive_index_url": "https://whaleresearch.wixsite.com/archives",
        "index_results": {
            str(year): {
                "url": f"https://whaleresearch.wixsite.com/{year}encounters",
                "response_sha256": "b" * 64,
                "entry_count": sum(
                    int(item["source_year"]) == year for item in entries
                ),
            }
            for year in years
        },
        "index_entries": entries,
        "record_pages": pages,
    }


def _entry(
    number: str,
    url: str,
    *,
    series: str = "encounter",
    sequence: str | None = None,
) -> dict[str, object]:
    return {
        "source_year": 2020,
        "record_series": series,
        "encounter_number": number,
        "index_sequence": sequence,
        "index_date_text": "03-Jun",
        "index_descriptor": "Bigg's killer whales T65A",
        "index_title": f"# {number} • 03-Jun • Bigg's killer whales T65A",
        "record_page_url": url,
    }


def _atlist_payload(markers: list[dict[str, object]]) -> dict[str, object]:
    return {
        "maps": {
            "2024": {
                "year": 2024,
                "map_id": "3d6d0c96-06fb-4087-959c-1ecdcab167af",
                "page_url": "https://www.whaleresearch.com/encounters2024",
                "fields_url": "https://api.atlist.com/fields",
                "markers_url": "https://api.atlist.com/markers",
                "fields_response": {"response_sha256": "c" * 64},
                "markers_response": {"response_sha256": "d" * 64},
                "markers_payload": {"markers": markers},
            }
        }
    }


def _marker(marker_id: str = "marker-1") -> dict[str, object]:
    return {
        "id": marker_id,
        "name": "Encounter #7 - Jun 3, 2024",
        "lat": 48.5,
        "long": -123.25,
        "createdAt": "2024-06-04T18:00:00Z",
        "tags": [{"name": "Bigg's Killer Whales"}],
        "notes": (
            "EncSummary: short summary<br>ObservBegin: 09:30<br>ObservEnd: 10:30<br>"
            "Pods: T65A<br>IDs Encountered: T65A, T65A3<br>LocationDescr: Haro Strait"
        ),
    }


def _normalization_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["SOURCE_RETRIEVED_AT_UTC"] = pd.Timestamp("2025-06-10T00:00:00Z")
    frame["SOURCE_PAYLOAD_CORRECTED"] = False
    frame["LAST_CORRECTED_AT_UTC"] = pd.Timestamp("2025-06-10T00:00:00Z")
    return frame


def test_cwr_configuration_and_catalog_contract():
    config = _config()
    source = config.collection.sources["cwr"]
    assert config.schema_version == 6
    assert source.archive_years == tuple(range(2017, 2024))
    assert set(source.atlist_maps) == {2024, 2025, 2026}
    assert source.archive_fetch_workers == 8
    assert source.timezone == "America/Los_Angeles"
    assert source.source_license == "UNKNOWN"
    assert source.source_use_class == "INTERNAL_ONLY"
    assert DATASETS.get("whale.sightings.source_cwr").schema_version == "5"
    dependencies = DATASETS.get("whale.sightings.source_records").dependencies
    assert "whale.sightings.source_cwr" in {str(item) for item in dependencies}


def test_cwr_configuration_rejects_missing_year(tmp_path):
    payload = yaml.safe_load(
        (project_root() / "config/data/sightings.yaml").read_text(encoding="utf-8")
    )
    payload["collection"]["sources"].pop("cwr")
    path = tmp_path / "sightings.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing source configuration"):
        load_sightings_config(path)


def test_wix_index_parsing_keeps_sequences_and_uav_distinct():
    content = b"""
    <p><a href="https://whaleresearch.wixsite.com/2020encounters/7">#7 Seq 1</a> \xe2\x80\xa2 03-Jun \xe2\x80\xa2 J Pod</p>
    <p><a href="https://whaleresearch.wixsite.com/2020encounters/uav-7">#7</a> \xe2\x80\xa2 03-Jun \xe2\x80\xa2 UAV</p>
    """
    rows = parse_archive_index(2020, content)
    assert [(row["encounter_number"], row["record_series"]) for row in rows] == [
        ("7", "encounter"),
        ("7", "uav_encounter"),
    ]
    assert rows[0]["index_sequence"] == "1"


def test_wix_multisequence_aggregation_and_coordinate_precedence():
    first_url = "https://whaleresearch.wixsite.com/2020encounters/7-1"
    second_url = "https://whaleresearch.wixsite.com/2020encounters/7-2"
    entries = [
        _entry("7", first_url, sequence="1"),
        _entry("7", second_url, sequence="2"),
    ]
    pages = {
        first_url: _archive_page(first_url, number="7", sequence="1", end=None),
        second_url: _archive_page(
            second_url,
            number="7",
            sequence="2",
            start="48 32.0/123 13.0",
            end="48 33.0/123 12.0",
        ),
    }
    row = _archive_rows(_archive_payload(entries, pages), _cwr_settings())[0]
    assert row["SOURCE_NATIVE_ID"] == "wix:2020:encounter:7"
    assert row["SOURCE_OCCURRENCE_COUNT"] == 2
    assert float(row["LATITUDE_RAW"]) == pytest.approx(48.5)
    assert float(row["LONGITUDE_RAW"]) == pytest.approx(-123.25)
    assert "MULTIPLE_SEQUENCE_PAGES_AGGREGATED" in row["SOURCE_QC_DETAIL"]


def test_wix_uses_final_end_only_when_no_start_exists():
    url = "https://whaleresearch.wixsite.com/2020encounters/8"
    row = _archive_rows(
        _archive_payload(
            [_entry("8", url)],
            {url: _archive_page(url, number="8", start=None, end="48 31.0/123 14.0")},
        ),
        _cwr_settings(),
    )[0]
    assert float(row["LATITUDE_RAW"]) == pytest.approx(48 + 31 / 60)
    assert "ARCHIVE_FINAL_SEQUENCE_END_FALLBACK" in row["SOURCE_QC_DETAIL"]


def test_known_wix_index_page_identity_mismatch_withholds_page_fields():
    url = "https://whaleresearch.wixsite.com/2020encounters/2020-1"
    row = _archive_rows(
        _archive_payload([_entry("2", url)], {url: _archive_page(url, number="01")}),
        _cwr_settings(),
    )[0]
    assert row["SOURCE_NATIVE_ID"] == "wix:2020:encounter:2"
    assert row["OBSERVED_DATE_RAW"] == "2020-06-03"
    assert row["LATITUDE_RAW"] is None
    assert row["SOURCE_QC_STATUS"] == "QUARANTINED"
    assert "ARCHIVE_INDEX_PAGE_IDENTITY_MISMATCH" in row["SOURCE_QC_DETAIL"]


def test_cwr_exact_response_bodies_are_persisted_as_deterministic_gzip(tmp_path):
    snapshot = tmp_path / "snapshot"
    body = b"<html>exact provider bytes</html>\n"
    payload = {
        "response": {
            "url": "https://example.test/archive",
            "content_type": "text/html",
            "response_sha256": "fixture",
            "_raw_body_b64": base64.b64encode(body).decode("ascii"),
        }
    }
    count = _persist_raw_bodies(payload, snapshot / "raw_responses", snapshot)
    assert count == 1
    raw_path = snapshot / payload["response"]["raw_response_path"]
    with gzip.open(raw_path, "rb") as handle:
        assert handle.read() == body
    assert "_raw_body_b64" not in payload["response"]


def test_coordinate_conversion_infers_west_and_rejects_invalid_values():
    longitude, method = parse_coordinate_component("123 30.0", "longitude")
    assert longitude == pytest.approx(-123.5)
    assert method == "DEGREES_DECIMAL_MINUTES_WEST_INFERRED_FROM_CWR_DOMAIN"
    assert parse_coordinate_component("999", "latitude")[0] is None


def test_atlist_filtering_notes_native_ids_and_reconciliation():
    markers = [
        _marker(),
        {"id": "visitor", "name": "Visitor Center", "lat": 48, "long": -123},
    ]
    rows, metrics = _atlist_rows(_atlist_payload(markers), _cwr_settings())
    assert len(rows) == 1
    row = rows[0]
    assert row["SOURCE_NATIVE_ID"] == (
        "atlist:3d6d0c96-06fb-4087-959c-1ecdcab167af:marker-1"
    )
    assert row["OBSERVED_DATE_RAW"] == "2024-06-03"
    assert row["OBSERVED_AT_RAW"] == "2024-06-03T09:30:00"
    assert "T65A3" in str(row["POD_ECOTYPE_RAW"])
    assert metrics["raw_marker_count"] == 2
    assert metrics["encounter_marker_count"] == 1
    assert metrics["excluded_marker_count"] == 1


def test_atlist_title_without_year_infers_configured_map_year():
    marker = _marker()
    marker["name"] = "Encounter #15 - February 13"

    rows, _metrics = _atlist_rows(_atlist_payload([marker]), _cwr_settings())

    assert rows[0]["OBSERVED_DATE_RAW"] == "2024-02-13"
    assert "ATLIST_DATE_YEAR_INFERRED" in rows[0]["SOURCE_QC_DETAIL"]


def test_labeled_note_aliases_are_equivalent():
    first = parse_labeled_notes("EncSummary: one\nObservBegin: 08:15")
    second = parse_labeled_notes("Encounter Summary: one\nObservBegin: 08:15")
    assert first == second == {"summary": "one", "start_time": "08:15"}


def test_labeled_notes_accept_equals_and_inline_end_time():
    fields = parse_labeled_notes(
        "Encounter summary: one\n"
        "Start Latitude = 48 22.40, Start Longitude = 123 23.73.\n"
        "Start Time = 06:01 PM, End Time = 06:11 PM.\n"
        "Location = Victor Hotel."
    )

    assert fields == {
        "summary": "one",
        "start_latitude": "48 22.40",
        "start_longitude": "123 23.73.",
        "start_time": "06:01 PM",
        "end_time": "06:11 PM.",
        "location_description": "Victor Hotel.",
    }
    observed_at, flags = precise_local_timestamp(
        "2024-02-13", fields["start_time"], fields["end_time"]
    )
    assert observed_at == "2024-02-13T18:01:00"
    assert flags == []


@pytest.mark.parametrize(
    ("start", "end", "expected", "flag"),
    [
        ("08:30", "10:30", "2025-06-03T08:30:00", None),
        ("03:59", "04:30", None, "START_TIME_OUTSIDE_LOCAL_PLAUSIBILITY_GATE"),
        ("14:00", "13:59", None, "SOURCE_TIME_RANGE_ANOMALY"),
        ("09:00", "bad", None, "UNPARSED_END_TIME"),
    ],
)
def test_pacific_time_plausibility_gate(start, end, expected, flag):
    observed_at, flags = precise_local_timestamp("2025-06-03", start, end)
    assert observed_at == expected
    assert (flag in flags) if flag else not flags


def test_cwr_timestamp_localizes_with_dst_and_anomaly_falls_back_to_date():
    summer = _source_record(
        "CWR",
        "summer",
        {},
        observed_at="2025-07-03T10:00:00",
        observed_date="2025-07-03",
        latitude=48.5,
        longitude=-123.25,
        species="Orcinus orca",
        source_use_class="INTERNAL_ONLY",
    )
    winter = _source_record(
        "CWR",
        "winter",
        {},
        observed_at="2025-01-03T10:00:00",
        observed_date="2025-01-03",
        latitude=48.5,
        longitude=-123.25,
        species="Orcinus orca",
        source_use_class="INTERNAL_ONLY",
    )
    date_only = _source_record(
        "CWR",
        "date-only",
        {},
        observed_date="2025-06-03",
        latitude=48.5,
        longitude=-123.25,
        species="Orcinus orca",
        source_use_class="INTERNAL_ONLY",
    )
    records, _audit = _normalize_records(
        _normalization_frame([summer, winter, date_only]), _config()
    )
    by_id = {row["SOURCE_RECORD_ID"]: row for row in records}
    assert by_id["CWR:summer"]["SOURCE_EVENT_AT_UTC"].hour == 17
    assert by_id["CWR:winter"]["SOURCE_EVENT_AT_UTC"].hour == 18
    assert by_id["CWR:date-only"]["SOURCE_EVENT_AT_UTC"] is None
    assert by_id["CWR:date-only"]["SOURCE_TIME_PRECISION"] == "DATE"


def test_cwr_associations_merge_without_changing_global_thresholds():
    cwr = _source_record(
        "CWR",
        "cwr-1",
        {},
        observed_at="2025-06-03T10:00:00",
        observed_date="2025-06-03",
        latitude=48.5,
        longitude=-123.25,
        species="Orcinus orca",
        pod_ecotype="Bigg's Killer Whales | T65A | T65A3",
        source_use_class="INTERNAL_ONLY",
    )
    twm = _source_record(
        "TWM",
        "twm-1",
        {},
        observed_at="2025-06-03T10:10:00",
        observed_date="2025-06-03",
        latitude=48.5005,
        longitude=-123.2505,
        species="Orcinus orca",
        pod_ecotype="T65A",
    )
    evidence = _evidence(pd.Series(cwr))
    assert any(item["ASSOCIATION_VALUE"] == "T65A" for item in evidence)
    records, audit = _normalize_records(_normalization_frame([cwr, twm]), _config())
    groups = _cluster(records, _config(), audit)
    assert len(groups) == 1
    observations, associations = _materialize(
        groups, audit, ["existing-observation"], policy=OBSERVATION_POLICY
    )
    assert observations[0]["SOURCE"] == "TWM"
    assert not observations[0]["PUBLIC_RELEASE_ELIGIBLE"]
    assert any(item["SOURCE"] == "CWR" for item in associations)
    assert SOURCE_PRIORITY["CWR"] < SOURCE_PRIORITY["GBIF"]


def test_cwr_archive_reuse_and_full_refresh(monkeypatch, tmp_path):
    settings = _cwr_settings().model_copy(
        update={
            "archive_years": (2020,),
            "atlist_maps": {2024: _cwr_settings().atlist_maps[2024]},
        }
    )
    archive_url = "https://whaleresearch.wixsite.com/2020encounters/1"
    archive = _archive_payload(
        [_entry("1", archive_url)],
        {archive_url: _archive_page(archive_url, number="1")},
    )
    atlist = _atlist_payload([_marker()])
    calls = {"archive": 0, "atlist": 0}

    def fetch_archive(_settings):
        calls["archive"] += 1
        return archive

    def fetch_atlist(_settings):
        calls["atlist"] += 1
        return atlist

    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.collect.sources.cwr._fetch_archive",
        fetch_archive,
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.collect.sources.cwr._fetch_atlist",
        fetch_atlist,
    )
    first = tmp_path / "first"
    first.mkdir()
    metrics = collect_cwr_snapshot(
        settings, first, previous_snapshot=None, full_refresh=False
    )
    assert metrics["source_event_count"] == 2
    assert not metrics["archive_reused"]

    second = tmp_path / "second"
    second.mkdir()
    reused = collect_cwr_snapshot(
        settings, second, previous_snapshot=first, full_refresh=False
    )
    assert reused["archive_reused"]
    assert calls == {"archive": 1, "atlist": 2}

    third = tmp_path / "third"
    third.mkdir()
    refreshed = collect_cwr_snapshot(
        settings, third, previous_snapshot=second, full_refresh=True
    )
    assert not refreshed["archive_reused"]
    assert calls == {"archive": 2, "atlist": 3}


def test_cwr_collection_is_full_replace_observed_only_and_reuses_offline(
    monkeypatch, tmp_path
):
    payload = yaml.safe_load(
        (project_root() / "config/data/sightings.yaml").read_text(encoding="utf-8")
    )
    for name, source in payload["collection"]["sources"].items():
        source["enabled"] = name == "cwr"
    payload["collection"]["sources"]["cwr"]["archive_years"] = [2020]
    payload["collection"]["sources"]["cwr"]["atlist_maps"] = {
        2024: payload["collection"]["sources"]["cwr"]["atlist_maps"][2024]
    }
    config_path = tmp_path / "sightings.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    url = "https://whaleresearch.wixsite.com/2020encounters/1"
    archive = _archive_payload(
        [_entry("1", url)], {url: _archive_page(url, number="1")}
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.collect.sources.cwr._fetch_archive",
        lambda _settings: archive,
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.collect.sources.cwr._fetch_atlist",
        lambda _settings: _atlist_payload([_marker()]),
    )
    request_fields = {
        "config": config_path,
        "data_root": tmp_path / "data",
        "artifact_root": tmp_path / "artifacts",
        "output_root": tmp_path / "outputs",
        "end_date": date(2025, 1, 1),
        "force": True,
    }
    online = collect_sightings(
        SightingsCollectionRequest(run_id="online", **request_fields)
    )
    snapshot = online.outputs[0].path
    metadata = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
    assert metadata["snapshot_mode"] == "FULL_REPLACE"
    assert metadata["coverage_status"] == "observed_only"
    assert metadata["source_event_count"] == 2
    assert metadata["raw_marker_count"] == 1
    assert {"archive_extracts.json", "atlist_maps.json", "cwr_policy.json"} <= set(
        metadata["file_checksums"]
    )

    offline = collect_sightings(
        SightingsCollectionRequest(run_id="offline", offline=True, **request_fields)
    )
    assert offline.outputs[0].path == snapshot
    assert offline.outputs[0].freshness == "offline"


def test_cwr_remote_errors_fail_closed_without_retrying_permanent_4xx(monkeypatch):
    calls = 0

    class Response:
        status_code = 404
        content = b"missing"
        headers: dict[str, str] = {}

        def raise_for_status(self):
            raise requests.HTTPError("missing")

    def get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.collect.sources.cwr.requests.get", get
    )
    with pytest.raises(requests.HTTPError):
        _request_bytes("https://example.test/missing", timeout=1, retries=3)
    assert calls == 1


def test_cwr_malformed_json_fails_closed(monkeypatch):
    class Response:
        status_code = 200
        content = b"not json"
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.collect.sources.cwr.requests.get",
        lambda *args, **kwargs: Response(),
    )
    with pytest.raises(ValueError, match="Malformed CWR JSON"):
        _request_json("https://example.test/data", timeout=1, retries=0)


def test_cwr_snapshot_aborts_on_duplicate_native_ids(monkeypatch, tmp_path):
    settings = _cwr_settings().model_copy(
        update={
            "archive_years": (2020,),
            "atlist_maps": {2024: _cwr_settings().atlist_maps[2024]},
        }
    )
    url = "https://whaleresearch.wixsite.com/2020encounters/1"
    archive = _archive_payload(
        [_entry("1", url)], {url: _archive_page(url, number="1")}
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.collect.sources.cwr._fetch_archive",
        lambda _settings: archive,
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.collect.sources.cwr._fetch_atlist",
        lambda _settings: _atlist_payload([_marker("same"), _marker("same")]),
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="duplicate native IDs"):
        collect_cwr_snapshot(
            settings, snapshot, previous_snapshot=None, full_refresh=True
        )


def test_offline_archive_file_is_immutable_copy(tmp_path):
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    previous.mkdir()
    current.mkdir()
    (previous / "archive_extracts.json").write_text("{}", encoding="utf-8")
    shutil.copy2(previous / "archive_extracts.json", current / "archive_extracts.json")
    assert (current / "archive_extracts.json").read_bytes() == (
        previous / "archive_extracts.json"
    ).read_bytes()


from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    OBSERVATION_POLICY,
)
