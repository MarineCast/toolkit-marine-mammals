from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from marine_mammal_toolkit.cli import cli
from marine_mammal_toolkit.cetaceans.killer_whales.query import (
    preflight_observations,
    query_observations,
    run_demo,
)
from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
from marine_mammal_toolkit.tools.observations.collect import pipeline as collect
from marine_mammal_toolkit.tools.observations.process.pipeline import (
    normalize_sightings,
)
from marine_mammal_toolkit.tools.schemas.observations import (
    SightingsCollectionRequest,
    NormalizationRequest,
)


def _configuration(root, *sources):
    payload = yaml.safe_load(config_path().read_text())
    payload["collection"]["sources"] = {
        name: value
        for name, value in payload["collection"]["sources"].items()
        if name in sources
    }
    path = root / "query.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _inat_record():
    return {
        "id": 1,
        "observed_on": "2025-06-03",
        "latitude": 48.5,
        "longitude": -123.0,
        "taxon": {"name": "Orcinus orca"},
    }


def test_demo_uses_production_api_without_network(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("network"))
    result = run_demo(tmp_path)
    assert result.observations.row_count == 2
    assert set(result.read().SOURCE) == {"TWM"}
    assert not result.read().PUBLIC_RELEASE_ELIGIBLE.any()
    assert result.manifest.is_file()
    # Repeat the same query and verify row identities remain stable.
    again = run_demo(tmp_path)
    assert result.read().OBSERVATION_ID.tolist() == again.read().OBSERVATION_ID.tolist()


def test_missing_twm_continues_and_keeps_coverage_unknown(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(collect, "_fetch_inaturalist", lambda *a: [_inat_record()])
    result = query_observations(
        workspace_root=tmp_path,
        sources=("twm", "inaturalist"),
        start=date(2025, 6, 1),
        end=date(2025, 6, 8),
    )
    assert set(result.read().SOURCE) == {"INATURALIST"}
    assert "Continuing with other sources" in caplog.text
    assert any("source_unavailable" in warning for warning in result.warnings)
    snapshot = json.loads(result.manifest.read_text())["collection"]["data_snapshot"]
    assert snapshot["coverage_status"] == "unverified"
    twm = next(
        item for item in snapshot["source_watermarks"] if item["source"] == "twm"
    )
    assert twm["coverage_status"] == "source_unavailable"
    assert twm["coverage_through"] is None


def test_first_bounded_query_and_scope_isolation(tmp_path, monkeypatch):
    calls = []

    def fetch(settings, start, end):
        calls.append((start, end))
        return [_inat_record()]

    monkeypatch.setattr(collect, "_fetch_inaturalist", fetch)
    result = query_observations(
        workspace_root=tmp_path, start=date(2025, 6, 1), end=date(2025, 6, 8)
    )
    assert result.read().shape[0] == 1
    assert calls == [(date(2025, 6, 1), date(2025, 6, 8))]
    other = query_observations(
        workspace_root=tmp_path, start=date(2025, 6, 4), end=date(2025, 6, 8)
    )
    assert other.query_root != result.query_root
    assert other.read().empty
    assert result.read().shape[0] == 1


def test_disabled_source_is_removed_from_active_state_not_history(
    tmp_path, monkeypatch
):
    config = _configuration(tmp_path, "twm")
    csv = tmp_path / "twm.csv"
    csv.write_text("id,date,latitude,longitude,pod\na,2025-06-03,48.5,-123.0,J pod\n")
    common = dict(
        config=config,
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
        force=True,
    )
    first = collect.collect_sightings(
        SightingsCollectionRequest(
            **common, twm_files=(csv,), end_date=date(2025, 6, 8)
        )
    )
    normalized = normalize_sightings(
        NormalizationRequest(**common, inputs=first.outputs)
    )
    assert (
        next(
            a
            for a in normalized.outputs
            if a.dataset_id == "whale.sightings.observations"
        ).row_count
        == 1
    )
    _configuration(tmp_path, "inaturalist")
    monkeypatch.setattr(collect, "_fetch_inaturalist", lambda *args: [])
    second = collect.collect_sightings(
        SightingsCollectionRequest(**common, end_date=date(2025, 6, 8))
    )
    normalized = normalize_sightings(
        NormalizationRequest(**common, inputs=second.outputs)
    )
    artifacts = {a.dataset_id: a for a in normalized.outputs}
    assert artifacts["whale.sightings.observations"].row_count == 0
    assert artifacts["whale.sightings.source_records"].row_count == 0
    assert artifacts["whale.sightings.source_record_history"].row_count == 1


def test_changed_collection_scope_restarts_provider_history(tmp_path, monkeypatch):
    config = _configuration(tmp_path, "inaturalist")
    starts = []
    monkeypatch.setattr(
        collect,
        "_fetch_inaturalist",
        lambda config, start, end: starts.append(start) or [],
    )
    common = dict(
        config=config,
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
        force=True,
    )
    collect.collect_sightings(
        SightingsCollectionRequest(**common, end_date=date(2025, 6, 8))
    )
    collect.collect_sightings(
        SightingsCollectionRequest(**common, end_date=date(2025, 6, 9))
    )
    payload = yaml.safe_load(config.read_text())
    payload["full_area"]["min_lat"] = 40
    config.write_text(yaml.safe_dump(payload))
    latest = collect.collect_sightings(
        SightingsCollectionRequest(**common, end_date=date(2025, 6, 9))
    )
    assert starts == [date(1980, 1, 1), date(2025, 6, 6), date(1980, 1, 1)]
    assert (
        json.loads((latest.outputs[0].path / "snapshot.json").read_text())[
            "snapshot_mode"
        ]
        == "FULL_REPLACE"
    )


def test_preflight_is_read_only_and_missing_twm_is_warning(tmp_path):
    before = list(tmp_path.iterdir())
    result = preflight_observations(config_path(), workspace_root=tmp_path)
    assert result["ready"]
    assert result["sources"]["twm"]["status"] == "source_unavailable"
    assert not result["network_checked"]
    assert list(tmp_path.iterdir()) == before
    derived = preflight_observations(
        config_path(), workspace_root=tmp_path, profile="imputation-only"
    )
    assert not derived["ready"]


def test_public_cli_demo_and_query_plan(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("network"))
    runner = CliRunner()
    base = ["--workspace-root", str(tmp_path), "killer-whales", "observations"]
    result = runner.invoke(cli, base + ["demo"])
    assert result.exit_code == 0, str(result.exception)
    assert json.loads(result.output)["row_count"] == 2
    result = runner.invoke(
        cli,
        base
        + [
            "query",
            "--source",
            "gbif",
            "--dataset",
            "e0da2d53-86f0-440c-a11a-42ffb0b3fd3e",
            "--start",
            "2025-06-01",
            "--end",
            "2025-06-08",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, str(result.exception)
    assert json.loads(result.output)["sources"] == ["gbif"]


def test_missing_twm_only_is_unavailable_even_without_configured_path(tmp_path):
    path = _configuration(tmp_path, "twm")
    payload = yaml.safe_load(path.read_text())
    payload["collection"]["sources"]["twm"]["local_path"] = None
    path.write_text(yaml.safe_dump(payload))
    result = query_observations(
        workspace_root=tmp_path,
        sources=("twm",),
        config=path,
        start=date(2025, 6, 1),
        end=date(2025, 6, 8),
        twm_files=("missing.csv",),
    )
    assert result.read().empty
    assert any("source_unavailable" in warning for warning in result.warnings)
    snapshot = json.loads(result.manifest.read_text())["collection"]["data_snapshot"]
    assert snapshot["coverage_through"] is None
    assert snapshot["coverage_status"] == "unverified"


def test_workspace_lock_rejects_another_writer_and_cleans_up(tmp_path):
    from contextvars import Context
    from marine_mammal_toolkit.tools._core.locking import workspace_write_lock

    def another_writer():
        with workspace_write_lock(tmp_path):
            pytest.fail("a second writer acquired the lock")

    with workspace_write_lock(tmp_path):
        with workspace_write_lock(tmp_path):
            assert (tmp_path / ".marine-mammals-write.lock").is_file()
        with pytest.raises(FileExistsError, match="another writer"):
            Context().run(another_writer)
    assert not (tmp_path / ".marine-mammals-write.lock").exists()
