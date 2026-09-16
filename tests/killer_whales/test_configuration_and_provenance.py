from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import DataSnapshotMetadata
from marine_mammal_toolkit.tools.schemas.artifacts import SourceWatermark
from marine_mammal_toolkit.tools._core.config.paths import project_root
from marine_mammal_toolkit.tools._core.persistence import checksum_path
from marine_mammal_toolkit.cetaceans.killer_whales.populations.prepare import (
    load_population_config,
)
from marine_mammal_toolkit.tools.observations.collect import pipeline as collection
from marine_mammal_toolkit.tools.observations.collect.pipeline import (
    _observed_date_bounds,
)
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.observations.runtime import build_data_snapshot


def test_population_paths_resolve_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = load_population_config(project_root() / "config/data/project.yaml")
    assert config["base_directory"] == project_root()
    assert (config["base_directory"] / config["source_path"]).is_file()


def test_default_imputation_requires_purged_certification():
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    assert "purged_blocked" in config.imputation.evaluate_strategies
    assert config.imputation.model.hard_label_certification_strategy == "purged_blocked"
    assert config.imputation.inputs.water_network_config == Path(
        "config/data/environment_seascape.yaml"
    )


def test_configuration_rejects_missing_purged_evaluation(tmp_path):
    source = project_root() / "config/data/sightings.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["imputation"]["evaluate_strategies"] = ["encounter"]
    path = tmp_path / "sightings.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="purged_blocked"):
        load_sightings_config(path)


def test_observed_bounds_are_descriptive_not_coverage():
    table = pa.table(
        {
            "OBSERVED_AT_RAW": ["2025-02-03T09:00:00-08:00", None],
            "OBSERVED_DATE_RAW": [None, "2025-02-05"],
        }
    )
    assert _observed_date_bounds(table) == ("2025-02-03", "2025-02-05")


def test_maplify_requires_response_completeness_metadata(monkeypatch):
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    monkeypatch.setattr(
        collection, "_request_json", lambda *args, **kwargs: {"results": []}
    )
    with pytest.raises(ValueError, match="count completeness"):
        collection._fetch_maplify(config, date(2025, 1, 1), date(2025, 1, 2))


def test_inaturalist_rejects_a_changed_declared_total(monkeypatch):
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )

    def response(*args, **kwargs):
        page = kwargs["params"]["page"]
        if page == 1:
            return {
                "total_results": 201,
                "results": [{"id": index} for index in range(200)],
            }
        return {"total_results": 202, "results": [{"id": 201}]}

    monkeypatch.setattr(collection, "_request_json", response)
    with pytest.raises(ValueError, match="changed during pagination"):
        collection._fetch_inaturalist(config, date(2025, 1, 1), date(2025, 1, 2))


def _source_artifact(tmp_path: Path, source: str, metadata: dict) -> ArtifactRef:
    root = tmp_path / source
    root.mkdir()
    (root / "snapshot.json").write_text(json.dumps(metadata), encoding="utf-8")
    return ArtifactRef(
        kind="source",
        dataset_id=f"whale.sightings.source_{source}",
        path=root,
        producer="test",
        checksum=checksum_path(root),
        run_id="test-run",
    )


def test_verified_snapshot_uses_interval_intersection(tmp_path):
    first = _source_artifact(
        tmp_path,
        "first",
        {
            "source": "first",
            "retrieval_id": "one",
            "retrieved_at": "2025-02-10T00:00:00+00:00",
            "coverage_start": "2020-01-01",
            "coverage_through": "2025-02-09",
            "coverage_status": "configured_verified",
        },
    )
    second = _source_artifact(
        tmp_path,
        "second",
        {
            "source": "second",
            "retrieval_id": "two",
            "retrieved_at": "2025-02-11T00:00:00+00:00",
            "coverage_start": "2021-01-01",
            "coverage_through": "2025-02-08",
            "coverage_status": "request_complete",
        },
    )
    snapshot = build_data_snapshot((first, second), default_coverage_start="1980-01-01")
    assert snapshot.coverage_start == "2021-01-01"
    assert snapshot.coverage_through == "2025-02-08"
    assert snapshot.coverage_status == "verified_intersection"


def test_any_unverified_source_removes_composite_verified_coverage(tmp_path):
    verified = _source_artifact(
        tmp_path,
        "verified",
        {
            "source": "verified",
            "retrieval_id": "one",
            "retrieved_at": "2025-02-10T00:00:00+00:00",
            "coverage_start": "2020-01-01",
            "coverage_through": "2025-02-09",
            "coverage_status": "request_complete",
        },
    )
    observed = _source_artifact(
        tmp_path,
        "observed",
        {
            "source": "observed",
            "retrieval_id": "two",
            "retrieved_at": "2025-02-11T00:00:00+00:00",
            "coverage_start": "2024-01-01",
            "coverage_through": "2025-02-10",
            "observed_start": "2024-01-01",
            "observed_through": "2025-02-10",
            "coverage_status": "observed_only",
        },
    )
    snapshot = build_data_snapshot(
        (verified, observed), default_coverage_start="1980-01-01"
    )
    assert snapshot.coverage_start is None
    assert snapshot.coverage_through is None
    assert snapshot.coverage_status == "unverified"


def test_legacy_manifest_coverage_is_not_promoted_to_verified():
    watermark = SourceWatermark.from_dict(
        {
            "source": "legacy",
            "retrieval_id": "old-run",
            "retrieved_at": "2025-02-10T00:00:00+00:00",
            "coverage_start": "1980-01-01",
            "coverage_through": "2025-02-09",
            "checksum": "abc123",
        }
    )
    snapshot = DataSnapshotMetadata.from_dict(
        {
            "snapshot_id": "legacy-snapshot",
            "snapshot_created_at": "2025-02-10T00:00:00+00:00",
            "coverage_start": "1980-01-01",
            "coverage_through": "2025-02-09",
            "source_watermarks": [watermark.__dict__],
        }
    )

    assert watermark.coverage_start is None
    assert watermark.coverage_through is None
    assert watermark.requested_start == "1980-01-01"
    assert watermark.requested_through == "2025-02-09"
    assert watermark.coverage_status == "legacy_unverified"
    assert snapshot.coverage_start is None
    assert snapshot.coverage_through is None
    assert snapshot.coverage_status == "legacy_unverified"
