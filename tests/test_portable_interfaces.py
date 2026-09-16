import json
from pathlib import Path

import joblib
import pytest
from openpyxl import Workbook

from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
from marine_mammal_toolkit.cetaceans.killer_whales.observations.imputer import (
    KillerWhaleImputer,
)
from marine_mammal_toolkit.cetaceans.killer_whales.populations.prepare import (
    load_population_rows,
    build_population_payload,
)
from marine_mammal_toolkit.tools._core.config import ConfigDocument, workspace
from marine_mammal_toolkit.tools.observations.collect.sources.inaturalist import (
    fetch_inaturalist,
)
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.populations.workbook import read_annual_counts


def test_legacy_model_rejected_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "old.joblib"
    path.write_bytes(b"legacy-pickle")
    monkeypatch.setattr(
        joblib, "load", lambda *args: pytest.fail("Legacy data was deserialized")
    )
    with pytest.raises(ValueError, match="Refit"):
        KillerWhaleImputer.load(path)


def test_source_accepts_another_taxon_without_orca_defaults():
    _, config = load_sightings_config(config_path())
    captured = []

    def transport(url, **kwargs):
        captured.append(kwargs["params"])
        return {"total_results": 0, "results": []}

    from datetime import date

    assert (
        fetch_inaturalist(
            config.collection.sources["inaturalist"],
            date(2020, 1, 1),
            date(2020, 1, 2),
            bbox=config.full_area,
            taxon_id=123,
            request_json=transport,
        )
        == []
    )
    assert captured[0]["taxon_id"] == 123


def test_config_includes_use_declaring_directory_and_paths_use_workspace(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text("input: data/example.parquet\nvalues: [1, 2]\n")
    (configs / "child.yaml").write_text("extends: base.yaml\nvalues: [3]\n")
    document = ConfigDocument.load(
        configs / "child.yaml", workspace_root=tmp_path / "other"
    )
    assert document.data["values"] == [3]
    assert (
        document.resolve_path(document.data["input"])
        == tmp_path / "other/data/example.parquet"
    )


def _book(path, headers, rows):
    book = Workbook()
    book.active.title = "Counts"
    book.active.append(headers)
    for row in rows:
        book.active.append(row)
    book.save(path)


def test_generic_population_reader_accepts_other_group_names(tmp_path):
    path = tmp_path / "counts.xlsx"
    _book(path, ["Year", "North", "South", "Total"], [[2022, 2, 3, 6], [2021, 1, 2, 3]])
    rows = read_annual_counts(
        path,
        sheet_name="Counts",
        aliases={"year": "census_year"},
        count_columns=("north", "south", "total"),
        component_columns=("north", "south"),
        total_column="total",
    )
    assert [row["census_year"] for row in rows] == [2021, 2022]
    assert rows[1]["pod_sum"] == 5
    assert rows[1]["total_matches_pod_sum"] is False


@pytest.mark.parametrize(
    "rows",
    [
        [[2020, 1, 1, 1, 3], [2020, 1, 1, 1, 3]],
        [[2020, 1, None, 1, 2]],
        [[2020, 1, -1, 1, 1]],
        [[2020, 1, 0.5, 1, 2.5]],
    ],
)
def test_population_rejects_invalid_records(tmp_path, rows):
    path = tmp_path / "bad.xlsx"
    _book(path, ["Census Year", "J Pod", "K Pod", "L Pod", "All Pods"], rows)
    with pytest.raises(ValueError):
        load_population_rows(path, sheet_name="Counts")


def test_population_duplicate_headers_are_rejected(tmp_path):
    path = tmp_path / "duplicate.xlsx"
    _book(
        path,
        ["Census Year", "J Pod", "J_Pod", "K Pod", "L Pod", "All Pods"],
        [[2020, 1, 1, 1, 1, 3]],
    )
    with pytest.raises(ValueError, match="duplicate normalized"):
        load_population_rows(path, sheet_name="Counts")


def test_population_mismatch_and_failed_atomic_export_preserve_previous_file(
    tmp_path, monkeypatch
):
    from marine_mammal_toolkit.cetaceans.killer_whales.populations.prepare import (
        export_population_numbers,
    )
    from marine_mammal_toolkit.tools.schemas import artifacts

    source = tmp_path / "counts.xlsx"
    _book(
        source,
        ["Census Year", "J Pod", "K Pod", "L Pod", "All Pods"],
        [[2020, 1, 2, 3, 8]],
    )
    config = tmp_path / "population.yaml"
    config.write_text(
        "population_source_path: counts.xlsx\npopulation_output_path: population.json\npopulation_sheet_name: Counts\n"
    )
    target = tmp_path / "population.json"
    target.write_text("previous")
    with pytest.raises(ValueError, match="No output was written"):
        export_population_numbers(
            config_path=config, workspace_root=tmp_path, fail_on_total_mismatch=True
        )
    assert target.read_text() == "previous"

    def fail_replace(*args):
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic rename"):
        export_population_numbers(config_path=config, workspace_root=tmp_path)
    assert target.read_text() == "previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_config_paths_and_manifest_pointers_are_workspace_relative(
    tmp_path, monkeypatch
):
    from click.testing import CliRunner
    from marine_mammal_toolkit.cli import cli, _manifest_payload

    root = tmp_path / "external"
    root.mkdir()
    (root / "config.yaml").write_text(config_path().read_text())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = CliRunner().invoke(
        cli,
        [
            "--workspace-root",
            str(root),
            "killer-whales",
            "observations",
            "impute",
            "fit",
            "--config",
            "config.yaml",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert str(root / "models") in result.output
    (root / "run.json").write_text('{"outputs": []}')
    (root / "latest.json").write_text('{"manifest": "run.json"}')
    assert _manifest_payload(str(root / "latest.json")) == {"outputs": []}


def test_cli_fit_selects_species_composition(synthetic_workspace, monkeypatch):
    from types import SimpleNamespace
    from click.testing import CliRunner
    from marine_mammal_toolkit.cli import cli
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import imputation

    captured = []

    def fit(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(model_path=synthetic_workspace / "model.joblib")

    monkeypatch.setattr(imputation, "fit_imputation_model", fit)
    result = CliRunner().invoke(
        cli,
        [
            "--workspace-root",
            str(synthetic_workspace),
            "killer-whales",
            "observations",
            "impute",
            "fit",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[0]["source_config_path"] == config_path()


def test_stage_default_roots_use_document_workspace(tmp_path, monkeypatch):
    from marine_mammal_toolkit.tools.schemas.stages import StageRequest

    root = tmp_path / "selected"
    document = ConfigDocument.load(config_path(), workspace_root=root)
    monkeypatch.chdir(tmp_path)
    request = StageRequest(config=document)
    assert request.data_root == root / "data"
    assert request.artifact_root == root / "artifacts"
