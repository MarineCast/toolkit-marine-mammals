"""Public demography workflow and legacy census compatibility."""

from __future__ import annotations

import json

from click.testing import CliRunner
from openpyxl import Workbook

from marine_mammal_toolkit.cli import cli
from marine_mammal_toolkit.cetaceans.killer_whales.demography import (
    export_population_numbers,
)
from marine_mammal_toolkit.cetaceans.killer_whales.populations.prepare import (
    export_population_numbers as legacy_export,
)


def test_demography_census_validates_without_writing_then_exports(tmp_path):
    workbook = Workbook()
    workbook.active.title = "Counts"
    workbook.active.append(["Census Year", "J Pod", "K Pod", "L Pod", "All Pods"])
    workbook.active.append([2024, 4, 3, 2, 9])
    workbook.active.append([2025, 5, 3, 2, 10])
    workbook.save(tmp_path / "census.xlsx")

    command = [
        "--workspace-root",
        str(tmp_path),
        "killer-whales",
        "demography",
        "census",
        "--workbook",
        "census.xlsx",
        "--sheet",
        "Counts",
        "--output",
        "results/census.json",
    ]
    preview = CliRunner().invoke(cli, [*command, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    plan = json.loads(preview.output)
    assert plan["validation"]["row_count"] == 2
    assert plan["latest"]["census_year"] == 2025
    assert plan["written"] is False
    assert not (tmp_path / "results/census.json").exists()

    exported = CliRunner().invoke(cli, command)
    assert exported.exit_code == 0, exported.output
    payload = json.loads((tmp_path / "results/census.json").read_text())
    assert payload["rows"][0]["census_year"] == 2024
    assert payload["latest"]["all_pods"] == 10
    assert payload["validation"]["total_mismatch_count"] == 0
    assert legacy_export is export_population_numbers


def test_demography_census_mismatch_gate_preserves_output(tmp_path):
    workbook = Workbook()
    workbook.active.title = "Counts"
    workbook.active.append(["Census Year", "J Pod", "K Pod", "L Pod", "All Pods"])
    workbook.active.append([2025, 5, 3, 2, 11])
    workbook.save(tmp_path / "census.xlsx")
    output = tmp_path / "existing.json"
    output.write_text("previous")

    result = CliRunner().invoke(
        cli,
        [
            "--workspace-root",
            str(tmp_path),
            "killer-whales",
            "demography",
            "census",
            "--workbook",
            "census.xlsx",
            "--sheet",
            "Counts",
            "--output",
            "existing.json",
            "--fail-on-total-mismatch",
        ],
    )
    assert result.exit_code != 0
    assert "No output was written" in result.output
    assert output.read_text() == "previous"
