from __future__ import annotations

import json
import shutil
from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from marine_mammal_toolkit.cetaceans.killer_whales.observations.report import (
    build_sightings_report_html,
)
from marine_mammal_toolkit.cli import cli


def _write_product(root: Path) -> tuple[Path, Path, Path]:
    processed = root / "processed/sightings/final"
    processed.mkdir(parents=True)
    composite = processed / "composite-sightings.parquet"
    imputed = processed / "imputed-sightings.parquet"
    manifest = processed / "imputation-model-manifest.json"
    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": ["orca-1", "orca-2", "orca-3"],
            "SIGHTING_DATE_UTC": pd.to_datetime(
                ["2025-01-01", "2025-01-03", "2025-02-01"], utc=True
            ),
            "LATITUDE": [48.50, 48.52, 49.00],
            "LONGITUDE": [-123.20, -123.18, -124.00],
            "SOURCE": ["TWM", "TWM", "GBIF"],
        }
    )
    frame.to_parquet(composite, index=False)
    frame.assign(IMPUTATION_APPLIED=[False, True, False]).to_parquet(
        imputed, index=False
    )
    manifest.write_text(
        json.dumps({"release_id": "fixture-release", "public_eligible": False}),
        encoding="utf-8",
    )
    return composite, imputed, manifest


def test_builds_density_map_and_daily_timeline(tmp_path: Path) -> None:
    composite, imputed, manifest = _write_product(tmp_path)
    output = tmp_path / "processed/sightings/final/sightings-report.html"

    result = build_sightings_report_html(
        composite_path=composite,
        imputed_path=imputed,
        model_manifest_path=manifest,
        output_path=output,
    )

    assert result == output.resolve()
    rendered = output.read_text(encoding="utf-8")
    assert "Killer-whale sightings report" in rendered
    assert "All-time reported-sighting density" in rendered
    assert "Reported sightings count over time" in rendered
    assert 'id="reported-sightings-density"' in rendered
    assert 'id="reported-sightings-timeline"' in rendered
    assert "3</strong><span>Composite sightings" in rendered
    assert "1</strong><span>Hard-imputed labels" in rendered
    assert "fixture-release" in rendered
    assert "not evidence of survey coverage or whale absence" in rendered


def test_report_cli_uses_product_defaults(tmp_path: Path) -> None:
    product_root = tmp_path / "data/marine-mammals/killer-whales"
    composite, imputed, manifest = _write_product(product_root)
    generation = (
        product_root / "processed/sightings/final/by-date/2025-02-01/fixture-release"
    )
    generation.mkdir(parents=True)
    payload = json.loads(manifest.read_text())
    payload["release_manifest"] = "fixture-release.json"
    payload["tables"] = {
        "composite": {"checksum": checksum_path(composite)},
        "imputed": {"checksum": checksum_path(imputed)},
    }
    manifest.write_text(json.dumps(payload))
    for path in (composite, imputed, manifest):
        shutil.copy2(path, generation / path.name)
    (product_root / "processed/sightings/final/latest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "release_id": "fixture-release",
                "generation": "processed/sightings/final/by-date/2025-02-01/fixture-release",
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli,
        [
            "--workspace-root",
            str(tmp_path),
            "--data-root",
            "data/marine-mammals/killer-whales",
            "killer-whales",
            "observations",
            "report",
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    output = Path(result.output.strip())
    assert output.parent == product_root / "processed/sightings/final/reports"
    assert output.is_file()
    latest = json.loads(
        (product_root / "processed/sightings/final/latest.json").read_text()
    )
    assert latest["report"] == output.relative_to(product_root).as_posix()
    assert len(latest["report_checksum"]) == 64
