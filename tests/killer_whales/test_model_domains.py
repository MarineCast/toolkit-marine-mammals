from __future__ import annotations

import json
from pathlib import Path

import h3
import polars as pl
import pytest

from marine_mammal_toolkit.cetaceans.killer_whales.observations.domains import (
    build_operational_model_domains,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    load_domain_geometry,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    load_domain_provenance,
)


def test_operational_model_domains_materialize_with_water_cell_parity(
    tmp_path: Path,
) -> None:
    areas_config = tmp_path / "areas.yaml"
    areas_config.write_text(
        """
areas:
  srkw_range:
    description: Fixture SRKW area.
    bbox_wgs84: {min_lon: -125, min_lat: 47, max_lon: -122, max_lat: 50}
  transient_range:
    description: Fixture Transient area.
    bbox_wgs84: {min_lon: -150, min_lat: 40, max_lon: -120, max_lat: 62}
""".strip() + "\n",
        encoding="utf-8",
    )
    domain_config = tmp_path / "model-domains.yaml"
    domain_config.write_text(
        f"""
schema_version: 1
domain_kind: operational_model_extent
source_authority: Fixture review
source_release: fixture-v1
approved_at: "2026-08-19T00:00:00Z"
review_status: approved_for_model_domain
areas_config: {areas_config}
water_universe_root: data/processed/environment/seascape/full_counting
resolutions: [4, 5, 6]
domains:
  SRKW:
    area_key: srkw_range
    source_reference: repo://fixture#srkw
    output: data/processed/domain/whale_layer/spatial_support/SRKW_MODEL_DOMAIN.parquet
    expected_cells: {{4: 1, 5: 1, 6: 1}}
  TRANSIENT:
    area_key: transient_range
    source_reference: repo://fixture#transient
    output: data/processed/domain/whale_layer/spatial_support/TRANSIENT_MODEL_DOMAIN.parquet
    expected_cells: {{4: 2, 5: 2, 6: 2}}
expected_union_cells: {{4: 2, 5: 2, 6: 2}}
""".strip() + "\n",
        encoding="utf-8",
    )
    universe_root = tmp_path / "processed/environment/seascape/full_counting"
    universe_root.mkdir(parents=True)
    for resolution in (4, 5, 6):
        cells = [
            h3.latlng_to_cell(48.5, -123.2, resolution),
            h3.latlng_to_cell(60.0, -140.0, resolution),
            h3.latlng_to_cell(30.0, -100.0, resolution),
        ]
        pl.DataFrame(
            {"H3_INDEX": cells, "H3_RESOLUTION": [resolution] * len(cells)}
        ).write_parquet(universe_root / f"H3_WATER_UNIVERSE_{resolution}.parquet")

    result = build_operational_model_domains(
        sightings_config_path="config/data/sightings.yaml",
        domain_config_path=domain_config,
        data_root=tmp_path,
        run_id="fixture-model-domains",
    )

    assert result.effective_cell_counts == {
        4: {"SRKW": 1, "TRANSIENT": 2, "OTHER": 2},
        5: {"SRKW": 1, "TRANSIENT": 2, "OTHER": 2},
        6: {"SRKW": 1, "TRANSIENT": 2, "OTHER": 2},
    }
    for ecotype, polygon, metadata in zip(
        ("SRKW", "TRANSIENT"),
        result.polygon_paths,
        result.metadata_paths,
        strict=True,
    ):
        assert not load_domain_geometry(polygon).is_empty
        payload = load_domain_provenance(
            metadata, polygon_path=polygon, ecotype=ecotype
        )
        assert payload["domain_kind"] == "operational_model_extent"
        assert payload["water_membership"].startswith("canonical_h3_full_counting")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["effective_cell_counts"]["4"] == {
        "SRKW": 1,
        "TRANSIENT": 2,
        "OTHER": 2,
    }

    with pytest.raises(FileExistsError, match="pass --force"):
        build_operational_model_domains(
            sightings_config_path="config/data/sightings.yaml",
            domain_config_path=domain_config,
            data_root=tmp_path,
            run_id="fixture-model-domains-second",
        )
