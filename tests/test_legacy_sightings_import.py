from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/import_legacy_sightings_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_sightings_import", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importer_copies_only_supported_local_inputs_with_checksums(tmp_path):
    legacy = tmp_path / "legacy"
    source_root = legacy / "raw/sightings"
    (source_root / "twm_export").mkdir(parents=True)
    (source_root / "acartia_export/Archive").mkdir(parents=True)
    (source_root / "bc_sightings_export").mkdir(parents=True)
    (source_root / "inaturalist_export").mkdir(parents=True)
    (source_root / "twm_export/twm1975.csv").write_text("id\n1\n")
    (source_root / "acartia_export/current.csv").write_text("id\n2\n")
    (source_root / "acartia_export/Archive/older.csv").write_text("id\n3\n")
    (source_root / "bc_sightings_export/unsupported.csv").write_text("id\n4\n")
    (source_root / "inaturalist_export/redundant.csv").write_text("id\n5\n")

    destination = tmp_path / "workspace/data/sightings/source_inputs"
    manifest = _module().import_inputs(
        legacy_data_root=legacy,
        destination=destination,
    )

    assert (destination / "twm/twm1975.csv").read_text() == "id\n1\n"
    assert (destination / "acartia/current.csv").read_text() == "id\n2\n"
    assert (destination / "acartia/Archive/older.csv").read_text() == "id\n3\n"
    assert not (destination / "bc_sightings_export").exists()
    assert not (destination / "inaturalist_export").exists()
    payload = json.loads(manifest.read_text())
    assert payload["sources"]["twm"]["file_count"] == 1
    assert payload["sources"]["acartia"]["file_count"] == 2
    assert payload["rights"]["redistribution_approved"] is False


def test_importer_refuses_to_replace_changed_inputs_without_overwrite(tmp_path):
    legacy = tmp_path / "legacy"
    for relative in ("twm_export/twm1975.csv", "acartia_export/current.csv"):
        path = legacy / "raw/sightings" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("id\n1\n")
    destination = tmp_path / "workspace/data/sightings/source_inputs"
    importer = _module()
    importer.import_inputs(legacy_data_root=legacy, destination=destination)
    (destination / "twm/twm1975.csv").write_text("changed\n")

    with pytest.raises(FileExistsError, match="--overwrite"):
        importer.import_inputs(legacy_data_root=legacy, destination=destination)


def test_product_importer_provisions_local_inputs_and_support(tmp_path):
    legacy = tmp_path / "legacy"
    for relative in ("twm_export/twm1975.csv", "acartia_export/current.csv"):
        path = legacy / "raw/sightings" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("id\n1\n")
    domains = legacy / "processed/domain/whale_layer/spatial_support"
    domains.mkdir(parents=True)
    for name in _module().MODEL_DOMAIN_FILES:
        (domains / name).write_text(name)
    support = legacy / "processed/domain/environmental_layer/seascape/spatial_support"
    for relative in _module().SEASCAPE_SUPPORT_FILES:
        path = support / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    config = tmp_path / "seascape-config"
    (config / "data").mkdir(parents=True)
    (config / "common.yaml").write_text("areas: {}\n")
    (config / "data/project.yaml").write_text("base_directory: .\n")
    product = tmp_path / "data/marine-mammals/killer-whales"

    manifest = _module().import_product_inputs(
        legacy_data_root=legacy,
        product_root=product,
        seascape_config_source=config,
    )

    assert (product / "raw/source-inputs/twm/twm1975.csv").is_file()
    assert (product / "raw/support/model-domains/SRKW_MODEL_DOMAIN.parquet").is_file()
    assert (
        product
        / "raw/support/seascape/data/processed/domain/environmental_layer/seascape/"
        "spatial_support/water_network/H3_WATER_PASSABLE_EDGES_RES_6.parquet"
    ).is_file()
    assert (product / "raw/support/seascape/config/data/project.yaml").is_file()
    portable_project = (
        product / "raw/support/seascape/config/data/product-project.yaml"
    ).read_text()
    assert "${env:SEASCAPE_WORKSPACE}" in portable_project
    assert "SEASCAPE_LAYER: environment_seascape.yaml" in portable_project
    payload = json.loads(manifest.read_text())
    assert payload["rights"]["redistribution_approved"] is False
    assert len(payload["model_domains"]) == len(_module().MODEL_DOMAIN_FILES)
