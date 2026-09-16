"""Offline tests use temporary workspaces, never local scientific artifacts."""

from pathlib import Path
import shutil

import pytest
from openpyxl import Workbook

from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
from marine_mammal_toolkit.tools._core.config.paths import workspace
from marine_mammal_toolkit.cetaceans.killer_whales.catalog import (
    register_builtin_datasets,
)


@pytest.fixture(autouse=True)
def synthetic_workspace(tmp_path):
    root = tmp_path / "workspace"
    config = root / "config/data"
    config.mkdir(parents=True)
    for name, target in (
        ("sightings", "sightings.yaml"),
        ("populations", "whale.yaml"),
        ("model_domains", "whale/model_domains.yaml"),
    ):
        dest = config / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path(name), dest)
    shutil.copy2(config_path("areas"), root / "config/common.yaml")
    (config / "project.yaml").write_text("base_directory: .\nWHALE_LAYER: whale.yaml\n")
    population = (
        root / "data/raw/orca_population/Number of Southern Resident killer whales.xlsx"
    )
    population.parent.mkdir(parents=True)
    book = Workbook()
    book.active.title = "Chart Data"
    book.active.append(["Census Year", "J Pod", "K Pod", "L Pod", "All Pods"])
    book.active.append([2020, 1, 1, 1, 3])
    book.save(population)
    register_builtin_datasets()
    with workspace(root):
        yield root
