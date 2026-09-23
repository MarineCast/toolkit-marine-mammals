"""Installed-package configuration resources for the killer-whale run."""

from importlib.resources import files
from pathlib import Path


def config_path(name: str = "sightings") -> Path:
    if name not in {
        "sightings",
        "sightings_product",
        "populations",
        "model_domains",
        "areas",
    }:
        raise ValueError(f"Unknown killer-whale configuration: {name}")
    return Path(
        str(
            files("marine_mammal_toolkit").joinpath(
                "resources", "killer_whales", name + ".yaml"
            )
        )
    )
