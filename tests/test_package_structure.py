"""Smoke tests for the public namespace scaffold."""

from importlib import import_module

import marine_mammal_toolkit

NAMESPACES = (
    "observations",
    "populations",
    "taxonomy",
    "telemetry",
    "acoustic",
    "schemas",
    "quality",
    "cetaceans",
    "cetaceans.killer_whales",
    "cetaceans.humpbacks",
    "cetaceans.gray_whales",
    "pinnipeds",
    "pinnipeds.haulouts",
    "pinnipeds.seals",
    "pinnipeds.sea_lions",
)


def test_package_version() -> None:
    assert marine_mammal_toolkit.__version__ == "0.1.0"


def test_documented_namespaces_are_importable() -> None:
    for namespace in NAMESPACES:
        prefix = "" if namespace.startswith(("cetaceans", "pinnipeds")) else "tools."
        assert import_module(f"marine_mammal_toolkit.{prefix}{namespace}")
