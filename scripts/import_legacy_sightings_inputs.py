#!/usr/bin/env python3
"""Import local-only sightings inputs from a legacy OrcaCast data workspace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_LAYOUTS = {
    "twm": Path("raw/sightings/twm_export"),
    "acartia": Path("raw/sightings/acartia_export"),
}

EXCLUDED_LEGACY_INPUTS = {
    "raw/sightings/bc_sightings_export": (
        "BC ArcGIS is not a supported canonical sightings source."
    ),
    "raw/sightings/inaturalist_export": (
        "iNaturalist is collected through the complete paginated provider API."
    ),
}

MODEL_DOMAIN_FILES = (
    "SRKW_MODEL_DOMAIN.parquet",
    "SRKW_MODEL_DOMAIN.metadata.json",
    "TRANSIENT_MODEL_DOMAIN.parquet",
    "TRANSIENT_MODEL_DOMAIN.metadata.json",
    "MODEL_DOMAINS_MANIFEST.json",
)

SEASCAPE_SUPPORT_FILES = (
    "h3_geometry/H3_MARINE_SUPPORT_RES_6.parquet",
    "water_network/H3_WATER_PASSABLE_EDGES_RES_6.parquet",
    "water_network/H3_WATER_CONNECTORS_RES_6.parquet",
    "water_network/H3_WATER_NEIGHBORHOODS_RES_6.parquet",
    "water_network/_dataset_manifest.json",
)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_source(source: Path, destination: Path, *, overwrite: bool) -> dict[str, Any]:
    files = sorted(path for path in source.rglob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV inputs found beneath {source}")

    inventory: list[dict[str, Any]] = []
    for source_file in files:
        relative = source_file.relative_to(source)
        target = destination / relative
        source_checksum = _checksum(source_file)
        if target.exists():
            target_checksum = _checksum(target)
            if target_checksum != source_checksum and not overwrite:
                raise FileExistsError(
                    f"Refusing to replace changed input without --overwrite: {target}"
                )
        if not target.exists() or _checksum(target) != source_checksum:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
        copied_checksum = _checksum(target)
        if copied_checksum != source_checksum:
            raise OSError(f"Copied input checksum mismatch: {target}")
        inventory.append(
            {
                "path": relative.as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": copied_checksum,
            }
        )

    return {
        "source_path": str(source.resolve()),
        "destination_path": str(destination.resolve()),
        "file_count": len(inventory),
        "size_bytes": sum(item["size_bytes"] for item in inventory),
        "files": inventory,
    }


def _copy_file(source: Path, destination: Path, *, overwrite: bool) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"Required legacy support file is missing: {source}")
    source_checksum = _checksum(source)
    if destination.exists():
        target_checksum = _checksum(destination)
        if target_checksum != source_checksum and not overwrite:
            raise FileExistsError(
                f"Refusing to replace changed input without --overwrite: {destination}"
            )
    if not destination.exists() or _checksum(destination) != source_checksum:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if _checksum(destination) != source_checksum:
        raise OSError(f"Copied input checksum mismatch: {destination}")
    return {
        "source_path": str(source.resolve()),
        "path": destination.as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": source_checksum,
    }


def _copy_tree(
    source: Path, destination: Path, *, overwrite: bool
) -> list[dict[str, Any]]:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No support files found beneath {source}")
    return [
        _copy_file(
            item,
            destination / item.relative_to(source),
            overwrite=overwrite,
        )
        for item in files
    ]


def import_inputs(
    *, legacy_data_root: Path, destination: Path, overwrite: bool = False
) -> Path:
    """Copy supported local inputs and write a checksum-complete manifest."""

    root = legacy_data_root.expanduser().resolve()
    target_root = destination.expanduser().resolve()
    sources = {
        name: _copy_source(root / relative, target_root / name, overwrite=overwrite)
        for name, relative in SOURCE_LAYOUTS.items()
    }
    payload = {
        "schema_version": "1",
        "producer": "scripts/import_legacy_sightings_inputs.py",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_data_root": str(root),
        "rights": {
            "source_license": "UNKNOWN",
            "source_use_class": "INTERNAL_ONLY",
            "redistribution_approved": False,
        },
        "sources": sources,
        "excluded_legacy_inputs": EXCLUDED_LEGACY_INPUTS,
    }
    target_root.mkdir(parents=True, exist_ok=True)
    manifest = target_root / "_import_manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest.name}.", dir=manifest.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, manifest)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return manifest


def import_product_inputs(
    *,
    legacy_data_root: Path,
    product_root: Path,
    overwrite: bool = False,
    seascape_config_source: Path | None = None,
) -> Path:
    """Provision the local inputs needed by the consumer-facing product."""

    legacy = legacy_data_root.expanduser().resolve()
    product = product_root.expanduser().resolve()
    raw = product / "raw"
    source_manifest = import_inputs(
        legacy_data_root=legacy,
        destination=raw / "source-inputs",
        overwrite=overwrite,
    )
    model_source = legacy / "processed/domain/whale_layer/spatial_support"
    model_destination = raw / "support/model-domains"
    model_domains = [
        _copy_file(
            model_source / name,
            model_destination / name,
            overwrite=overwrite,
        )
        for name in MODEL_DOMAIN_FILES
    ]

    seascape_source = (
        legacy / "processed/domain/environmental_layer/seascape/spatial_support"
    )
    seascape_destination = (
        raw / "support/seascape/data/processed/domain/environmental_layer/seascape/"
        "spatial_support"
    )
    seascape_support = [
        _copy_file(
            seascape_source / relative,
            seascape_destination / relative,
            overwrite=overwrite,
        )
        for relative in SEASCAPE_SUPPORT_FILES
    ]

    config_destination = raw / "support/seascape/config"
    if seascape_config_source is not None:
        seascape_config = _copy_tree(
            seascape_config_source.expanduser().resolve(),
            config_destination,
            overwrite=overwrite,
        )
    else:
        resource = importlib.resources.files("seascape.resources").joinpath("config")
        with importlib.resources.as_file(resource) as config_source:
            seascape_config = _copy_tree(
                config_source,
                config_destination,
                overwrite=overwrite,
            )
    product_project_resource = importlib.resources.files(
        "marine_mammal_toolkit"
    ).joinpath("resources/killer_whales/seascape_product_project.yaml")
    with importlib.resources.as_file(product_project_resource) as product_project:
        seascape_config.append(
            _copy_file(
                product_project,
                config_destination / "data/product-project.yaml",
                overwrite=overwrite,
            )
        )

    payload = {
        "schema_version": "1",
        "producer": "scripts/import_legacy_sightings_inputs.py",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_data_root": str(legacy),
        "product_root": str(product),
        "source_input_manifest": str(source_manifest.relative_to(product)),
        "rights": {
            "source_license": "UNKNOWN",
            "source_use_class": "INTERNAL_ONLY",
            "redistribution_approved": False,
        },
        "model_domains": model_domains,
        "seascape_support": seascape_support,
        "seascape_config": seascape_config,
    }
    manifest = raw / "_input_manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest.name}.", dir=manifest.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, manifest)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-data-root", required=True, type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/sightings/source_inputs"),
    )
    parser.add_argument(
        "--product-root",
        type=Path,
        help=(
            "Provision source inputs, model domains, and H3 r6 water support "
            "beneath a killer-whale product root."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = (
        import_product_inputs(
            legacy_data_root=args.legacy_data_root,
            product_root=args.product_root,
            overwrite=args.overwrite,
        )
        if args.product_root is not None
        else import_inputs(
            legacy_data_root=args.legacy_data_root,
            destination=args.destination,
            overwrite=args.overwrite,
        )
    )
    print(result)


if __name__ == "__main__":
    main()
