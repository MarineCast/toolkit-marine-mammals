"""Build reviewed operational whale model-domain polygon artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import polars as pl
from shapely.geometry import box

from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools._core.config import ConfigDocument
from marine_mammal_toolkit.tools._core.config.paths import project_root
from marine_mammal_toolkit.tools._core.persistence import checksum_path

from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    cells_in_domain,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    load_domain_geometry,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    load_domain_provenance,
)

_ECOTYPES = ("SRKW", "TRANSIENT")
_DOMAIN_KIND = "operational_model_extent"
_REVIEW_STATUS = "approved_for_model_domain"


@dataclass(frozen=True)
class OperationalDomainBuildResult:
    polygon_paths: tuple[Path, ...]
    metadata_paths: tuple[Path, ...]
    manifest_path: Path
    effective_cell_counts: Mapping[int, Mapping[str, int]]


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _resolve_data_path(value: str | Path, data_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == "data":
        return (data_root / Path(*candidate.parts[1:])).resolve()
    return (project_root() / candidate).resolve()


def _bbox_from_area(
    areas: Mapping[str, Any], area_key: str
) -> tuple[float, float, float, float]:
    area = _require_mapping(areas.get(area_key), f"areas.{area_key}")
    bounds = _require_mapping(area.get("bbox_wgs84"), f"areas.{area_key}.bbox_wgs84")
    required = ("min_lon", "min_lat", "max_lon", "max_lat")
    missing = [name for name in required if name not in bounds]
    if missing:
        raise ValueError(f"areas.{area_key}.bbox_wgs84 is missing {missing}")
    result = tuple(float(bounds[name]) for name in required)
    min_lon, min_lat, max_lon, max_lat = result
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError(f"areas.{area_key}.bbox_wgs84 is invalid: {result}")
    return result


def _expected_counts(raw: Mapping[str, Any], label: str) -> dict[int, int]:
    return {int(resolution): int(count) for resolution, count in raw.items()}


def _domain_metadata(
    *,
    ecotype: str,
    area_key: str,
    source_reference: str,
    source_authority: str,
    source_release: str,
    approved_at: str,
    polygon_path: Path,
    areas_config: Path,
    areas_config_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ecotype": ecotype,
        "domain_kind": _DOMAIN_KIND,
        "source_authority": source_authority,
        "source_url": source_reference,
        "source_release": source_release,
        "retrieved_at": approved_at,
        "derivation": (
            f"Polygonized areas.{area_key}.bbox_wgs84 from {areas_config}. "
            "The model-grid stage applies representative-point membership to the canonical "
            "H3 full-counting water universe; this polygon is intentionally not water-clipped."
        ),
        "geometry_sha256": checksum_path(polygon_path),
        "review_status": _REVIEW_STATUS,
        "area_key": area_key,
        "areas_config": str(areas_config),
        "areas_config_hash": areas_config_hash,
        "water_membership": "canonical_h3_full_counting_universe_then_representative_point",
    }


def build_operational_model_domains(
    *,
    sightings_config_path: str | Path,
    domain_config_path: str | Path,
    data_root: str | Path,
    run_id: str,
    force: bool = False,
) -> OperationalDomainBuildResult:
    """Materialize AOI polygons and verify their effective water-cell domains."""

    data_root = Path(data_root).expanduser().resolve()
    sightings_document, sightings = load_sightings_config(sightings_config_path)
    domain_document = ConfigDocument.load(
        domain_config_path,
        allowed_keys={
            "schema_version",
            "domain_kind",
            "source_authority",
            "source_release",
            "approved_at",
            "review_status",
            "areas_config",
            "water_universe_root",
            "resolutions",
            "domains",
            "expected_union_cells",
        },
    )
    raw = domain_document.data
    if raw.get("schema_version") != 1:
        raise ValueError("Operational model-domain config schema_version must be 1")
    if raw.get("domain_kind") != _DOMAIN_KIND:
        raise ValueError(f"domain_kind must be {_DOMAIN_KIND!r}")
    if raw.get("review_status") != _REVIEW_STATUS:
        raise ValueError(f"review_status must be {_REVIEW_STATUS!r}")

    areas_path = Path(str(raw["areas_config"])).expanduser()
    if not areas_path.is_absolute():
        areas_path = domain_document.source.parent / areas_path
    areas_document = ConfigDocument.load(areas_path)
    areas = _require_mapping(areas_document.data.get("areas"), "areas")
    definitions = _require_mapping(raw.get("domains"), "domains")
    if set(definitions) != set(_ECOTYPES):
        raise ValueError(f"domains must contain exactly {list(_ECOTYPES)}")
    resolutions = tuple(int(value) for value in raw.get("resolutions", ()))
    if not resolutions or len(resolutions) != len(set(resolutions)):
        raise ValueError("resolutions must be nonempty and unique")

    water_root = _resolve_data_path(str(raw["water_universe_root"]), data_root)
    water_cells: dict[int, set[str]] = {}
    water_inputs: dict[int, dict[str, Any]] = {}
    for resolution in resolutions:
        path = water_root / f"H3_WATER_UNIVERSE_{resolution}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing canonical H{resolution} water universe: {path}"
            )
        frame = pl.read_parquet(path, columns=["H3_INDEX", "H3_RESOLUTION"])
        if frame.is_empty() or frame.get_column("H3_RESOLUTION").unique().to_list() != [
            resolution
        ]:
            raise ValueError(f"Invalid H{resolution} water universe: {path}")
        cells = set(frame.get_column("H3_INDEX").to_list())
        if len(cells) != frame.height:
            raise ValueError(
                f"H{resolution} water universe contains duplicate cells: {path}"
            )
        water_cells[resolution] = cells
        water_inputs[resolution] = {
            "path": str(path),
            "checksum": checksum_path(path),
            "row_count": len(cells),
        }

    output_paths: dict[str, Path] = {}
    domain_geometries: dict[str, Any] = {}
    domain_details: dict[str, dict[str, Any]] = {}
    for ecotype in _ECOTYPES:
        definition = _require_mapping(definitions[ecotype], f"domains.{ecotype}")
        area_key = str(definition["area_key"])
        output = _resolve_data_path(str(definition["output"]), data_root)
        configured_output = _resolve_data_path(
            str(sightings.model_universes[ecotype].polygon), data_root
        )
        if output != configured_output:
            raise ValueError(
                f"domains.{ecotype}.output does not match sightings model_universes: "
                f"{output} != {configured_output}"
            )
        bounds = _bbox_from_area(areas, area_key)
        output_paths[ecotype] = output
        domain_geometries[ecotype] = box(*bounds)
        domain_details[ecotype] = {
            "area_key": area_key,
            "source_reference": str(definition["source_reference"]),
            "bounds_wgs84": list(bounds),
            "expected_cells": _expected_counts(
                _require_mapping(
                    definition.get("expected_cells"),
                    f"domains.{ecotype}.expected_cells",
                ),
                f"domains.{ecotype}.expected_cells",
            ),
        }

    parent_paths = {path.parent for path in output_paths.values()}
    if len(parent_paths) != 1:
        raise ValueError("Operational model-domain outputs must share one directory")
    output_root = next(iter(parent_paths))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "MODEL_DOMAINS_MANIFEST.json"
    final_metadata = {
        ecotype: path.with_suffix(".metadata.json")
        for ecotype, path in output_paths.items()
    }
    final_targets = [*output_paths.values(), *final_metadata.values(), manifest_path]
    existing = [path for path in final_targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Operational model-domain outputs already exist; pass --force to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    staging = Path(tempfile.mkdtemp(prefix=".model-domains-", dir=output_root))
    staged_polygons: dict[str, Path] = {}
    staged_metadata: dict[str, Path] = {}
    try:
        for ecotype in _ECOTYPES:
            staged_polygon = staging / output_paths[ecotype].name
            gpd.GeoDataFrame(
                {
                    "ECOTYPE": [ecotype],
                    "DOMAIN_KIND": [_DOMAIN_KIND],
                    "SOURCE_RELEASE": [str(raw["source_release"])],
                    "AREA_KEY": [domain_details[ecotype]["area_key"]],
                },
                geometry=[domain_geometries[ecotype]],
                crs="EPSG:4326",
            ).to_parquet(staged_polygon, index=False)
            metadata = _domain_metadata(
                ecotype=ecotype,
                area_key=domain_details[ecotype]["area_key"],
                source_reference=domain_details[ecotype]["source_reference"],
                source_authority=str(raw["source_authority"]),
                source_release=str(raw["source_release"]),
                approved_at=str(raw["approved_at"]),
                polygon_path=staged_polygon,
                areas_config=areas_document.source,
                areas_config_hash=areas_document.config_hash,
            )
            staged_sidecar = staged_polygon.with_suffix(".metadata.json")
            atomic_write_json(staged_sidecar, metadata)
            load_domain_geometry(staged_polygon)
            load_domain_provenance(
                staged_sidecar, polygon_path=staged_polygon, ecotype=ecotype
            )
            staged_polygons[ecotype] = staged_polygon
            staged_metadata[ecotype] = staged_sidecar

        effective_counts: dict[int, dict[str, int]] = {}
        expected_union = _expected_counts(
            _require_mapping(raw.get("expected_union_cells"), "expected_union_cells"),
            "expected_union_cells",
        )
        selected_by_resolution: dict[int, dict[str, set[str]]] = {}
        for resolution in resolutions:
            selected = {
                ecotype: cells_in_domain(
                    water_cells[resolution], domain_geometries[ecotype]
                )
                for ecotype in _ECOTYPES
            }
            selected_by_resolution[resolution] = selected
            counts = {ecotype: len(cells) for ecotype, cells in selected.items()}
            counts["OTHER"] = len(selected["SRKW"] | selected["TRANSIENT"])
            effective_counts[resolution] = counts
            for ecotype in _ECOTYPES:
                expected = domain_details[ecotype]["expected_cells"].get(resolution)
                if expected != counts[ecotype]:
                    raise ValueError(
                        f"{ecotype} H{resolution} parity failed: expected {expected}, "
                        f"observed {counts[ecotype]}"
                    )
            if expected_union.get(resolution) != counts["OTHER"]:
                raise ValueError(
                    f"OTHER H{resolution} parity failed: expected {expected_union.get(resolution)}, "
                    f"observed {counts['OTHER']}"
                )

        manifest = {
            "schema_version": 1,
            "dataset_family": "whale.spatial_support.operational_model_domains",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "producer": "whale.spatial_support.operational_model_domain.v1",
            "domain_kind": _DOMAIN_KIND,
            "membership_rule": "representative_point",
            "water_membership": "canonical_h3_full_counting_universe_then_domain",
            "sightings_config": str(sightings_document.source),
            "sightings_config_hash": sightings_document.config_hash,
            "domain_config": str(domain_document.source),
            "domain_config_hash": domain_document.config_hash,
            "areas_config": str(areas_document.source),
            "areas_config_hash": areas_document.config_hash,
            "water_universes": {str(key): value for key, value in water_inputs.items()},
            "effective_cell_counts": {
                str(key): value for key, value in effective_counts.items()
            },
            "outputs": [
                {
                    "dataset_id": f"whale.spatial_support.{ecotype.lower()}_model_domain",
                    "ecotype": ecotype,
                    "path": str(output_paths[ecotype]),
                    "metadata_path": str(final_metadata[ecotype]),
                    "geometry_sha256": checksum_path(staged_polygons[ecotype]),
                    "metadata_sha256": checksum_path(staged_metadata[ecotype]),
                    "bounds_wgs84": domain_details[ecotype]["bounds_wgs84"],
                    "effective_cell_counts": {
                        str(resolution): len(
                            selected_by_resolution[resolution][ecotype]
                        )
                        for resolution in resolutions
                    },
                }
                for ecotype in _ECOTYPES
            ],
        }
        staged_manifest = staging / manifest_path.name
        atomic_write_json(staged_manifest, manifest)

        for ecotype in _ECOTYPES:
            os.replace(staged_polygons[ecotype], output_paths[ecotype])
            os.replace(staged_metadata[ecotype], final_metadata[ecotype])
        os.replace(staged_manifest, manifest_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return OperationalDomainBuildResult(
        polygon_paths=tuple(output_paths[ecotype] for ecotype in _ECOTYPES),
        metadata_paths=tuple(final_metadata[ecotype] for ecotype in _ECOTYPES),
        manifest_path=manifest_path,
        effective_cell_counts=effective_counts,
    )
