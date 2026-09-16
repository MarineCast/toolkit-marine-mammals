"""Explicit polygon-backed whale model-universe contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import h3
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools._core.persistence import checksum_path

if TYPE_CHECKING:
    from marine_mammal_toolkit.tools._core.config import ConfigDocument

    from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
        SightingsPipelineConfig,
    )


@dataclass(frozen=True)
class ModelDomains:
    geometries: dict[str, Polygon | MultiPolygon]
    artifacts: tuple[ArtifactRef, ...]


_PROVENANCE_FIELDS = {
    "schema_version",
    "ecotype",
    "source_authority",
    "source_url",
    "source_release",
    "retrieved_at",
    "derivation",
    "geometry_sha256",
    "review_status",
}
_DOMAIN_PRODUCERS = {
    "external_ecological_domain": "approved_external_model_domain",
    "operational_model_extent": "whale.spatial_support.operational_model_domain.v1",
}


def load_domain_geometry(path: str | Path) -> Polygon | MultiPolygon:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Whale model-universe polygon does not exist: {source}. "
            "Supply an ecotype-specific, provenance-reviewed polygon; bounding-box "
            "fallbacks are intentionally unsupported."
        )
    frame = (
        gpd.read_parquet(source)
        if source.suffix.lower() == ".parquet"
        else gpd.read_file(source)
    )
    if frame.empty or "geometry" not in frame:
        raise ValueError(f"Whale model-universe polygon is empty: {source}")
    if frame.crs is None:
        raise ValueError(f"Whale model-universe polygon lacks a CRS: {source}")
    frame = frame.to_crs("EPSG:4326")
    geometry: Any = shapely.make_valid(shapely.union_all(frame.geometry.to_numpy()))
    if not isinstance(geometry, (Polygon, MultiPolygon)) or geometry.is_empty:
        raise ValueError(f"Whale model-universe artifact is not polygonal: {source}")
    return geometry


def load_domain_provenance(
    path: str | Path, *, polygon_path: str | Path, ecotype: str
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    polygon = Path(polygon_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Whale model-universe provenance does not exist: {source}. "
            "The polygon and its source review are both required."
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model-universe provenance must be a JSON object: {source}")
    missing = sorted(_PROVENANCE_FIELDS - set(payload))
    if missing:
        raise ValueError(f"Model-universe provenance is missing {missing}: {source}")
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported model-universe provenance schema: {source}")
    if str(payload["ecotype"]).upper() != ecotype.upper():
        raise ValueError(
            f"Model-universe provenance ecotype does not match {ecotype}: {source}"
        )
    if payload["review_status"] != "approved_for_model_domain":
        raise ValueError(
            f"Model-universe provenance is not approved for modeling: {source}"
        )
    domain_kind = str(payload.get("domain_kind") or "external_ecological_domain")
    if domain_kind not in _DOMAIN_PRODUCERS:
        raise ValueError(
            f"Unsupported model-universe domain_kind={domain_kind!r}: {source}"
        )
    if payload["geometry_sha256"] != checksum_path(polygon):
        raise ValueError(
            f"Model-universe polygon checksum does not match provenance: {source}"
        )
    for field in _PROVENANCE_FIELDS - {"schema_version"}:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(
                f"Model-universe provenance field {field} is empty: {source}"
            )
    return payload


def load_model_domains(
    document: "ConfigDocument", config: "SightingsPipelineConfig"
) -> ModelDomains:
    geometries: dict[str, Polygon | MultiPolygon] = {}
    artifacts: list[ArtifactRef] = []
    for ecotype in ("SRKW", "TRANSIENT"):
        settings = config.model_universes[ecotype]
        path = document.resolve_path(settings.polygon)
        geometries[ecotype] = load_domain_geometry(path)
        provenance_path = path.with_suffix(".metadata.json")
        provenance = load_domain_provenance(
            provenance_path, polygon_path=path, ecotype=ecotype
        )
        artifacts.append(
            ArtifactRef(
                kind="spatial_support",
                dataset_id=f"whale.spatial_support.{ecotype.lower()}_model_domain",
                path=path,
                producer=_DOMAIN_PRODUCERS[
                    str(provenance.get("domain_kind") or "external_ecological_domain")
                ],
                schema_version="1",
                checksum=checksum_path(path),
                inputs=(checksum_path(provenance_path),),
                spatial_coverage={
                    "membership_rule": settings.membership_rule,
                    "provenance_path": str(provenance_path),
                    "source_authority": provenance["source_authority"],
                    "source_url": provenance["source_url"],
                    "source_release": provenance["source_release"],
                    "review_status": provenance["review_status"],
                    "domain_kind": provenance.get(
                        "domain_kind", "external_ecological_domain"
                    ),
                    "derivation": provenance["derivation"],
                },
            )
        )
    return ModelDomains(geometries=geometries, artifacts=tuple(artifacts))


def cells_in_domain(cells: set[str], geometry: Polygon | MultiPolygon) -> set[str]:
    ordered = sorted(cells)
    centers = np.asarray([h3.cell_to_latlng(cell) for cell in ordered], dtype=float)
    selected = shapely.covers(geometry, shapely.points(centers[:, 1], centers[:, 0]))
    return {cell for cell, keep in zip(ordered, selected, strict=True) if bool(keep)}
