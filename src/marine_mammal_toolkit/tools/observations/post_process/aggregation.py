from __future__ import annotations

import heapq
import json
import math
import os
import shutil
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import h3  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools._core.data import ProcessingMode
from marine_mammal_toolkit.tools._core.data import StageResult
from marine_mammal_toolkit.tools._core.data import ValidationReport
from marine_mammal_toolkit.tools._core.persistence import checksum_path
from marine_mammal_toolkit.tools._core.seascape import load_water_graph
from marine_mammal_toolkit.tools._core.seascape import load_water_network_config

from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    SightingsPipelineConfig,
)
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.schemas.observations import MODEL_GRID_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import MODEL_INTENSITY_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import IntensityRequest
from marine_mammal_toolkit.tools.schemas.observations import ModelGridRequest
from marine_mammal_toolkit.tools.observations.post_process.counts import (
    UNVERIFIED_PERIOD_STATUS,
)
from marine_mammal_toolkit.tools.observations.post_process.counts import (
    _coverage_contract,
)
from marine_mammal_toolkit.tools.observations.post_process.counts import _universe
from marine_mammal_toolkit.tools.observations.runtime import code_revision
from marine_mammal_toolkit.tools.observations.runtime import resume_result
from marine_mammal_toolkit.tools.observations.runtime import stage_signature
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    cells_in_domain,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    load_model_domains,
)


def make_ring_weights_power(
    ring_count: int, *, w0: float, w_k: float, alpha: float, round_to: int
) -> dict[int, float]:
    if ring_count < 1:
        raise ValueError("ring_count must be at least one")
    if ring_count == 1:
        return {0: round(float(w0), round_to)}
    outer = ring_count - 1
    weights = {
        ring: round(float(w_k + (w0 - w_k) * ((1 - ring / outer) ** alpha)), round_to)
        for ring in range(ring_count)
    }
    weights[0], weights[outer] = round(w0, round_to), round(w_k, round_to)
    return weights


def _kernel_geometry(
    config: SightingsPipelineConfig, resolution: int
) -> dict[str, float | int | str]:
    edge = float(h3.average_hexagon_edge_length(resolution, unit="km"))
    step = edge * math.sqrt(3)
    max_ring = max(1, math.ceil(config.kernel.radius_km / step))
    half_weight_distance = config.kernel.radius_km * (
        1 - 0.5 ** (1 / config.kernel.exponent)
    )
    return {
        "configured_radius_km": config.kernel.radius_km,
        "routing_method": "bounded_dijkstra",
        "distance_metric": (
            "canonical_r6_edge_distance_km"
            if resolution == 6
            else "canonical_r6_passability_with_parent_center_edge_distance_km"
        ),
        "edge_length_km": edge,
        "center_step_km": step,
        "max_ring": max_ring,
        "ring_count": max_ring + 1,
        "effective_half_weight_km": half_weight_distance,
    }


def _model_cells(
    all_cells: set[str], geometries: dict[str, Any], bucket: str
) -> set[str]:
    if bucket in geometries:
        return cells_in_domain(all_cells, geometries[bucket])
    return set().union(
        *(cells_in_domain(all_cells, geometry) for geometry in geometries.values())
    )


def _periods(
    start: date,
    end: date,
    frequency: str,
    mode: ProcessingMode,
    *,
    zero_fill_verified: bool = True,
) -> pd.DataFrame:
    if frequency == "daily":
        frame = pd.DataFrame({"PERIOD_START": pd.date_range(start, end, freq="D").date})
        frame["PERIOD_END"] = frame.PERIOD_START
        frame["PERIOD_STATUS"] = "COMPLETE"
        frame["IS_COMPLETE_PERIOD"] = True
        if not zero_fill_verified:
            frame["PERIOD_STATUS"] = UNVERIFIED_PERIOD_STATUS
            frame["IS_COMPLETE_PERIOD"] = False
        return frame
    monday = start - timedelta(days=start.weekday())
    starts = pd.date_range(monday, end, freq="7D").date
    frame = pd.DataFrame({"PERIOD_START": starts})
    frame["PERIOD_END"] = frame.PERIOD_START.map(
        lambda value: value + timedelta(days=6)
    )
    complete = frame.PERIOD_START.ge(start) & frame.PERIOD_END.le(end)
    final_start = end - timedelta(days=end.weekday())
    open_mask = frame.PERIOD_START.eq(final_start) & ~complete
    frame = frame.loc[
        complete | (open_mask if mode is ProcessingMode.AS_OF else False)
    ].copy()
    frame["PERIOD_STATUS"] = np.where(complete.loc[frame.index], "COMPLETE", "OPEN")
    frame["IS_COMPLETE_PERIOD"] = frame.PERIOD_STATUS.eq("COMPLETE")
    if not zero_fill_verified:
        frame["PERIOD_STATUS"] = UNVERIFIED_PERIOD_STATUS
        frame["IS_COMPLETE_PERIOD"] = False
    return frame


def _resolved_coverage_contract(
    artifact: ArtifactRef,
    manifest: dict[str, Any],
    start: date,
    end: date,
) -> dict[str, Any]:
    """Resolve verified zero semantics from typed lineage or its dataset manifest."""

    source = manifest.get("coverage_contract")
    if isinstance(source, dict) and source.get("zero_fill_verified"):
        coverage_start = source.get("coverage_start")
        coverage_through = source.get("coverage_through")
        if coverage_start and coverage_through:
            verified = start >= date.fromisoformat(
                str(coverage_start)
            ) and end <= date.fromisoformat(str(coverage_through))
            if verified:
                return {
                    **source,
                    "requested_start": start.isoformat(),
                    "requested_end": end.isoformat(),
                    "zero_fill_verified": True,
                    "zero_semantics": "verified_no_report",
                }
    return _coverage_contract(artifact, start, end)


def _require_dense_coverage(contract: dict[str, Any]) -> None:
    if not contract.get("zero_fill_verified"):
        raise ValueError(
            "Dense reported-sighting grids require verified source coverage because "
            "the current schema cannot represent unavailable cell-periods as nullable. "
            "Refusing to materialize unverified coverage as no-report zeros."
        )


def _period_columns(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    out = frame.copy()
    start = pd.to_datetime(out.PERIOD_START)
    out["YEAR"] = start.dt.year.astype("int16")
    out["FREQUENCY"] = frequency
    if frequency == "daily":
        out["DAY_OF_YEAR"] = start.dt.dayofyear.astype("Int16")
        out["ISO_YEAR"], out["ISO_WEEK"] = pd.NA, pd.NA
    else:
        iso = start.dt.isocalendar()
        out["DAY_OF_YEAR"] = pd.NA
        out["ISO_YEAR"] = iso.year.astype("Int16")
        out["ISO_WEEK"] = iso.week.astype("Int8")
    return out


def _read_counts(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def _dense_chunk(
    counts: pd.DataFrame,
    cells: set[str],
    periods: pd.DataFrame,
    bucket: str,
    resolution: int,
    frequency: str,
    coverage_contract: dict[str, Any],
) -> pd.DataFrame:
    counts = counts.copy()
    grid = pd.MultiIndex.from_product(
        [sorted(cells), periods.PERIOD_START], names=["H3_INDEX", "PERIOD_START"]
    ).to_frame(index=False)
    grid = grid.merge(periods, on="PERIOD_START", how="left")
    measure_columns = [
        "SIGHTING_COUNT",
        "SOURCE_REPORT_COUNT",
        "OBSERVED_SIGHTING_COUNT",
        "HARD_IMPUTED_COUNT",
        "EXPECTED_SIGHTING_COUNT",
        "MATURE_EXPECTED_SIGHTING_COUNT",
        "PROVISIONAL_EXPECTED_COUNT",
        "EXPECTED_UNKNOWN_COUNT",
    ]
    for column in measure_columns:
        if column not in counts:
            if column in {"EXPECTED_SIGHTING_COUNT", "MATURE_EXPECTED_SIGHTING_COUNT"}:
                counts[column] = counts.get("SIGHTING_COUNT", 0)
            else:
                counts[column] = 0
    selected = counts[
        counts.ECOTYPE_BUCKET.eq(bucket)
        & counts.H3_RESOLUTION.eq(resolution)
        & counts.FREQUENCY.eq(frequency)
    ][["H3_INDEX", "PERIOD_START", *measure_columns]]
    dense = grid.merge(selected, on=["H3_INDEX", "PERIOD_START"], how="left")
    dense[measure_columns] = dense[measure_columns].fillna(0)
    integer_measures = [
        "SIGHTING_COUNT",
        "SOURCE_REPORT_COUNT",
        "OBSERVED_SIGHTING_COUNT",
        "HARD_IMPUTED_COUNT",
    ]
    dense[integer_measures] = dense[integer_measures].astype("int32")
    dense["REPORTED_SIGHTING"] = dense.SIGHTING_COUNT.gt(0).astype("int8")
    dense["TARGET_COHORT_ID"] = str(coverage_contract["target_cohort_id"])
    dense["TARGET_COHORT_STATUS"] = str(coverage_contract["target_cohort_status"])
    represented_report = dense.REPORTED_SIGHTING.eq(
        1
    ) | dense.EXPECTED_SIGHTING_COUNT.gt(0)
    dense["REPORTING_STATE"] = np.where(represented_report, "REPORTED", "NO_REPORT")
    dense["H3_RESOLUTION"] = resolution
    dense["ECOTYPE_BUCKET"] = bucket
    return _period_columns(dense, frequency)[MODEL_GRID_SCHEMA.names]


def _write_partitions(
    frame: pd.DataFrame, root: Path, schema: pa.Schema, part_index: int
) -> list[dict[str, Any]]:
    frequency = str(frame.FREQUENCY.iloc[0])
    year_column = "ISO_YEAR" if frequency == "weekly" else "YEAR"
    year_name = "iso_year" if frequency == "weekly" else "year"
    metadata: list[dict[str, Any]] = []
    for suffix, (year, group) in enumerate(frame.groupby(year_column, dropna=False)):
        if pd.isna(year):
            raise ValueError(
                f"Missing {year_column} in {frequency} model-grid partition"
            )
        path = (
            root / f"{year_name}={int(year)}/part-{part_index:05d}-{suffix:02d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(group, preserve_index=False).cast(schema),
            path,
            compression="zstd",
        )
        metadata.append(
            {
                "path": str(path.relative_to(root)),
                "checksum": checksum_path(path),
                "expected_cells": int(group.H3_INDEX.nunique()),
                "expected_periods": int(group.PERIOD_START.nunique()),
                "expected_rows": len(group),
                "period_statuses": sorted(set(group.PERIOD_STATUS)),
            }
        )
    return metadata


def build_model_grid(request: ModelGridRequest) -> StageResult:
    if request.ecotype_counts is None:
        raise ValueError("Model grid requires an ecotype_counts artifact")
    if (
        request.ecotype_counts.processing_mode
        and request.ecotype_counts.processing_mode != request.mode.value
    ):
        raise ValueError("Model-grid processing mode must match the count artifact")
    if request.mode is ProcessingMode.AS_OF and not (
        request.knowledge_cutoff or request.ecotype_counts.knowledge_cutoff
    ):
        raise ValueError("AS_OF model grids require a count artifact knowledge cutoff")
    from marine_mammal_toolkit.tools.quality.observations import (
        validate_sightings_artifact,
    )

    input_report = validate_sightings_artifact(request.ecotype_counts)
    input_report.require_valid()
    document, config = load_sightings_config(request.config)
    model_domains = load_model_domains(document, config)
    counts = _read_counts(request.ecotype_counts.path)
    count_manifest_path = request.ecotype_counts.path / "_dataset_manifest.json"
    count_manifest = (
        json.loads(count_manifest_path.read_text())
        if count_manifest_path.exists()
        else {}
    )
    manifest_start = (
        date.fromisoformat(count_manifest["start_date"])
        if count_manifest.get("start_date")
        else None
    )
    manifest_end = (
        date.fromisoformat(count_manifest["end_date"])
        if count_manifest.get("end_date")
        else None
    )
    start = request.start_date or manifest_start
    end = request.end_date or manifest_end
    if start is None or end is None:
        raise ValueError(
            "Empty counts require explicit model-grid start_date and end_date"
        )
    if start > end:
        raise ValueError("Model-grid start_date must be on or before end_date")
    coverage_contract = _resolved_coverage_contract(
        request.ecotype_counts, count_manifest, start, end
    )
    _require_dense_coverage(coverage_contract)
    inputs = (
        request.ecotype_counts,
        *request.water_universes,
        *model_domains.artifacts,
    )
    signature, signature_payload = stage_signature(
        stage="whale.sightings.model_grid",
        semantic_version="7",
        config_hash=document.config_hash,
        inputs=inputs,
        parameters={
            "start_date": start,
            "end_date": end,
            "resolutions": request.resolutions,
            "frequencies": request.frequencies,
            "mode": request.mode.value,
            "knowledge_cutoff": request.knowledge_cutoff
            or request.ecotype_counts.knowledge_cutoff,
        },
    )
    manifest_path = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/manifests/model_grid/{signature}.json"
    )
    resumed = resume_result(
        enabled=request.resume,
        manifest_path=manifest_path,
        config_hash=document.config_hash,
        inputs=inputs,
        signature=signature,
    )
    if resumed is not None:
        return resumed
    staging = (
        request.data_root / ".staging" / request.run_id / "whale.sightings.model_grid"
    )
    if staging.exists():
        shutil.rmtree(staging)
    partition_metadata: list[dict[str, Any]] = []
    group_expectations: list[dict[str, Any]] = []
    for resolution in request.resolutions:
        full_cells, _ = _universe(request.water_universes, resolution)
        for bucket in config.count_policy.buckets:
            cells = _model_cells(full_cells, model_domains.geometries, bucket)
            if not cells:
                raise ValueError(f"Model universe is empty for {bucket} H{resolution}")
            for frequency in request.frequencies:
                periods = _periods(
                    start,
                    end,
                    frequency,
                    request.mode,
                    zero_fill_verified=bool(coverage_contract["zero_fill_verified"]),
                )
                group_expectations.append(
                    {
                        "bucket": bucket,
                        "frequency": frequency,
                        "resolution": resolution,
                        "expected_cells": len(cells),
                        "expected_periods": len(periods),
                        "expected_rows": len(cells) * len(periods),
                    }
                )
                chunk_size = config.aggregation_chunk_periods[frequency]  # type: ignore[index]
                root = (
                    staging
                    / f"ecotype={bucket}/frequency={frequency}/resolution={resolution}"
                )
                for part_index, offset in enumerate(range(0, len(periods), chunk_size)):
                    chunk = periods.iloc[offset : offset + chunk_size]
                    dense = _dense_chunk(
                        counts,
                        cells,
                        chunk,
                        bucket,
                        resolution,
                        frequency,
                        coverage_contract,
                    )
                    for metadata in _write_partitions(
                        dense, root, MODEL_GRID_SCHEMA, part_index
                    ):
                        metadata["path"] = str(
                            (root / metadata["path"]).relative_to(staging)
                        )
                        partition_metadata.append(
                            {
                                "bucket": bucket,
                                "frequency": frequency,
                                "resolution": resolution,
                                **metadata,
                            }
                        )
    destination = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/dense/mode={request.mode.value}/reported_sighting"
    )
    if destination.exists() and not request.force:
        raise FileExistsError(f"Model grid exists; pass --force: {destination}")
    atomic_write_json(
        staging / "_dataset_manifest.json",
        {
            "dataset_id": "whale.sightings.reported_sighting_grid",
            "schema_version": "7",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "processing_mode": request.mode.value,
            "coverage_contract": coverage_contract,
            "quantity_contract": {
                "reported_sighting": "reported-sighting presence indicator",
                "zero_semantics": coverage_contract["zero_semantics"],
                "not_interpretable_as": [
                    "biological_absence",
                    "observer-effort-adjusted occurrence",
                ],
            },
            "partitions": partition_metadata,
            "expected_groups": group_expectations,
            "inputs": [
                request.ecotype_counts.checksum
                or checksum_path(request.ecotype_counts.path),
                *[
                    artifact.checksum or checksum_path(artifact.path)
                    for artifact in request.water_universes
                ],
                *[artifact.checksum for artifact in model_domains.artifacts],
            ],
        },
    )
    candidate_report = validate_sightings_artifact(
        ArtifactRef(
            kind="domain",
            dataset_id="whale.sightings.reported_sighting_grid",
            path=staging,
            producer="whale.sightings.model_grid.v7",
            schema_version="7",
            checksum=checksum_path(staging),
        )
    )
    if not candidate_report.valid:
        quarantine = (
            request.data_root / "quarantine/whale.sightings.model_grid" / request.run_id
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            shutil.rmtree(quarantine)
        os.replace(staging, quarantine)
        raise ValueError(
            "Model-grid candidate validation failed: "
            + "; ".join(candidate_report.errors)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = staging.parent / "canonical-model-grid-backup"
    if backup.exists():
        shutil.rmtree(backup)
    try:
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staging, destination)
    except OSError:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    shutil.rmtree(staging.parent, ignore_errors=True)
    artifact = ArtifactRef(
        kind="domain",
        dataset_id="whale.sightings.reported_sighting_grid",
        path=destination,
        producer="whale.sightings.model_grid.v7",
        schema_version="7",
        run_id=request.run_id,
        config_hash=document.config_hash,
        checksum=checksum_path(destination),
        row_count=sum(int(item["expected_rows"]) for item in partition_metadata),
        file_count=len(partition_metadata),
        inputs=tuple(item.checksum or str(item.path) for item in inputs),
        processing_mode=request.mode.value,
        knowledge_cutoff=request.knowledge_cutoff
        or request.ecotype_counts.knowledge_cutoff,
        data_snapshot=request.ecotype_counts.data_snapshot,
        temporal_coverage=coverage_contract,
    )
    report = ValidationReport(
        True,
        artifact.dataset_id or artifact.kind,
        metrics={"partitions": len(partition_metadata)},
    )
    manifest = RunManifest(
        run_id=request.run_id,
        workflow="whale.sightings.model_grid.v7",
        config_hash=document.config_hash,
        resolved_config=document.redacted_data(),
        inputs=inputs,
        outputs=(artifact,),
        data_snapshot=request.ecotype_counts.data_snapshot,
        schema_version="7",
        code_revision=code_revision(),
        stage_signature=signature,
        stages=(
            {
                "name": "model_grid",
                "semantic_version": "7",
                "signature": signature_payload,
                "partition_count": len(partition_metadata),
            },
        ),
    )
    manifest.write(manifest_path, overwrite=request.force)
    latest_pointer = (
        request.data_root
        / "processed/domain/whale_layer/sightings/manifests/model_grid/latest.json"
    )
    atomic_write_json(
        latest_pointer,
        {
            "manifest": manifest_path.relative_to(latest_pointer.parent).as_posix(),
            "stage_signature": signature,
        },
        overwrite=True,
    )
    return StageResult((artifact,), (input_report, candidate_report, report), manifest)


def _water_network_lineage(config_path: Path) -> tuple[ArtifactRef, ...]:
    """Return the exact config, manifest, support, and edge inputs used for routing."""

    config = load_water_network_config(config_path)
    paths = (
        ("configuration", "environment.seascape.water_network_config", config_path),
        (
            "manifest",
            "environment.seascape.water_network_manifest",
            config.manifest_path,
        ),
        ("domain", "environment.seascape.marine_support_r6", config.support_path(6)),
        ("domain", "environment.seascape.water_edges_r6", config.edge_path(6)),
    )
    missing = [str(path) for _kind, _dataset_id, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Canonical water-network lineage is incomplete: {missing}"
        )
    return tuple(
        ArtifactRef(
            kind=kind,
            dataset_id=dataset_id,
            path=path,
            producer="environment.seascape.water_network",
            checksum=checksum_path(path),
        )
        for kind, dataset_id, path in paths
    )


def _center_distance_km(left: str, right: str) -> float:
    return float(
        h3.great_circle_distance(
            h3.cell_to_latlng(left), h3.cell_to_latlng(right), unit="km"
        )
    )


@lru_cache(maxsize=16)
def _water_adjacency(
    resolution: int,
    config_path: str,
    lineage_token: tuple[str, ...],
) -> dict[str, tuple[tuple[str, float], ...]]:
    """Collapse canonical R6 passable edges to a weighted requested grid."""

    del lineage_token  # Its value intentionally participates in the cache key.
    if resolution > 6:
        raise ValueError("Whale intensity currently supports H3 resolutions through R6")
    graph = load_water_graph(6, config_path)
    adjacency: dict[str, dict[str, float]] = {}
    for left_position, left_cell in enumerate(graph.cells):
        left = (
            str(left_cell)
            if resolution == 6
            else h3.cell_to_parent(str(left_cell), resolution)
        )
        adjacency.setdefault(left, {})
        neighbors, weights_m = graph.neighbors_of(left_position)
        for right_position, weight_m in zip(neighbors, weights_m, strict=True):
            right_cell = str(graph.cells[int(right_position)])
            right = (
                right_cell
                if resolution == 6
                else h3.cell_to_parent(right_cell, resolution)
            )
            if left == right:
                continue
            distance_km = (
                float(weight_m) / 1_000.0
                if resolution == 6
                else _center_distance_km(left, right)
            )
            if not math.isfinite(distance_km) or distance_km <= 0:
                raise ValueError(f"Invalid water-edge distance for {left} -> {right}")
            prior = adjacency[left].get(right)
            adjacency[left][right] = (
                distance_km if prior is None else min(prior, distance_km)
            )
            adjacency.setdefault(right, {})[left] = adjacency[left][right]
    return {
        cell: tuple(sorted(neighbors.items())) for cell, neighbors in adjacency.items()
    }


def _bounded_water_neighbors(
    origin: str,
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    maximum_distance_km: float,
) -> dict[str, float]:
    """Return shortest water-path distances bounded by a physical radius."""

    if not math.isfinite(maximum_distance_km) or maximum_distance_km < 0:
        raise ValueError("maximum_distance_km must be finite and non-negative")
    distances = {origin: 0.0}
    frontier: list[tuple[float, str]] = [(0.0, origin)]
    while frontier:
        distance, current = heapq.heappop(frontier)
        if distance > distances[current]:
            continue
        for neighbor, edge_distance in adjacency.get(current, ()):
            if not math.isfinite(edge_distance) or edge_distance <= 0:
                raise ValueError(
                    f"Invalid weighted adjacency edge: {current} -> {neighbor}"
                )
            candidate = distance + edge_distance
            if candidate > maximum_distance_km + 1e-12:
                continue
            if candidate >= distances.get(neighbor, math.inf):
                continue
            distances[neighbor] = candidate
            heapq.heappush(frontier, (candidate, neighbor))
    return distances


def _distance_weight(
    distance_km: float,
    radius_km: float,
    *,
    center_weight: float,
    outer_weight: float,
    exponent: float,
) -> float:
    if distance_km < 0 or distance_km > radius_km + 1e-12:
        raise ValueError("Kernel distance must lie inside the configured radius")
    if radius_km <= 0:
        raise ValueError("Kernel radius must be positive")
    fraction = min(1.0, max(0.0, distance_km / radius_km))
    return float(
        outer_weight + (center_weight - outer_weight) * ((1 - fraction) ** exponent)
    )


def _intensity(
    frame: pd.DataFrame,
    config: SightingsPipelineConfig,
    *,
    water_network_config_path: Path,
    lineage_token: tuple[str, ...],
) -> pd.DataFrame:
    resolution = int(frame.H3_RESOLUTION.iloc[0])
    geometry = _kernel_geometry(config, resolution)
    expected = pd.to_numeric(
        frame["MATURE_EXPECTED_SIGHTING_COUNT"], errors="raise"
    ).astype(float)
    if not np.isfinite(expected).all() or (expected < 0).any():
        raise ValueError(
            "Intensity inputs require finite non-negative mature expected counts"
        )
    frame = frame.copy()
    frame["MATURE_EXPECTED_SIGHTING_COUNT"] = expected
    cells = set(frame.H3_INDEX)
    adjacency = _water_adjacency(
        resolution, str(water_network_config_path), lineage_token
    )
    support: dict[tuple[date, str], float] = {}
    for row in frame.loc[frame.MATURE_EXPECTED_SIGHTING_COUNT.gt(0)].itertuples():
        routed = _bounded_water_neighbors(
            row.H3_INDEX,
            adjacency,
            float(geometry["configured_radius_km"]),
        )
        for neighbor, distance in routed.items():
            if neighbor not in cells:
                continue
            weight = _distance_weight(
                distance,
                float(geometry["configured_radius_km"]),
                center_weight=config.kernel.center_weight,
                outer_weight=config.kernel.outer_weight,
                exponent=config.kernel.exponent,
            )
            expected_count = float(row.MATURE_EXPECTED_SIGHTING_COUNT)
            contribution_probability = min(1.0, max(0.0, expected_count * weight))
            contribution = (
                -math.inf
                if contribution_probability >= 1.0
                else math.log1p(-contribution_probability)
            )
            key = (row.PERIOD_START, neighbor)
            support[key] = support.get(key, 0.0) + contribution
    values = []
    for row in frame.itertuples():
        value = support.get((row.PERIOD_START, row.H3_INDEX))
        intensity = -math.expm1(value) if value is not None else 0.0
        values.append(
            1.0
            if float(row.MATURE_EXPECTED_SIGHTING_COUNT) >= 1.0
            else min(1.0, max(0.0, intensity))
        )
    result = frame.copy()
    activity = np.asarray(values, dtype="float32")
    result["RELATIVE_REPORTED_ACTIVITY"] = activity
    # One-release wire compatibility. New consumers must use the explicit
    # reported-activity name and semantics.
    result["RELATIVE_SIGHTING_INTENSITY"] = activity
    return result


def _promote_activity_directories(
    candidates: tuple[tuple[Path, Path], ...],
    *,
    transaction_root: Path,
    force: bool,
) -> None:
    """Replace the canonical surface and its compatibility alias as one unit."""

    existing = [
        destination for _candidate, destination in candidates if destination.exists()
    ]
    if existing and not force:
        raise FileExistsError(
            "Activity products exist; pass --force: "
            + ", ".join(str(path) for path in existing)
        )
    backups_root = transaction_root / "backups"
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for _candidate, destination in candidates:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = backups_root / destination.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
                backups.append((backup, destination))
        for candidate, destination in candidates:
            os.replace(candidate, destination)
            promoted.append(destination)
    except OSError:
        for destination in reversed(promoted):
            if destination.exists():
                shutil.rmtree(destination)
        for backup, destination in reversed(backups):
            if backup.exists():
                os.replace(backup, destination)
        raise
    shutil.rmtree(backups_root, ignore_errors=True)


def build_intensity(request: IntensityRequest) -> StageResult:
    if request.model_grid is None:
        raise ValueError("Intensity requires a model-grid artifact")
    if (
        request.model_grid.processing_mode
        and request.model_grid.processing_mode != request.mode.value
    ):
        raise ValueError("Intensity processing mode must match the model-grid artifact")
    from marine_mammal_toolkit.tools.quality.observations import (
        validate_sightings_artifact,
    )

    input_report = validate_sightings_artifact(request.model_grid)
    input_report.require_valid()
    document, config = load_sightings_config(request.config)
    source_root = request.model_grid.path
    source_manifest_path = source_root / "_dataset_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text())
        if source_manifest_path.exists()
        else {}
    )
    coverage_contract = source_manifest.get("coverage_contract")
    if not isinstance(coverage_contract, dict):
        coverage_contract = request.model_grid.temporal_coverage or {
            "zero_fill_verified": False,
            "zero_semantics": "unavailable_unverified_coverage_not_no_report",
        }
    _require_dense_coverage(dict(coverage_contract))
    water_network_config_path = document.resolve_path(config.water_network_config)
    water_lineage = _water_network_lineage(water_network_config_path)
    lineage_token = tuple(
        item.checksum or checksum_path(item.path) for item in water_lineage
    )
    inputs = (request.model_grid, *water_lineage)
    signature, signature_payload = stage_signature(
        stage="whale.sightings.intensity",
        semantic_version="8",
        config_hash=document.config_hash,
        inputs=inputs,
        parameters={
            "mode": request.mode.value,
            "knowledge_cutoff": request.knowledge_cutoff
            or request.model_grid.knowledge_cutoff,
            "normalization_schema_version": "5",
        },
    )
    manifest_path = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/manifests/intensity/{signature}.json"
    )
    resumed = resume_result(
        enabled=request.resume,
        manifest_path=manifest_path,
        config_hash=document.config_hash,
        inputs=inputs,
        signature=signature,
    )
    if resumed is not None:
        return resumed
    destination = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/dense/mode={request.mode.value}/relative_reported_activity"
    )
    alias_destination = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/dense/mode={request.mode.value}/relative_intensity"
    )
    transaction_root = (
        request.data_root / ".staging" / request.run_id / "whale.sightings.intensity"
    )
    if transaction_root.exists():
        shutil.rmtree(transaction_root)
    staging = transaction_root / "relative_reported_activity"
    metadata: list[dict[str, Any]] = []
    for part_index, source in enumerate(sorted(source_root.rglob("*.parquet"))):
        frame = pd.read_parquet(source)
        result = _intensity(
            frame,
            config,
            water_network_config_path=water_network_config_path,
            lineage_token=lineage_token,
        )
        target = staging / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(result, preserve_index=False).cast(
                MODEL_INTENSITY_SCHEMA
            ),
            target,
            compression="zstd",
        )
        metadata.append(
            {
                "path": str(target.relative_to(staging)),
                "checksum": checksum_path(target),
                "rows": len(result),
            }
        )
    if not metadata:
        raise ValueError("Model-grid artifact contains no Parquet partitions")
    atomic_write_json(
        staging / "_dataset_manifest.json",
        {
            "dataset_id": "whale.sightings.relative_reported_activity",
            "schema_version": "8",
            "partitions": metadata,
            "coverage_contract": coverage_contract,
            "quantity_contract": {
                "quantity": "relative_reported_sighting_activity_proxy",
                "interpretation": (
                    "Water-network-smoothed relative activity derived from reported sightings "
                    "and imputed ecotype mass. It is not occurrence probability, abundance, "
                    "detection probability, observer effort, or biological absence."
                ),
                "zero_semantics": coverage_contract.get(
                    "zero_semantics", "unavailable_unverified_coverage_not_no_report"
                ),
            },
            "kernel": {
                str(resolution): _kernel_geometry(config, resolution)
                for resolution in config.h3_resolutions
            },
            "input_checksum": request.model_grid.checksum or checksum_path(source_root),
            "water_network_lineage": [
                {
                    "dataset_id": item.dataset_id,
                    "producer": item.producer,
                    "schema_version": item.schema_version,
                    "checksum": item.checksum,
                }
                for item in water_lineage
            ],
        },
    )
    candidate_report = validate_sightings_artifact(
        ArtifactRef(
            kind="domain",
            dataset_id="whale.sightings.relative_reported_activity",
            path=staging,
            producer="whale.sightings.intensity.v8",
            schema_version="8",
            checksum=checksum_path(staging),
        )
    )
    if not candidate_report.valid:
        quarantine = (
            request.data_root / "quarantine/whale.sightings.intensity" / request.run_id
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            shutil.rmtree(quarantine)
        os.replace(transaction_root, quarantine)
        raise ValueError(
            "Intensity candidate validation failed: "
            + "; ".join(candidate_report.errors)
        )
    alias_staging = transaction_root / "relative_intensity"
    shutil.copytree(staging, alias_staging, copy_function=shutil.copy2)
    alias_manifest_path = alias_staging / "_dataset_manifest.json"
    alias_manifest = json.loads(alias_manifest_path.read_text())
    alias_manifest.update(
        {
            "dataset_id": "whale.sightings.relative_intensity",
            "compatibility_alias_for": "whale.sightings.relative_reported_activity",
            "deprecation": "one_release",
        }
    )
    atomic_write_json(alias_manifest_path, alias_manifest, overwrite=True)
    alias_candidate_report = validate_sightings_artifact(
        ArtifactRef(
            kind="domain",
            dataset_id="whale.sightings.relative_intensity",
            path=alias_staging,
            producer="whale.sightings.intensity.compatibility_alias.v8",
            schema_version="8",
            checksum=checksum_path(alias_staging),
        )
    )
    if not alias_candidate_report.valid:
        quarantine = (
            request.data_root / "quarantine/whale.sightings.intensity" / request.run_id
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            shutil.rmtree(quarantine)
        os.replace(transaction_root, quarantine)
        raise ValueError(
            "Intensity alias candidate validation failed: "
            + "; ".join(alias_candidate_report.errors)
        )
    _promote_activity_directories(
        ((staging, destination), (alias_staging, alias_destination)),
        transaction_root=transaction_root,
        force=request.force,
    )
    shutil.rmtree(transaction_root, ignore_errors=True)
    artifact = ArtifactRef(
        kind="domain",
        dataset_id="whale.sightings.relative_reported_activity",
        path=destination,
        producer="whale.sightings.intensity.v8",
        schema_version="8",
        run_id=request.run_id,
        config_hash=document.config_hash,
        checksum=checksum_path(destination),
        row_count=sum(int(item["rows"]) for item in metadata),
        file_count=len(metadata),
        inputs=tuple(item.checksum or str(item.path) for item in inputs),
        processing_mode=request.mode.value,
        knowledge_cutoff=request.knowledge_cutoff
        or request.model_grid.knowledge_cutoff,
        data_snapshot=request.model_grid.data_snapshot,
        temporal_coverage=coverage_contract,
    )
    alias_artifact = ArtifactRef(
        **{
            **artifact.__dict__,
            "dataset_id": "whale.sightings.relative_intensity",
            "path": alias_destination,
            "producer": "whale.sightings.intensity.compatibility_alias.v8",
            "checksum": checksum_path(alias_destination),
        }
    )
    report = ValidationReport(
        True,
        artifact.dataset_id or artifact.kind,
        metrics={
            "partitions": len(metadata),
            "compatibility_alias": str(alias_destination),
        },
    )
    manifest = RunManifest(
        run_id=request.run_id,
        workflow="whale.sightings.intensity.v8",
        config_hash=document.config_hash,
        resolved_config=document.redacted_data(),
        inputs=inputs,
        outputs=(artifact, alias_artifact),
        data_snapshot=request.model_grid.data_snapshot,
        schema_version="8",
        code_revision=code_revision(),
        stage_signature=signature,
        stages=(
            {
                "name": "intensity",
                "semantic_version": "8",
                "signature": signature_payload,
                "partition_count": len(metadata),
            },
        ),
    )
    manifest.write(manifest_path, overwrite=request.force)
    latest_pointer = (
        request.data_root
        / "processed/domain/whale_layer/sightings/manifests/intensity/latest.json"
    )
    atomic_write_json(
        latest_pointer,
        {
            "manifest": manifest_path.relative_to(latest_pointer.parent).as_posix(),
            "stage_signature": signature,
        },
        overwrite=True,
    )
    return StageResult(
        (artifact, alias_artifact),
        (input_report, candidate_report, alias_candidate_report, report),
        manifest,
    )
