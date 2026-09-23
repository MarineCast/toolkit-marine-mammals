"""Public observation queries without regional models or an application checkout."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

import yaml

from .configuration import SightingsPipelineConfig, load_sightings_config
from .resources import config_path
from marine_mammal_toolkit.tools._core.config import workspace
from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef, atomic_write_json
from marine_mammal_toolkit.tools.schemas.sources import BBox

SOURCE_CATALOG = {
    "twm": {
        "access": "local_csv",
        "date_query": False,
        "bounds_query": False,
        "notes": "Supply TWM-format CSVs; missing files warn and record unavailable coverage.",
    },
    "acartia": {
        "access": "current_endpoint_and_local_csv",
        "date_query": False,
        "bounds_query": False,
        "notes": "Current feed; historical records require supplemental local inputs.",
    },
    "maplify": {
        "access": "http",
        "date_query": True,
        "bounds_query": True,
        "notes": "Regional provider; no guarantee of coverage outside its region.",
    },
    "inaturalist": {
        "access": "public_api",
        "date_query": True,
        "bounds_query": True,
        "notes": "Orcinus orca observations; optional INATURALIST_TOKEN.",
    },
    "cwr": {
        "access": "web_archives",
        "date_query": False,
        "bounds_query": False,
        "notes": "Configured archive years/maps only; internal-use policy applies.",
    },
    "gbif": {
        "access": "public_api",
        "date_query": True,
        "bounds_query": True,
        "notes": "Select dataset UUIDs; search capped at 100,000 records, narrow larger queries.",
    },
}


@dataclass(frozen=True)
class ObservationQueryResult:
    observations: ArtifactRef
    associations: ArtifactRef
    manifest: Path
    query_root: Path
    warnings: tuple[str, ...]

    def read(self):
        """Read canonical observations, including source identity and quality flags."""
        import pandas as pd

        return pd.read_parquet(self.observations.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": str(self.observations.path),
            "associations": str(self.associations.path),
            "manifest": str(self.manifest),
            "query_root": str(self.query_root),
            "row_count": self.observations.row_count,
            "warnings": list(self.warnings),
        }


def query_configuration(
    *,
    workspace_root: str | Path,
    start: date,
    end: date,
    sources: Sequence[str] = ("inaturalist",),
    bbox: tuple[float, float, float, float] = (-180, 32, -109, 72),
    dataset_keys: Sequence[str] = (),
    twm_files: Sequence[str | Path] = (),
    config: str | Path | None = None,
) -> dict[str, Any]:
    """Compose a bounded observation-only configuration, preserving source policies."""
    if start > end:
        raise ValueError("start must not follow end")
    if (
        not sources
        or len(set(sources)) != len(sources)
        or set(sources) - set(SOURCE_CATALOG)
    ):
        raise ValueError(
            "Select one or more unique sources from: " + ", ".join(SOURCE_CATALOG)
        )
    if dataset_keys and "gbif" not in sources:
        raise ValueError("dataset_keys requires the gbif source")
    if len(set(dataset_keys)) != len(dataset_keys):
        raise ValueError("dataset_keys must be unique")
    for key in dataset_keys:
        UUID(key)
    bounds = BBox(min_lon=bbox[0], min_lat=bbox[1], max_lon=bbox[2], max_lat=bbox[3])
    root = Path(workspace_root).expanduser().resolve()
    _, settings = load_sightings_config(config or config_path(), workspace_root=root)
    payload = settings.model_dump(mode="json")
    payload.update(
        min_date=start.isoformat(),
        max_date=end.isoformat(),
        full_area=bounds.model_dump(),
    )
    selected = {}
    for name in sources:
        source = dict(payload["collection"]["sources"][name])
        source.update(enabled=True, min_date=start.isoformat())
        if name in {"gbif", "maplify", "inaturalist"}:
            source["bbox"] = bounds.model_dump()
        if name == "gbif" and dataset_keys:
            known = {entry["key"]: entry for entry in source["dataset_allowlist"]}
            source["dataset_allowlist"] = [
                known.get(key, {"key": key, "title": key, "use_class": "INTERNAL_ONLY"})
                for key in dataset_keys
            ]
        selected[name] = source
    payload["collection"]["sources"] = selected
    SightingsPipelineConfig.model_validate(payload)
    # Absolute paths make the query identity independent of the caller's cwd.
    for name, source in selected.items():
        if source.get("local_path"):
            source["local_path"] = str((root / source["local_path"]).resolve())
    return payload


def preflight_observations(
    config: str | Path,
    *,
    workspace_root: str | Path,
    profile: str = "observations-only",
    data_root: str | Path = "data",
) -> dict[str, Any]:
    """Check local prerequisites only; never contact a provider or expose credentials."""
    root = Path(workspace_root).expanduser().resolve()
    document, settings = load_sightings_config(config, workspace_root=root)
    warnings, errors, statuses = [], [], {}
    for name, source in settings.collection.sources.items():
        if not source.enabled:
            continue
        statuses[name] = {**SOURCE_CATALOG[name], "status": "not_probed"}
        if source.credential_env:
            statuses[name]["credential_configured"] = bool(
                os.getenv(source.credential_env)
            )
        if name == "twm":
            path = (
                document.resolve_path(source.local_path) if source.local_path else None
            )
            available = bool(
                path and path.is_dir() and any(p.is_file() for p in path.glob("*.csv"))
            )
            statuses[name]["status"] = (
                "local_input_present" if available else "source_unavailable"
            )
            if not available:
                warnings.append(
                    "TWM files are missing; collection will continue with unknown TWM coverage."
                )
    if profile != "observations-only":
        from .observations.release import release_profile
        from importlib.util import find_spec

        selected_profile = release_profile(profile)
        if selected_profile.include_imputation or selected_profile.include_counts:
            needed = (
                ("seascape", "geopandas", "sklearn")
                if selected_profile.include_imputation
                else ("geopandas",)
            )
            if any(find_spec(name) is None for name in needed):
                errors.append(
                    "Install marine-mammal-toolkit[imputation] for this profile."
                )
        if selected_profile.include_imputation:
            water = document.resolve_path(
                settings.imputation.inputs.water_network_config
            )
            if not water.is_file():
                errors.append(
                    f"Missing imputation water-network configuration: {water}"
                )
            seascape = os.getenv("SEASCAPE_WORKSPACE")
            if (
                not seascape
                or not Path(seascape).is_absolute()
                or not (Path(seascape) / "config/common.yaml").is_file()
            ):
                errors.append(
                    "SEASCAPE_WORKSPACE must contain config/common.yaml for imputation."
                )
            for name, domain in settings.model_universes.items():
                if not document.resolve_path(domain.polygon).is_file():
                    errors.append(
                        f"Missing {name} model domain: {document.resolve_path(domain.polygon)}"
                    )
        if selected_profile.include_counts:
            from .pipeline import _universe_artifacts

            try:
                _universe_artifacts(
                    (root / data_root).resolve(), selected_profile.resolutions
                )
            except (ValueError, OSError, KeyError) as exc:
                errors.append(f"Missing or invalid count water-universe inputs: {exc}")
    if not statuses:
        errors.append("No sources are enabled")
    if statuses and all(
        item["status"] == "source_unavailable" for item in statuses.values()
    ):
        warnings.append(
            "All selected sources are unavailable; no observations can be reported."
        )
    return {
        "ready": not errors,
        "profile": profile,
        "sources": statuses,
        "warnings": warnings,
        "errors": errors,
        "network_checked": False,
    }


def query_observations(
    *,
    workspace_root: str | Path,
    start: date,
    end: date,
    sources: Sequence[str] = ("inaturalist",),
    bbox: tuple[float, float, float, float] = (-180, 32, -109, 72),
    dataset_keys: Sequence[str] = (),
    twm_files: Sequence[str | Path] = (),
    data_root: str | Path = "data",
    config: str | Path | None = None,
    offline: bool = False,
) -> ObservationQueryResult:
    """Collect and normalize selected datasets into a query-scoped workspace.

    Dates are inclusive model-local dates. No model fitting, counts, pruning,
    public redistribution, or regional seascape products are involved.
    """
    from marine_mammal_toolkit.tools.observations.collect.pipeline import (
        collect_sightings,
    )
    from marine_mammal_toolkit.tools.observations.process.pipeline import (
        normalize_sightings,
    )
    from marine_mammal_toolkit.tools.schemas.observations import (
        SightingsCollectionRequest,
        NormalizationRequest,
    )

    root = Path(workspace_root).expanduser().resolve()
    files = tuple((root / Path(p).expanduser()).resolve() for p in twm_files)
    payload = query_configuration(
        workspace_root=root,
        start=start,
        end=end,
        sources=sources,
        bbox=bbox,
        dataset_keys=dataset_keys,
        config=config,
    )
    identity = {"config": payload, "twm_files": [str(p) for p in files]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    query_root = (root / data_root).resolve() / "queries" / key
    query_root.mkdir(parents=True, exist_ok=True)
    settings_file = query_root / "query.yaml"
    if not settings_file.exists():
        from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_text

        atomic_write_text(settings_file, yaml.safe_dump(payload), overwrite=False)
    from marine_mammal_toolkit.tools._core.locking import workspace_write_lock

    with workspace_write_lock(query_root), workspace(root):
        common = dict(
            config=settings_file,
            data_root=query_root,
            artifact_root=query_root / "artifacts",
            output_root=query_root / "outputs",
            force=True,
        )
        collected = collect_sightings(
            SightingsCollectionRequest(
                **common,
                start_date=start,
                end_date=end,
                twm_files=files,
                offline=offline,
            )
        )
        normalized = normalize_sightings(
            NormalizationRequest(**common, inputs=collected.outputs)
        )
        artifacts = {a.dataset_id: a for a in normalized.outputs}
        warnings = tuple(w for report in collected.validations for w in report.warnings)
        manifest = query_root / "query-result.json"
        atomic_write_json(
            manifest,
            {
                "schema_version": 1,
                "query": identity,
                "collection": collected.manifest.to_dict(),
                "normalization": normalized.manifest.to_dict(),
                "warnings": warnings,
            },
            overwrite=True,
        )
        return ObservationQueryResult(
            artifacts["whale.sightings.observations"],
            artifacts["whale.sightings.associations"],
            manifest,
            query_root,
            warnings,
        )


def run_demo(workspace_root: str | Path) -> ObservationQueryResult:
    """Run the production query API on a tiny, explicitly synthetic local fixture."""
    root = Path(workspace_root).expanduser().resolve()
    path = root / "demo-inputs/synthetic-twm.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "id,date,latitude,longitude,pod\nsynthetic-1,2025-06-03,48.5,-123.0,J pod\nsynthetic-2,2025-06-04,48.6,-123.1,Transient\n"
    if path.exists() and path.read_text() != content:
        raise FileExistsError(f"Refusing to overwrite an existing demo input: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return query_observations(
        workspace_root=root,
        sources=("twm",),
        twm_files=(path,),
        start=date(2025, 6, 1),
        end=date(2025, 6, 8),
    )
