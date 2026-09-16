"""Compose counts, grid and intensity stages using explicit artifact inputs."""

from .counts import build_counts
from .aggregation import build_model_grid, build_intensity
from marine_mammal_toolkit.tools.schemas.observations import (
    CountRequest,
    ModelGridRequest,
    IntensityRequest,
)


def _one(artifacts, names, *, required=True):
    matches = [artifact for artifact in artifacts if artifact.dataset_id in names]
    if len(matches) > 1 or (required and not matches):
        raise ValueError(
            f"Expected one artifact in {sorted(names)}, found {len(matches)}"
        )
    return matches[0] if matches else None


def post_process_observations(
    *,
    config,
    inputs,
    parent_inputs=(),
    water_universes,
    data_root,
    artifact_root,
    output_root,
    run_id,
    resolutions,
    frequencies,
    start_date=None,
    end_date=None,
    force=False,
    resume=False,
):
    observations = _one(
        inputs,
        {"whale.sightings.observations", "whale.sightings.imputed_retrospective"},
    )
    associations = _one(inputs, {"whale.sightings.associations"}, required=False)
    if associations is None:
        associations = _one(
            parent_inputs, {"whale.sightings.associations"}, required=False
        )
    common = dict(
        config=config,
        data_root=data_root,
        artifact_root=artifact_root,
        output_root=output_root,
        run_id=run_id,
        force=force,
        resume=resume,
    )
    counts = build_counts(
        CountRequest(
            **common,
            observations=observations,
            associations=associations,
            water_universes=water_universes,
            resolutions=resolutions,
            start_date=start_date,
            end_date=end_date,
        )
    )
    for report in counts.validations:
        report.require_valid()
    grid = build_model_grid(
        ModelGridRequest(
            **common,
            ecotype_counts=_one(counts.outputs, {"whale.sightings.ecotype_counts"}),
            water_universes=water_universes,
            resolutions=resolutions,
            frequencies=frequencies,
            start_date=start_date,
            end_date=end_date,
        )
    )
    for report in grid.validations:
        report.require_valid()
    intensity = build_intensity(
        IntensityRequest(
            **common,
            model_grid=_one(grid.outputs, {"whale.sightings.reported_sighting_grid"}),
        )
    )
    for report in intensity.validations:
        report.require_valid()
    return counts, grid, intensity
