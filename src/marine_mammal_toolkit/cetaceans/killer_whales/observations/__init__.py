"""Lazy public workflow interfaces."""

from importlib import import_module

_EXPORTS = {
    "query_observations": (
        "marine_mammal_toolkit.cetaceans.killer_whales.query",
        "query_observations",
    ),
    "resolve_sightings_product": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.product",
        "resolve_sightings_product",
    ),
    "CountRequest": (
        "marine_mammal_toolkit.tools.schemas.observations",
        "CountRequest",
    ),
    "ImputationRequest": (
        "marine_mammal_toolkit.tools.schemas.observations",
        "ImputationRequest",
    ),
    "IntensityRequest": (
        "marine_mammal_toolkit.tools.schemas.observations",
        "IntensityRequest",
    ),
    "ModelGridRequest": (
        "marine_mammal_toolkit.tools.schemas.observations",
        "ModelGridRequest",
    ),
    "NormalizationRequest": (
        "marine_mammal_toolkit.tools.schemas.observations",
        "NormalizationRequest",
    ),
    "SightingsCollectionRequest": (
        "marine_mammal_toolkit.tools.schemas.observations",
        "SightingsCollectionRequest",
    ),
    "validate_sightings_release": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.release",
        "validate_sightings_release",
    ),
    "build_sightings_report_html": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.report",
        "build_sightings_report_html",
    ),
    "SightingsPipelineRunRequest": (
        "marine_mammal_toolkit.cetaceans.killer_whales.pipeline",
        "SightingsPipelineRunRequest",
    ),
    "SightingsPipelineRunResult": (
        "marine_mammal_toolkit.cetaceans.killer_whales.pipeline",
        "SightingsPipelineRunResult",
    ),
    "SightingsReleaseBlocked": (
        "marine_mammal_toolkit.cetaceans.killer_whales.pipeline",
        "SightingsReleaseBlocked",
    ),
    "run_sightings_pipeline": (
        "marine_mammal_toolkit.cetaceans.killer_whales.pipeline",
        "run_sightings_pipeline",
    ),
    "collect": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.service",
        "collect",
    ),
    "counts": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.service",
        "counts",
    ),
    "impute": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.service",
        "impute",
    ),
    "intensity": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.service",
        "intensity",
    ),
    "model_grid": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.service",
        "model_grid",
    ),
    "normalize": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.service",
        "normalize",
    ),
    "process": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.service",
        "process",
    ),
    "validate": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.service",
        "validate",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
