"""Lazy public workflow interfaces."""

from importlib import import_module

_EXPORTS = {
    "FeatureConfig": (
        "marine_mammal_toolkit.tools.observations.impute.config",
        "FeatureConfig",
    ),
    "ImputationConfig": (
        "marine_mammal_toolkit.tools.observations.impute.config",
        "ImputationConfig",
    ),
    "ModelConfig": (
        "marine_mammal_toolkit.tools.observations.impute.config",
        "ModelConfig",
    ),
    "attach_encounter_ids": (
        "marine_mammal_toolkit.tools.observations.impute.encounters",
        "attach_encounter_ids",
    ),
    "load_preprocessed_sightings": (
        "marine_mammal_toolkit.tools.observations.impute.io",
        "load_preprocessed_sightings",
    ),
    "EvaluationResult": (
        "marine_mammal_toolkit.tools.observations.impute.model",
        "EvaluationResult",
    ),
    "SelectiveDateContextImputer": (
        "marine_mammal_toolkit.tools.observations.impute.model",
        "SelectiveDateContextImputer",
    ),
    "FitResult": (
        "marine_mammal_toolkit.tools.observations.impute.pipeline",
        "FitResult",
    ),
    "ImputationResult": (
        "marine_mammal_toolkit.tools.observations.impute.pipeline",
        "ImputationResult",
    ),
    "WorkflowResult": (
        "marine_mammal_toolkit.tools.observations.impute.pipeline",
        "WorkflowResult",
    ),
    "apply_imputation_model": (
        "marine_mammal_toolkit.tools.observations.impute.pipeline",
        "apply_imputation_model",
    ),
    "fit_imputation_model": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.imputation",
        "fit_imputation_model",
    ),
    "resolve_model_path": (
        "marine_mammal_toolkit.tools.observations.impute.pipeline",
        "resolve_model_path",
    ),
    "run_imputation_workflow": (
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.imputation",
        "run_imputation_workflow",
    ),
    "impute_sightings": (
        "marine_mammal_toolkit.tools.observations.impute.service",
        "impute_sightings",
    ),
    "ImputationWorkflowSettings": (
        "marine_mammal_toolkit.tools.observations.impute.settings",
        "ImputationWorkflowSettings",
    ),
    "load_imputation_settings": (
        "marine_mammal_toolkit.tools.observations.impute.settings",
        "load_imputation_settings",
    ),
    "build_weekly_probability_surface": (
        "marine_mammal_toolkit.tools.observations.impute.surface",
        "build_weekly_probability_surface",
    ),
    "make_probability_surface_map": (
        "marine_mammal_toolkit.tools.observations.impute.surface",
        "make_probability_surface_map",
    ),
    "make_animated_map": (
        "marine_mammal_toolkit.tools.observations.impute.visualization",
        "make_animated_map",
    ),
    "plot_confusion": (
        "marine_mammal_toolkit.tools.observations.impute.visualization",
        "plot_confusion",
    ),
    "plot_reliability": (
        "marine_mammal_toolkit.tools.observations.impute.visualization",
        "plot_reliability",
    ),
    "plot_risk_coverage": (
        "marine_mammal_toolkit.tools.observations.impute.visualization",
        "plot_risk_coverage",
    ),
    "save_map_html": (
        "marine_mammal_toolkit.tools.observations.impute.visualization",
        "save_map_html",
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
