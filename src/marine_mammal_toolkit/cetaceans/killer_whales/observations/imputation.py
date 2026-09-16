"""Compose the reusable binary engine with killer-whale features and policies."""

from marine_mammal_toolkit.tools.observations.impute import pipeline as engine
from .imputer import KillerWhaleImputer


def fit_imputation_model(*, workspace_root=None, **kwargs):
    if workspace_root is not None:
        from marine_mammal_toolkit.tools._core.config import workspace

        with workspace(workspace_root):
            return fit_imputation_model(**kwargs)
    source = kwargs.get("source_config_path")
    domains = None
    if source is not None:
        from ..configuration import load_sightings_config
        from marine_mammal_toolkit.tools.observations.post_process.spatial import (
            load_model_domains,
        )

        document, config = load_sightings_config(source)
        domains = load_model_domains(document, config)
    return engine.fit_imputation_model(
        imputer_factory=KillerWhaleImputer, model_domains=domains, **kwargs
    )


def run_imputation_workflow(*, workspace_root=None, **kwargs):
    if workspace_root is not None:
        from marine_mammal_toolkit.tools._core.config import workspace

        with workspace(workspace_root):
            return run_imputation_workflow(**kwargs)
    return engine.run_imputation_workflow(imputer_factory=KillerWhaleImputer, **kwargs)


apply_imputation_model = engine.apply_imputation_model
