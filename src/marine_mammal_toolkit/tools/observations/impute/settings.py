from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)

from marine_mammal_toolkit.tools.observations.impute.config import FeatureConfig
from marine_mammal_toolkit.tools.observations.impute.config import ImputationConfig
from marine_mammal_toolkit.tools.observations.impute.config import ModelConfig


@dataclass(frozen=True)
class ImputationWorkflowSettings:
    config_path: Path
    observations_path: Path
    associations_path: Path | None
    models_dir: Path
    output_path: Path
    evaluate_strategies: tuple[str, ...]
    config: ImputationConfig


def _resolve(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )


def _strict_overrides(defaults: Any, values: dict[str, Any], section: str) -> Any:
    unknown = sorted(set(values) - set(defaults.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown {section} settings: {', '.join(unknown)}")
    return replace(defaults, **values)


def load_imputation_settings(path: str | Path) -> ImputationWorkflowSettings:
    """Load the sightings imputation section from an OrcaCast configuration file."""

    document, pipeline = load_sightings_config(path)
    source = document.source
    section = pipeline.imputation
    root = document.resolve_path(".")
    inputs = section.inputs
    artifacts = section.artifacts

    observations = _resolve(root, inputs.observations)
    models_dir = _resolve(root, artifacts.models_dir)
    output = _resolve(root, artifacts.output)
    if observations is None or models_dir is None or output is None:
        raise ValueError(
            "imputation inputs.observations, artifacts.models_dir, and artifacts.output are required"
        )
    feature_values = section.feature.model_dump()
    feature_values["temporal_windows"] = tuple(
        tuple(window) for window in feature_values["temporal_windows"]
    )
    water_network_config = _resolve(root, inputs.water_network_config)
    if water_network_config is None:
        raise ValueError("imputation.inputs.water_network_config is required")
    feature_values["water_network_config_path"] = str(water_network_config)
    feature = _strict_overrides(FeatureConfig(), feature_values, "imputation.feature")
    model = _strict_overrides(
        ModelConfig(), section.model.model_dump(), "imputation.model"
    )
    config = ImputationConfig(
        regime=section.regime,
        feature=feature,
        model=model,
    )
    config.validate()
    strategies = tuple(section.evaluate_strategies)
    allowed = {"reconstruction", "encounter", "blocked", "purged_blocked"}
    invalid = sorted(set(strategies) - allowed)
    if not strategies or invalid:
        raise ValueError(
            f"Invalid imputation evaluate_strategies: {invalid or strategies}"
        )
    return ImputationWorkflowSettings(
        config_path=source,
        observations_path=observations,
        associations_path=_resolve(root, inputs.associations),
        models_dir=models_dir,
        output_path=output,
        evaluate_strategies=strategies,
        config=config,
    )
