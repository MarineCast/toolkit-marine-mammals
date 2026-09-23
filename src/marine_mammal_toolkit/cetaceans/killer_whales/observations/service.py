from __future__ import annotations

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools._core.data import StageResult
from marine_mammal_toolkit.tools._core.data import ValidationReport

from marine_mammal_toolkit.tools.observations.collect.pipeline import collect_sightings
from marine_mammal_toolkit.tools.schemas.observations import CountRequest
from marine_mammal_toolkit.tools.schemas.observations import ImputationRequest
from marine_mammal_toolkit.tools.schemas.observations import IntensityRequest
from marine_mammal_toolkit.tools.schemas.observations import ModelGridRequest
from marine_mammal_toolkit.tools.schemas.observations import NormalizationRequest
from marine_mammal_toolkit.tools.schemas.observations import SightingsCollectionRequest
from marine_mammal_toolkit.tools.observations.process.pipeline import (
    normalize_sightings,
)
from marine_mammal_toolkit.tools.quality.observations import validate_sightings_artifact


def collect(request: SightingsCollectionRequest) -> StageResult:
    return collect_sightings(request)


def normalize(request: NormalizationRequest) -> StageResult:
    return normalize_sightings(request)


def process(request: NormalizationRequest) -> StageResult:
    """Canonical processing stops after source-neutral normalization."""
    return normalize(request)


def impute(request: ImputationRequest) -> StageResult:
    # The estimator stack is intentionally loaded only for the imputation stage.
    from marine_mammal_toolkit.tools.observations.impute.service import impute_sightings

    return impute_sightings(request)


def counts(request: CountRequest) -> StageResult:
    from marine_mammal_toolkit.tools.observations.post_process.counts import (
        build_counts,
    )

    return build_counts(request)


def model_grid(request: ModelGridRequest) -> StageResult:
    from marine_mammal_toolkit.tools.observations.post_process.aggregation import (
        build_model_grid,
    )

    return build_model_grid(request)


def intensity(request: IntensityRequest) -> StageResult:
    from marine_mammal_toolkit.tools.observations.post_process.aggregation import (
        build_intensity,
    )

    return build_intensity(request)


def validate(artifact: ArtifactRef) -> ValidationReport:
    return validate_sightings_artifact(artifact)
