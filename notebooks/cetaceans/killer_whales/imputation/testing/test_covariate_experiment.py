from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import pandas as pd

from covariate_experiment import (
    SeascapeSpec,
    _select_multistratum_constrained_blend,
    build_seasonal_features,
    load_static_seascape_features,
)


def test_seasonal_features_are_finite_cyclic_and_geographically_aware() -> None:
    frame = pd.DataFrame(
        {
            "SIGHTING_DATE": ["2024-01-01", "2024-07-01", "2025-01-01"],
            "LATITUDE": [48.5, 48.5, 48.5],
            "LONGITUDE": [-123.2, -123.2, -123.2],
        }
    )
    result = build_seasonal_features(frame)

    assert result.shape == (3, 16)
    assert np.isfinite(result.to_numpy()).all()
    assert result.loc[1, "season__daylight_hours"] > result.loc[0, "season__daylight_hours"]
    assert np.isclose(result.loc[0, "season__sin_1"], result.loc[2, "season__sin_1"])
    season_columns = [
        "season__winter",
        "season__spring",
        "season__summer",
        "season__autumn",
    ]
    assert result[season_columns].sum(axis=1).eq(1.0).all()


def test_static_seascape_join_preserves_missingness(tmp_path: Path) -> None:
    latitude, longitude = 48.5, -123.2
    cell = h3.latlng_to_cell(latitude, longitude, 8)
    artifact = tmp_path / "physical.parquet"
    pd.DataFrame(
        {
            "H3_INDEX": [cell],
            "DEPTH": [-100.0],
            "DISTANCE": [np.nan],
        }
    ).to_parquet(artifact, index=False)
    spec = SeascapeSpec("physical", artifact.name, 8, ("DEPTH", "DISTANCE"))
    targets = pd.DataFrame(
        {
            "LATITUDE": [latitude, 47.0],
            "LONGITUDE": [longitude, -125.0],
        }
    )

    features, coverage, lineage = load_static_seascape_features(targets, tmp_path, specs=(spec,))

    assert features.loc[0, "seascape__physical__depth"] == -100.0
    assert np.isnan(features.loc[0, "seascape__physical__distance"])
    assert features.loc[0, "seascape__physical__distance__missing"] == 1.0
    assert features.loc[1, "seascape__physical__cell_available"] == 0.0
    assert coverage.loc[0, "matched_cell_n"] == 1
    assert lineage[0]["sha256"]


def test_multistratum_guardrail_can_fall_back_to_incumbent() -> None:
    target = np.asarray([0, 0, 1, 1] * 30)
    current = np.where(target == 1, 0.8, 0.2)
    challenger = current.copy()
    challenger[:60] = 1.0 - current[:60]
    strata = pd.DataFrame(
        {
            "SOURCE": ["A"] * 60 + ["B"] * 60,
            "ERA": ["EARLY"] * 60 + ["LATE"] * 60,
            "REGION": ["NORTH"] * 60 + ["SOUTH"] * 60,
        }
    )

    current_weight, _loss, maximum_degradation = _select_multistratum_constrained_blend(
        current,
        challenger,
        target,
        strata,
        np.ones(len(target)),
    )

    assert current_weight >= 0.9
    assert maximum_degradation <= 0.10
