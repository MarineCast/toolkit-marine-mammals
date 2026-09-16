from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_bakeoff_experiment import (
    DIFFUSION_PROBABILITY_COLUMNS,
    _rolling_splits,
    fit_graph_diffusion_stack,
    group_robust_sample_weights,
)


def test_group_robust_weights_reduce_group_mass_imbalance() -> None:
    groups = np.asarray(["LARGE"] * 90 + ["SMALL"] * 10)
    result = group_robust_sample_weights(np.ones(100), groups, strength=1.0)

    assert np.isfinite(result).all()
    assert np.isclose(result.mean(), 1.0)
    assert np.isclose(result[groups == "LARGE"].sum(), result[groups == "SMALL"].sum())


def test_graph_diffusion_stack_is_a_probability_conserving_simplex() -> None:
    target = np.asarray([0, 1] * 60)
    good = np.where(target == 1, 0.8, 0.2)
    bad = 1.0 - good
    features = pd.DataFrame(
        {
            column: good if index == 0 else bad
            for index, column in enumerate(DIFFUSION_PROBABILITY_COLUMNS)
        }
    )
    model = fit_graph_diffusion_stack(
        features,
        target,
        np.ones(len(target)),
        np.asarray(["A"] * 60 + ["B"] * 60),
    )
    probability = model.predict(features, pd.DataFrame(index=features.index))

    assert np.isclose(model.weights.sum(), 1.0)
    assert np.all(model.weights >= 0.0)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert model.weights[0] > 0.99


def test_rolling_splits_are_strictly_past_to_future() -> None:
    frame = pd.DataFrame(
        {
            "SIGHTING_DATE": [f"{year}-06-01" for year in range(1990, 2027) for _ in range(2)],
            "Y_TRUE": [0, 1] * 37,
        }
    )
    splits = _rolling_splits(frame)

    assert len(splits) == 5
    dates = pd.to_datetime(frame["SIGHTING_DATE"])
    for _name, train, test in splits:
        assert dates.iloc[train].max() < dates.iloc[test].min()
        assert not set(train).intersection(test)
