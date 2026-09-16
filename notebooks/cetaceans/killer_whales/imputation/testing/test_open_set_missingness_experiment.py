from __future__ import annotations

import numpy as np
import pandas as pd

from open_set_missingness_experiment import _hierarchical_multiclass, propensity_weights


def test_propensity_weights_are_finite_bounded_and_normalized() -> None:
    result = propensity_weights([0.0, 0.01, 0.5, 1.0], lower_bound=0.05)

    assert np.isfinite(result).all()
    assert np.isclose(result.mean(), 1.0)
    assert result.max() / result.min() <= 20.0 + 1e-12


def test_hierarchical_open_set_conserves_mass_and_does_not_certify_counts() -> None:
    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": ["a", "b", "c"],
            "ENCOUNTER_ID": ["ea", "eb", "ec"],
            "THREE_CLASS": ["SRKW", "TRANSIENT", "OTHER"],
            "OPEN_TARGET": [1, 1, 0],
            "P_SRKW": [0.8, 0.2, 0.5],
            "open_model": [0.9, 0.8, 0.1],
            "EVALUATION_WEIGHT": [1.0, 1.0, 1.0],
        }
    )
    _comparison, probabilities = _hierarchical_multiclass(frame, "open_model")

    mass_columns = [
        "P_SRKW_UNCONDITIONAL",
        "P_TRANSIENT_UNCONDITIONAL",
        "P_OTHER_UNCONDITIONAL",
    ]
    assert np.allclose(probabilities[mass_columns].sum(axis=1), 1.0)
    assert not probabilities["SOFT_COUNT_CERTIFIED"].any()
    assert probabilities["EXPECTED_UNKNOWN_COUNT_IF_UNLABELED"].eq(1.0).all()
    assert probabilities.loc[2, "P_OTHER_UNCONDITIONAL"] == 0.9
