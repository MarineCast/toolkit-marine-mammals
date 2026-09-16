from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit

from guarded_residual_experiment import (
    _fit_offset_residual,
    _fit_intercept_only_calibration,
    fit_adaptive_blender,
)


def test_offset_residual_is_bounded_and_finite() -> None:
    design = pd.DataFrame(
        {
            "season": np.linspace(-1.0, 1.0, 120),
            "seascape": np.tile([0.0, 1.0, np.nan], 40),
        }
    )
    target = (design["season"].to_numpy() > 0).astype(int)
    baseline = np.full(len(target), 0.5)
    model = _fit_offset_residual(
        design,
        target,
        baseline,
        np.ones(len(target)),
        l2_penalty=0.1,
    )

    probability, correction = model.predict_probability(design, baseline, maximum_correction=0.5)

    assert np.isfinite(probability).all()
    assert np.max(np.abs(correction)) <= 0.5 + 1e-12
    assert np.max(np.abs(logit(probability) - logit(baseline))) <= 0.5 + 1e-10


def test_adaptive_blender_shrinks_a_failing_group() -> None:
    target = np.asarray([0, 1] * 120)
    baseline = np.where(target == 1, 0.75, 0.25)
    challenger = np.where(target == 1, 0.9, 0.1)
    challenger[:20] = 1.0 - challenger[:20]
    strata = pd.DataFrame(
        {
            "SOURCE": ["FAIL"] * 20 + ["HELP"] * 220,
            "ERA": ["ALL"] * 240,
            "REGION": ["NORTH"] * 20 + ["SOUTH"] * 220,
            "EVIDENCE_REGIME": ["LOCAL"] * 240,
            "SEASCAPE_COVERAGE": ["COMPLETE"] * 240,
        }
    )

    blender, diagnostics = fit_adaptive_blender(
        baseline,
        challenger,
        target,
        strata,
        np.ones(len(target)),
        minimum_group_n=10,
        shrinkage_n=50.0,
    )
    weights = blender.weights(strata)

    assert weights[:20].mean() < weights[20:].mean()
    assert (
        diagnostics.loc[diagnostics["stratum"].eq("FAIL"), "local_challenger_weight"].iat[0] == 0.0
    )
    assert np.all((weights >= 0.0) & (weights <= 1.0))


def test_adaptive_blender_falls_back_for_insufficient_or_unseen_strata() -> None:
    target = np.asarray([0, 1] * 120)
    baseline = np.where(target == 1, 0.75, 0.25)
    challenger = np.where(target == 1, 0.9, 0.1)
    strata = pd.DataFrame(
        {
            "SOURCE": ["SMALL"] * 20 + ["LARGE"] * 220,
            "ERA": ["ALL"] * 240,
            "REGION": ["ALL"] * 240,
            "EVIDENCE_REGIME": ["LOCAL"] * 240,
            "SEASCAPE_COVERAGE": ["COMPLETE"] * 240,
        }
    )
    blender, _diagnostics = fit_adaptive_blender(
        baseline, challenger, target, strata, np.ones(len(target))
    )

    assert np.all(blender.weights(strata.iloc[:20]) == 0.0)
    unseen = strata.iloc[[20]].copy()
    unseen["SOURCE"] = "UNSEEN"
    assert blender.weights(unseen)[0] == 0.0


def test_intercept_only_calibration_corrects_mean_bias_without_changing_rank() -> None:
    target = np.asarray([0, 1] * 100)
    probability = np.where(target == 1, 0.6, 0.1)
    adjustment = _fit_intercept_only_calibration(probability, target, np.ones(len(target)))
    calibrated = 1.0 / (1.0 + np.exp(-(logit(probability) + adjustment)))

    assert np.isclose(calibrated.mean(), target.mean(), atol=1e-6)
    assert np.array_equal(np.argsort(calibrated), np.argsort(probability))
