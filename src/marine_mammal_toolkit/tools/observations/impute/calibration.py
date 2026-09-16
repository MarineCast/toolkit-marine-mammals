from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_probability(p)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class ProbabilityCalibrator:
    method: str = "auto"
    random_state: int = 42
    fitted_method_: str | None = None
    model_: IsotonicRegression | LogisticRegression | None = None

    def fit(
        self, raw_probability: np.ndarray, y: np.ndarray
    ) -> "ProbabilityCalibrator":
        p = _clip_probability(raw_probability)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration requires both classes")
        method = self.method
        class_counts = np.bincount(y, minlength=2)
        if method == "auto":
            method = (
                "isotonic" if len(y) >= 300 and class_counts.min() >= 75 else "sigmoid"
            )
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(p, y)
        elif method == "sigmoid":
            model = LogisticRegression(
                C=1.0, solver="lbfgs", random_state=self.random_state
            )
            model.fit(_logit(p), y)
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
        self.fitted_method_ = method
        self.model_ = model
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.fitted_method_ is None:
            raise RuntimeError("Calibrator is not fitted")
        p = _clip_probability(raw_probability)
        if self.fitted_method_ == "isotonic":
            if not isinstance(self.model_, IsotonicRegression):
                raise RuntimeError(
                    "Isotonic calibrator has the wrong fitted model type"
                )
            out = self.model_.predict(p)
        else:
            if not isinstance(self.model_, LogisticRegression):
                raise RuntimeError("Sigmoid calibrator has the wrong fitted model type")
            out = self.model_.predict_proba(_logit(p))[:, 1]
        return _clip_probability(np.asarray(out, dtype=float))


@dataclass
class ClassConditionalConformal:
    alpha: float = 0.10
    scores_0_: np.ndarray | None = None
    scores_1_: np.ndarray | None = None

    def fit(
        self, probability_srkw: np.ndarray, y: np.ndarray
    ) -> "ClassConditionalConformal":
        p = _clip_probability(probability_srkw)
        y = np.asarray(y, dtype=int)
        scores_0 = p[y == 0]  # 1 - P(class 0)
        scores_1 = 1 - p[y == 1]  # 1 - P(class 1)
        if len(scores_0) == 0 or len(scores_1) == 0:
            raise ValueError("Conformal calibration requires both classes")
        self.scores_0_ = np.sort(scores_0)
        self.scores_1_ = np.sort(scores_1)
        return self

    @staticmethod
    def _pvalue(scores: np.ndarray, test_score: np.ndarray) -> np.ndarray:
        # Number of calibration scores >= test score, with the finite-sample +1 correction.
        idx = np.searchsorted(scores, test_score, side="left")
        ge = len(scores) - idx
        return (ge + 1.0) / (len(scores) + 1.0)

    def pvalues(self, probability_srkw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.scores_0_ is None or self.scores_1_ is None:
            raise RuntimeError("Conformal calibrator is not fitted")
        p = _clip_probability(probability_srkw)
        p0 = self._pvalue(self.scores_0_, p)
        p1 = self._pvalue(self.scores_1_, 1 - p)
        return p0, p1

    def prediction_sets(
        self, probability_srkw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p0, p1 = self.pvalues(probability_srkw)
        include_0 = p0 > self.alpha
        include_1 = p1 > self.alpha
        size = include_0.astype(int) + include_1.astype(int)
        label = np.full(len(size), -1, dtype=int)
        label[(size == 1) & include_0] = 0
        label[(size == 1) & include_1] = 1
        return label, size, np.column_stack([p0, p1])


def crossfit_calibration(
    raw_probability: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    method: str,
    random_state: int,
) -> tuple[np.ndarray, ProbabilityCalibrator]:
    raw_probability = _clip_probability(raw_probability)
    y = np.asarray(y, dtype=int)
    calibrated = np.full(len(y), np.nan, dtype=float)
    for fold_index, (train_idx, test_idx) in enumerate(folds):
        calibrator = ProbabilityCalibrator(
            method=method, random_state=random_state + fold_index
        )
        calibrator.fit(raw_probability[train_idx], y[train_idx])
        calibrated[test_idx] = calibrator.predict(raw_probability[test_idx])
    if np.isnan(calibrated).any():
        raise RuntimeError("Cross-fitted calibration did not cover every row")
    final = ProbabilityCalibrator(method=method, random_state=random_state)
    final.fit(raw_probability, y)
    return calibrated, final


def crossfit_conformal(
    calibrated_probability: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ClassConditionalConformal]:
    p = _clip_probability(calibrated_probability)
    y = np.asarray(y, dtype=int)
    singleton_label = np.full(len(y), -1, dtype=int)
    set_size = np.full(len(y), -1, dtype=int)
    pvalues = np.full((len(y), 2), np.nan, dtype=float)
    for train_idx, test_idx in folds:
        conformal = ClassConditionalConformal(alpha=alpha).fit(
            p[train_idx], y[train_idx]
        )
        labels, sizes, fold_pvalues = conformal.prediction_sets(p[test_idx])
        singleton_label[test_idx] = labels
        set_size[test_idx] = sizes
        pvalues[test_idx] = fold_pvalues
    if (set_size < 0).any() or np.isnan(pvalues).any():
        raise RuntimeError("Cross-fitted conformal predictions did not cover every row")
    final = ClassConditionalConformal(alpha=alpha).fit(p, y)
    return singleton_label, set_size, pvalues, final
