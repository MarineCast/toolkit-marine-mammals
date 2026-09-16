from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from marine_mammal_toolkit.tools.observations.impute.metrics import wilson_upper_bound


@dataclass(frozen=True)
class PolicyRule:
    threshold: float
    calibration_n: int
    accepted_n: int
    errors: int
    error_upper: float
    scope: str
    status: str = "selected"


@dataclass
class AcceptancePolicy:
    target_error: float = 0.05
    confidence: float = 0.95
    min_examples: int = 20
    min_accepts: int = 12
    hard_abstain_regimes: tuple[str, ...] = ()
    rules: dict[str, PolicyRule] = field(default_factory=dict)

    @staticmethod
    def _key(predicted_class: int | str, regime: str | None) -> str:
        cls = str(predicted_class)
        return f"class={cls}|regime={regime or '*'}"

    def _select_rule(
        self,
        y: np.ndarray,
        p: np.ndarray,
        base_eligible: np.ndarray,
        *,
        scope: str,
    ) -> PolicyRule:
        n_scope = len(y)
        if n_scope < self.min_examples:
            return PolicyRule(
                float("inf"), n_scope, 0, 0, 1.0, scope, "insufficient_examples"
            )
        pred = (p >= 0.5).astype(int)
        maxp = np.maximum(p, 1 - p)
        candidate_values = np.unique(
            np.concatenate(
                [
                    np.asarray(
                        [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
                    ),
                    (
                        np.quantile(maxp[base_eligible], np.linspace(0, 1, 31))
                        if base_eligible.any()
                        else np.asarray([])
                    ),
                ]
            )
        )
        candidate_values = np.sort(
            candidate_values[(candidate_values >= 0.5) & (candidate_values <= 1)]
        )
        eligible_n = int(base_eligible.sum())
        if eligible_n < self.min_accepts:
            return PolicyRule(
                float("inf"),
                n_scope,
                0,
                0,
                1.0,
                scope,
                "insufficient_accepts",
            )
        best: PolicyRule | None = None
        for threshold in candidate_values:
            accepted = base_eligible & (maxp >= threshold)
            n = int(accepted.sum())
            if n < self.min_accepts:
                continue
            errors = int((pred[accepted] != y[accepted]).sum())
            upper = wilson_upper_bound(errors, n, self.confidence)
            if upper <= self.target_error:
                candidate = PolicyRule(
                    float(threshold), n_scope, n, errors, upper, scope, "selected"
                )
                if best is None or candidate.accepted_n > best.accepted_n:
                    best = candidate
                elif best is not None and candidate.accepted_n == best.accepted_n:
                    if candidate.threshold < best.threshold:
                        best = candidate
        return best or PolicyRule(
            float("inf"),
            n_scope,
            0,
            0,
            1.0,
            scope,
            "risk_failed",
        )

    def fit(
        self,
        y: np.ndarray,
        p: np.ndarray,
        singleton_label: np.ndarray,
        ood_inlier: np.ndarray,
        regimes: np.ndarray,
        other_veto: np.ndarray,
    ) -> "AcceptancePolicy":
        y = np.asarray(y, dtype=int)
        p = np.asarray(p, dtype=float)
        predicted = (p >= 0.5).astype(int)
        singleton_label = np.asarray(singleton_label, dtype=int)
        ood_inlier = np.asarray(ood_inlier, dtype=bool)
        regimes = np.asarray(regimes, dtype=str)
        other_veto = np.asarray(other_veto, dtype=bool)
        base = (singleton_label == predicted) & ood_inlier & ~other_veto

        rules: dict[str, PolicyRule] = {}
        for cls in (0, 1):
            class_mask = predicted == cls
            for regime in sorted(np.unique(regimes[class_mask])):
                mask = class_mask & (regimes == regime)
                key = self._key(cls, regime)
                rules[key] = self._select_rule(
                    y[mask], p[mask], base[mask], scope=f"class={cls},regime={regime}"
                )
            key = self._key(cls, None)
            rules[key] = self._select_rule(
                y[class_mask], p[class_mask], base[class_mask], scope=f"class={cls}"
            )
        rules[self._key("*", None)] = self._select_rule(y, p, base, scope="global")
        self.rules = rules
        return self

    def _rule_for(self, cls: int, regime: str) -> PolicyRule:
        specific = self.rules.get(self._key(cls, regime))
        if specific is not None:
            if specific.status not in {
                "insufficient_examples",
                "insufficient_accepts",
            }:
                # A sufficiently sampled regime that fails its risk requirement
                # must never inherit a more permissive pooled class rule.
                return specific
        class_rule = self.rules.get(self._key(cls, None))
        if class_rule is not None:
            return class_rule
        return self.rules.get(
            self._key("*", None),
            PolicyRule(float("inf"), 0, 0, 0, 1.0, "none"),
        )

    def apply(
        self,
        p: np.ndarray,
        singleton_label: np.ndarray,
        ood_inlier: np.ndarray,
        regimes: np.ndarray,
        other_veto: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(p, dtype=float)
        predicted = (p >= 0.5).astype(int)
        maxp = np.maximum(p, 1 - p)
        singleton_label = np.asarray(singleton_label, dtype=int)
        ood_inlier = np.asarray(ood_inlier, dtype=bool)
        regimes = np.asarray(regimes, dtype=str)
        other_veto = np.asarray(other_veto, dtype=bool)

        accepted = np.zeros(len(p), dtype=bool)
        threshold_used = np.full(len(p), np.inf, dtype=float)
        reason = np.full(len(p), "RISK_THRESHOLD_NOT_MET", dtype=object)
        for i in range(len(p)):
            if regimes[i] in self.hard_abstain_regimes:
                reason[i] = str(regimes[i])
                continue
            if other_veto[i]:
                reason[i] = "OTHER_CLASS_PLAUSIBLE"
                continue
            if not ood_inlier[i]:
                reason[i] = "OUT_OF_DISTRIBUTION"
                continue
            if singleton_label[i] < 0:
                reason[i] = "AMBIGUOUS_CONFORMAL_SET"
                continue
            if singleton_label[i] != predicted[i]:
                reason[i] = "MODEL_CONFORMAL_DISAGREEMENT"
                continue
            rule = self._rule_for(int(predicted[i]), str(regimes[i]))
            threshold_used[i] = rule.threshold
            if maxp[i] >= rule.threshold:
                accepted[i] = True
                reason[i] = "ACCEPTED"
        return accepted, threshold_used, reason.astype(str)

    def rule_metadata(
        self, p: np.ndarray, regimes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the fitted class/regime rule scope and error bound per row."""

        probability = np.asarray(p, dtype=float)
        predicted = (probability >= 0.5).astype(int)
        regime_values = np.asarray(regimes, dtype=str)
        scopes = np.empty(len(probability), dtype=object)
        upper = np.ones(len(probability), dtype=float)
        for index, (cls, regime) in enumerate(
            zip(predicted, regime_values, strict=True)
        ):
            rule = self._rule_for(int(cls), str(regime))
            scopes[index] = rule.scope
            upper[index] = rule.error_upper
        return scopes.astype(str), upper

    def to_dict(self) -> dict:
        return {
            "target_error": self.target_error,
            "confidence": self.confidence,
            "min_examples": self.min_examples,
            "min_accepts": self.min_accepts,
            "hard_abstain_regimes": list(self.hard_abstain_regimes),
            "rules": {key: asdict(value) for key, value in self.rules.items()},
        }


def crossfit_policy(
    y: np.ndarray,
    p: np.ndarray,
    singleton_label: np.ndarray,
    ood_inlier: np.ndarray,
    regimes: np.ndarray,
    other_veto: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    target_error: float,
    confidence: float,
    min_examples: int,
    min_accepts: int,
    hard_abstain_regimes: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AcceptancePolicy]:
    accepted = np.zeros(len(y), dtype=bool)
    threshold = np.full(len(y), np.inf, dtype=float)
    reason = np.full(len(y), "NOT_EVALUATED", dtype=object)
    for train_idx, test_idx in folds:
        policy = AcceptancePolicy(
            target_error=target_error,
            confidence=confidence,
            min_examples=min_examples,
            min_accepts=min_accepts,
            hard_abstain_regimes=hard_abstain_regimes,
        ).fit(
            np.asarray(y)[train_idx],
            np.asarray(p)[train_idx],
            np.asarray(singleton_label)[train_idx],
            np.asarray(ood_inlier)[train_idx],
            np.asarray(regimes)[train_idx],
            np.asarray(other_veto)[train_idx],
        )
        a, t, r = policy.apply(
            np.asarray(p)[test_idx],
            np.asarray(singleton_label)[test_idx],
            np.asarray(ood_inlier)[test_idx],
            np.asarray(regimes)[test_idx],
            np.asarray(other_veto)[test_idx],
        )
        accepted[test_idx] = a
        threshold[test_idx] = t
        reason[test_idx] = r

    final = AcceptancePolicy(
        target_error=target_error,
        confidence=confidence,
        min_examples=min_examples,
        min_accepts=min_accepts,
        hard_abstain_regimes=hard_abstain_regimes,
    ).fit(y, p, singleton_label, ood_inlier, regimes, other_veto)
    return accepted, threshold, reason.astype(str), final
