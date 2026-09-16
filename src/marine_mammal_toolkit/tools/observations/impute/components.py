"""Species-supplied components for the selective binary inference engine."""

from dataclasses import dataclass
from typing import Callable, Any
import pandas as pd


@dataclass
class FeatureBatch:
    X: pd.DataFrame
    meta: pd.DataFrame


@dataclass(frozen=True)
class ImputationComponents:
    feature_builder: Callable[..., Any]
    prepare_frame: Callable[..., pd.DataFrame]
    acceptance_policy: Callable[..., Any]
    support_veto: Callable[..., Any]
