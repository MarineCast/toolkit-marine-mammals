"""Killer-whale evidence veto; separate from generic risk calibration."""

import numpy as np
import pandas as pd


def other_support_veto(
    meta: pd.DataFrame,
    *,
    ratio: float,
    floor: float,
) -> np.ndarray:
    other = meta["SAME_DAY_OTHER_SUPPORT"].to_numpy(dtype=float)
    binary_max = np.maximum(
        meta["SAME_DAY_SRKW_SUPPORT"].to_numpy(dtype=float),
        meta["SAME_DAY_TRANSIENT_SUPPORT"].to_numpy(dtype=float),
    )
    return (other >= floor) & (other > ratio * np.maximum(binary_max, 1e-9))
