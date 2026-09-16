from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088


def make_spatiotemporal_groups(
    frame: pd.DataFrame,
    *,
    block_days: int = 7,
    block_km: float = 20.0,
) -> pd.Series:
    """Create coarse date × space blocks for leakage-resistant evaluation."""

    dates = pd.to_datetime(frame["SIGHTING_DATE"], format="mixed").dt.normalize()
    day_block = (dates.astype("int64") // (86_400 * 10**9) // block_days).astype(str)
    lat = frame["LATITUDE"].to_numpy(dtype=float)
    lon = frame["LONGITUDE"].to_numpy(dtype=float)
    y = np.floor((lat * 110.574) / block_km).astype(int)
    x = np.floor((lon * 111.320 * np.cos(np.deg2rad(lat))) / block_km).astype(int)
    groups = pd.Series(
        [f"{d}:{xi}:{yi}" for d, xi, yi in zip(day_block, x, y, strict=True)],
        index=frame.index,
        name="CV_GROUP",
    )
    if "ENCOUNTER_ID" in frame:
        encounter = frame["ENCOUNTER_ID"].astype("string")
        fallback = pd.Series(
            [f"__row__{position}" for position in range(len(frame))],
            index=frame.index,
            dtype="string",
        )
        encounter = encounter.fillna(fallback).astype(str)
        groups = groups.groupby(encounter, sort=False).transform("min")
        groups.name = "CV_GROUP"
    return groups


def _effective_splits(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        raise ValueError("Both SRKW and TRANSIENT labels are required")
    n = int(min(requested, counts.min()))
    if n < 2:
        raise ValueError("Not enough examples per class for cross-validation")
    return n


def iter_splits(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    strategy: str,
    n_splits: int,
    random_state: int,
    groups: pd.Series | np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = _effective_splits(y, n_splits)
    if strategy == "reconstruction":
        splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=random_state)
        yield from splitter.split(frame, y)
        return
    if strategy not in {"encounter", "blocked", "purged_blocked"}:
        raise ValueError(f"Unknown split strategy: {strategy}")
    if groups is None:
        raise ValueError("Blocked splitting requires groups")
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    n = min(n, len(unique_groups))
    if n < 2:
        raise ValueError("Not enough spatiotemporal groups for blocked validation")
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=random_state)
    try:
        yield from splitter.split(frame, y, group_array)
    except ValueError as exc:
        raise ValueError(
            "Blocked CV could not produce class-balanced folds. Increase data, reduce "
            "n_splits, or use larger/smaller block settings."
        ) from exc


def purge_spatiotemporal_neighbors(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    purge_days: int,
    purge_km: float,
) -> np.ndarray:
    """Remove training rows that could provide direct context to held-out rows."""

    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.asarray(train_idx, dtype=int)
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]
    train_rad = np.deg2rad(train[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float))
    test_rad = np.deg2rad(test[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float))
    tree = BallTree(test_rad, metric="haversine")
    neighbors = tree.query_radius(train_rad, r=purge_km / EARTH_RADIUS_KM)
    train_dates = pd.to_datetime(train["SIGHTING_DATE"]).to_numpy(dtype="datetime64[D]")
    test_dates = pd.to_datetime(test["SIGHTING_DATE"]).to_numpy(dtype="datetime64[D]")
    keep = np.ones(len(train_idx), dtype=bool)
    for position, candidate_test_idx in enumerate(neighbors):
        if len(candidate_test_idx) == 0:
            continue
        day_gap = np.abs(
            (
                test_dates[np.asarray(candidate_test_idx, dtype=int)]
                - train_dates[position]
            )
            .astype("timedelta64[D]")
            .astype(int)
        )
        if np.any(day_gap <= purge_days):
            keep[position] = False
    return np.asarray(train_idx, dtype=int)[keep]
