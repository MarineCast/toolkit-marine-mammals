from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import h3  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools._core.data import ProcessingMode
from marine_mammal_toolkit.tools._core.data import StageResult
from marine_mammal_toolkit.tools._core.data import ValidationReport
from marine_mammal_toolkit.tools._core.persistence import checksum_path

from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.schemas.observations import COUNT_EXCLUSION_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import COUNT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import DETAIL_COUNT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import PERIOD_TOTAL_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import POD_COUNT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import TOTAL_COUNT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import CountRequest
from marine_mammal_toolkit.tools.observations.runtime import code_revision
from marine_mammal_toolkit.tools.observations.runtime import resume_result
from marine_mammal_toolkit.tools.observations.runtime import stage_signature

COUNT_MEASURES = [
    "SIGHTING_COUNT",
    "SOURCE_REPORT_COUNT",
    "OBSERVED_SIGHTING_COUNT",
    "HARD_IMPUTED_COUNT",
    "EXPECTED_SIGHTING_COUNT",
    "MATURE_EXPECTED_SIGHTING_COUNT",
    "PROVISIONAL_EXPECTED_COUNT",
    "EXPECTED_UNKNOWN_COUNT",
]

UNVERIFIED_PERIOD_STATUS = "UNVERIFIED_COVERAGE"


def _coverage_contract(artifact: ArtifactRef, start: date, end: date) -> dict[str, Any]:
    """Describe whether zero-filled periods are backed by verified source coverage."""

    snapshot = artifact.data_snapshot
    coverage_start = snapshot.coverage_start if snapshot is not None else None
    coverage_through = snapshot.coverage_through if snapshot is not None else None
    snapshot_verified = bool(
        snapshot is not None
        and snapshot.coverage_status == "verified_intersection"
        and coverage_start
        and coverage_through
    )
    window_verified = bool(
        snapshot_verified
        and start >= date.fromisoformat(str(coverage_start))
        and end <= date.fromisoformat(str(coverage_through))
    )
    cohort_identity = {
        "snapshot_id": snapshot.snapshot_id if snapshot is not None else None,
        "input_checksum": artifact.checksum or checksum_path(artifact.path),
        "coverage_start": coverage_start,
        "coverage_through": coverage_through,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
    }
    cohort_digest = hashlib.sha256(
        json.dumps(cohort_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return {
        "target_cohort_id": f"sightings-target-cohort-v1:{cohort_digest}",
        "target_cohort_status": (
            "VERIFIED_COMPLETE" if window_verified else "UNVERIFIED"
        ),
        "snapshot_coverage_status": (
            snapshot.coverage_status if snapshot is not None else "missing"
        ),
        "coverage_start": coverage_start,
        "coverage_through": coverage_through,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "zero_fill_verified": window_verified,
        "zero_semantics": (
            "verified_no_report"
            if window_verified
            else "unavailable_unverified_coverage_not_no_report"
        ),
    }


def _validated_expected_weights(
    work: pd.DataFrame, *, policy
) -> dict[str, pd.Series] | None:
    """Return finite per-row class mass or fail before any aggregation.

    Target-class rows must carry exactly one unit of expected mass. Explicit
    non-target classes (for example NRKW or OFFSHORE) may carry zero target
    mass; `_expected_contributions` represents those rows as one unit in their
    observed OTHER detail instead of silently dropping them.
    """

    present = set(policy.expected_columns.values()).intersection(work.columns)
    if not present:
        return None
    if present != set(policy.expected_columns.values()):
        raise ValueError(
            "Expected-mass inputs must be all-or-none; missing "
            f"{sorted(set(policy.expected_columns.values()) - present)}"
        )
    mass_columns = [*policy.expected_columns.values()]
    has_other_mass = policy.other_column in work
    if has_other_mass:
        mass_columns.append(policy.other_column)
    numeric = work.loc[:, mass_columns].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    bounded = ((values >= 0.0) & (values <= 1.0)).all(axis=1)
    totals = values.sum(axis=1)
    unit_mass = np.isclose(totals, 1.0, rtol=0.0, atol=1e-9)
    zero_mass = np.isclose(totals, 0.0, rtol=0.0, atol=1e-12)
    detail_column = (
        "ECOTYPE_DETAIL_EFFECTIVE"
        if "ECOTYPE_DETAIL_EFFECTIVE" in work
        else "ECOTYPE_DETAIL"
    )
    detail = work[detail_column].astype(str).str.upper()
    probabilistic = (
        work.get("USE_FOR_PROBABILISTIC_COUNTS", pd.Series(False, index=work.index))
        .fillna(False)
        .astype(bool)
    )
    explicit_non_target = ~detail.isin(set(policy.expected_columns)) & ~probabilistic
    valid_mass = (
        unit_mass
        if has_other_mass
        else unit_mass | (zero_mass & explicit_non_target.to_numpy())
    )
    valid = finite & bounded & valid_mass
    if not valid.all():
        invalid_positions = np.flatnonzero(~valid)
        sample = []
        for position in invalid_positions[:5]:
            observation_id = (
                str(work.iloc[position]["OBSERVATION_ID"])
                if "OBSERVATION_ID" in work
                else str(work.index[position])
            )
            sample.append(
                {
                    "observation_id": observation_id,
                    "mass": float(totals[position]),
                    "values": values[position].tolist(),
                }
            )
        raise ValueError(
            "Expected class mass must be finite, bounded in [0, 1], and sum to one "
            "per row (legacy three-column inputs may use zero only for an explicit "
            "non-target class); "
            f"invalid_rows={len(invalid_positions)}, sample={sample}"
        )
    return {
        column: pd.Series(numeric[column].to_numpy(dtype=float), index=work.index)
        for column in mass_columns
    }


def _expected_contributions(work: pd.DataFrame, *, policy) -> pd.DataFrame:
    """Expand each observation into additive ecotype probability contributions."""

    if work.empty:
        return work.assign(
            EXPECTED_SIGHTING_COUNT=pd.Series(dtype="float64"),
            MATURE_EXPECTED_SIGHTING_COUNT=pd.Series(dtype="float64"),
            PROVISIONAL_EXPECTED_COUNT=pd.Series(dtype="float64"),
            EXPECTED_UNKNOWN_COUNT=pd.Series(dtype="float64"),
        )
    validated_weights = _validated_expected_weights(work, policy=policy)
    expected_columns = policy.expected_columns
    pieces: list[pd.DataFrame] = []
    total_binary = pd.Series(0.0, index=work.index)
    mature = ~work.get(
        "IMPUTATION_CONTEXT_STATUS", pd.Series("MATURE_RETROSPECTIVE", index=work.index)
    ).eq("PROVISIONAL_RECENT")
    for label, column in expected_columns.items():
        if validated_weights is not None:
            weight = validated_weights[column]
        else:
            weight = work["ECOTYPE_DETAIL"].eq(label).astype(float)
        total_binary = total_binary + weight
        keep = weight.gt(0)
        if not keep.any():
            continue
        contribution = work.loc[keep].copy()
        contribution["ECOTYPE_DETAIL"] = label
        contribution["ECOTYPE_BUCKET"] = (
            label if label != policy.unknown_label else policy.other_bucket
        )
        contribution["EXPECTED_SIGHTING_COUNT"] = weight.loc[keep]
        contribution["MATURE_EXPECTED_SIGHTING_COUNT"] = (
            weight.loc[keep] * mature.loc[keep]
        )
        contribution["PROVISIONAL_EXPECTED_COUNT"] = (
            weight.loc[keep] * ~mature.loc[keep]
        )
        contribution["EXPECTED_UNKNOWN_COUNT"] = (
            weight.loc[keep] if label == policy.unknown_label else 0.0
        )
        pieces.append(contribution)

    if validated_weights is not None and policy.other_column in validated_weights:
        other_weight = validated_weights[policy.other_column]
        keep = other_weight.gt(0)
        if keep.any():
            contribution = work.loc[keep].copy()
            contribution["ECOTYPE_BUCKET"] = policy.other_bucket
            contribution["EXPECTED_SIGHTING_COUNT"] = other_weight.loc[keep]
            contribution["MATURE_EXPECTED_SIGHTING_COUNT"] = (
                other_weight.loc[keep] * mature.loc[keep]
            )
            contribution["PROVISIONAL_EXPECTED_COUNT"] = (
                other_weight.loc[keep] * ~mature.loc[keep]
            )
            contribution["EXPECTED_UNKNOWN_COUNT"] = 0.0
            pieces.append(contribution)
        represented_mass = total_binary + other_weight
    else:
        represented_mass = total_binary

    other = represented_mass.eq(0)
    if other.any():
        contribution = work.loc[other].copy()
        contribution["EXPECTED_SIGHTING_COUNT"] = 1.0
        contribution["MATURE_EXPECTED_SIGHTING_COUNT"] = mature.loc[other].astype(float)
        contribution["PROVISIONAL_EXPECTED_COUNT"] = (~mature.loc[other]).astype(float)
        contribution["EXPECTED_UNKNOWN_COUNT"] = 0.0
        pieces.append(contribution)
    return pd.concat(pieces, ignore_index=True)


def _universe(
    artifacts: tuple[ArtifactRef, ...], resolution: int
) -> tuple[set[str], ArtifactRef]:
    matches = [
        artifact
        for artifact in artifacts
        if (artifact.dataset_id or "").endswith(f"_r{resolution}")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Exactly one full counting universe is required for H{resolution}"
        )
    artifact = matches[0]
    if artifact.checksum and artifact.checksum != checksum_path(artifact.path):
        raise ValueError(f"Counting universe checksum mismatch for H{resolution}")
    frame = pd.read_parquet(artifact.path)
    column = next(
        (name for name in ("H3_INDEX", "h3", "H3", "hex_id") if name in frame), None
    )
    if column is None:
        raise ValueError(f"Counting universe has no H3_INDEX column: {artifact.path}")
    raw_cells = [str(value) for value in frame[column].dropna()]
    if len(raw_cells) != len(set(raw_cells)):
        raise ValueError(
            f"Counting universe contains duplicate cells for H{resolution}"
        )
    invalid = [cell for cell in raw_cells if not h3.is_valid_cell(cell)]
    wrong_resolution = [
        cell
        for cell in raw_cells
        if h3.is_valid_cell(cell) and h3.get_resolution(cell) != resolution
    ]
    if invalid:
        raise ValueError(
            f"Counting universe contains {len(invalid)} invalid H3 cells for H{resolution}"
        )
    if wrong_resolution:
        raise ValueError(
            f"Counting universe contains {len(wrong_resolution)} wrong-resolution cells for H{resolution}"
        )
    cells = set(raw_cells)
    if not cells:
        raise ValueError(f"Counting universe is empty for H{resolution}")
    return cells, artifact


def _period_fields(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    out = frame.copy()
    start = pd.to_datetime(out.PERIOD_START)
    out["YEAR"] = start.dt.year.astype("int16")
    if frequency == "daily":
        out["PERIOD_END"] = out.PERIOD_START
        out["DAY_OF_YEAR"] = start.dt.dayofyear.astype("Int16")
        out["ISO_YEAR"] = pd.NA
        out["ISO_WEEK"] = pd.NA
    else:
        iso = start.dt.isocalendar()
        out["PERIOD_END"] = (start + pd.to_timedelta(6, unit="D")).dt.date
        out["DAY_OF_YEAR"] = pd.NA
        out["ISO_YEAR"] = iso.year.astype("Int16")
        out["ISO_WEEK"] = iso.week.astype("Int8")
    out["FREQUENCY"] = frequency
    return out


def _daily_counts(
    observations: pd.DataFrame,
    associations: pd.DataFrame,
    cells: set[str],
    resolution: int,
    *,
    policy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = observations.copy()
    work["H3_INDEX"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(work.LATITUDE, work.LONGITUDE, strict=True)
    ]
    outside = ~work.H3_INDEX.isin(cells)
    excluded = work.loc[
        outside,
        ["OBSERVATION_ID", "SIGHTING_DATE", "LATITUDE", "LONGITUDE", "H3_INDEX"],
    ].copy()
    excluded["H3_RESOLUTION"] = resolution
    excluded["REASON"] = "OUTSIDE_FULL_AREA_WATER_UNIVERSE"
    work = work.loc[~outside].copy()
    work["H3_RESOLUTION"] = resolution
    work["PERIOD_START"] = work.SIGHTING_DATE
    applied = work.get("IMPUTATION_APPLIED", pd.Series(False, index=work.index)).fillna(
        False
    )
    if "ECOTYPE_DETAIL_EFFECTIVE" in work:
        work["ECOTYPE_DETAIL"] = work["ECOTYPE_DETAIL_EFFECTIVE"]
    if "ECOTYPE_BUCKET_EFFECTIVE" in work:
        work["ECOTYPE_BUCKET"] = work["ECOTYPE_BUCKET_EFFECTIVE"]
    work["OBSERVED_SIGHTING_COUNT"] = (~applied.astype(bool)).astype("int32")
    work["HARD_IMPUTED_COUNT"] = applied.astype(bool).astype("int32")
    expected = _expected_contributions(work, policy=policy)

    def grouped(category: str, output: str) -> pd.DataFrame:
        columns = ["H3_INDEX", "H3_RESOLUTION", "PERIOD_START", category]
        hard = work.groupby(columns, as_index=False).agg(
            SIGHTING_COUNT=("OBSERVATION_ID", "nunique"),
            SOURCE_REPORT_COUNT=("SOURCE_REPORT_COUNT", "sum"),
            OBSERVED_SIGHTING_COUNT=("OBSERVED_SIGHTING_COUNT", "sum"),
            HARD_IMPUTED_COUNT=("HARD_IMPUTED_COUNT", "sum"),
        )
        probabilistic = expected.groupby(columns, as_index=False)[
            [
                "EXPECTED_SIGHTING_COUNT",
                "MATURE_EXPECTED_SIGHTING_COUNT",
                "PROVISIONAL_EXPECTED_COUNT",
                "EXPECTED_UNKNOWN_COUNT",
            ]
        ].sum()
        result = hard.merge(probabilistic, on=columns, how="outer").fillna(0)
        result = result.rename(columns={category: output})
        return _period_fields(result, "daily")

    bucket = grouped("ECOTYPE_BUCKET", "ECOTYPE_BUCKET")
    detail = grouped("ECOTYPE_DETAIL", "ECOTYPE_DETAIL")
    total_columns = ["H3_INDEX", "H3_RESOLUTION", "PERIOD_START"]
    hard_total = work.groupby(total_columns, as_index=False).agg(
        SIGHTING_COUNT=("OBSERVATION_ID", "nunique"),
        SOURCE_REPORT_COUNT=("SOURCE_REPORT_COUNT", "sum"),
        OBSERVED_SIGHTING_COUNT=("OBSERVED_SIGHTING_COUNT", "sum"),
        HARD_IMPUTED_COUNT=("HARD_IMPUTED_COUNT", "sum"),
    )
    expected_total = expected.groupby(total_columns, as_index=False)[
        [
            "EXPECTED_SIGHTING_COUNT",
            "MATURE_EXPECTED_SIGHTING_COUNT",
            "PROVISIONAL_EXPECTED_COUNT",
            "EXPECTED_UNKNOWN_COUNT",
        ]
    ].sum()
    total = _period_fields(
        hard_total.merge(expected_total, on=total_columns, how="outer").fillna(0),
        "daily",
    )
    pods = associations[
        associations.ASSOCIATION_KIND.eq("POD")
        & associations.ASSOCIATION_VALUE.isin(["J", "K", "L"])
    ][["OBSERVATION_ID", "ASSOCIATION_VALUE"]].drop_duplicates()
    pod_work = work.merge(pods, on="OBSERVATION_ID", how="inner")
    pod = _period_fields(
        pod_work.groupby(
            ["H3_INDEX", "H3_RESOLUTION", "PERIOD_START", "ASSOCIATION_VALUE"],
            as_index=False,
        )
        .agg(
            SIGHTING_COUNT=("OBSERVATION_ID", "nunique"),
            SOURCE_REPORT_COUNT=("SOURCE_REPORT_COUNT", "sum"),
            OBSERVED_SIGHTING_COUNT=("OBSERVED_SIGHTING_COUNT", "sum"),
            HARD_IMPUTED_COUNT=("HARD_IMPUTED_COUNT", "sum"),
        )
        .rename(columns={"ASSOCIATION_VALUE": "POD"}),
        "daily",
    )
    if not pod.empty:
        pod["EXPECTED_SIGHTING_COUNT"] = pod["SIGHTING_COUNT"].astype(float)
        pod["MATURE_EXPECTED_SIGHTING_COUNT"] = pod["SIGHTING_COUNT"].astype(float)
        pod["PROVISIONAL_EXPECTED_COUNT"] = 0.0
        pod["EXPECTED_UNKNOWN_COUNT"] = 0.0
    for frame in (bucket, detail, total, pod):
        frame["PERIOD_STATUS"] = "COMPLETE"
        frame["IS_COMPLETE_PERIOD"] = True
    return bucket, detail, total, pod, excluded


def _weekly_from_daily(
    daily: pd.DataFrame,
    category_column: str | None,
    start: date,
    end: date,
    mode: ProcessingMode,
) -> tuple[pd.DataFrame, int]:
    if daily.empty:
        return daily.copy(), 0
    work = daily.copy()
    day = pd.to_datetime(work.PERIOD_START)
    work["PERIOD_START"] = (day - pd.to_timedelta(day.dt.weekday, unit="D")).dt.date
    columns = ["H3_INDEX", "H3_RESOLUTION", "PERIOD_START"]
    if category_column:
        columns.append(category_column)
    weekly = work.groupby(columns, as_index=False)[COUNT_MEASURES].sum()
    weekly = _period_fields(weekly, "weekly")
    complete = weekly.PERIOD_START.map(
        lambda value: value >= start
    ) & weekly.PERIOD_END.map(lambda value: value <= end)
    final_week_start = end - timedelta(days=end.weekday())
    open_mask = weekly.PERIOD_START.eq(final_week_start) & ~complete
    keep = complete | (open_mask if mode is ProcessingMode.AS_OF else False)
    dropped = int((~keep).sum())
    weekly = weekly.loc[keep].copy()
    weekly["PERIOD_STATUS"] = "COMPLETE"
    weekly["IS_COMPLETE_PERIOD"] = True
    if mode is ProcessingMode.AS_OF:
        weekly.loc[
            weekly.PERIOD_START.eq(final_week_start) & ~complete.loc[weekly.index],
            "PERIOD_STATUS",
        ] = "OPEN"
        weekly.loc[weekly.PERIOD_STATUS.eq("OPEN"), "IS_COMPLETE_PERIOD"] = False
    return weekly, dropped


def _period_totals(
    observations: pd.DataFrame,
    start: date,
    end: date,
    mode: ProcessingMode,
    *,
    zero_fill_verified: bool,
    policy,
) -> pd.DataFrame:
    calendar = pd.DataFrame({"PERIOD_START": pd.date_range(start, end, freq="D").date})
    work = observations.copy()
    applied = work.get("IMPUTATION_APPLIED", pd.Series(False, index=work.index)).fillna(
        False
    )
    work["OBSERVED_SIGHTING_COUNT"] = (~applied.astype(bool)).astype("int32")
    work["HARD_IMPUTED_COUNT"] = applied.astype(bool).astype("int32")
    expected = _expected_contributions(work, policy=policy)
    observed = (
        work.groupby("SIGHTING_DATE", as_index=False)
        .agg(
            SIGHTING_COUNT=("OBSERVATION_ID", "nunique"),
            SOURCE_REPORT_COUNT=("SOURCE_REPORT_COUNT", "sum"),
            OBSERVED_SIGHTING_COUNT=("OBSERVED_SIGHTING_COUNT", "sum"),
            HARD_IMPUTED_COUNT=("HARD_IMPUTED_COUNT", "sum"),
        )
        .rename(columns={"SIGHTING_DATE": "PERIOD_START"})
    )
    probabilistic = (
        expected.groupby("SIGHTING_DATE", as_index=False)[
            [
                "EXPECTED_SIGHTING_COUNT",
                "MATURE_EXPECTED_SIGHTING_COUNT",
                "PROVISIONAL_EXPECTED_COUNT",
                "EXPECTED_UNKNOWN_COUNT",
            ]
        ]
        .sum()
        .rename(columns={"SIGHTING_DATE": "PERIOD_START"})
    )
    daily = calendar.merge(observed, on="PERIOD_START", how="left")
    daily = daily.merge(probabilistic, on="PERIOD_START", how="left")
    daily[COUNT_MEASURES] = daily[COUNT_MEASURES].fillna(0)
    daily[
        [
            "SIGHTING_COUNT",
            "SOURCE_REPORT_COUNT",
            "OBSERVED_SIGHTING_COUNT",
            "HARD_IMPUTED_COUNT",
        ]
    ] = daily[
        [
            "SIGHTING_COUNT",
            "SOURCE_REPORT_COUNT",
            "OBSERVED_SIGHTING_COUNT",
            "HARD_IMPUTED_COUNT",
        ]
    ].astype(
        "int32"
    )
    daily = _period_fields(daily, "daily")
    daily["PERIOD_STATUS"] = "COMPLETE"
    daily["IS_COMPLETE_PERIOD"] = True
    weekly_source = daily.copy()
    day = pd.to_datetime(weekly_source.PERIOD_START)
    weekly_source["PERIOD_START"] = (
        day - pd.to_timedelta(day.dt.weekday, unit="D")
    ).dt.date
    weekly = weekly_source.groupby("PERIOD_START", as_index=False)[COUNT_MEASURES].sum()
    weekly = _period_fields(weekly, "weekly")
    complete = weekly.PERIOD_START.map(
        lambda value: value >= start
    ) & weekly.PERIOD_END.map(lambda value: value <= end)
    final_week_start = end - timedelta(days=end.weekday())
    open_mask = weekly.PERIOD_START.eq(final_week_start) & ~complete
    weekly = weekly.loc[
        complete | (open_mask if mode is ProcessingMode.AS_OF else False)
    ].copy()
    weekly["PERIOD_STATUS"] = "COMPLETE"
    weekly["IS_COMPLETE_PERIOD"] = True
    weekly.loc[
        weekly.PERIOD_START.eq(final_week_start) & ~complete.loc[weekly.index],
        "PERIOD_STATUS",
    ] = "OPEN"
    weekly.loc[weekly.PERIOD_STATUS.eq("OPEN"), "IS_COMPLETE_PERIOD"] = False
    combined = pd.concat([daily, weekly], ignore_index=True)
    if not zero_fill_verified:
        combined["PERIOD_STATUS"] = UNVERIFIED_PERIOD_STATUS
        combined["IS_COMPLETE_PERIOD"] = False
    return combined


def _apply_coverage_status(
    frame: pd.DataFrame, *, zero_fill_verified: bool
) -> pd.DataFrame:
    """Prevent sparse observations from advertising complete period coverage."""

    if zero_fill_verified or frame.empty:
        return frame
    result = frame.copy()
    result["PERIOD_STATUS"] = UNVERIFIED_PERIOD_STATUS
    result["IS_COMPLETE_PERIOD"] = False
    return result


def _apply_reporting_contract(
    frame: pd.DataFrame,
    *,
    coverage_contract: dict[str, Any],
) -> pd.DataFrame:
    """Attach explicit cohort and missingness semantics to every count row."""

    result = _apply_coverage_status(
        frame,
        zero_fill_verified=bool(coverage_contract["zero_fill_verified"]),
    )
    result["TARGET_COHORT_ID"] = str(coverage_contract["target_cohort_id"])
    result["TARGET_COHORT_STATUS"] = str(coverage_contract["target_cohort_status"])
    reported = pd.Series(False, index=result.index)
    if "SIGHTING_COUNT" in result:
        reported = reported | pd.to_numeric(
            result["SIGHTING_COUNT"], errors="coerce"
        ).fillna(0).gt(0)
    if "EXPECTED_SIGHTING_COUNT" in result:
        reported = reported | pd.to_numeric(
            result["EXPECTED_SIGHTING_COUNT"], errors="coerce"
        ).fillna(0).gt(0)
    missing_state = (
        "NO_REPORT" if coverage_contract["zero_fill_verified"] else "UNAVAILABLE"
    )
    result["REPORTING_STATE"] = np.where(reported, "REPORTED", missing_state)
    return result


def _ordered(frame: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    for name in schema.names:
        if name not in frame:
            frame[name] = pd.NA
    return frame[schema.names]


def _write_dataset(frame: pd.DataFrame, root: Path, schema: pa.Schema) -> None:
    if frame.empty:
        root.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist([], schema=schema), root / "part-00000.parquet"
        )
        return
    frame = frame.copy()
    if "FREQUENCY" not in frame:
        root.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(
            _ordered(frame, schema), preserve_index=False
        ).cast(schema)
        pq.write_table(table, root / "part-00000.parquet", compression="zstd")
        return
    frame["_PARTITION_YEAR"] = frame.ISO_YEAR.fillna(frame.YEAR)
    for (frequency, year), group in frame.groupby(["FREQUENCY", "_PARTITION_YEAR"]):
        partition_name = "iso_year" if frequency == "weekly" else "year"
        path = (
            root
            / f"frequency={frequency}/{partition_name}={int(year)}/part-00000.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(
            _ordered(group.copy(), schema), preserve_index=False
        ).cast(schema)
        pq.write_table(table, path, compression="zstd")


def build_counts(request: CountRequest) -> StageResult:
    if request.observations is None:
        raise ValueError("Counts require an explicit normalized observations artifact")
    supported_inputs = {
        "whale.sightings.observations",
        "whale.sightings.imputed_retrospective",
        "whale.sightings.imputed_as_of",
    }
    if request.observations.dataset_id not in supported_inputs:
        raise ValueError(
            "Counts require normalized or pipeline-imputed observations; "
            f"found {request.observations.dataset_id!r}"
        )
    if request.mode is ProcessingMode.AS_OF:
        raise NotImplementedError(
            "AS_OF counts are disabled until source state is reconstructed at the knowledge cutoff"
        )
    document, config = load_sightings_config(request.config)
    from marine_mammal_toolkit.tools.quality.observations import (
        validate_sightings_artifact,
    )

    input_reports = [validate_sightings_artifact(request.observations)]
    if request.associations is not None:
        input_reports.append(validate_sightings_artifact(request.associations))
    for input_report in input_reports:
        input_report.require_valid()
    observations = pd.read_parquet(request.observations.path)
    if {
        "ECOTYPE_DETAIL_EFFECTIVE",
        "ECOTYPE_BUCKET_EFFECTIVE",
    } <= set(observations):
        observations["ECOTYPE_DETAIL"] = observations["ECOTYPE_DETAIL_EFFECTIVE"]
        observations["ECOTYPE_BUCKET"] = observations["ECOTYPE_BUCKET_EFFECTIVE"]
    associations = (
        pd.read_parquet(request.associations.path)
        if request.associations is not None
        else pd.DataFrame(
            columns=["OBSERVATION_ID", "ASSOCIATION_KIND", "ASSOCIATION_VALUE"]
        )
    )
    if observations.empty:
        if request.start_date is None or request.end_date is None:
            raise ValueError(
                "Empty observations require explicit start_date and end_date"
            )
        start, end = request.start_date, request.end_date
    else:
        start = request.start_date or observations.SIGHTING_DATE.min()
        end = request.end_date or observations.SIGHTING_DATE.max()
    if start > end:
        raise ValueError("Count start_date must be on or before end_date")
    coverage_contract = _coverage_contract(request.observations, start, end)
    signature_inputs = tuple(
        item
        for item in (
            request.observations,
            request.associations,
            *request.water_universes,
        )
        if item is not None
    )
    signature, signature_payload = stage_signature(
        stage="whale.sightings.counts",
        semantic_version="7",
        config_hash=document.config_hash,
        inputs=signature_inputs,
        parameters={
            "start_date": start,
            "end_date": end,
            "resolutions": request.resolutions,
            "mode": request.mode.value,
            "knowledge_cutoff": request.knowledge_cutoff,
        },
    )
    manifest_path = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/manifests/counts/{signature}.json"
    )
    resumed = resume_result(
        enabled=request.resume,
        manifest_path=manifest_path,
        config_hash=document.config_hash,
        inputs=signature_inputs,
        signature=signature,
    )
    if resumed is not None:
        return resumed
    window = (
        observations.SIGHTING_DATE.between(start, end)
        if not observations.empty
        else pd.Series([], dtype=bool)
    )
    observations = observations.loc[window].copy()
    all_bucket: list[pd.DataFrame] = []
    all_detail: list[pd.DataFrame] = []
    all_total: list[pd.DataFrame] = []
    all_pod: list[pd.DataFrame] = []
    all_exclusions: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {"coverage_contract": coverage_contract}
    used_universes: list[ArtifactRef] = []
    for resolution in request.resolutions:
        cells, universe_artifact = _universe(request.water_universes, resolution)
        used_universes.append(universe_artifact)
        bucket, detail, total, pod, excluded = _daily_counts(
            observations, associations, cells, resolution, policy=config.count_policy
        )
        all_exclusions.append(excluded)
        metrics[f"outside_counting_universe_r{resolution}"] = len(excluded)
        for daily, category, destination in (
            (bucket, "ECOTYPE_BUCKET", all_bucket),
            (detail, "ECOTYPE_DETAIL", all_detail),
            (total, None, all_total),
            (pod, "POD", all_pod),
        ):
            weekly, dropped = _weekly_from_daily(
                daily, category, start, end, request.mode
            )
            metrics[
                f"partial_week_rows_dropped_{category or 'total'}_r{resolution}"
            ] = dropped
            destination.extend([daily, weekly])
    products = {
        "whale.sightings.ecotype_counts": (
            pd.concat(all_bucket, ignore_index=True) if all_bucket else pd.DataFrame(),
            COUNT_SCHEMA,
        ),
        "whale.sightings.ecotype_detail_counts": (
            pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame(),
            DETAIL_COUNT_SCHEMA,
        ),
        "whale.sightings.orca_total_counts": (
            pd.concat(all_total, ignore_index=True) if all_total else pd.DataFrame(),
            TOTAL_COUNT_SCHEMA,
        ),
        "whale.sightings.pod_counts": (
            pd.concat(all_pod, ignore_index=True) if all_pod else pd.DataFrame(),
            POD_COUNT_SCHEMA,
        ),
        "whale.sightings.period_totals": (
            _period_totals(
                observations,
                start,
                end,
                request.mode,
                zero_fill_verified=bool(coverage_contract["zero_fill_verified"]),
                policy=config.count_policy,
            ),
            PERIOD_TOTAL_SCHEMA,
        ),
        "whale.sightings.count_exclusions": (
            (
                pd.concat(all_exclusions, ignore_index=True)
                if all_exclusions
                else pd.DataFrame()
            ),
            COUNT_EXCLUSION_SCHEMA,
        ),
    }
    products = {
        dataset_id: (
            (
                _apply_reporting_contract(frame, coverage_contract=coverage_contract),
                schema,
            )
            if dataset_id != "whale.sightings.count_exclusions"
            else (frame, schema)
        )
        for dataset_id, (frame, schema) in products.items()
    }
    report_measures = [
        "SIGHTING_COUNT",
        "SOURCE_REPORT_COUNT",
        "OBSERVED_SIGHTING_COUNT",
        "HARD_IMPUTED_COUNT",
        "EXPECTED_SIGHTING_COUNT",
        "MATURE_EXPECTED_SIGHTING_COUNT",
        "PROVISIONAL_EXPECTED_COUNT",
        "EXPECTED_UNKNOWN_COUNT",
    ]

    def totals(frame: pd.DataFrame) -> dict[str, float]:
        return {
            column: float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
            for column in report_measures
            if column in frame
        }

    prior_period_root = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/counts/mode={request.mode.value}/period_totals"
    )
    prior_files = sorted(prior_period_root.rglob("*.parquet"))
    prior_periods = (
        pd.concat((pd.read_parquet(path) for path in prior_files), ignore_index=True)
        if prior_files
        else pd.DataFrame()
    )

    def totals_by_frequency(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        if frame.empty or "FREQUENCY" not in frame:
            return {}
        return {
            str(frequency): totals(group)
            for frequency, group in frame.groupby("FREQUENCY", sort=True)
        }

    before_by_frequency = totals_by_frequency(prior_periods)
    after_by_frequency = totals_by_frequency(
        products["whale.sightings.period_totals"][0]
    )
    before_totals = before_by_frequency.get("daily", {})
    after_totals = after_by_frequency.get("daily", {})
    metrics["count_totals_reconciliation_frequency"] = "daily"
    metrics["count_totals_before_by_frequency"] = before_by_frequency
    metrics["count_totals_after_by_frequency"] = after_by_frequency
    metrics["count_totals_before"] = before_totals
    metrics["count_totals_after"] = after_totals
    metrics["count_total_deltas"] = {
        column: after_totals.get(column, 0.0) - before_totals.get(column, 0.0)
        for column in sorted(set(before_totals) | set(after_totals))
    }
    staging = request.data_root / ".staging" / request.run_id / "whale.sightings.counts"
    if staging.exists():
        shutil.rmtree(staging)
    destinations: list[tuple[str, Path, Path]] = []
    candidate_reports: list[ValidationReport] = []
    for dataset_id, (frame, schema) in products.items():
        name = dataset_id.rsplit(".", 1)[-1]
        root = staging / name
        _write_dataset(frame, root, schema)
        partitions = sorted(
            str(path.relative_to(root)) for path in root.rglob("*.parquet")
        )
        atomic_write_json(
            root / "_dataset_manifest.json",
            {
                "dataset_id": dataset_id,
                "schema_version": "7",
                "partitions": partitions,
                "row_count": len(frame),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "processing_mode": request.mode.value,
                "coverage_contract": coverage_contract,
                "universe_checksums": [
                    item.checksum or checksum_path(item.path) for item in used_universes
                ],
            },
        )
        destinations.append(
            (
                dataset_id,
                root,
                request.data_root
                / f"processed/domain/whale_layer/sightings/counts/mode={request.mode.value}/{name}",
            )
        )
    candidate_reports.extend(
        validate_sightings_artifact(
            ArtifactRef(
                kind="domain",
                dataset_id=dataset_id,
                path=root,
                producer="whale.sightings.counts.v7",
                schema_version="7",
                checksum=checksum_path(root),
            )
        )
        for dataset_id, root, _ in destinations
    )
    invalid = [report for report in candidate_reports if not report.valid]
    if invalid:
        quarantine = (
            request.data_root / "quarantine/whale.sightings.counts" / request.run_id
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        if quarantine.exists():
            shutil.rmtree(quarantine)
        os.replace(staging, quarantine)
        raise ValueError(
            "Count candidate validation failed: "
            + "; ".join(error for report in invalid for error in report.errors)
        )
    destination_root = (
        request.data_root
        / f"processed/domain/whale_layer/sightings/counts/mode={request.mode.value}"
    )
    if destination_root.exists() and not request.force:
        raise FileExistsError(f"Count products exist; pass --force: {destination_root}")
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    backup = staging.parent / "canonical-counts-backup"
    if backup.exists():
        shutil.rmtree(backup)
    try:
        if destination_root.exists():
            os.replace(destination_root, backup)
        os.replace(staging, destination_root)
    except OSError:
        if backup.exists() and not destination_root.exists():
            os.replace(backup, destination_root)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    shutil.rmtree(staging.parent, ignore_errors=True)
    inputs = tuple(
        item
        for item in (
            request.observations,
            request.associations,
            *request.water_universes,
        )
        if item is not None
    )
    outputs = tuple(
        ArtifactRef(
            kind="domain",
            dataset_id=dataset_id,
            path=destination,
            producer="whale.sightings.counts.v7",
            schema_version="7",
            run_id=request.run_id,
            config_hash=document.config_hash,
            checksum=checksum_path(destination),
            row_count=len(products[dataset_id][0]),
            file_count=sum(1 for _ in destination.rglob("*.parquet")),
            inputs=tuple(item.checksum or str(item.path) for item in inputs),
            processing_mode=request.mode.value,
            knowledge_cutoff=request.knowledge_cutoff
            or request.observations.knowledge_cutoff,
            data_snapshot=request.observations.data_snapshot,
            temporal_coverage=coverage_contract,
        )
        for dataset_id, _, destination in destinations
    )
    report = ValidationReport(True, "whale.sightings.counts", metrics=metrics)
    manifest = RunManifest(
        run_id=request.run_id,
        workflow="whale.sightings.counts.v7",
        config_hash=document.config_hash,
        resolved_config=document.redacted_data(),
        inputs=inputs,
        outputs=outputs,
        data_snapshot=request.observations.data_snapshot,
        schema_version="7",
        code_revision=code_revision(),
        stage_signature=signature,
        stages=(
            {
                "name": "counts",
                "semantic_version": "7",
                "signature": signature_payload,
                **metrics,
            },
        ),
    )
    manifest.write(manifest_path, overwrite=request.force)
    latest_pointer = (
        request.data_root
        / "processed/domain/whale_layer/sightings/manifests/counts/latest.json"
    )
    atomic_write_json(
        latest_pointer,
        {
            "manifest": manifest_path.relative_to(latest_pointer.parent).as_posix(),
            "stage_signature": signature,
        },
        overwrite=True,
    )
    return StageResult(outputs, (*input_reports, *candidate_reports, report), manifest)
