from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_polars: Any
try:
    import polars as _polars_module

    _polars = _polars_module
except ImportError:  # Optional acceleration; pandas remains supported.
    _polars = None

pl: Any = _polars


REQUIRED_OBSERVATION_COLUMNS = {
    "OBSERVATION_ID",
    "SIGHTING_DATE",
    "LATITUDE",
    "LONGITUDE",
    "ECOTYPE_DETAIL",
}


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        if p.is_dir() or p.suffix.lower() in {".parquet", ".pq"}:
            if pl is not None:
                sources = (
                    [str(item) for item in sorted(p.rglob("*.parquet"))]
                    if p.is_dir()
                    else [str(p)]
                )
                if not sources:
                    raise ValueError(f"No Parquet files found under {p}")
                try:
                    return pl.scan_parquet(sources).collect().to_pandas()
                except Exception:
                    # Preserve compatibility with uncommon Arrow extension types.
                    pass
            return pd.read_parquet(p)
        if p.suffix.lower() in {".csv", ".txt"}:
            return pd.read_csv(p)
        if p.suffix.lower() in {".feather", ".arrow"}:
            return pd.read_feather(p)
    except ImportError as exc:
        raise ImportError(
            "Reading Parquet/Arrow requires a notebook kernel with pyarrow available."
        ) from exc
    raise ValueError(f"Unsupported table path: {p}")


def load_preprocessed_sightings(
    observations_path: str | Path,
    associations_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load and validate the pipeline's canonical preprocessed sightings tables."""

    observations = _read_table(observations_path)
    missing = sorted(REQUIRED_OBSERVATION_COLUMNS - set(observations.columns))
    if missing:
        raise ValueError(f"Observations table is missing required columns: {missing}")

    observations = observations.copy()
    observations["OBSERVATION_ID"] = observations["OBSERVATION_ID"].astype(str)
    if observations["OBSERVATION_ID"].duplicated().any():
        examples = observations.loc[
            observations["OBSERVATION_ID"].duplicated(False), "OBSERVATION_ID"
        ].head(10)
        raise ValueError(
            "OBSERVATION_ID must be unique. Duplicate examples: "
            + ", ".join(examples.astype(str))
        )

    observations["SIGHTING_DATE"] = pd.to_datetime(
        observations["SIGHTING_DATE"], errors="coerce", format="mixed"
    ).dt.normalize()
    if observations["SIGHTING_DATE"].isna().any():
        raise ValueError("SIGHTING_DATE contains unparseable values")

    observations["LATITUDE"] = pd.to_numeric(observations["LATITUDE"], errors="coerce")
    observations["LONGITUDE"] = pd.to_numeric(
        observations["LONGITUDE"], errors="coerce"
    )
    invalid_coord = (
        observations["LATITUDE"].isna()
        | observations["LONGITUDE"].isna()
        | ~observations["LATITUDE"].between(-90, 90)
        | ~observations["LONGITUDE"].between(-180, 180)
    )
    if invalid_coord.any():
        raise ValueError(
            f"{int(invalid_coord.sum())} observations have invalid coordinates"
        )

    observations["ECOTYPE_DETAIL"] = (
        observations["ECOTYPE_DETAIL"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    if "SOURCE_REPORT_COUNT" not in observations:
        observations["SOURCE_REPORT_COUNT"] = 1
    observations["SOURCE_REPORT_COUNT"] = pd.to_numeric(
        observations["SOURCE_REPORT_COUNT"], errors="coerce"
    ).fillna(1)
    observations["SOURCE_REPORT_COUNT"] = observations["SOURCE_REPORT_COUNT"].clip(
        lower=1
    )

    if "SOURCE_TIME_PRECISION" not in observations:
        observations["SOURCE_TIME_PRECISION"] = "DATE"
    observations["SOURCE_TIME_PRECISION"] = (
        observations["SOURCE_TIME_PRECISION"].fillna("DATE").astype(str).str.upper()
    )

    associations = None
    if associations_path is not None:
        associations = _read_table(associations_path).copy()
        required_assoc = {
            "OBSERVATION_ID",
            "ASSOCIATION_KIND",
            "ASSOCIATION_VALUE",
            "CONFIDENCE",
            "CONFLICTING",
        }
        missing_assoc = sorted(required_assoc - set(associations.columns))
        if missing_assoc:
            raise ValueError(f"Associations table is missing columns: {missing_assoc}")
        associations["OBSERVATION_ID"] = associations["OBSERVATION_ID"].astype(str)
        associations["ASSOCIATION_KIND"] = (
            associations["ASSOCIATION_KIND"].astype(str).str.upper()
        )
        associations["ASSOCIATION_VALUE"] = (
            associations["ASSOCIATION_VALUE"].astype(str).str.upper()
        )
        associations["CONFIDENCE"] = associations["CONFIDENCE"].astype(str).str.upper()
        associations["CONFLICTING"] = (
            associations["CONFLICTING"].fillna(False).astype(bool)
        )

    return observations, associations


def add_anchor_quality(
    observations: pd.DataFrame,
    associations: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add a conservative observation-level anchor weight.

    Associations are not used as deterministic assignments. They only control
    how much a known label contributes as training/context evidence.
    """

    out = observations.copy()
    out["ANCHOR_WEIGHT"] = 1.0
    out["LABEL_CONFLICT"] = out["ECOTYPE_DETAIL"].eq("MIXED")
    out["LABEL_QUALITY"] = "OBSERVED"
    low_quality = (
        out.get("OBSERVATION_QUALITY_TIER", pd.Series("", index=out.index))
        .astype(str)
        .eq("LOW_QUALITY_NON_ANCHOR")
    )

    def apply_observation_quality(frame: pd.DataFrame) -> pd.DataFrame:
        frame.loc[low_quality, "ANCHOR_WEIGHT"] = 0.0
        frame.loc[low_quality, "LABEL_QUALITY"] = "LOW_QUALITY_NON_ANCHOR"
        frame.loc[frame["LABEL_CONFLICT"], "ANCHOR_WEIGHT"] = 0.0
        frame.loc[frame["LABEL_CONFLICT"], "LABEL_QUALITY"] = "CONFLICT"
        return frame

    if associations is None or associations.empty:
        return apply_observation_quality(out)

    assoc = associations.loc[associations["ASSOCIATION_KIND"].eq("ECOTYPE")].copy()
    if assoc.empty:
        return apply_observation_quality(out)

    field = (
        assoc["EVIDENCE_FIELD"].astype(str).str.upper()
        if "EVIDENCE_FIELD" in assoc
        else pd.Series("", index=assoc.index)
    )
    confidence = assoc["CONFIDENCE"].astype(str).str.upper()
    weights = np.select(
        [
            confidence.eq("STRONG"),
            confidence.eq("EXPLICIT") & field.eq("POD_ECOTYPE_RAW"),
            confidence.eq("EXPLICIT"),
        ],
        [1.0, 1.0, 0.75],
        default=0.60,
    ).astype(float)
    weights[assoc["CONFLICTING"].to_numpy(dtype=bool)] = 0.0
    assoc["_WEIGHT"] = weights

    detail = out.set_index("OBSERVATION_ID")["ECOTYPE_DETAIL"]
    assoc["_EXPECTED"] = assoc["OBSERVATION_ID"].map(detail)
    assoc = assoc.loc[assoc["ASSOCIATION_VALUE"].eq(assoc["_EXPECTED"])]
    best = assoc.groupby("OBSERVATION_ID")["_WEIGHT"].max()
    out["ANCHOR_WEIGHT"] = out["OBSERVATION_ID"].map(best).fillna(out["ANCHOR_WEIGHT"])
    association_conflict = (
        out["OBSERVATION_ID"].map(best.eq(0)).fillna(False).astype(bool)
    )

    out.loc[out["ANCHOR_WEIGHT"].ge(0.99), "LABEL_QUALITY"] = "GOLD"
    out.loc[
        out["ANCHOR_WEIGHT"].between(0.60, 0.99, inclusive="left"), "LABEL_QUALITY"
    ] = "SILVER"
    out.loc[out["ANCHOR_WEIGHT"].le(0), "LABEL_QUALITY"] = "CONFLICT"
    out["LABEL_CONFLICT"] = (
        out["LABEL_CONFLICT"]
        | (out["ANCHOR_WEIGHT"].le(0) & ~low_quality)
        | association_conflict
    )
    return apply_observation_quality(out)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def write_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        if pl is not None:
            try:
                pl.from_pandas(frame, include_index=False).write_csv(p)
                return p
            except Exception:
                pass
        frame.to_csv(p, index=False)
    else:
        try:
            if pl is not None:
                try:
                    pl.from_pandas(frame, include_index=False).write_parquet(
                        p, compression="zstd"
                    )
                    return p
                except Exception:
                    pass
            frame.to_parquet(p, index=False)
        except ImportError as exc:
            raise ImportError("Writing Parquet requires pyarrow") from exc
    return p
