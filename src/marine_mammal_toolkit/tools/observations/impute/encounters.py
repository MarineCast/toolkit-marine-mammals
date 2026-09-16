"""Deterministic encounter clustering for canonical sighting observations."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088
LINK_ASSOCIATION_KINDS = {"MEMBER", "SOCIAL_GROUP", "POD"}
TRUSTED_ASSOCIATION_CONFIDENCE = {"EXPLICIT", "STRONG"}


def _association_keys(
    associations: pd.DataFrame | None,
) -> dict[str, frozenset[tuple[str, str]]]:
    if associations is None or associations.empty:
        return {}
    required = {
        "OBSERVATION_ID",
        "ASSOCIATION_KIND",
        "ASSOCIATION_VALUE",
        "CONFIDENCE",
        "CONFLICTING",
    }
    if not required <= set(associations):
        return {}
    frame = associations.loc[
        associations["ASSOCIATION_KIND"]
        .astype(str)
        .str.upper()
        .isin(LINK_ASSOCIATION_KINDS)
    ].copy()
    frame = frame.loc[~frame["CONFLICTING"].fillna(True).astype(bool)]
    frame = frame.loc[
        frame["CONFIDENCE"]
        .fillna("")
        .astype(str)
        .str.upper()
        .isin(TRUSTED_ASSOCIATION_CONFIDENCE)
    ]
    if frame.empty:
        return {}
    frame["OBSERVATION_ID"] = frame["OBSERVATION_ID"].astype(str)
    frame["ASSOCIATION_KIND"] = frame["ASSOCIATION_KIND"].astype(str).str.upper()
    frame["ASSOCIATION_VALUE"] = frame["ASSOCIATION_VALUE"].astype(str).str.upper()
    return {
        observation_id: frozenset(
            zip(group["ASSOCIATION_KIND"], group["ASSOCIATION_VALUE"], strict=True)
        )
        for observation_id, group in frame.groupby("OBSERVATION_ID", sort=False)
    }


def attach_encounter_ids(
    observations: pd.DataFrame,
    associations: pd.DataFrame | None = None,
    *,
    radius_km: float = 3.0,
    association_radius_km: float = 20.0,
    timestamp_tolerance_hours: float = 6.0,
) -> pd.DataFrame:
    """Attach stable-in-run encounter IDs without changing observation identity.

    Same-date observations are clustered when every pair in an encounter falls
    within ``radius_km``. A shared, non-conflicting, trusted member,
    social-group, or pod association permits that pair to use the wider
    ``association_radius_km``. When both observations have precise source
    timestamps, they must also fall within ``timestamp_tolerance_hours``.

    Clusters use deterministic complete-link compatibility rather than
    transitive single-link components. This bounds every encounter's diameter
    and prevents a chain of date-only sightings from joining distant endpoints.
    """

    if radius_km <= 0 or association_radius_km < radius_km:
        raise ValueError(
            "Encounter radii must satisfy 0 < radius <= association radius"
        )
    if timestamp_tolerance_hours <= 0:
        raise ValueError("timestamp_tolerance_hours must be positive")
    if observations.empty:
        return observations.assign(
            ENCOUNTER_ID=pd.Series(dtype=str), ENCOUNTER_SIZE=pd.Series(dtype=int)
        )
    required = {"OBSERVATION_ID", "SIGHTING_DATE", "LATITUDE", "LONGITUDE"}
    missing = sorted(required - set(observations))
    if missing:
        raise ValueError(f"Encounter clustering is missing columns: {missing}")

    out = observations.copy().reset_index(names="_ORIGINAL_INDEX")
    out["OBSERVATION_ID"] = out["OBSERVATION_ID"].astype(str)
    out["SIGHTING_DATE"] = pd.to_datetime(
        out["SIGHTING_DATE"], format="mixed"
    ).dt.normalize()
    association_keys = _association_keys(associations)
    event_time = (
        pd.to_datetime(out["SOURCE_EVENT_AT_UTC"], utc=True, errors="coerce")
        if "SOURCE_EVENT_AT_UTC" in out
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )
    precise = (
        out.get("SOURCE_TIME_PRECISION", pd.Series("DATE", index=out.index))
        .astype(str)
        .str.upper()
        .eq("TIMESTAMP")
        & event_time.notna()
    )
    max_radius = association_radius_km / EARTH_RADIUS_KM
    members: list[list[int]] = []

    for _, positions_series in out.groupby("SIGHTING_DATE", sort=False).groups.items():
        positions = np.asarray(list(positions_series), dtype=int)
        positions = np.asarray(
            sorted(positions, key=lambda item: (out.at[item, "OBSERVATION_ID"], item)),
            dtype=int,
        )
        if len(positions) == 1:
            members.append([int(positions[0])])
            continue
        coordinates = np.deg2rad(
            out.loc[positions, ["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)
        )
        tree = BallTree(coordinates, metric="haversine")
        neighbors, distances = tree.query_radius(
            coordinates, r=max_radius, return_distance=True, sort_results=False
        )
        compatible = np.eye(len(positions), dtype=bool)
        for local_left, (local_neighbors, local_distances) in enumerate(
            zip(neighbors, distances, strict=True)
        ):
            left = int(positions[local_left])
            left_keys = association_keys.get(
                out.at[left, "OBSERVATION_ID"], frozenset()
            )
            for local_right, distance_rad in zip(
                local_neighbors, local_distances, strict=True
            ):
                if int(local_right) <= local_left:
                    continue
                right = int(positions[int(local_right)])
                distance_km = float(distance_rad) * EARTH_RADIUS_KM
                right_keys = association_keys.get(
                    out.at[right, "OBSERVATION_ID"], frozenset()
                )
                shared_association = bool(left_keys & right_keys)
                if distance_km > radius_km and not shared_association:
                    continue
                if precise.iat[left] and precise.iat[right]:
                    hours = (
                        abs(
                            (
                                event_time.iat[left] - event_time.iat[right]
                            ).total_seconds()
                        )
                        / 3600
                    )
                    if hours > timestamp_tolerance_hours:
                        continue
                compatible[local_left, int(local_right)] = True
                compatible[int(local_right), local_left] = True

        day_clusters: list[list[int]] = []
        local_by_position = {
            int(position): local for local, position in enumerate(positions)
        }
        for position in positions:
            local_position = local_by_position[int(position)]
            candidates = [
                cluster
                for cluster in day_clusters
                if all(
                    compatible[local_position, local_by_position[member]]
                    for member in cluster
                )
            ]
            if candidates:
                selected = min(
                    candidates,
                    key=lambda cluster: (
                        -len(cluster),
                        tuple(out.at[member, "OBSERVATION_ID"] for member in cluster),
                    ),
                )
                selected.append(int(position))
            else:
                day_clusters.append([int(position)])
        members.extend(day_clusters)

    encounter_id = np.empty(len(out), dtype=object)
    encounter_size = np.zeros(len(out), dtype=int)
    for group_positions in members:
        identifiers = sorted(out.loc[group_positions, "OBSERVATION_ID"].astype(str))
        digest = hashlib.sha1("|".join(identifiers).encode("utf-8")).hexdigest()[:20]
        value = f"orca:encounter:v2:{digest}"
        encounter_id[group_positions] = value
        encounter_size[group_positions] = len(group_positions)
    out["ENCOUNTER_ID"] = encounter_id
    out["ENCOUNTER_SIZE"] = encounter_size
    return (
        out.sort_values("_ORIGINAL_INDEX")
        .drop(columns="_ORIGINAL_INDEX")
        .set_axis(observations.index)
    )
