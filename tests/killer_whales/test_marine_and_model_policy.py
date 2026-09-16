from __future__ import annotations

from types import SimpleNamespace

import h3
import numpy as np
import pandas as pd

from seascape.spatial_support.water_network import WaterGraph
from marine_mammal_toolkit.tools.observations.impute import marine
from marine_mammal_toolkit.tools.observations.impute.model import (
    certify_hard_label_classes,
)


def _small_graph() -> tuple[WaterGraph, str, str, str]:
    left = h3.latlng_to_cell(48.5, -123.2, 6)
    right = next(cell for cell in h3.grid_disk(left, 1) if cell != left)
    disconnected = h3.latlng_to_cell(47.0, -125.0, 6)
    support = pd.DataFrame(
        {
            "H3_INDEX": [left, right, disconnected],
            "WATER_MASK_VERSION": ["test"] * 3,
            "SPATIAL_SUPPORT_VERSION": ["test"] * 3,
            "CONNECTOR_TARGET_H3_INDEX": [None, None, None],
            "CONNECTOR_DISTANCE_M": [np.nan, np.nan, np.nan],
        }
    )
    graph = WaterGraph(
        resolution=6,
        cells=np.asarray([left, right], dtype=object),
        offsets=np.asarray([0, 1, 2], dtype=np.int64),
        neighbors=np.asarray([1, 0], dtype=np.int64),
        weights_m=np.asarray([1200.0, 1200.0]),
        support=support,
        cell_to_position={left: 0, right: 1},
        water_mask_version="test",
        spatial_support_version="test",
    )
    return graph, left, right, disconnected


def test_marine_lookup_uses_passable_edges_and_preserves_disconnection(
    monkeypatch, tmp_path
):
    graph, left, right, disconnected = _small_graph()
    monkeypatch.setattr(marine, "load_water_graph", lambda *_args, **_kwargs: graph)
    lookup = marine.MarineDistanceLookup(
        tmp_path / "seascape.yaml", resolution=6, required_radius_km=40
    )
    distances, keep, fallback = lookup.resolve(
        left,
        np.asarray([right, disconnected, "outside"], dtype=object),
        np.asarray([1.0, 30.0, 4.0]),
        fallback_to_haversine=True,
    )
    assert distances[0] == 1.2
    assert keep.tolist() == [True, False, True]
    assert fallback.tolist() == [False, False, True]


def test_hard_label_certification_uses_only_purged_results():
    encounter = SimpleNamespace(
        metrics={
            "ENCOUNTER_LEVEL": {
                "PREDICTED_CLASS_RISK": {
                    label: {"accepted_n": 100, "errors": 0, "error_upper": 0.01}
                    for label in ("TRANSIENT", "SRKW")
                }
            }
        }
    )
    purged = SimpleNamespace(
        metrics={
            "ENCOUNTER_LEVEL": {
                "PREDICTED_CLASS_RISK": {
                    label: {"accepted_n": 0, "errors": 0, "error_upper": 1.0}
                    for label in ("TRANSIENT", "SRKW")
                }
            }
        }
    )
    certification = certify_hard_label_classes(
        {"encounter": encounter, "purged_blocked": purged},
        strategy="purged_blocked",
        target_selective_error=0.05,
    )
    assert not certification["SRKW"]["CERTIFIED"]
    assert not certification["TRANSIENT"]["CERTIFIED"]
    assert certification["SRKW"]["STRATEGY"] == "purged_blocked"


def test_hard_label_certification_requires_release_and_stratum_gates():
    purged = SimpleNamespace(
        metrics={
            "ENCOUNTER_LEVEL": {
                "PREDICTED_CLASS_RISK": {
                    label: {"accepted_n": 100, "errors": 0, "error_upper": 0.01}
                    for label in ("TRANSIENT", "SRKW")
                }
            },
            "HARD_LABEL_PROMOTION_GATE": {
                "ZERO_LINEAGE_LEAKAGE": True,
                "REQUIRED_STRATA_PASSED": True,
            },
        }
    )
    disabled = certify_hard_label_classes(
        {"purged_blocked": purged},
        strategy="purged_blocked",
        target_selective_error=0.05,
    )
    assert not disabled["SRKW"]["CERTIFIED"]

    enabled = certify_hard_label_classes(
        {"purged_blocked": purged},
        strategy="purged_blocked",
        target_selective_error=0.05,
        minimum_accepted_per_class=100,
        release_enabled=True,
    )
    assert enabled["SRKW"]["CERTIFIED"]
    assert enabled["TRANSIENT"]["CERTIFIED"]
