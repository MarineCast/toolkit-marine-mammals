from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
import pytest
from shapely.geometry import box

from marine_mammal_toolkit.tools._core.persistence import checksum_path
from marine_mammal_toolkit.tools.observations.post_process.aggregation import (
    _bounded_water_neighbors,
)
from marine_mammal_toolkit.tools.observations.post_process.aggregation import (
    _promote_activity_directories,
)
from marine_mammal_toolkit.tools.observations.post_process.aggregation import (
    _require_dense_coverage,
)
from marine_mammal_toolkit.tools.observations.post_process.counts import (
    _apply_coverage_status,
)
from marine_mammal_toolkit.tools.observations.post_process.counts import (
    _apply_reporting_contract,
)
from marine_mammal_toolkit.tools.observations.post_process.counts import _daily_counts
from marine_mammal_toolkit.tools.observations.post_process.counts import (
    _expected_contributions,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    cells_in_domain,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    load_domain_geometry,
)
from marine_mammal_toolkit.tools.observations.post_process.spatial import (
    load_domain_provenance,
)
from marine_mammal_toolkit.tools.quality.observations import (
    _validate_weekly_reconciliation,
)


def test_hard_counts_use_effective_imputed_ecotype():
    latitude, longitude = 48.5, -123.2
    cell = h3.latlng_to_cell(latitude, longitude, 4)
    observations = pd.DataFrame(
        {
            "OBSERVATION_ID": ["unknown-1"],
            "SIGHTING_DATE": [date(2025, 1, 1)],
            "LATITUDE": [latitude],
            "LONGITUDE": [longitude],
            "SOURCE_REPORT_COUNT": [1],
            "ECOTYPE_DETAIL": ["UNKNOWN"],
            "ECOTYPE_BUCKET": ["OTHER"],
            "ECOTYPE_DETAIL_EFFECTIVE": ["SRKW"],
            "ECOTYPE_BUCKET_EFFECTIVE": ["SRKW"],
            "IMPUTATION_APPLIED": [True],
            "EXPECTED_SRKW_COUNT": [0.9],
            "EXPECTED_TRANSIENT_COUNT": [0.1],
            "EXPECTED_UNKNOWN_COUNT": [0.0],
            "IMPUTATION_CONTEXT_STATUS": ["MATURE_RETROSPECTIVE"],
        }
    )
    associations = pd.DataFrame(
        columns=["OBSERVATION_ID", "ASSOCIATION_KIND", "ASSOCIATION_VALUE"]
    )
    bucket, _detail, _total, _pod, _excluded = _daily_counts(
        observations, associations, {cell}, 4, policy=COUNT_POLICY
    )
    hard_srkw = bucket.loc[bucket.ECOTYPE_BUCKET.eq("SRKW"), "SIGHTING_COUNT"].sum()
    assert hard_srkw == 1


def test_water_neighbor_traversal_does_not_jump_disconnected_components():
    adjacency = {
        "a": (("b", 2.0), ("c", 10.0)),
        "b": (("a", 2.0), ("c", 2.0)),
        "c": (("a", 10.0), ("b", 2.0), ("d", 1.0)),
        "d": (("c", 1.0),),
        "disconnected": (),
    }
    assert _bounded_water_neighbors("a", adjacency, 4.0) == {
        "a": 0.0,
        "b": 2.0,
        "c": 4.0,
    }


def test_dense_grid_refuses_unverified_zero_fill():
    with pytest.raises(ValueError, match="Refusing to materialize"):
        _require_dense_coverage({"zero_fill_verified": False})


def test_sparse_counts_do_not_claim_complete_periods_when_coverage_is_unverified():
    frame = pd.DataFrame(
        {
            "PERIOD_STATUS": ["COMPLETE"],
            "IS_COMPLETE_PERIOD": [True],
            "SIGHTING_COUNT": [1],
        }
    )
    result = _apply_coverage_status(frame, zero_fill_verified=False)
    assert result["PERIOD_STATUS"].tolist() == ["UNVERIFIED_COVERAGE"]
    assert result["IS_COMPLETE_PERIOD"].tolist() == [False]


def test_reporting_state_never_turns_unverified_missingness_into_no_report():
    frame = pd.DataFrame(
        {
            "SIGHTING_COUNT": [1, 0],
            "EXPECTED_SIGHTING_COUNT": [1.0, 0.0],
            "PERIOD_STATUS": ["COMPLETE", "COMPLETE"],
            "IS_COMPLETE_PERIOD": [True, True],
        }
    )
    result = _apply_reporting_contract(
        frame,
        coverage_contract={
            "zero_fill_verified": False,
            "target_cohort_id": "cohort-fixture",
            "target_cohort_status": "UNVERIFIED",
        },
    )
    assert result["REPORTING_STATE"].tolist() == ["REPORTED", "UNAVAILABLE"]
    assert result["PERIOD_STATUS"].eq("UNVERIFIED_COVERAGE").all()
    assert not result["IS_COMPLETE_PERIOD"].any()

    verified = _apply_reporting_contract(
        frame.iloc[[1]],
        coverage_contract={
            "zero_fill_verified": True,
            "target_cohort_id": "cohort-verified",
            "target_cohort_status": "VERIFIED_COMPLETE",
        },
    )
    assert verified["REPORTING_STATE"].item() == "NO_REPORT"


def test_unverified_sparse_weekly_counts_still_reconcile_to_daily_facts():
    measures = {
        "SIGHTING_COUNT": [1, 2, 3],
        "SOURCE_REPORT_COUNT": [1, 2, 3],
        "OBSERVED_SIGHTING_COUNT": [1, 2, 3],
        "HARD_IMPUTED_COUNT": [0, 0, 0],
        "EXPECTED_SIGHTING_COUNT": [1.0, 2.0, 3.0],
        "MATURE_EXPECTED_SIGHTING_COUNT": [1.0, 2.0, 3.0],
        "PROVISIONAL_EXPECTED_COUNT": [0.0, 0.0, 0.0],
        "EXPECTED_UNKNOWN_COUNT": [0.0, 0.0, 0.0],
    }
    frame = pd.DataFrame(
        {
            "H3_INDEX": ["cell", "cell", "cell"],
            "H3_RESOLUTION": [4, 4, 4],
            "FREQUENCY": ["daily", "daily", "weekly"],
            "PERIOD_START": [date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 17)],
            "ECOTYPE_BUCKET": ["SRKW", "SRKW", "SRKW"],
            "IS_COMPLETE_PERIOD": [False, False, False],
            **measures,
        }
    )
    errors: list[str] = []

    _validate_weekly_reconciliation(frame, errors, date(2026, 8, 17), date(2026, 8, 23))

    assert errors == []


def test_activity_alias_promotion_rolls_back_both_destinations(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    transaction_root = tmp_path / "transaction"
    canonical_candidate = transaction_root / "relative_reported_activity"
    alias_candidate = transaction_root / "relative_intensity"
    canonical_candidate.mkdir(parents=True)
    alias_candidate.mkdir(parents=True)
    (canonical_candidate / "value.txt").write_text("new-canonical")
    (alias_candidate / "value.txt").write_text("new-alias")
    destination_root = tmp_path / "published"
    canonical_destination = destination_root / "relative_reported_activity"
    alias_destination = destination_root / "relative_intensity"
    canonical_destination.mkdir(parents=True)
    alias_destination.mkdir(parents=True)
    (canonical_destination / "value.txt").write_text("old-canonical")
    (alias_destination / "value.txt").write_text("old-alias")

    from marine_mammal_toolkit.tools.observations.post_process import aggregation

    real_replace = aggregation.os.replace

    def fail_alias_candidate(source, destination):
        if Path(source) == alias_candidate and Path(destination) == alias_destination:
            raise OSError("fault injection")
        return real_replace(source, destination)

    monkeypatch.setattr(aggregation.os, "replace", fail_alias_candidate)
    with pytest.raises(OSError, match="fault injection"):
        _promote_activity_directories(
            (
                (canonical_candidate, canonical_destination),
                (alias_candidate, alias_destination),
            ),
            transaction_root=transaction_root,
            force=True,
        )
    assert (canonical_destination / "value.txt").read_text() == "old-canonical"
    assert (alias_destination / "value.txt").read_text() == "old-alias"


@pytest.mark.parametrize(
    "weights",
    [
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
        (1.1, -0.1, 0.0),
        (0.4, 0.4, 0.0),
    ],
)
def test_expected_mass_rejects_invalid_or_partial_rows(weights):
    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": ["bad-mass"],
            "ECOTYPE_DETAIL": ["UNKNOWN"],
            "EXPECTED_SRKW_COUNT": [weights[0]],
            "EXPECTED_TRANSIENT_COUNT": [weights[1]],
            "EXPECTED_UNKNOWN_COUNT": [weights[2]],
        }
    )
    with pytest.raises(ValueError, match="Expected class mass"):
        _expected_contributions(frame, policy=COUNT_POLICY)


def test_explicit_non_target_zero_mass_is_retained_as_one_report():
    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": ["non-target"],
            "ECOTYPE_DETAIL": ["NRKW"],
            "ECOTYPE_BUCKET": ["OTHER"],
            "USE_FOR_PROBABILISTIC_COUNTS": [False],
            "EXPECTED_SRKW_COUNT": [0.0],
            "EXPECTED_TRANSIENT_COUNT": [0.0],
            "EXPECTED_OTHER_COUNT": [1.0],
            "EXPECTED_UNKNOWN_COUNT": [0.0],
        }
    )
    expanded = _expected_contributions(frame, policy=COUNT_POLICY)
    assert expanded["EXPECTED_SIGHTING_COUNT"].sum() == 1.0
    assert expanded["ECOTYPE_DETAIL"].tolist() == ["NRKW"]


def test_four_way_expected_mass_preserves_fractional_other_contribution():
    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": ["mixed-mass"],
            "ECOTYPE_DETAIL": ["MIXED"],
            "ECOTYPE_BUCKET": ["OTHER"],
            "EXPECTED_SRKW_COUNT": [0.25],
            "EXPECTED_TRANSIENT_COUNT": [0.25],
            "EXPECTED_OTHER_COUNT": [0.25],
            "EXPECTED_UNKNOWN_COUNT": [0.25],
        }
    )
    expanded = _expected_contributions(frame, policy=COUNT_POLICY)
    assert expanded["EXPECTED_SIGHTING_COUNT"].sum() == pytest.approx(1.0)
    assert expanded.loc[
        expanded["ECOTYPE_DETAIL"].eq("MIXED"), "EXPECTED_SIGHTING_COUNT"
    ].item() == pytest.approx(0.25)


def test_polygon_domain_membership_is_explicit(tmp_path):
    path = tmp_path / "domain.parquet"
    gpd.GeoDataFrame(
        {"domain": ["test"]},
        geometry=[box(-123.4, 48.3, -123.0, 48.7)],
        crs="EPSG:4326",
    ).to_parquet(path)
    geometry = load_domain_geometry(path)
    inside = h3.latlng_to_cell(48.5, -123.2, 6)
    outside = h3.latlng_to_cell(47.0, -125.0, 6)
    assert cells_in_domain({inside, outside}, geometry) == {inside}


def test_polygon_provenance_must_match_geometry_checksum(tmp_path):
    polygon = tmp_path / "domain.parquet"
    provenance = tmp_path / "domain.metadata.json"
    gpd.GeoDataFrame(
        {"domain": ["test"]},
        geometry=[box(-123.4, 48.3, -123.0, 48.7)],
        crs="EPSG:4326",
    ).to_parquet(polygon)
    payload = {
        "schema_version": 1,
        "ecotype": "SRKW",
        "source_authority": "Test authority",
        "source_url": "https://example.test/source",
        "source_release": "test-v1",
        "retrieved_at": "2026-08-01T00:00:00Z",
        "derivation": "Test-only polygon.",
        "geometry_sha256": checksum_path(polygon),
        "review_status": "approved_for_model_domain",
    }
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        load_domain_provenance(provenance, polygon_path=polygon, ecotype="SRKW")
        == payload
    )

    payload["geometry_sha256"] = "incorrect"
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_domain_provenance(provenance, polygon_path=polygon, ecotype="SRKW")


from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    COUNT_POLICY,
)
