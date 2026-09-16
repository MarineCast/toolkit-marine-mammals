from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from marine_mammal_toolkit.tools._core.config.paths import project_root
from marine_mammal_toolkit.tools.observations.process.adapters import _source_record
from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.tools.observations.impute.io import add_anchor_quality
from marine_mammal_toolkit.tools.observations.process.pipeline import _cluster
from marine_mammal_toolkit.tools.observations.process.pipeline import _materialize
from marine_mammal_toolkit.tools.observations.process.pipeline import _normalize_records


def _record(
    source: str,
    identifier: str,
    *,
    latitude: float,
    longitude: float,
    uncertainty_m: float | None,
    quality_grade: str | None = None,
) -> dict:
    retrieved = pd.Timestamp("2026-08-19T12:00:00Z")
    row = _source_record(
        source,
        identifier,
        {"id": identifier},
        observed_date="2026-08-18",
        latitude=latitude,
        longitude=longitude,
        species="Orcinus orca",
        pod_ecotype="SRKW",
        source_license="UNKNOWN",
        source_use_class="INTERNAL_ONLY",
        coordinate_uncertainty_m=uncertainty_m,
        source_qc_status="ACCEPTED",
        source_qc_detail=(
            json.dumps({"quality_grade": quality_grade}) if quality_grade else None
        ),
    )
    row.update(
        {
            "SOURCE_RETRIEVED_AT_UTC": retrieved,
            "SOURCE_PAYLOAD_CORRECTED": False,
            "LAST_CORRECTED_AT_UTC": retrieved,
        }
    )
    return row


def test_normalization_selects_best_coordinate_and_persists_provenance() -> None:
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    source = pd.DataFrame(
        [
            _record(
                "TWM",
                "coarse",
                latitude=48.0,
                longitude=-123.0,
                uncertainty_m=1_000,
            ),
            _record(
                "INATURALIST",
                "precise",
                latitude=48.0002,
                longitude=-123.0002,
                uncertainty_m=25,
                quality_grade="research",
            ),
        ]
    )
    records, audit = _normalize_records(source, config)
    observations, _associations = _materialize(
        [records], audit, ["observation-1"], policy=OBSERVATION_POLICY
    )

    observation = observations[0]
    assert observation["LATITUDE"] == 48.0002
    assert observation["LONGITUDE"] == -123.0002
    assert observation["COORDINATE_UNCERTAINTY_M"] == 25
    assert observation["COORDINATE_SELECTION_METHOD"] == "MINIMUM_REPORTED_UNCERTAINTY"
    assert json.loads(observation["FIELD_PROVENANCE"])[
        "coordinate_source_record_id"
    ].endswith(":precise")
    licenses = json.loads(observation["CONTRIBUTOR_LICENSE_SUMMARY"])
    assert {item["use_class"] for item in licenses} == {"INTERNAL_ONLY"}
    assert observation["PUBLIC_RELEASE_ELIGIBLE"] is False


def test_low_quality_observation_is_retained_but_not_an_anchor() -> None:
    observations = pd.DataFrame(
        {
            "OBSERVATION_ID": ["low-quality"],
            "ECOTYPE_DETAIL": ["SRKW"],
            "OBSERVATION_QUALITY_TIER": ["LOW_QUALITY_NON_ANCHOR"],
        }
    )
    result = add_anchor_quality(observations, associations=None)

    assert result.loc[0, "ANCHOR_WEIGHT"] == 0
    assert result.loc[0, "LABEL_QUALITY"] == "LOW_QUALITY_NON_ANCHOR"
    assert not bool(result.loc[0, "LABEL_CONFLICT"])


def test_normalization_chain_match_is_retained_for_review_not_transitively_merged() -> (
    None
):
    _document, config = load_sightings_config(
        project_root() / "config/data/sightings.yaml"
    )
    source = pd.DataFrame(
        [
            _record(
                "TWM",
                "left",
                latitude=48.0,
                longitude=-123.000,
                uncertainty_m=10,
            ),
            _record(
                "ACARTIA",
                "middle",
                latitude=48.0,
                longitude=-123.004,
                uncertainty_m=10,
            ),
            _record(
                "MAPLIFY",
                "right",
                latitude=48.0,
                longitude=-123.008,
                uncertainty_m=10,
            ),
        ]
    )
    records, audit = _normalize_records(source, config)
    for record in records:
        record["SOURCE_EVENT_AT_UTC"] = pd.Timestamp(
            "2026-08-18T18:00:00Z"
        ).to_pydatetime()
    groups = _cluster(records, config, audit)

    assert sorted(len(group) for group in groups) == [1, 2]
    uncertain = [item for item in audit if item["REASON"] == "UNCERTAIN_MATCH"]
    assert len(uncertain) == 1
    assert json.loads(uncertain[0]["DETAIL"])["partial_chain_groups"]


from marine_mammal_toolkit.cetaceans.killer_whales.observations.interpretation import (
    OBSERVATION_POLICY,
)
