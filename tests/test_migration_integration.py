"""Small offline runs: all products are confined to pytest temporary storage."""

from datetime import date
import json

import h3
import numpy as np
import pandas as pd
import pytest
import yaml

from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import (
    SightingsPipelineRunRequest,
    run_sightings_pipeline,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    resolve_sightings_release_artifact,
    validate_sightings_release,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.imputer import (
    KillerWhaleImputer,
)
from marine_mammal_toolkit.tools.observations.impute.config import (
    FeatureConfig,
    ImputationConfig,
    ModelConfig,
)
from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef, checksum_path


def test_offline_observed_pipeline_and_replay(synthetic_workspace, monkeypatch):
    import requests

    def no_network(*args, **kwargs):
        raise AssertionError("Synthetic pipeline must never access the network")

    monkeypatch.setattr(requests, "get", no_network)
    root = synthetic_workspace
    raw = yaml.safe_load(config_path().read_text())
    for name, source in raw["collection"]["sources"].items():
        source["enabled"] = name == "twm"
    config = root / "synthetic.yaml"
    config.write_text(yaml.safe_dump(raw))
    csv = root / "input.csv"
    csv.write_text(
        "id,date,latitude,longitude,pod\na,2025-06-03,48.5,-123.0,J pod\nb,2025-06-04,48.5,-123.0,Transient\n"
    )
    universe = root / "data/processed/environment/seascape/full_counting"
    universe.mkdir(parents=True)
    refs = []
    for resolution in (4, 5, 6):
        path = universe / f"H3_WATER_UNIVERSE_{resolution}.parquet"
        pd.DataFrame(
            {"H3_INDEX": [h3.latlng_to_cell(48.5, -123.0, resolution)]}
        ).to_parquet(path)
        refs.append(
            ArtifactRef(
                kind="domain",
                dataset_id=f"environment.seascape.h3_full_counting_universe_r{resolution}",
                path=path,
                producer="synthetic",
                checksum=checksum_path(path),
            )
        )
    (universe / "_dataset_manifest.json").write_text(
        json.dumps({"outputs": [r.to_dict() for r in refs]})
    )
    common = dict(
        config=config,
        data_root=root / "data",
        artifact_root=root / "artifacts",
        output_root=root / "outputs",
        profile="observed-only",
        start_date=date(2025, 6, 2),
        end_date=date(2025, 6, 8),
    )
    result = run_sightings_pipeline(
        SightingsPipelineRunRequest(
            **common, twm_files=(csv,), run_id="synthetic-first"
        )
    )
    assert validate_sightings_release(result.release_manifest).valid
    observed = resolve_sightings_release_artifact(
        result.release_manifest, "whale.sightings.observations"
    )
    first = pd.read_parquet(observed.path)
    assert len(first) == 2
    assert first.OBSERVATION_ID.str.startswith("orca:v4:").all()
    assert not first.PUBLIC_RELEASE_ELIGIBLE.any()
    assert all(
        stage.manifest.code_revision.startswith("marine-mammal-toolkit:")
        for stage in result.stage_results.values()
        if stage.manifest
    )
    replay = run_sightings_pipeline(
        SightingsPipelineRunRequest(**common, offline=True, run_id="synthetic-replay")
    )
    second = resolve_sightings_release_artifact(
        replay.release_manifest, "whale.sightings.observations"
    )
    # A fresh normalization run records its own correction-as-of timestamp.
    pd.testing.assert_frame_equal(
        first.drop(columns="LAST_CORRECTED_AT_UTC"),
        pd.read_parquet(second.path).drop(columns="LAST_CORRECTED_AT_UTC"),
    )


def test_deterministic_synthetic_fit_and_versioned_roundtrip(tmp_path, monkeypatch):
    from seascape.spatial_support.water_network import WaterGraph
    from marine_mammal_toolkit.tools.observations.impute import marine

    longitudes = np.linspace(-160.0, -110.0, 60)
    cells = [h3.latlng_to_cell(48.5, lon, 6) for lon in longitudes]
    graph = WaterGraph(
        resolution=6,
        cells=np.array(cells),
        offsets=np.zeros(61, dtype=int),
        neighbors=np.array([], dtype=int),
        weights_m=np.array([]),
        support=pd.DataFrame({"H3_INDEX": cells}),
        cell_to_position={cell: i for i, cell in enumerate(cells)},
        water_mask_version="synthetic",
        spatial_support_version="synthetic",
    )
    monkeypatch.setattr(marine, "load_water_graph", lambda *args, **kwargs: graph)
    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": [f"synthetic-{i}" for i in range(60)],
            "SIGHTING_DATE": pd.date_range("2020-01-01", periods=60, freq="10D").date,
            "LATITUDE": [48.5] * 60,
            "LONGITUDE": longitudes,
            "ECOTYPE_DETAIL": ["SRKW" if i % 2 else "TRANSIENT" for i in range(60)],
            "ECOTYPE_BUCKET": ["SRKW" if i % 2 else "TRANSIENT" for i in range(60)],
            "SOURCE_SET": ["TWM"] * 60,
        }
    )
    config = ImputationConfig(
        feature=FeatureConfig(
            max_radius_km=10,
            max_day_lag=1,
            temporal_windows=(
                ("same_day", 0, 0),
                ("past_1d", -1, -1),
                ("future_1d", 1, 1),
            ),
            marine_fallback_to_haversine=False,
            water_network_config_path=str(tmp_path / "synthetic-water.yaml"),
        ),
        model=ModelConfig(n_estimators=8, min_samples_leaf=2, n_splits=3),
    )
    query = frame.iloc[:4].copy()
    query["OBSERVATION_ID"] = [f"unknown-{i}" for i in range(4)]
    query["ECOTYPE_DETAIL"] = "UNKNOWN"
    query["ECOTYPE_BUCKET"] = "OTHER"
    query["SIGHTING_DATE"] = (
        pd.to_datetime(query["SIGHTING_DATE"]) + pd.Timedelta(days=5)
    ).dt.date
    cohort = pd.concat([frame, query], ignore_index=True)
    a = KillerWhaleImputer(config).fit(
        cohort, evaluate_strategies=("encounter", "purged_blocked")
    )
    b = KillerWhaleImputer(config).fit(
        cohort, evaluate_strategies=("encounter", "purged_blocked")
    )
    expected = a.apply_to_all(query)
    pd.testing.assert_frame_equal(expected, b.apply_to_all(query))
    path = a.save(tmp_path / "model.joblib")
    assert path.read_bytes().startswith(b"MMTK-IMPUTER\x00\x01\n")
    restored = KillerWhaleImputer.load(path)
    pd.testing.assert_frame_equal(expected, restored.apply_to_all(query))
    assert restored.soft_mass_certification_["CERTIFIED"] is False
    np.testing.assert_allclose(
        expected[
            [
                "EXPECTED_SRKW_COUNT",
                "EXPECTED_TRANSIENT_COUNT",
                "EXPECTED_OTHER_COUNT",
                "EXPECTED_UNKNOWN_COUNT",
            ]
        ].sum(axis=1),
        1.0,
    )


def test_seascape_bridge_resolves_base_without_changing_working_directory(
    synthetic_workspace, monkeypatch
):
    from marine_mammal_toolkit.tools._core import seascape

    root = synthetic_workspace
    config = root / "water.yaml"
    config.write_text("base_directory: data\nwater_network: {}\n")
    monkeypatch.setenv("SEASCAPE_WORKSPACE", str(root))

    def inspect(resolution, path):
        payload = yaml.safe_load(path.read_text())
        assert payload["base_directory"] == str(root / "data")
        assert resolution == 6
        return "graph"

    monkeypatch.setattr(seascape, "_load_water_graph", inspect)
    assert seascape.load_water_graph(6, config) == "graph"
    monkeypatch.delenv("SEASCAPE_WORKSPACE")
    with pytest.raises(ValueError, match="SEASCAPE_WORKSPACE"):
        seascape.load_water_graph(6, config)
