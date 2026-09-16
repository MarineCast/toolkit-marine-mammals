from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from marine_mammal_toolkit.cli import cli
from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import DataSnapshotMetadata
from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    promote_sightings_release,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    release_profile,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    resolve_sightings_release_artifact,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
    validate_sightings_release,
)
from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import _coverage_gate
from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import _imputation_gates
from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import _latest_collect


def _artifact(
    path: Path, *, dataset_id: str = "whale.sightings.fixture"
) -> ArtifactRef:
    return ArtifactRef(
        kind="domain",
        dataset_id=dataset_id,
        path=path,
        producer="fixture",
        schema_version="1",
        run_id="fixture-run",
        config_hash="config-hash",
        checksum=checksum_path(path),
        row_count=1,
        file_count=1,
        processing_mode="retrospective",
    )


def _stage(artifact: ArtifactRef) -> RunManifest:
    return RunManifest(
        run_id="fixture-run",
        workflow="whale.sightings.fixture",
        config_hash="config-hash",
        resolved_config={},
        outputs=(artifact,),
        schema_version="1",
        stage_signature="fixture-signature",
    )


def test_release_is_immutable_relative_and_exactly_inventoried(tmp_path: Path) -> None:
    source = tmp_path / "candidate.parquet"
    source.write_bytes(b"immutable fixture")
    artifact = _artifact(source)
    release_root = tmp_path / "releases"

    manifest = promote_sightings_release(
        release_root=release_root,
        profile=release_profile("observed-only"),
        end_date=date(2026, 8, 19),
        config_hash="config-hash",
        run_id="fixture-run",
        artifacts=(artifact,),
        stage_manifests=(_stage(artifact),),
        gates=({"name": "fixture", "required": True, "passed": True},),
    )

    pointer = json.loads((release_root / "latest.json").read_text())
    assert not Path(pointer["manifest"]).is_absolute()
    assert manifest == release_root / pointer["manifest"]
    assert validate_sightings_release(release_root / "latest.json").valid

    payload = json.loads(manifest.read_text())
    copied = manifest.parent / payload["inventory"][0]["path"]
    copied.write_bytes(b"mutated")
    report = validate_sightings_release(manifest)
    assert not report.valid
    assert any("Checksum mismatch" in error for error in report.errors)


def test_release_artifact_resolution_is_checksum_verified(tmp_path: Path) -> None:
    source = tmp_path / "candidate.parquet"
    source.write_bytes(b"immutable fixture")
    artifact = _artifact(source, dataset_id="whale.sightings.imputed_retrospective")
    release_root = tmp_path / "releases"
    manifest = promote_sightings_release(
        release_root=release_root,
        profile=release_profile("imputation-only"),
        end_date=date(2026, 8, 19),
        config_hash="config-hash",
        run_id="fixture-run",
        artifacts=(artifact,),
        stage_manifests=(_stage(artifact),),
        gates=(
            {
                "name": "verified_target_cohort",
                "required": False,
                "passed": False,
                "coverage_status": "unverified",
                "coverage_through": None,
            },
        ),
    )

    resolved = resolve_sightings_release_artifact(
        release_root / "latest.json", "whale.sightings.imputed_retrospective"
    )
    assert resolved.manifest_path == manifest
    assert resolved.release_id == manifest.parent.name
    assert resolved.path.read_bytes() == b"immutable fixture"
    assert resolved.checksum == artifact.checksum
    assert resolved.coverage_status == "unverified"
    assert resolved.coverage_through is None

    resolved.path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_sightings_release_artifact(
            release_root / "latest.json", "whale.sightings.imputed_retrospective"
        )


def test_offline_collection_prefers_sources_from_authoritative_release(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate-source"
    source.mkdir()
    (source / "snapshot.json").write_text('{"source":"twm"}', encoding="utf-8")
    artifact = _artifact(source, dataset_id="whale.sightings.source_twm")
    release_root = tmp_path / "processed/domain/whale_layer/sightings/releases"
    manifest = promote_sightings_release(
        release_root=release_root,
        profile=release_profile("observed-only"),
        end_date=date(2026, 8, 19),
        config_hash="config-hash",
        run_id="released-run",
        artifacts=(artifact,),
        stage_manifests=(_stage(artifact),),
        gates=({"name": "fixture", "required": True, "passed": True},),
    )

    stale = tmp_path / "stale-source"
    stale.mkdir()
    (stale / "snapshot.json").write_text('{"source":"twm"}', encoding="utf-8")
    stale_artifact = _artifact(stale, dataset_id="whale.sightings.source_twm")
    stale_manifest = _stage(stale_artifact)
    raw_pointer = tmp_path / "raw/whale/sightings/manifests/latest.json"
    stale_manifest_path = raw_pointer.parent / "collect/stale.json"
    stale_manifest.write(stale_manifest_path)
    raw_pointer.parent.mkdir(parents=True, exist_ok=True)
    raw_pointer.write_text(
        json.dumps({"manifest": "collect/stale.json"}), encoding="utf-8"
    )

    result = _latest_collect(tmp_path)
    released_inventory = json.loads(manifest.read_text())["inventory"][0]
    expected = (manifest.parent / released_inventory["path"]).resolve()
    assert result.skipped is True
    assert result.outputs[0].path == expected
    assert result.outputs[0].freshness == "released"


def test_failed_gate_does_not_advance_existing_release_pointer(tmp_path: Path) -> None:
    source = tmp_path / "candidate.parquet"
    source.write_bytes(b"fixture")
    artifact = _artifact(source)
    release_root = tmp_path / "releases"
    release_root.mkdir()
    previous = {"release_id": "prior", "manifest": "generations/prior/manifest.json"}
    (release_root / "latest.json").write_text(json.dumps(previous))

    with pytest.raises(ValueError, match="coverage"):
        promote_sightings_release(
            release_root=release_root,
            profile=release_profile("production-retrospective"),
            end_date=date(2026, 8, 19),
            config_hash="config-hash",
            run_id="failed-run",
            artifacts=(artifact,),
            stage_manifests=(_stage(artifact),),
            gates=({"name": "coverage", "required": True, "passed": False},),
        )

    assert json.loads((release_root / "latest.json").read_text()) == previous
    assert not (release_root / "generations").exists()


def test_release_rejects_unlisted_files(tmp_path: Path) -> None:
    source = tmp_path / "candidate.parquet"
    source.write_bytes(b"fixture")
    artifact = _artifact(source)
    manifest = promote_sightings_release(
        release_root=tmp_path / "releases",
        profile=release_profile("authoritative-counts"),
        end_date=date(2026, 8, 19),
        config_hash="config-hash",
        run_id="fixture-run",
        artifacts=(artifact,),
        stage_manifests=(_stage(artifact),),
        gates=({"name": "mass", "required": True, "passed": True},),
    )
    (manifest.parent / "unexpected.txt").write_text("not inventoried")

    report = validate_sightings_release(manifest)
    assert not report.valid
    assert any("unlisted files" in error for error in report.errors)


def test_release_cli_supports_dry_run_and_manifest_validation(tmp_path: Path) -> None:
    runner = CliRunner()
    dry_run = runner.invoke(
        cli,
        [
            "--workspace-root",
            str(tmp_path),
            "--data-root",
            str(tmp_path / "data"),
            "killer-whales",
            "observations",
            "run",
            "--profile",
            "production-retrospective",
            "--end-date",
            "2026-08-19",
            "--dry-run",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert '"estimated_dense_rows"' in dry_run.output
    assert '"operation": "killer-whales.observations.run"' in dry_run.output

    source = tmp_path / "candidate.parquet"
    source.write_bytes(b"fixture")
    artifact = _artifact(source)
    manifest = promote_sightings_release(
        release_root=tmp_path / "releases",
        profile=release_profile("observed-only"),
        end_date=date(2026, 8, 19),
        config_hash="config-hash",
        run_id="fixture-run",
        artifacts=(artifact,),
        stage_manifests=(_stage(artifact),),
        gates=({"name": "fixture", "required": True, "passed": True},),
    )
    validated = runner.invoke(
        cli,
        [
            "--workspace-root",
            str(tmp_path),
            "killer-whales",
            "observations",
            "validate",
            "sightings-release",
            "--manifest",
            str(manifest),
        ],
    )
    assert validated.exit_code == 0, validated.output
    assert "PASS\twhale.sightings.release" in validated.output


def test_imputation_only_profile_stops_before_downstream_products() -> None:
    profile = release_profile("imputation-only")

    assert profile.include_imputation is True
    assert profile.include_counts is False
    assert profile.include_model_grid is False
    assert profile.include_intensity is False
    assert profile.require_verified_cohort is False
    assert profile.public_by_default is False


def test_production_coverage_and_uncertified_soft_mass_fail_closed(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations.parquet"
    pd.DataFrame(
        {
            "USE_FOR_PROBABILISTIC_COUNTS": [True],
            "IMPUTATION_APPLIED": [False],
            "CLASS_CERTIFIED_FOR_HARD_LABEL": [False],
            "PREDICTION_STABLE": [True],
            "EXPECTED_SRKW_COUNT": [0.8],
            "EXPECTED_TRANSIENT_COUNT": [0.2],
            "EXPECTED_UNKNOWN_COUNT": [0.0],
        }
    ).to_parquet(observations, index=False)
    snapshot = DataSnapshotMetadata(
        snapshot_id="snapshot",
        snapshot_created_at="2026-08-19T00:00:00+00:00",
        coverage_start=None,
        coverage_through=None,
        source_watermarks=(),
        coverage_status="unverified",
    )
    artifact = ArtifactRef(
        **{
            **_artifact(
                observations, dataset_id="whale.sightings.imputed_retrospective"
            ).__dict__,
            "data_snapshot": snapshot,
        }
    )

    coverage = _coverage_gate(
        artifact,
        profile=release_profile("production-retrospective"),
        end_date=date(2026, 8, 19),
    )
    imputation = _imputation_gates(artifact)
    assert coverage["required"] is True and coverage["passed"] is False
    assert imputation[0]["required"] is True and imputation[0]["passed"] is False


def test_imputation_release_gate_does_not_treat_observed_mass_as_soft(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "imputed.parquet"
    pd.DataFrame(
        {
            "ECOTYPE_DETAIL_OBSERVED": ["SRKW", "UNKNOWN"],
            "USE_FOR_PROBABILISTIC_COUNTS": [True, True],
            "IMPUTATION_APPLIED": [False, False],
            "CLASS_CERTIFIED_FOR_HARD_LABEL": [False, False],
            "HARD_LABEL_CERTIFIED": [False, False],
            "SOFT_COUNT_CERTIFIED": [False, False],
            "STABILITY_EVALUATED": [False, False],
            "PREDICTION_STABLE": [False, False],
            "EXPECTED_SRKW_COUNT": [1.0, 0.0],
            "EXPECTED_TRANSIENT_COUNT": [0.0, 0.0],
            "EXPECTED_OTHER_COUNT": [0.0, 0.0],
            "EXPECTED_UNKNOWN_COUNT": [0.0, 1.0],
        }
    ).to_parquet(observations, index=False)
    artifact = _artifact(
        observations, dataset_id="whale.sightings.imputed_retrospective"
    )
    gate = _imputation_gates(artifact)[0]
    assert gate["soft_count_rows"] == 1
    assert gate["unsafe_soft_rows"] == 1
