from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner

from marine_mammal_toolkit.cli import cli
from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
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
from marine_mammal_toolkit.cetaceans.killer_whales.observations.product import (
    cleanup_completed_sightings_run,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.product import (
    current_sightings_row_count,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.product import (
    materialize_sightings_product,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.product import (
    prune_prior_sightings_artifacts,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.product import (
    sightings_product_layout,
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
    assert not (release_root / ".staging").exists()
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


def test_materializes_daily_product_and_model_manifest(tmp_path: Path) -> None:
    observations = tmp_path / "candidate/observations.parquet"
    imputed = tmp_path / "candidate/imputed.parquet"
    model = tmp_path / "candidate/model"
    observations.parent.mkdir()
    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": ["orca-1", "orca-2"],
            "SIGHTING_DATE_UTC": pd.to_datetime(["2026-08-18", "2026-08-19"], utc=True),
            "ECOTYPE": ["SRKW", "UNKNOWN"],
            "LATITUDE": [48.50, 48.62],
            "LONGITUDE": [-123.20, -123.05],
            "SOURCE": ["TWM", "GBIF"],
        }
    )
    frame.to_parquet(observations, index=False)
    frame.assign(
        ECOTYPE_IMPUTED=["SRKW", "UNKNOWN"],
        IMPUTATION_STATUS=["OBSERVED", "ABSTAINED"],
        IMPUTATION_APPLIED=[False, False],
    ).to_parquet(imputed, index=False)
    model.mkdir()
    (model / "ecotype_imputer.joblib").write_bytes(b"fixture-model")
    (model / "model_metrics.json").write_text(
        json.dumps(
            {
                "fit_at_utc": "2026-08-20T01:02:03+00:00",
                "fit_run_id": "fixture-model-run",
                "model_sha256": "fixture-sha",
                "training_summary": {"LABELED_N": 101},
                "evaluations": {"purged_blocked": {"metrics": {"accuracy": 0.91}}},
                "config": {"model": {"random_state": 42}},
            }
        ),
        encoding="utf-8",
    )
    artifacts = (
        _artifact(observations, dataset_id="whale.sightings.observations"),
        _artifact(imputed, dataset_id="whale.sightings.imputed_retrospective"),
        _artifact(model, dataset_id="whale.sightings.imputation_model"),
    )
    release = promote_sightings_release(
        release_root=(tmp_path / "product/processed/sightings/final/releases"),
        profile=release_profile("imputation-only"),
        end_date=date(2026, 8, 20),
        config_hash="config-hash",
        run_id="fixture-run",
        artifacts=artifacts,
        stage_manifests=tuple(_stage(artifact) for artifact in artifacts),
        gates=(
            {
                "name": "verified_target_cohort",
                "required": False,
                "passed": False,
                "coverage_status": "unverified",
            },
        ),
    )

    product = materialize_sightings_product(
        release,
        product_root=tmp_path / "product",
    )

    assert product.composite_path.is_file()
    assert product.imputed_path.is_file()
    assert product.report_path.is_file()
    assert product.dated_root == (
        tmp_path
        / "product/processed/sightings/final/by-date/2026-08-20"
        / release.parent.name
    )
    assert len(pd.read_parquet(product.composite_path)) == 2
    payload = json.loads(product.model_manifest_path.read_text())
    assert payload["release_profile"] == "imputation-only"
    assert payload["tables"]["composite"]["temporal_grain"] == "daily"
    assert payload["tables"]["composite"]["observed_start"].startswith("2026-08-18")
    assert payload["imputation_model"]["training_summary"]["LABELED_N"] == 101
    assert (
        payload["imputation_model"]["evaluations"]["purged_blocked"]["metrics"][
            "accuracy"
        ]
        == 0.91
    )
    latest = json.loads(
        (tmp_path / "product/processed/sightings/final/latest.json").read_text()
    )
    assert latest["release_id"] == release.parent.name
    assert (
        latest["report"]
        == product.report_path.relative_to(tmp_path / "product").as_posix()
    )
    assert (
        latest["manifest"]
        == product.model_manifest_path.relative_to(tmp_path / "product").as_posix()
    )
    assert latest["report_checksum"] == checksum_path(product.report_path)
    rendered = product.report_path.read_text(encoding="utf-8")
    assert "All-time reported-sighting density" in rendered
    assert "Reported sightings count over time" in rendered


def test_completed_product_cleanup_removes_only_completed_run_workspaces(
    tmp_path: Path,
) -> None:
    product_root = tmp_path / "product"
    completed_run_id = "sightings-20260920-fixture-120000"
    failed_run_id = "sightings-20260920-fixture-130000"
    completed_candidate = product_root / "_sightings_product_runs" / completed_run_id
    completed_stage = product_root / ".staging" / f"{completed_run_id}-normalize"
    failed_candidate = product_root / "_sightings_product_runs" / failed_run_id
    failed_stage = product_root / ".staging" / f"{failed_run_id}-normalize"
    for directory in (
        completed_candidate,
        completed_stage,
        failed_candidate,
        failed_stage,
    ):
        directory.mkdir(parents=True)
        (directory / "marker.txt").write_text("fixture", encoding="utf-8")

    cleanup_completed_sightings_run(
        product_root=product_root,
        completed_run_id=completed_run_id,
    )

    assert not completed_candidate.exists()
    assert not completed_stage.exists()
    assert failed_candidate.is_dir()
    assert failed_stage.is_dir()


def test_completed_product_cleanup_removes_empty_workspace_roots(
    tmp_path: Path,
) -> None:
    product_root = tmp_path / "product"
    completed_run_id = "sightings-20260920-fixture-120000"
    (product_root / "_sightings_product_runs" / completed_run_id).mkdir(parents=True)
    (product_root / ".staging" / f"{completed_run_id}-normalize").mkdir(parents=True)

    cleanup_completed_sightings_run(
        product_root=product_root,
        completed_run_id=completed_run_id,
    )

    assert not (product_root / "_sightings_product_runs").exists()
    assert not (product_root / ".staging").exists()


@pytest.mark.parametrize(
    "keep,pins,removed", [(1, (), True), (3, (), False), (1, ("old-release",), False)]
)
def test_guarded_retention_keeps_only_current_sightings_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep, pins, removed
) -> None:
    product_root = tmp_path / "product"
    processed = product_root / "processed/sightings/final"
    release_id = "current-release"
    fit_run_id = "current-model"
    current_generation = processed / "by-date/2026-09-20" / release_id
    current_release = processed / "releases/generations" / release_id
    current_model = processed.parent / "imputed/models/sighting_imputation" / fit_run_id
    for directory in (current_generation, current_release, current_model):
        directory.mkdir(parents=True)

    frame = pd.DataFrame(
        {
            "OBSERVATION_ID": ["orca-1", "orca-2"],
            "SIGHTING_DATE_UTC": pd.to_datetime(["2026-09-19", "2026-09-20"], utc=True),
        }
    )
    stable_composite = processed / "composite-sightings.parquet"
    frame.to_parquet(stable_composite, index=False)
    frame.to_parquet(current_generation / stable_composite.name, index=False)
    release_manifest = current_release / "manifest.json"
    release_manifest.write_text("{}", encoding="utf-8")
    manifest = {
        "release_id": release_id,
        "release_manifest": str(release_manifest.relative_to(product_root)),
        "tables": {
            "composite": {
                "row_count": 2,
                "checksum": checksum_path(stable_composite),
            }
        },
        "imputation_model": {"fit_run_id": fit_run_id},
    }
    (current_generation / "imputation-model-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (processed / "latest.json").write_text(
        json.dumps(
            {
                "release_id": release_id,
                "generation": str(current_generation.relative_to(product_root)),
            }
        ),
        encoding="utf-8",
    )
    model_root = current_model.parent
    (model_root / "latest.json").write_text(
        json.dumps({"fit_run_id": fit_run_id}), encoding="utf-8"
    )

    old_generation = processed / "by-date/2026-09-19/old-release"
    old_release = current_release.parent / "old-release"
    old_model = model_root / "old-model"
    for directory in (old_generation, old_release, old_model):
        directory.mkdir(parents=True)
        (directory / "marker.txt").write_text("old", encoding="utf-8")

    (old_generation / "imputation-model-manifest.json").write_text(
        json.dumps(
            {
                "release_id": "old-release",
                "imputation_model": {"fit_run_id": "old-model"},
            }
        )
    )

    class _ValidRelease:
        def require_valid(self) -> None:
            return None

    monkeypatch.setattr(
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.product.validate_sightings_release",
        lambda _: _ValidRelease(),
    )

    retention = prune_prior_sightings_artifacts(
        product_root=product_root,
        previous_row_count=2,
        keep_generations=keep,
        pinned_release_ids=pins,
    )

    assert current_sightings_row_count(product_root) == 2
    assert retention.applied is True
    assert retention.growth_fraction == 0.0
    assert current_generation.is_dir()
    assert current_release.is_dir()
    assert current_model.is_dir()
    assert old_generation.exists() == (not removed)
    assert old_release.exists() == (not removed)
    assert old_model.exists() == (not removed)
    report = json.loads((processed / "retention.json").read_text())
    assert report["reason"] == "retention_gate_passed"
    assert len(report["removed_paths"]) == (3 if removed else 0)


@pytest.mark.parametrize(
    ("previous_row_count", "current_row_count", "reason"),
    [
        (3, 2, "current_count_below_previous"),
        (10, 12, "growth_exceeds_threshold"),
    ],
)
def test_retention_gate_preserves_prior_artifacts_on_suspicious_counts(
    previous_row_count: int, current_row_count: int, reason: str
) -> None:
    from marine_mammal_toolkit.cetaceans.killer_whales.observations.product import (
        _retention_decision,
    )

    applied, actual_reason, _, _ = _retention_decision(
        previous_row_count=previous_row_count,
        current_row_count=current_row_count,
        max_growth_fraction=0.10,
    )

    assert applied is False
    assert actual_reason == reason


def test_product_cli_dry_run_declares_incremental_daily_outputs(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "--workspace-root",
            str(tmp_path),
            "--data-root",
            "data/marine-mammals/killer-whales",
            "killer-whales",
            "observations",
            "product",
            "--end-date",
            "2026-08-20",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "incremental-update"
    assert payload["temporal_grain"] == "daily"
    assert payload["raw_root"].endswith("marine-mammals/killer-whales/raw")
    assert payload["processed_root"].endswith(
        "marine-mammals/killer-whales/processed/sightings"
    )
    assert payload["normalized_root"].endswith(
        "processed/sightings/normalized"
    )
    assert payload["imputed_root"].endswith("processed/sightings/imputed")
    assert payload["final_root"].endswith("processed/sightings/final")
    assert payload["artifact_root"] == payload["imputed_root"]
    assert payload["output_root"] == payload["imputed_root"]
    assert payload["composite"].endswith(
        "processed/sightings/final/composite-sightings.parquet"
    )
    assert payload["imputed"].endswith(
        "processed/sightings/final/imputed-sightings.parquet"
    )
    assert payload["report"].endswith("processed/sightings/final/sightings-report.html")
    assert payload["previous_row_count"] is None
    assert payload["max_growth_fraction"] == 0.10


def test_product_cli_accepts_optional_custom_yaml(tmp_path: Path) -> None:
    custom_config = tmp_path / "custom-sightings.yaml"
    custom_config.write_text(
        "\n".join(
            (
                f"extends: {json.dumps(str(config_path('sightings_product')))}",
                'min_date: "2001-02-03"',
                "collection:",
                "  overlap_days: 9",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "--workspace-root",
            str(tmp_path),
            "--data-root",
            "product",
            "killer-whales",
            "observations",
            "product",
            "--config",
            "custom-sightings.yaml",
            "--end-date",
            "2026-08-20",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config"] == str(custom_config)
    assert payload["start_date"] == "2001-02-03"


def test_product_layout_prepares_owned_stage_directories(tmp_path: Path) -> None:
    layout = sightings_product_layout(tmp_path / "killer-whales")

    layout.prepare()

    assert layout.raw_root.is_dir()
    assert layout.normalized_root.is_dir()
    assert layout.imputed_root.is_dir()
    assert layout.final_root.is_dir()
    assert layout.sightings_root == tmp_path / "killer-whales/processed/sightings"


def test_product_cli_wires_pipeline_to_layout_owned_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(request):
        captured["request"] = request
        return SimpleNamespace(
            release_manifest=tmp_path / "fixture-release.json",
            candidate_root=tmp_path / "product/_sightings_product_runs/fixture-run",
        )

    def fake_materialize(release_manifest, *, product_root):
        captured["materialize_root"] = product_root
        final = product_root / "processed/sightings/final"
        return SimpleNamespace(
            release_manifest=release_manifest,
            dated_root=final / "by-date/2026-08-20/fixture-release",
            composite_path=final / "composite-sightings.parquet",
            imputed_path=final / "imputed-sightings.parquet",
            model_manifest_path=final / "imputation-model-manifest.json",
            report_path=final / "sightings-report.html",
        )

    class FakeRetention:
        def as_dict(self, *, product_root):
            return {"applied": False, "product_root": str(product_root)}

    monkeypatch.setattr(
        "marine_mammal_toolkit.cetaceans.killer_whales.pipeline.run_sightings_pipeline",
        fake_run,
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.product.current_sightings_row_count",
        lambda _: None,
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.product.materialize_sightings_product",
        fake_materialize,
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.product.prune_prior_sightings_artifacts",
        lambda **_: FakeRetention(),
    )
    monkeypatch.setattr(
        "marine_mammal_toolkit.cetaceans.killer_whales.observations.product.cleanup_completed_sightings_run",
        lambda **kwargs: captured.update(cleanup=kwargs),
    )

    result = CliRunner().invoke(
        cli,
        [
            "--workspace-root",
            str(tmp_path),
            "--data-root",
            "product",
            "--artifact-root",
            "outside-artifacts",
            "--output-root",
            "outside-outputs",
            "--run-id",
            "fixture-run",
            "killer-whales",
            "observations",
            "product",
            "--end-date",
            "2026-08-20",
        ],
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.data_root == tmp_path / "product"
    assert request.artifact_root == tmp_path / "product/processed/sightings/imputed"
    assert request.output_root == tmp_path / "product/processed/sightings/imputed"
    assert request.persistent_state is True
    assert request.profile == "imputation-only"
    assert captured["materialize_root"] == tmp_path / "product"
    assert captured["cleanup"] == {
        "product_root": tmp_path / "product",
        "completed_run_id": "fixture-run",
    }
    assert (tmp_path / "product/raw").is_dir()
    assert (tmp_path / "product/processed/sightings/normalized").is_dir()
    assert (tmp_path / "product/processed/sightings/imputed").is_dir()
    assert (tmp_path / "product/processed/sightings/final").is_dir()


def test_offline_collection_prefers_sources_from_authoritative_release(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate-source"
    source.mkdir()
    (source / "snapshot.json").write_text('{"source":"twm"}', encoding="utf-8")
    artifact = _artifact(source, dataset_id="whale.sightings.source_twm")
    release_root = tmp_path / "processed/sightings/final/releases"
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


@pytest.mark.parametrize("failure", ["copy", "report", "pointer"])
def test_failed_product_publication_preserves_previous_generation(
    tmp_path, monkeypatch, failure
):
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        product as module,
    )

    test_materializes_daily_product_and_model_manifest(tmp_path)
    root = tmp_path / "product"
    final = root / "processed/sightings/final"
    previous = module.resolve_sightings_product(root)
    files = [
        final / name
        for name in (
            "latest.json",
            "composite-sightings.parquet",
            "imputed-sightings.parquet",
            "imputation-model-manifest.json",
            "sightings-report.html",
        )
    ]
    before = {path: path.read_bytes() for path in files}
    observations = tmp_path / "candidate/observations.parquet"
    imputed = tmp_path / "candidate/imputed.parquet"
    model = tmp_path / "candidate/model"
    for path in (observations, imputed):
        frame = pd.read_parquet(path)
        frame.loc[0, "OBSERVATION_ID"] = "new-generation"
        frame.to_parquet(path, index=False)
    artifacts = tuple(
        _artifact(path, dataset_id=dataset)
        for path, dataset in (
            (observations, "whale.sightings.observations"),
            (imputed, "whale.sightings.imputed_retrospective"),
            (model, "whale.sightings.imputation_model"),
        )
    )
    release = promote_sightings_release(
        release_root=final / "releases",
        profile=release_profile("imputation-only"),
        end_date=date(2026, 8, 21),
        config_hash="new",
        run_id="new",
        artifacts=artifacts,
        stage_manifests=tuple(_stage(a) for a in artifacts),
        gates=({"name": "fixture", "required": True, "passed": True},),
    )
    if failure == "copy":
        original = module._atomic_copy
        calls = []

        def fail_copy(source, destination):
            calls.append(destination)
            # The pointer reader sees old, internally consistent paths throughout.
            assert module.resolve_sightings_product(root) == previous
            if len(calls) == 2:
                raise OSError("injected copy failure")
            return original(source, destination)

        monkeypatch.setattr(module, "_atomic_copy", fail_copy)
    elif failure == "report":

        def fail_report(**kwargs):
            raise OSError("injected report failure")

        monkeypatch.setattr(module, "build_sightings_report_html", fail_report)
    else:
        original = module.atomic_write_json

        def fail_pointer(path, *args, **kwargs):
            if path == final / "latest.json":
                raise OSError("injected pointer failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(module, "atomic_write_json", fail_pointer)
    with pytest.raises(OSError, match="injected"):
        materialize_sightings_product(release, product_root=root)
    assert {path: path.read_bytes() for path in files} == before
    assert module.resolve_sightings_product(root) == previous
