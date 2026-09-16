"""Standalone marine-mammal workflow CLI."""

from __future__ import annotations
import json
from datetime import date
from pathlib import Path
from typing import Any, Callable
import click
from marine_mammal_toolkit.tools._core.config import ConfigDocument, workspace
from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
from marine_mammal_toolkit.cetaceans.killer_whales.catalog import (
    register_builtin_datasets,
)


def _input_path(ctx, parameter, value):
    """Resolve input paths against the explicitly selected data workspace."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_input_path(ctx, parameter, item) for item in value)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(ctx.find_root().obj["workspace_root"]) / candidate
    if not candidate.exists():
        raise click.BadParameter(f"Input does not exist: {candidate}", param=parameter)
    return str(candidate.resolve())


@click.group()
@click.option(
    "--workspace-root", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--data-root", default="data", type=click.Path(path_type=Path))
@click.option("--artifact-root", default="artifacts", type=click.Path(path_type=Path))
@click.option("--output-root", default="outputs", type=click.Path(path_type=Path))
@click.option("--run-id")
@click.pass_context
def cli(ctx, workspace_root, data_root, artifact_root, output_root, run_id):
    """Marine-mammal source processing and species workflows."""
    root = workspace_root.expanduser().resolve()
    ctx.with_resource(workspace(root))
    ctx.obj = {"workspace_root": root, "run_id": run_id}
    for name, value in (
        ("data_root", data_root),
        ("artifact_root", artifact_root),
        ("output_root", output_root),
    ):
        ctx.obj[name] = str(
            value.resolve() if value.is_absolute() else (root / value).resolve()
        )
    register_builtin_datasets()


@cli.group("killer-whales")
def killer_whales():
    """Run the configured killer-whale implementation."""


@killer_whales.group()
def observations():
    """Collect, process, impute, and post-process sightings."""


@observations.group()
def impute():
    """Fit or apply a certified selective imputation model."""


@observations.group("post-process")
def post_process():
    """Build counts, model grids, intensity, or all three sequentially."""


@killer_whales.group()
def populations():
    """Process annual population census inputs."""


@populations.command("run")
@click.option(
    "--config",
    default=lambda: str(config_path("populations")),
    type=click.Path(),
    callback=_input_path,
)
@click.option("--fail-on-total-mismatch", is_flag=True)
@click.option("--ecotype", default=None)
def population_run(config, fail_on_total_mismatch, ecotype):
    from .cetaceans.killer_whales.populations.prepare import export_population_numbers

    click.echo(
        export_population_numbers(
            config_path=config,
            ecotype=ecotype,
            fail_on_total_mismatch=fail_on_total_mismatch,
        )
    )


@impute.command("fit")
@click.option(
    "--config",
    default=lambda: str(config_path()),
    type=click.Path(),
    callback=_input_path,
)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def imputation_fit(ctx, config, dry_run):
    from .tools.observations.impute.settings import load_imputation_settings
    from .cetaceans.killer_whales.observations.imputation import fit_imputation_model

    settings = load_imputation_settings(config)
    if dry_run:
        click.echo(
            json.dumps(
                {
                    "operation": "killer-whales.observations.impute.fit",
                    "observations": str(settings.observations_path),
                    "models_dir": str(settings.models_dir),
                },
                indent=2,
            )
        )
        return
    result = fit_imputation_model(
        observations_path=settings.observations_path,
        associations_path=settings.associations_path,
        models_dir=settings.models_dir,
        run_id=ctx.find_root().obj["run_id"],
        config=settings.config,
        source_config_path=settings.config_path,
        evaluate_strategies=settings.evaluate_strategies,
    )
    click.echo(result.model_path)


@post_process.command("all")
@click.option(
    "--config",
    default=lambda: str(config_path()),
    type=click.Path(),
    callback=_input_path,
)
@click.option(
    "--input-manifest", required=True, type=click.Path(), callback=_input_path
)
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--force", is_flag=True)
@click.option("--resume", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def post_process_all(
    ctx, config, input_manifest, start_date, end_date, force, resume, dry_run
):
    from .tools.observations.post_process.pipeline import post_process_observations
    from .cetaceans.killer_whales.configuration import load_sightings_config

    document, settings = load_sightings_config(config)
    if dry_run:
        _show_sightings_plan(
            ctx, "killer-whales.observations.post-process.all", config, input_manifest
        )
        return
    roots = ctx.find_root().obj
    results = post_process_observations(
        config=document,
        inputs=_manifest_artifacts(input_manifest),
        parent_inputs=_manifest_inputs(input_manifest),
        water_universes=_sightings_universe_artifacts(
            Path(roots["data_root"]), settings.h3_resolutions
        ),
        data_root=Path(roots["data_root"]),
        artifact_root=Path(roots["artifact_root"]),
        output_root=Path(roots["output_root"]),
        run_id=roots["run_id"] or "sightings-post-process",
        resolutions=settings.h3_resolutions,
        frequencies=settings.frequencies,
        start_date=start_date.date() if start_date else None,
        end_date=end_date.date() if end_date else None,
        force=force,
        resume=resume,
    )
    for result in results:
        for artifact in result.outputs:
            click.echo(artifact.path)


def _common_options(function: Callable[..., Any]) -> Callable[..., Any]:
    options = [
        click.option(
            "--force", is_flag=True, help="Replace an existing product explicitly."
        ),
        click.option(
            "--resume", is_flag=True, help="Skip products validated by a run manifest."
        ),
        click.option(
            "--dry-run",
            is_flag=True,
            help="Resolve and display the operation without writing.",
        ),
        click.option(
            "--config",
            default=lambda: str(config_path()),
            type=click.Path(dir_okay=False),
            callback=_input_path,
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _show_plan(ctx: click.Context, operation: str, config: str) -> None:
    document = ConfigDocument.load(config)
    roots = ctx.find_root().obj
    click.echo(f"operation: {operation}")
    click.echo(f"config: {document.source}")
    click.echo(f"config_hash: {document.config_hash}")
    for key in ("data_root", "artifact_root", "output_root", "run_id"):
        click.echo(f"{key}: {roots.get(key)}")


def _manifest_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if "outputs" not in payload and payload.get("manifest"):
        target = Path(payload["manifest"])
        if not target.is_absolute():
            target = Path(path).resolve().parent / target
        payload = json.loads(target.read_text())
    return payload


def _manifest_artifacts(path: str) -> tuple[Any, ...]:
    from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef

    payload = _manifest_payload(path)
    return tuple(ArtifactRef.from_dict(item) for item in payload.get("outputs", ()))


def _manifest_inputs(path: str) -> tuple[Any, ...]:
    from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef

    return tuple(
        ArtifactRef.from_dict(item)
        for item in _manifest_payload(path).get("inputs", ())
    )


def _sightings_universe_artifacts(
    data_root: Path, resolutions: tuple[int, ...]
) -> tuple[Any, ...]:
    from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef

    root = data_root / "processed/environment/seascape/full_counting"
    manifest = root / "_dataset_manifest.json"
    if not manifest.exists():
        raise click.ClickException(
            f"Missing full counting universe manifest: {manifest}. "
            "Supply a validated toolkit-seascape full-counting universe."
        )
    registered = {
        item.dataset_id: item
        for item in (
            ArtifactRef.from_dict(payload)
            for payload in json.loads(manifest.read_text()).get("outputs", ())
        )
    }
    artifacts = []
    for resolution in resolutions:
        dataset_id = f"environment.seascape.h3_full_counting_universe_r{resolution}"
        artifact = registered.get(dataset_id)
        if artifact is None or not artifact.path.exists():
            raise click.ClickException(
                f"Missing registered full counting universe H{resolution}. "
                "Supply a validated toolkit-seascape full-counting universe."
            )
        artifacts.append(artifact)
    return tuple(artifacts)


def _show_sightings_plan(
    ctx: click.Context, operation: str, config: str, input_manifest: str | None = None
) -> None:
    from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
        load_sightings_config,
    )

    document, settings = load_sightings_config(config)
    roots = ctx.find_root().obj
    click.echo(f"operation: {operation}")
    click.echo(f"config: {document.source}")
    click.echo(f"config_hash: {document.config_hash}")
    click.echo(f"full_area: {settings.full_area.tuple()}")
    click.echo(
        "model_universes: "
        + ", ".join(
            f"{name}={document.resolve_path(universe.polygon)} "
            f"(provenance={document.resolve_path(universe.polygon).with_suffix('.metadata.json')})"
            for name, universe in settings.model_universes.items()
        )
    )
    click.echo(f"sources: {', '.join(settings.collection.sources)}")
    click.echo(f"h3_resolutions: {settings.h3_resolutions}")
    click.echo(f"frequencies: {settings.frequencies}")
    click.echo(
        "imputation_model_dir: "
        f"{document.resolve_path(settings.imputation.artifacts.models_dir)}"
    )
    click.echo(
        "imputation_output: "
        f"{document.resolve_path(settings.imputation.artifacts.output)}"
    )
    click.echo("water_universes: explicit registered ArtifactRef inputs (H4-H6)")
    click.echo(f"dest_data_root: {roots['data_root']}")
    if input_manifest:
        click.echo(f"input_manifest: {Path(input_manifest).resolve()}")
        for artifact in _manifest_artifacts(input_manifest):
            click.echo(f"input: {artifact.dataset_id} -> {artifact.path}")


@observations.command("run")
@click.option(
    "--profile",
    type=click.Choice(
        [
            "production-retrospective",
            "imputation-only",
            "authoritative-counts",
            "research-h6",
            "observed-only",
        ]
    ),
    default="production-retrospective",
    show_default=True,
)
@click.option("--start-date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--config",
    default=lambda: str(config_path()),
    show_default=True,
    type=click.Path(dir_okay=False),
    callback=_input_path,
)
@click.option(
    "--twm-file", multiple=True, type=click.Path(dir_okay=False), callback=_input_path
)
@click.option(
    "--offline", is_flag=True, help="Replay the latest complete collected cohort."
)
@click.option("--resume", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def run_sightings_release(
    ctx: click.Context,
    profile: str,
    start_date: Any,
    end_date: Any,
    config: str,
    twm_file: tuple[str, ...],
    offline: bool,
    resume: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Build sightings in isolation and atomically promote one generation."""

    from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
        load_sightings_config,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
        release_profile,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import (
        SightingsPipelineRunRequest,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import (
        SightingsReleaseBlocked,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import (
        estimate_sightings_build,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import (
        run_sightings_pipeline,
    )

    roots = ctx.find_root().obj
    end = end_date.date()
    document, settings = load_sightings_config(config)
    start = start_date.date() if start_date else date.fromisoformat(settings.min_date)
    selected_profile = release_profile(profile)
    if dry_run:
        estimate = estimate_sightings_build(
            data_root=Path(roots["data_root"]),
            profile=selected_profile,
            start_date=start,
            end_date=end,
        )
        estimate.update(
            {
                "operation": "killer-whales.observations.run",
                "config": str(document.source),
                "config_hash": document.config_hash,
                "offline": offline,
                "release_pointer": str(
                    Path(roots["data_root"])
                    / "processed/domain/whale_layer/sightings/releases/latest.json"
                ),
            }
        )
        click.echo(json.dumps(estimate, indent=2, sort_keys=True))
        return
    try:
        result = run_sightings_pipeline(
            SightingsPipelineRunRequest(
                config=Path(config),
                data_root=Path(roots["data_root"]),
                artifact_root=Path(roots["artifact_root"]),
                output_root=Path(roots["output_root"]),
                start_date=start_date.date() if start_date else None,
                end_date=end,
                profile=profile,
                run_id=roots["run_id"],
                offline=offline,
                twm_files=tuple(Path(item) for item in twm_file),
                force=force,
                resume=resume,
            )
        )
    except SightingsReleaseBlocked as exc:
        raise click.ClickException(
            f"{exc}. Candidate report: {exc.candidate_report}. "
            "The prior release pointer was not changed."
        ) from exc
    click.echo(result.release_manifest)


@observations.command("collect")
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--twm-file", multiple=True, type=click.Path(dir_okay=False), callback=_input_path
)
@click.option(
    "--offline", is_flag=True, help="Use the latest immutable local source snapshot."
)
@click.option(
    "--full-refresh", is_flag=True, help="Ignore watermarks and recollect full history."
)
@_common_options
@click.pass_context
def collect_sightings(
    ctx: click.Context,
    config: str,
    dry_run: bool,
    resume: bool,
    force: bool,
    start_date: Any,
    end_date: Any,
    twm_file: tuple[str, ...],
    offline: bool,
    full_refresh: bool,
) -> None:
    if dry_run:
        _show_sightings_plan(ctx, "data.collect.sightings", config)
        return
    from marine_mammal_toolkit.tools._core.config import ConfigDocument
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        SightingsCollectionRequest,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        collect as run,
    )

    roots = ctx.find_root().obj
    run(
        SightingsCollectionRequest(
            config=ConfigDocument.load(config),
            data_root=Path(roots["data_root"]),
            artifact_root=Path(roots["artifact_root"]),
            output_root=Path(roots["output_root"]),
            run_id=roots["run_id"] or "sightings-collect",
            force=force,
            resume=resume,
            offline=offline,
            start_date=start_date.date() if start_date else None,
            end_date=end_date.date() if end_date else None,
            twm_files=tuple(Path(path) for path in twm_file),
            full_refresh=full_refresh,
        )
    )


@observations.command("process")
@click.option(
    "--input-manifest",
    required=False,
    type=click.Path(dir_okay=False),
    callback=_input_path,
)
@_common_options
@click.pass_context
def process_sightings(
    ctx: click.Context,
    config: str,
    dry_run: bool,
    resume: bool,
    force: bool,
    input_manifest: str | None,
) -> None:
    if dry_run:
        _show_sightings_plan(ctx, "data.process.sightings", config, input_manifest)
        return
    from marine_mammal_toolkit.tools._core.config import ConfigDocument
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        NormalizationRequest,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        process as run,
    )

    if not input_manifest:
        raise click.UsageError(
            "--input-manifest from `data collect sightings` is required"
        )
    roots = ctx.find_root().obj
    run(
        NormalizationRequest(
            config=ConfigDocument.load(config),
            inputs=_manifest_artifacts(input_manifest),
            data_root=Path(roots["data_root"]),
            artifact_root=Path(roots["artifact_root"]),
            output_root=Path(roots["output_root"]),
            run_id=roots["run_id"] or "sightings-process",
            force=force,
            resume=resume,
        )
    )


@impute.command("apply")
@click.option(
    "--input-manifest",
    required=True,
    type=click.Path(dir_okay=False),
    callback=_input_path,
)
@click.option(
    "--model",
    "model_path",
    default=None,
    type=click.Path(),
    callback=_input_path,
    help="Optional override for imputation.artifacts.models_dir.",
)
@click.option(
    "--mode", type=click.Choice(["retrospective", "as_of"]), default="retrospective"
)
@click.option("--knowledge-cutoff")
@_common_options
@click.pass_context
def impute_sightings(
    ctx: click.Context,
    input_manifest: str,
    model_path: str | None,
    mode: str,
    knowledge_cutoff: str | None,
    config: str,
    dry_run: bool,
    resume: bool,
    force: bool,
) -> None:
    if dry_run:
        _show_sightings_plan(ctx, "data.impute.sightings", config, input_manifest)
        return
    from marine_mammal_toolkit.tools._core.data import ProcessingMode
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        ImputationRequest,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import impute as run
    from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
        load_sightings_config,
    )

    artifacts = _manifest_artifacts(input_manifest)
    observations = next(
        (
            item
            for item in artifacts
            if item.dataset_id == "whale.sightings.observations"
        ),
        None,
    )
    associations = next(
        (
            item
            for item in artifacts
            if item.dataset_id == "whale.sightings.associations"
        ),
        None,
    )
    if observations is None:
        raise click.ClickException(
            "Input manifest has no normalized observations artifact"
        )
    if mode == "as_of" and not knowledge_cutoff:
        raise click.UsageError("--knowledge-cutoff is required for as_of imputation")
    document, settings = load_sightings_config(config)
    configured_model = document.resolve_path(settings.imputation.artifacts.models_dir)
    roots = ctx.find_root().obj
    result = run(
        ImputationRequest(
            config=document,
            observations=observations,
            associations=associations,
            model_path=Path(model_path).resolve() if model_path else configured_model,
            data_root=Path(roots["data_root"]),
            artifact_root=Path(roots["artifact_root"]),
            output_root=Path(roots["output_root"]),
            run_id=roots["run_id"] or "sightings-impute",
            mode=ProcessingMode(mode),
            knowledge_cutoff=knowledge_cutoff,
            force=force,
            resume=resume,
        )
    )
    for artifact in result.outputs:
        click.echo(artifact.path)


@observations.command("validate")
@click.argument("dataset_id", default="all")
@click.option("--manifest", type=click.Path(dir_okay=False), callback=_input_path)
@click.option("--require-public-eligible", is_flag=True)
@click.pass_context
def data_validate(
    ctx: click.Context,
    dataset_id: str,
    manifest: str | None,
    require_public_eligible: bool,
) -> None:
    """Validate one registered canonical artifact, or every existing artifact."""
    from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
    from marine_mammal_toolkit.tools._core.data import DATASETS
    from marine_mammal_toolkit.tools.quality.tables import validate_path
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        validate as validate_sightings,
    )

    roots = ctx.find_root().obj
    if dataset_id == "sightings-release":
        from marine_mammal_toolkit.cetaceans.killer_whales.observations.release import (
            validate_sightings_release,
        )

        selected = (
            Path(manifest)
            if manifest
            else (
                Path(roots["data_root"])
                / "processed/domain/whale_layer/sightings/releases/latest.json"
            )
        )
        report = validate_sightings_release(
            selected, require_public_eligible=require_public_eligible
        )
        click.echo(
            f"{'PASS' if report.valid else 'FAIL'}\twhale.sightings.release\t{selected}"
        )
        for error in report.errors:
            click.echo(f"ERROR\t{error}")
        for warning in report.warnings:
            click.echo(f"WARN\t{warning}")
        if not report.valid:
            raise click.ClickException("Sightings release failed validation")
        return
    if manifest is not None or require_public_eligible:
        raise click.UsageError(
            "--manifest and --require-public-eligible apply only to sightings-release"
        )
    specs = tuple(DATASETS) if dataset_id == "all" else (DATASETS.get(dataset_id),)
    invalid = 0
    for spec in specs:
        path = spec.path(
            data_root=Path(roots["data_root"]),
            artifact_root=Path(roots["artifact_root"]),
            output_root=Path(roots["output_root"]),
        )
        if dataset_id == "all" and not path.exists():
            continue
        report = (
            validate_sightings(
                ArtifactRef(
                    kind=spec.layer.value,
                    dataset_id=str(spec.dataset_id),
                    path=path,
                    producer="marine-mammals.validate",
                )
            )
            if str(spec.dataset_id).startswith("whale.sightings.")
            else validate_path(path, spec)
        )
        click.echo(f"{'PASS' if report.valid else 'FAIL'}\t{spec.dataset_id}\t{path}")
        invalid += int(not report.valid)
    if invalid:
        raise click.ClickException(f"{invalid} dataset(s) failed validation")


@post_process.command("model-domains")
@click.option(
    "--domain-config",
    default=lambda: str(config_path("model_domains")),
    show_default=True,
    type=click.Path(dir_okay=False),
    callback=_input_path,
)
@_common_options
@click.pass_context
def build_sightings_model_domains(
    ctx: click.Context,
    domain_config: str,
    config: str,
    dry_run: bool,
    resume: bool,
    force: bool,
) -> None:
    """Build reviewed operational SRKW and Transient model domains."""

    del resume
    if dry_run:
        _show_sightings_plan(ctx, "data.build.sightings.model-domains", config)
        click.echo(f"domain_config: {Path(domain_config).resolve()}")
        return
    from marine_mammal_toolkit.cetaceans.killer_whales.observations.domains import (
        build_operational_model_domains,
    )

    roots = ctx.find_root().obj
    result = build_operational_model_domains(
        sightings_config_path=config,
        domain_config_path=domain_config,
        data_root=Path(roots["data_root"]),
        run_id=roots["run_id"] or "sightings-model-domains-v1",
        force=force,
    )
    for path in (*result.polygon_paths, *result.metadata_paths, result.manifest_path):
        click.echo(path)


@post_process.command("counts")
@click.option(
    "--input-manifest",
    required=True,
    type=click.Path(dir_okay=False),
    callback=_input_path,
)
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--resolution", "resolutions", multiple=True, type=click.IntRange(4, 6))
@click.option(
    "--mode", type=click.Choice(["retrospective", "as_of"]), default="retrospective"
)
@click.option("--knowledge-cutoff")
@_common_options
@click.pass_context
def build_sightings_counts(
    ctx: click.Context,
    input_manifest: str,
    start_date: Any,
    end_date: Any,
    resolutions: tuple[int, ...],
    mode: str,
    knowledge_cutoff: str | None,
    config: str,
    dry_run: bool,
    resume: bool,
    force: bool,
) -> None:
    if dry_run:
        _show_sightings_plan(ctx, "data.build.sightings.counts", config, input_manifest)
        return
    from marine_mammal_toolkit.tools._core.config import ConfigDocument
    from marine_mammal_toolkit.tools._core.data import ProcessingMode
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import CountRequest
    from marine_mammal_toolkit.tools.observations.post_process import counts as run

    artifacts = _manifest_artifacts(input_manifest)
    observations = next(
        (
            item
            for item in artifacts
            if item.dataset_id
            in {
                "whale.sightings.observations",
                "whale.sightings.imputed_retrospective",
                "whale.sightings.imputed_as_of",
            }
        ),
        None,
    )
    associations = next(
        (
            item
            for item in artifacts
            if item.dataset_id == "whale.sightings.associations"
        ),
        None,
    )
    if associations is None:
        associations = next(
            (
                item
                for item in _manifest_inputs(input_manifest)
                if item.dataset_id == "whale.sightings.associations"
            ),
            None,
        )
    if observations is None:
        raise click.ClickException(
            "Input manifest has no normalized or imputed observations artifact"
        )
    if mode == "as_of" and not knowledge_cutoff:
        raise click.UsageError("--knowledge-cutoff is required for as_of counts")
    roots = ctx.find_root().obj
    selected = resolutions or (4, 5, 6)
    run(
        CountRequest(
            config=ConfigDocument.load(config),
            observations=observations,
            associations=associations,
            water_universes=_sightings_universe_artifacts(
                Path(roots["data_root"]), selected
            ),
            data_root=Path(roots["data_root"]),
            artifact_root=Path(roots["artifact_root"]),
            output_root=Path(roots["output_root"]),
            run_id=roots["run_id"] or "sightings-counts-v4",
            start_date=start_date.date() if start_date else None,
            end_date=end_date.date() if end_date else None,
            resolutions=selected,
            mode=ProcessingMode(mode),
            knowledge_cutoff=knowledge_cutoff,
            force=force,
            resume=resume,
        )
    )


@post_process.command("model-grid")
@click.option(
    "--input-manifest",
    required=True,
    type=click.Path(dir_okay=False),
    callback=_input_path,
)
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--resolution", "resolutions", multiple=True, type=click.IntRange(4, 6))
@click.option(
    "--frequency", "frequencies", multiple=True, type=click.Choice(["daily", "weekly"])
)
@click.option(
    "--mode", type=click.Choice(["retrospective", "as_of"]), default="retrospective"
)
@_common_options
@click.pass_context
def build_sightings_model_grid(
    ctx: click.Context,
    input_manifest: str,
    start_date: Any,
    end_date: Any,
    resolutions: tuple[int, ...],
    frequencies: tuple[str, ...],
    mode: str,
    config: str,
    dry_run: bool,
    resume: bool,
    force: bool,
) -> None:
    if dry_run:
        _show_sightings_plan(
            ctx, "data.build.sightings.model-grid", config, input_manifest
        )
        return
    from marine_mammal_toolkit.tools._core.config import ConfigDocument
    from marine_mammal_toolkit.tools._core.data import ProcessingMode
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        ModelGridRequest,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        model_grid as run,
    )

    counts_artifact = next(
        (
            item
            for item in _manifest_artifacts(input_manifest)
            if item.dataset_id == "whale.sightings.ecotype_counts"
        ),
        None,
    )
    if counts_artifact is None:
        raise click.ClickException("Input manifest has no ecotype_counts artifact")
    roots = ctx.find_root().obj
    selected = resolutions or (4, 5, 6)
    run(
        ModelGridRequest(
            config=ConfigDocument.load(config),
            ecotype_counts=counts_artifact,
            water_universes=_sightings_universe_artifacts(
                Path(roots["data_root"]), selected
            ),
            data_root=Path(roots["data_root"]),
            artifact_root=Path(roots["artifact_root"]),
            output_root=Path(roots["output_root"]),
            run_id=roots["run_id"] or "sightings-model-grid-v4",
            start_date=start_date.date() if start_date else None,
            end_date=end_date.date() if end_date else None,
            resolutions=selected,
            frequencies=frequencies or ("daily", "weekly"),
            mode=ProcessingMode(mode),
            knowledge_cutoff=counts_artifact.knowledge_cutoff,
            force=force,
            resume=resume,
        )
    )


@post_process.command("intensity")
@click.option(
    "--input-manifest",
    required=True,
    type=click.Path(dir_okay=False),
    callback=_input_path,
)
@_common_options
@click.pass_context
def build_sightings_intensity(
    ctx: click.Context,
    input_manifest: str,
    config: str,
    dry_run: bool,
    resume: bool,
    force: bool,
) -> None:
    if dry_run:
        _show_sightings_plan(
            ctx, "data.build.sightings.intensity", config, input_manifest
        )
        return
    from marine_mammal_toolkit.tools._core.config import ConfigDocument
    from marine_mammal_toolkit.tools._core.data import ProcessingMode
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        IntensityRequest,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
        intensity as run,
    )

    grid = next(
        (
            item
            for item in _manifest_artifacts(input_manifest)
            if item.dataset_id == "whale.sightings.reported_sighting_grid"
        ),
        None,
    )
    if grid is None:
        raise click.ClickException(
            "Input manifest has no reported_sighting_grid artifact"
        )
    roots = ctx.find_root().obj
    run(
        IntensityRequest(
            config=ConfigDocument.load(config),
            model_grid=grid,
            data_root=Path(roots["data_root"]),
            artifact_root=Path(roots["artifact_root"]),
            output_root=Path(roots["output_root"]),
            run_id=roots["run_id"] or "sightings-intensity-v4",
            mode=ProcessingMode(grid.processing_mode or "retrospective"),
            knowledge_cutoff=grid.knowledge_cutoff,
            force=force,
            resume=resume,
        )
    )
