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


def _require_extra(extra: str, modules: tuple[str, ...]) -> None:
    from importlib.util import find_spec

    missing = [name for name in modules if find_spec(name) is None]
    if missing:
        raise click.ClickException(
            f"Missing optional dependencies: {', '.join(missing)}. "
            f"Install marine-mammal-toolkit[{extra}] for this command."
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


def _optional_twm_path(ctx, parameter, value):
    if isinstance(value, tuple):
        return tuple(_optional_twm_path(ctx, parameter, item) for item in value)
    return str(
        (
            Path(ctx.find_root().obj["workspace_root"]) / Path(value).expanduser()
        ).resolve()
    )


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


@observations.command("sources")
def list_observation_sources():
    """List providers, query capabilities, and configured GBIF datasets."""
    from .cetaceans.killer_whales.query import SOURCE_CATALOG
    from .cetaceans.killer_whales.configuration import load_sightings_config

    _, settings = load_sightings_config(config_path())
    click.echo(
        json.dumps(
            {
                "sources": SOURCE_CATALOG,
                "gbif_datasets": [
                    item.model_dump()
                    for item in settings.collection.sources["gbif"].dataset_allowlist
                ],
            },
            indent=2,
        )
    )


@observations.command("preflight")
@click.option("--config", default=lambda: str(config_path()), callback=_input_path)
@click.option("--profile", default="observations-only")
@click.pass_context
def preflight_observations_cli(ctx, config, profile):
    """Check local inputs and configuration without network access or writes."""
    from .cetaceans.killer_whales.query import preflight_observations

    try:
        result = preflight_observations(
            config,
            workspace_root=ctx.find_root().obj["workspace_root"],
            data_root=ctx.find_root().obj["data_root"],
            profile=profile,
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, indent=2))
    if not result["ready"]:
        raise click.ClickException("Preflight found missing prerequisites")


@observations.command("query")
@click.option(
    "--source",
    "sources",
    multiple=True,
    type=click.Choice(["twm", "acartia", "maplify", "inaturalist", "cwr", "gbif"]),
    help="Repeat to combine providers; default: inaturalist.",
)
@click.option(
    "--dataset",
    "dataset_keys",
    multiple=True,
    help="GBIF dataset UUID; repeat to combine datasets.",
)
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    default=(-180, 32, -109, 72),
    show_default=True,
    help="West south east north; default North Pacific observation extent.",
)
@click.option(
    "--twm-file",
    multiple=True,
    type=click.Path(dir_okay=False),
    help="Missing local TWM files warn and continue.",
)
@click.option("--config", callback=_input_path, type=click.Path(dir_okay=False))
@click.option(
    "--offline", is_flag=True, help="Replay this exact query's local source snapshots."
)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def query_observations_cli(
    ctx, sources, dataset_keys, start, end, bbox, twm_file, config, offline, dry_run
):
    """Query canonical observations without fitting models or building counts."""
    from .cetaceans.killer_whales.query import query_configuration, query_observations

    roots = ctx.find_root().obj
    arguments = dict(
        workspace_root=roots["workspace_root"],
        start=start.date(),
        end=end.date(),
        sources=sources or ("inaturalist",),
        dataset_keys=dataset_keys,
        bbox=bbox,
        twm_files=twm_file,
        config=config,
    )
    try:
        if dry_run:
            payload = query_configuration(**arguments)
            click.echo(
                json.dumps(
                    {
                        "operation": "observations.query",
                        "sources": list(payload["collection"]["sources"]),
                        "start": payload["min_date"],
                        "end": payload["max_date"],
                        "bbox": payload["full_area"],
                        "network_checked": False,
                    },
                    indent=2,
                )
            )
            return
        result = query_observations(
            **arguments, data_root=roots["data_root"], offline=offline
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.to_dict(), indent=2))


@observations.command("demo")
@click.pass_context
def observation_demo(ctx):
    """Run an offline synthetic example through the production query API."""
    from .cetaceans.killer_whales.query import run_demo

    try:
        result = run_demo(ctx.find_root().obj["workspace_root"])
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps({"synthetic": True, **result.to_dict()}, indent=2))


@observations.group()
def impute():
    """Fit or apply a certified selective imputation model."""
    _require_extra("imputation", ("seascape", "sklearn", "joblib"))


@observations.group("post-process")
def post_process():
    """Build counts, model grids, intensity, or all three sequentially."""
    _require_extra("imputation", ("geopandas", "seascape"))


@killer_whales.group()
def populations():
    """Legacy command for annual population census exports."""


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


@killer_whales.group()
def demography():
    """Validate and export killer-whale census data."""


@demography.command("census")
@click.option("--workbook", callback=_input_path, type=click.Path(dir_okay=False))
@click.option("--sheet", "sheet_name")
@click.option("--output", type=click.Path(dir_okay=False))
@click.option(
    "--config",
    default=lambda: str(config_path("populations")),
    callback=_input_path,
    type=click.Path(dir_okay=False),
)
@click.option("--fail-on-total-mismatch", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Validate and summarize without writing JSON.")
def demography_census(
    workbook, sheet_name, output, config, fail_on_total_mismatch, dry_run
):
    """Export annual SRKW J/K/L pod counts from a workbook."""
    from .cetaceans.killer_whales.demography import export_population_numbers
    from .cetaceans.killer_whales.demography import prepare_population_numbers

    arguments = dict(
        config_path=config,
        source_path=workbook,
        output_path=output,
        sheet_name=sheet_name,
        ecotype="SRKW",
    )
    try:
        if dry_run:
            payload, target = prepare_population_numbers(**arguments)
            mismatches = payload["validation"]["total_mismatch_count"]
            if fail_on_total_mismatch and mismatches:
                raise ValueError(
                    f"Found {mismatches} row(s) where all_pods does not equal "
                    "j_pod + k_pod + l_pod. No output was written."
                )
            click.echo(
                json.dumps(
                    {
                        "source": payload["source"],
                        "output": str(target),
                        "ecotype": payload["ecotype"],
                        "validation": payload["validation"],
                        "latest": payload["latest"],
                        "written": False,
                    },
                    indent=2,
                )
            )
            return
        click.echo(
            export_population_numbers(
                **arguments, fail_on_total_mismatch=fail_on_total_mismatch
            )
        )
    except (KeyError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


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
    click.echo(
        f"sources: {', '.join(name for name, source in settings.collection.sources.items() if source.enabled)}"
    )
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
            "observations-only",
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
    "--twm-file",
    multiple=True,
    type=click.Path(dir_okay=False),
    callback=_optional_twm_path,
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
        from .cetaceans.killer_whales.query import preflight_observations

        estimate = estimate_sightings_build(
            data_root=Path(roots["data_root"]),
            profile=selected_profile,
            start_date=start,
            end_date=end,
        )
        estimate.update(
            {
                "operation": "killer-whales.observations.run",
                "preflight": preflight_observations(
                    config,
                    workspace_root=roots["workspace_root"],
                    data_root=roots["data_root"],
                    profile=profile,
                ),
                "config": str(document.source),
                "config_hash": document.config_hash,
                "offline": offline,
                "release_pointer": str(
                    Path(roots["data_root"])
                    / "processed/sightings/final/releases/latest.json"
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


@observations.command("product")
@click.option(
    "--prune",
    is_flag=True,
    help="Opt in to guarded removal of older unpinned generations.",
)
@click.option(
    "--keep-generations", default=3, show_default=True, type=click.IntRange(min=1)
)
@click.option(
    "--pin-release",
    multiple=True,
    help="Release ID to protect during pruning; may be repeated.",
)
@click.option("--start-date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--config",
    type=click.Path(dir_okay=False),
    callback=_input_path,
    help=(
        "Optional sightings YAML. Omit it to use the packaged product " "configuration."
    ),
)
@click.option(
    "--full-refresh",
    is_flag=True,
    help="Ignore source watermarks and recollect the configured history.",
)
@click.option("--resume", is_flag=True)
@click.option("--force", is_flag=True)
@click.option(
    "--max-growth-fraction",
    default=0.10,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help=(
        "Retain prior product artifacts when the new sighting-count increase "
        "exceeds this fraction."
    ),
)
@click.option("--dry-run", is_flag=True)
@click.pass_context
def build_sightings_product(
    ctx: click.Context,
    start_date: Any,
    end_date: Any,
    config: str | None,
    full_refresh: bool,
    resume: bool,
    force: bool,
    max_growth_fraction: float,
    dry_run: bool,
    prune: bool,
    keep_generations: int,
    pin_release: tuple[str, ...],
) -> None:
    """Collect, impute, and publish the daily killer-whale sightings product."""
    _require_extra("imputation,report", ("seascape", "sklearn", "plotly"))

    from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
        load_sightings_config,
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
    from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import (
        SightingsPipelineRunRequest,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import (
        SightingsReleaseBlocked,
    )
    from marine_mammal_toolkit.cetaceans.killer_whales.pipeline import (
        run_sightings_pipeline,
    )

    if pin_release and not prune:
        raise click.UsageError(
            "--pin-release requires --prune; existing saved pins are always preserved"
        )
    roots = ctx.find_root().obj
    layout = sightings_product_layout(roots["data_root"])
    product_root = layout.product_root
    selected_config = config or str(config_path("sightings_product"))
    try:
        document, settings = load_sightings_config(selected_config)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Invalid sightings configuration: {exc}") from exc
    start = start_date.date() if start_date else date.fromisoformat(settings.min_date)
    end = end_date.date()
    previous_row_count = current_sightings_row_count(product_root)
    if dry_run:
        from .cetaceans.killer_whales.query import preflight_observations

        click.echo(
            json.dumps(
                {
                    "operation": "killer-whales.observations.product",
                    "preflight": preflight_observations(
                        selected_config,
                        workspace_root=roots["workspace_root"],
                        data_root=product_root,
                        profile="imputation-only",
                    ),
                    "mode": "full-refresh" if full_refresh else "incremental-update",
                    "config": str(document.source),
                    "config_hash": document.config_hash,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "product_root": str(product_root),
                    "raw_root": str(layout.raw_root),
                    "processed_root": str(layout.sightings_root),
                    "normalized_root": str(layout.normalized_root),
                    "imputed_root": str(layout.imputed_root),
                    "final_root": str(layout.final_root),
                    "artifact_root": str(layout.imputed_root),
                    "output_root": str(layout.imputed_root),
                    "composite": str(layout.final_root / "composite-sightings.parquet"),
                    "imputed": str(layout.final_root / "imputed-sightings.parquet"),
                    "model_manifest": str(
                        layout.final_root / "imputation-model-manifest.json"
                    ),
                    "report": str(layout.final_root / "sightings-report.html"),
                    "previous_row_count": previous_row_count,
                    "max_growth_fraction": max_growth_fraction,
                    "prune": prune,
                    "keep_generations": keep_generations,
                    "pinned_release_ids": pin_release,
                    "temporal_grain": "daily",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    from marine_mammal_toolkit.tools._core.locking import workspace_write_lock

    ctx.with_resource(workspace_write_lock(product_root))
    layout.prepare()
    try:
        result = run_sightings_pipeline(
            SightingsPipelineRunRequest(
                config=Path(selected_config),
                data_root=product_root,
                artifact_root=layout.imputed_root,
                output_root=layout.imputed_root,
                start_date=start_date.date() if start_date else None,
                end_date=end,
                profile="imputation-only",
                run_id=roots["run_id"],
                persistent_state=True,
                full_refresh=full_refresh,
                force=force,
                resume=resume,
            )
        )
    except SightingsReleaseBlocked as exc:
        raise click.ClickException(
            f"{exc}. Product run report: {exc.candidate_report}. "
            "The prior product aliases were not changed."
        ) from exc
    product = materialize_sightings_product(
        result.release_manifest,
        product_root=product_root,
    )
    retention = (
        prune_prior_sightings_artifacts(
            product_root=product_root,
            previous_row_count=previous_row_count,
            max_growth_fraction=max_growth_fraction,
            keep_generations=keep_generations,
            pinned_release_ids=pin_release,
        ).as_dict(product_root=product_root)
        if prune
        else {"applied": False, "reason": "pruning_not_requested", "removed_paths": []}
    )
    cleanup_completed_sightings_run(
        product_root=product_root,
        completed_run_id=result.candidate_root.name,
    )
    click.echo(
        json.dumps(
            {
                "release_manifest": str(product.release_manifest),
                "dated_root": str(product.dated_root),
                "composite": str(product.composite_path),
                "imputed": str(product.imputed_path),
                "model_manifest": str(product.model_manifest_path),
                "report": str(product.report_path),
                "retention": retention,
            },
            indent=2,
            sort_keys=True,
        )
    )


@observations.command("report")
@click.option("--composite", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--imputed", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--model-manifest", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--h3-resolution", default=6, show_default=True, type=click.IntRange(0, 15)
)
@click.option("--force", is_flag=True, help="Replace an existing HTML report.")
@click.option("--dry-run", is_flag=True)
@click.pass_context
def build_sightings_report(
    ctx: click.Context,
    composite: Path | None,
    imputed: Path | None,
    model_manifest: Path | None,
    output: Path | None,
    h3_resolution: int,
    force: bool,
    dry_run: bool,
) -> None:
    """Build the all-time sightings density map and daily count report."""
    _require_extra("report", ("plotly",))

    from marine_mammal_toolkit.cetaceans.killer_whales.observations.report import (
        build_sightings_report_html,
    )

    roots = ctx.find_root().obj
    workspace_root = Path(roots["workspace_root"])
    processed = Path(roots["data_root"]) / "processed/sightings/final"

    def resolve(value: Path | None, default: Path) -> Path:
        selected = value or default
        return (
            selected.expanduser().resolve()
            if selected.is_absolute()
            else (workspace_root / selected).resolve()
        )

    composite_path = resolve(composite, processed / "composite-sightings.parquet")
    imputed_path = resolve(imputed, processed / "imputed-sightings.parquet")
    manifest_path = resolve(
        model_manifest, processed / "imputation-model-manifest.json"
    )
    output_path = resolve(output, processed / "sightings-report.html")
    uses_product_defaults = all(
        value is None for value in (composite, imputed, model_manifest, output)
    )
    if uses_product_defaults and (processed / "latest.json").is_file():
        from .cetaceans.killer_whales.observations.product import (
            resolve_sightings_product,
        )
        from .tools._core.locking import workspace_write_lock
        from uuid import uuid4

        if not dry_run:
            ctx.with_resource(workspace_write_lock(roots["data_root"]))
        current = resolve_sightings_product(roots["data_root"], verify_report=False)
        composite_path = current.composite_path
        imputed_path = current.imputed_path
        manifest_path = current.model_manifest_path
        output_path = processed / "reports" / f"{uuid4().hex}.html"
    plan = {
        "operation": "killer-whales.observations.report",
        "composite": str(composite_path),
        "imputed": str(imputed_path),
        "model_manifest": str(manifest_path),
        "output": str(output_path),
        "h3_resolution": h3_resolution,
    }
    if dry_run:
        click.echo(json.dumps(plan, indent=2, sort_keys=True))
        return
    try:
        report_path = build_sightings_report_html(
            composite_path=composite_path,
            imputed_path=imputed_path,
            model_manifest_path=manifest_path,
            output_path=output_path,
            h3_resolution=h3_resolution,
            overwrite=force,
        )
        if uses_product_defaults:
            from marine_mammal_toolkit.cetaceans.killer_whales.observations.product import (
                record_sightings_report,
            )

            record_sightings_report(
                product_root=Path(roots["data_root"]), report_path=report_path
            )
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(report_path)


@observations.command("collect")
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option(
    "--twm-file",
    multiple=True,
    type=click.Path(dir_okay=False),
    callback=_optional_twm_path,
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
                / "processed/sightings/final/releases/latest.json"
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
