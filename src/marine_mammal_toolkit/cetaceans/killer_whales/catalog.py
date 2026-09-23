from __future__ import annotations

import pyarrow as pa

from marine_mammal_toolkit.tools.schemas.stages import DatasetFormat
from marine_mammal_toolkit.tools.schemas.stages import DatasetId
from marine_mammal_toolkit.tools.schemas.stages import DatasetLayer
from marine_mammal_toolkit.tools.schemas.stages import DatasetSpec
from marine_mammal_toolkit.tools.schemas.stages import ProcessingMode
from marine_mammal_toolkit.tools._core.registry import DATASETS

SIGHTINGS_SCHEMA = pa.schema(
    [
        pa.field("OBSERVATION_ID", pa.string(), nullable=False),
        pa.field("OBSERVATION_IDS", pa.string()),
        pa.field("DATETIME", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("LATITUDE", pa.float64(), nullable=False),
        pa.field("LONGITUDE", pa.float64(), nullable=False),
        pa.field("SOURCE", pa.string(), nullable=False),
        pa.field("SPECIES", pa.string()),
        pa.field("TYPE", pa.string()),
        pa.field("POD", pa.string()),
        pa.field("MEMBER", pa.string()),
        pa.field("SOURCE_OBSERVATION_ID", pa.string()),
        pa.field("SOURCE_OBSERVATION_IDS", pa.string()),
        pa.field("GLOBAL_OBSERVATION_ID", pa.string(), nullable=False),
    ]
)


def _register(
    dataset_id: str,
    layer: DatasetLayer,
    format: DatasetFormat,
    path: str,
    producer: str,
    *,
    dependencies: tuple[str, ...] = (),
    schema: pa.Schema | None = None,
    primary_key: tuple[str, ...] = (),
    partition_keys: tuple[str, ...] = (),
    modes: tuple[ProcessingMode, ...] | None = None,
    schema_version: str = "1",
) -> None:
    DATASETS.register(
        DatasetSpec(
            dataset_id=DatasetId(dataset_id),
            layer=layer,
            format=format,
            path_template=path,
            producer=producer,
            dependencies=tuple(DatasetId(item) for item in dependencies),
            schema=schema,
            primary_key=primary_key,
            partition_keys=partition_keys,
            schema_version=schema_version,
            allowed_modes=modes or (ProcessingMode.RETROSPECTIVE, ProcessingMode.AS_OF),
        )
    )


def register_builtin_datasets() -> None:
    if tuple(DATASETS):
        return
    for source in ("twm", "acartia", "maplify", "inaturalist", "cwr", "gbif"):
        _register(
            f"whale.sightings.source_{source}",
            DatasetLayer.SOURCE,
            DatasetFormat.DIRECTORY,
            f"{{data_root}}/raw/whale/sightings/{source}/snapshots",
            "whale.sightings.collect",
            schema_version="5",
        )
    for state_name, relative_path, primary_key in (
        (
            "source_record_history",
            "state/source_history.parquet",
            ("SOURCE_RECORD_ID", "RETRIEVAL_ID", "PAYLOAD_CHECKSUM"),
        ),
        ("source_records", "state/source_current.parquet", ("SOURCE_RECORD_ID",)),
    ):
        _register(
            f"whale.sightings.{state_name}",
            DatasetLayer.NORMALIZED,
            DatasetFormat.PARQUET,
            f"{{data_root}}/processed/sightings/normalized/{relative_path}",
            "whale.sightings.normalize.v9",
            dependencies=(
                "whale.sightings.source_twm",
                "whale.sightings.source_acartia",
                "whale.sightings.source_maplify",
                "whale.sightings.source_inaturalist",
                "whale.sightings.source_cwr",
                "whale.sightings.source_gbif",
            ),
            primary_key=primary_key,
            schema_version="9",
        )
    for name, relative_path, primary_key in (
        ("observations", "observations.parquet", ("OBSERVATION_ID",)),
        (
            "associations",
            "associations.parquet",
            (
                "OBSERVATION_ID",
                "SOURCE_RECORD_ID",
                "ASSOCIATION_KIND",
                "ASSOCIATION_VALUE",
                "RULE_ID",
                "EVIDENCE_FIELD",
            ),
        ),
        ("normalization_audit", "audit.parquet", ()),
        (
            "identity_resolution",
            "state/identity/assignments.parquet",
            ("SOURCE_RECORD_ID",),
        ),
        (
            "identity_aliases",
            "state/identity/aliases.parquet",
            ("ALIAS_OBSERVATION_ID",),
        ),
        ("identity_lineage", "state/identity/lineage.parquet", ()),
    ):
        _register(
            f"whale.sightings.{name}",
            DatasetLayer.NORMALIZED,
            DatasetFormat.PARQUET,
            f"{{data_root}}/processed/sightings/normalized/{relative_path}",
            "whale.sightings.normalize.v9",
            dependencies=("whale.sightings.source_records",),
            primary_key=primary_key,
            schema_version="9",
        )
    for mode in ProcessingMode:
        suffix = mode.value
        _register(
            f"whale.sightings.imputed_{suffix}",
            DatasetLayer.DOMAIN,
            DatasetFormat.PARQUET,
            f"{{data_root}}/processed/sightings/imputed/imputed_{suffix}.parquet",
            "whale.sightings.impute",
            dependencies=(
                "whale.sightings.observations",
                "whale.sightings.associations",
            ),
            primary_key=("OBSERVATION_ID",),
            modes=(mode,),
            schema_version="9",
        )
    count_keys = {
        "ecotype_counts": (
            "H3_INDEX",
            "H3_RESOLUTION",
            "FREQUENCY",
            "PERIOD_START",
            "ECOTYPE_BUCKET",
        ),
        "ecotype_detail_counts": (
            "H3_INDEX",
            "H3_RESOLUTION",
            "FREQUENCY",
            "PERIOD_START",
            "ECOTYPE_DETAIL",
        ),
        "orca_total_counts": ("H3_INDEX", "H3_RESOLUTION", "FREQUENCY", "PERIOD_START"),
        "pod_counts": (
            "H3_INDEX",
            "H3_RESOLUTION",
            "FREQUENCY",
            "PERIOD_START",
            "POD",
        ),
        "period_totals": ("FREQUENCY", "PERIOD_START"),
        "count_exclusions": ("OBSERVATION_ID", "H3_RESOLUTION"),
    }
    for count_name, primary_key in count_keys.items():
        _register(
            f"whale.sightings.{count_name}",
            DatasetLayer.DOMAIN,
            DatasetFormat.DIRECTORY,
            f"{{data_root}}/processed/sightings/final/counts/mode=retrospective/{count_name}",
            "whale.sightings.counts.v7",
            dependencies=(
                "whale.sightings.observations",
                "whale.sightings.associations",
                "environment.seascape.h3_full_counting_universe_r4",
                "environment.seascape.h3_full_counting_universe_r5",
                "environment.seascape.h3_full_counting_universe_r6",
            ),
            primary_key=primary_key,
            schema_version="7",
        )
    _register(
        "whale.sightings.reported_sighting_grid",
        DatasetLayer.DOMAIN,
        DatasetFormat.DIRECTORY,
        "{data_root}/processed/sightings/final/dense/mode=retrospective/reported_sighting",
        "whale.sightings.model_grid.v7",
        dependencies=(
            "whale.sightings.ecotype_counts",
            "environment.seascape.h3_full_counting_universe_r4",
            "environment.seascape.h3_full_counting_universe_r5",
            "environment.seascape.h3_full_counting_universe_r6",
        ),
        primary_key=(
            "H3_INDEX",
            "H3_RESOLUTION",
            "FREQUENCY",
            "PERIOD_START",
            "ECOTYPE_BUCKET",
        ),
        schema_version="7",
    )
    _register(
        "whale.sightings.relative_reported_activity",
        DatasetLayer.DOMAIN,
        DatasetFormat.DIRECTORY,
        "{data_root}/processed/sightings/final/dense/mode=retrospective/relative_reported_activity",
        "whale.sightings.intensity.v8",
        dependencies=(
            "whale.sightings.reported_sighting_grid",
            "environment.seascape.h3_full_marine_support_r6",
        ),
        primary_key=(
            "H3_INDEX",
            "H3_RESOLUTION",
            "FREQUENCY",
            "PERIOD_START",
            "ECOTYPE_BUCKET",
        ),
        schema_version="8",
    )
    _register(
        "whale.sightings.relative_intensity",
        DatasetLayer.DOMAIN,
        DatasetFormat.DIRECTORY,
        "{data_root}/processed/sightings/final/dense/mode=retrospective/relative_intensity",
        "whale.sightings.intensity.compatibility_alias.v8",
        dependencies=("whale.sightings.relative_reported_activity",),
        primary_key=(
            "H3_INDEX",
            "H3_RESOLUTION",
            "FREQUENCY",
            "PERIOD_START",
            "ECOTYPE_BUCKET",
        ),
        schema_version="8",
    )


register_builtin_datasets()
