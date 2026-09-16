from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from marine_mammal_toolkit.tools.schemas.artifacts import ArtifactRef
from marine_mammal_toolkit.tools.schemas.artifacts import DataSnapshotMetadata
from marine_mammal_toolkit.tools.schemas.artifacts import RunManifest
from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_json
from marine_mammal_toolkit.tools.schemas.artifacts import checksum_path

from marine_mammal_toolkit.tools.schemas.stages import DatasetSpec
from marine_mammal_toolkit.tools.schemas.stages import ProcessingMode
from marine_mammal_toolkit.tools.schemas.stages import ValidationReport
from marine_mammal_toolkit.tools.quality.tables import validate_path


class ArtifactStore:
    def __init__(self, *, data_root: Path, artifact_root: Path, output_root: Path):
        self.data_root = data_root.resolve()
        self.artifact_root = artifact_root.resolve()
        self.output_root = output_root.resolve()

    def destination(self, spec: DatasetSpec) -> Path:
        return spec.path(
            data_root=self.data_root,
            artifact_root=self.artifact_root,
            output_root=self.output_root,
        )

    def write_table(
        self,
        table: pa.Table | pl.DataFrame,
        spec: DatasetSpec,
        *,
        run_id: str,
        producer: str,
        config_hash: str,
        inputs: tuple[ArtifactRef, ...] = (),
        mode: ProcessingMode = ProcessingMode.RETROSPECTIVE,
        knowledge_cutoff: str | None = None,
        data_snapshot: DataSnapshotMetadata | None = None,
        force: bool = False,
    ) -> tuple[ArtifactRef, ValidationReport]:
        destination = self.destination(spec)
        if destination.exists() and not force:
            raise FileExistsError(
                f"Artifact already exists; pass --force: {destination}"
            )
        staging = self.data_root / ".staging" / run_id / str(spec.dataset_id)
        candidate = staging / destination.name
        staging.mkdir(parents=True, exist_ok=True)
        if isinstance(table, pl.DataFrame):
            table.write_parquet(candidate, compression="zstd")
            row_count = table.height
        else:
            pq.write_table(table, candidate, compression="zstd")
            row_count = table.num_rows
        report = validate_path(candidate, spec)
        if not report.valid:
            quarantine = self.data_root / "quarantine" / str(spec.dataset_id) / run_id
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            if quarantine.exists():
                shutil.rmtree(quarantine)
            os.replace(staging, quarantine)
            atomic_write_json(
                quarantine / "validation.json", report.__dict__, overwrite=True
            )
            report.require_valid()
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, destination)
        shutil.rmtree(staging, ignore_errors=True)
        ref = ArtifactRef(
            kind=spec.layer.value,
            dataset_id=str(spec.dataset_id),
            path=destination,
            producer=producer,
            schema_version=spec.schema_version,
            run_id=run_id,
            config_hash=config_hash,
            checksum=checksum_path(destination),
            row_count=row_count,
            file_count=1,
            processing_mode=mode.value,
            knowledge_cutoff=knowledge_cutoff,
            data_snapshot=data_snapshot,
            inputs=tuple(item.checksum or str(item.path) for item in inputs),
            sensitivity=spec.sensitivity,
        )
        return ref, report

    def reference_existing(
        self,
        spec: DatasetSpec,
        *,
        run_id: str,
        producer: str,
        config_hash: str,
        row_count: int,
        inputs: tuple[ArtifactRef, ...] = (),
        mode: ProcessingMode = ProcessingMode.RETROSPECTIVE,
        knowledge_cutoff: str | None = None,
        data_snapshot: DataSnapshotMetadata | None = None,
    ) -> tuple[ArtifactRef, ValidationReport]:
        """Reference an unchanged canonical artifact without rewriting it."""

        destination = self.destination(spec)
        report = validate_path(destination, spec)
        report.require_valid()
        checksum = checksum_path(destination)
        file_count = (
            len(list(destination.rglob("*.parquet"))) if destination.is_dir() else 1
        )
        return (
            ArtifactRef(
                kind=spec.layer.value,
                dataset_id=str(spec.dataset_id),
                path=destination,
                producer=producer,
                schema_version=spec.schema_version,
                run_id=run_id,
                config_hash=config_hash,
                checksum=checksum,
                row_count=row_count,
                file_count=file_count,
                processing_mode=mode.value,
                knowledge_cutoff=knowledge_cutoff,
                data_snapshot=data_snapshot,
                inputs=tuple(item.checksum or str(item.path) for item in inputs),
                sensitivity=spec.sensitivity,
            ),
            report,
        )

    def append_table(
        self,
        table: pl.DataFrame,
        spec: DatasetSpec,
        *,
        run_id: str,
        producer: str,
        config_hash: str,
        row_count: int,
        inputs: tuple[ArtifactRef, ...] = (),
        mode: ProcessingMode = ProcessingMode.RETROSPECTIVE,
        knowledge_cutoff: str | None = None,
        data_snapshot: DataSnapshotMetadata | None = None,
    ) -> tuple[ArtifactRef, ValidationReport]:
        """Append a validated Polars partition to a Parquet state dataset."""

        if table.is_empty():
            return self.reference_existing(
                spec,
                run_id=run_id,
                producer=producer,
                config_hash=config_hash,
                row_count=row_count,
                inputs=inputs,
                mode=mode,
                knowledge_cutoff=knowledge_cutoff,
                data_snapshot=data_snapshot,
            )
        destination = self.destination(spec)
        staging = self.data_root / ".staging" / run_id / str(spec.dataset_id)
        staging.mkdir(parents=True, exist_ok=True)
        candidate = staging / "append.parquet"
        table.write_parquet(candidate, compression="zstd")
        candidate_report = validate_path(candidate, spec)
        candidate_report.require_valid()

        if destination.is_file():
            migration = staging / "dataset"
            migration.mkdir()
            os.link(destination, migration / "part-base.parquet")
            backup = staging / "source-history-backup.parquet"
            os.replace(destination, backup)
            try:
                os.replace(migration, destination)
            except OSError:
                os.replace(backup, destination)
                raise
            backup.unlink()
        else:
            destination.mkdir(parents=True, exist_ok=True)

        digest = checksum_path(candidate)[:20]
        partition = destination / f"part-{digest}.parquet"
        if partition.exists():
            candidate.unlink()
        else:
            os.replace(candidate, partition)
        shutil.rmtree(staging, ignore_errors=True)

        report = validate_path(destination, spec)
        report.require_valid()
        checksum = checksum_path(destination)
        return (
            ArtifactRef(
                kind=spec.layer.value,
                dataset_id=str(spec.dataset_id),
                path=destination,
                producer=producer,
                schema_version=spec.schema_version,
                run_id=run_id,
                config_hash=config_hash,
                checksum=checksum,
                row_count=row_count,
                file_count=len(list(destination.rglob("*.parquet"))),
                processing_mode=mode.value,
                knowledge_cutoff=knowledge_cutoff,
                data_snapshot=data_snapshot,
                inputs=tuple(item.checksum or str(item.path) for item in inputs),
                sensitivity=spec.sensitivity,
            ),
            report,
        )

    def write_manifest(
        self, manifest: RunManifest, destination: Path, *, force: bool
    ) -> Path:
        return manifest.write(destination, overwrite=force)

    def is_resumable(
        self, artifact: ArtifactRef, spec: DatasetSpec, config_hash: str
    ) -> bool:
        return (
            artifact.dataset_id == str(spec.dataset_id)
            and artifact.config_hash == config_hash
            and artifact.path.exists()
            and artifact.checksum == checksum_path(artifact.path)
            and validate_path(artifact.path, spec).valid
        )
