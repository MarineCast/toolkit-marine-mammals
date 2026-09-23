from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from marine_mammal_toolkit.tools._core.data import DATASETS
from marine_mammal_toolkit.tools._core.data import ArtifactStore
from marine_mammal_toolkit.tools.schemas.observations import AUDIT_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import SOURCE_HISTORY_SCHEMA
from marine_mammal_toolkit.tools.schemas.observations import SOURCE_RECORD_SCHEMA
from marine_mammal_toolkit.tools.observations.process.pipeline import (
    _assemble_source_state,
)
from marine_mammal_toolkit.tools.quality.observations import validate_sightings_artifact


def _record(
    source_id: str,
    payload: str,
    retrieved_at: datetime,
    *,
    corrected: bool = False,
    last_corrected_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "SOURCE_RECORD_ID": source_id,
        "SOURCE": "TWM",
        "SOURCE_NATIVE_ID": source_id.removeprefix("TWM:"),
        "OBSERVED_AT_RAW": "2025-01-01",
        "OBSERVED_DATE_RAW": "2025-01-01",
        "CREATED_AT_RAW": None,
        "LATITUDE_RAW": "48.5",
        "LONGITUDE_RAW": "-123.0",
        "SPECIES_RAW": "Orcinus orca",
        "DESCRIPTION_RAW": None,
        "POD_ECOTYPE_RAW": None,
        "SOURCE_DATASET_ID": None,
        "SOURCE_EVENT_ID": None,
        "SOURCE_OCCURRENCE_IDS": None,
        "SOURCE_OCCURRENCE_COUNT": 1,
        "SOURCE_LICENSE": None,
        "SOURCE_USE_CLASS": "REDISTRIBUTABLE",
        "COORDINATE_UNCERTAINTY_M": None,
        "SOURCE_QC_STATUS": "ACCEPTED",
        "SOURCE_QC_DETAIL": None,
        "SOURCE_PAYLOAD": payload,
        "SOURCE_RETRIEVED_AT_UTC": retrieved_at,
        "SOURCE_PAYLOAD_CORRECTED": corrected,
        "LAST_CORRECTED_AT_UTC": last_corrected_at or retrieved_at,
    }


def _history(record: dict[str, object], retrieval_id: str) -> dict[str, object]:
    payload = str(record["SOURCE_PAYLOAD"])
    return {
        **record,
        "RETRIEVAL_ID": retrieval_id,
        "RETRIEVED_AT": record["SOURCE_RETRIEVED_AT_UTC"],
        "PAYLOAD_CHECKSUM": hashlib.sha256(payload.encode()).hexdigest(),
        "RAW_SCHEMA_FINGERPRINT": "fixture",
        "SNAPSHOT_MODE": "FULL_REPLACE",
    }


def _state_root(tmp_path: Path) -> Path:
    root = tmp_path / "processed/sightings/normalized"
    (root / "state").mkdir(parents=True)
    return root


def test_already_applied_snapshots_reuse_only_compatible_source_state(
    tmp_path: Path, monkeypatch
) -> None:
    root = _state_root(tmp_path)
    stamp = datetime(2025, 1, 2, tzinfo=timezone.utc)
    record = _record("TWM:1", '{"id":"1"}', stamp)
    pq.write_table(
        pa.Table.from_pylist([_history(record, "r1")], schema=SOURCE_HISTORY_SCHEMA),
        root / "state/source_history.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([record], schema=SOURCE_RECORD_SCHEMA),
        root / "state/source_current.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "SOURCE_RECORD_ID": "TWM:1",
                    "OBSERVATION_ID": None,
                    "STATUS": "NO_OP",
                    "REASON": "DUPLICATE_SOURCE_PAYLOAD",
                    "DETAIL": None,
                    "RULE_ID": "source_state.upsert",
                }
            ],
            schema=AUDIT_SCHEMA,
        ),
        root / "audit.parquet",
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "snapshot.json").write_text(
        json.dumps(
            {
                "retrieval_id": "r1",
                "retrieved_at": stamp.isoformat(),
                "snapshot_mode": "FULL_REPLACE",
                "raw_schema_fingerprint": "fixture",
            }
        )
    )
    manifest = tmp_path / "normalize.json"
    manifest.write_text(
        json.dumps(
            {
                "workflow": "whale.sightings.normalize.v10",
                "config_hash": "fixture-config",
                "inputs": [
                    {
                        "dataset_id": "whale.sightings.source_twm",
                        "path": str(snapshot.resolve()),
                    }
                ],
            }
        )
    )
    pointer = root / "manifests/normalize/latest.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"manifest": str(manifest)}))

    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.process.pipeline.adapt_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not adapt")
        ),
    )

    history, current, audit, changed, row_counts = _assemble_source_state(
        {"twm": snapshot},
        root,
        pd.Timestamp("2025-01-03T00:00:00Z"),
        expected_config_hash="fixture-config",
    )

    assert history.height == 1
    assert current.height == 1
    assert len(audit) == 1
    assert changed == {
        "whale.sightings.source_record_history": False,
        "whale.sightings.source_records": False,
    }
    assert row_counts["whale.sightings.source_record_history"] == 1

    calls: list[str] = []

    def adapt_after_config_change(source: str, _snapshot: Path) -> pa.Table:
        calls.append(source)
        return pa.Table.from_pylist([record], schema=SOURCE_RECORD_SCHEMA)

    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.process.pipeline.adapt_snapshot",
        adapt_after_config_change,
    )
    _assemble_source_state(
        {"twm": snapshot},
        root,
        pd.Timestamp("2025-01-03T00:00:00Z"),
        expected_config_hash="changed-config",
    )
    assert calls == ["twm"]


def test_history_delta_keeps_payload_transitions_without_rewriting_prior_rows(
    tmp_path: Path, monkeypatch
) -> None:
    root = _state_root(tmp_path)
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t3 = datetime(2025, 1, 3, tzinfo=timezone.utc)
    first = _record("TWM:1", '{"value":1}', t1)
    corrected = _record(
        "TWM:1", '{"value":2}', t3, corrected=True, last_corrected_at=t3
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_history(first, "r1")],
            schema=SOURCE_HISTORY_SCHEMA,
        ),
        root / "state/source_history.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([first], schema=SOURCE_RECORD_SCHEMA),
        root / "state/source_current.parquet",
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "snapshot.json").write_text(
        json.dumps(
            {
                "retrieval_id": "r3",
                "retrieved_at": t3.isoformat(),
                "snapshot_mode": "FULL_REPLACE",
                "raw_schema_fingerprint": "fixture",
            }
        )
    )
    incoming = pa.Table.from_pylist([corrected], schema=SOURCE_RECORD_SCHEMA)
    monkeypatch.setattr(
        "marine_mammal_toolkit.tools.observations.process.pipeline.adapt_snapshot",
        lambda *_args, **_kwargs: incoming,
    )

    history, current, _audit, changed, row_counts = _assemble_source_state(
        {"twm": snapshot}, root, pd.Timestamp("2025-01-04T00:00:00Z")
    )

    assert history.height == 1
    assert history.get_column("SOURCE_PAYLOAD").to_list() == ['{"value":2}']
    assert current.height == 1
    assert current.row(0, named=True)["SOURCE_PAYLOAD"] == '{"value":2}'
    assert changed["whale.sightings.source_record_history"] is True
    assert row_counts["whale.sightings.source_record_history"] == 2


def test_history_delta_append_migrates_single_file_to_partitioned_dataset(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "outputs"
    destination = (
        data_root / "processed/sightings/normalized/state/source_history.parquet"
    )
    destination.parent.mkdir(parents=True)
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
    first = _history(_record("TWM:1", '{"value":1}', t1), "r1")
    second = _history(
        _record("TWM:1", '{"value":2}', t2, corrected=True, last_corrected_at=t2),
        "r2",
    )
    pq.write_table(
        pa.Table.from_pylist([first], schema=SOURCE_HISTORY_SCHEMA), destination
    )
    store = ArtifactStore(
        data_root=data_root,
        artifact_root=artifact_root,
        output_root=output_root,
    )
    spec = replace(
        DATASETS.get("whale.sightings.source_record_history"),
        schema=SOURCE_HISTORY_SCHEMA,
        schema_version="8",
    )

    artifact, report = store.append_table(
        pl.from_arrow(pa.Table.from_pylist([second], schema=SOURCE_HISTORY_SCHEMA)),
        spec,
        run_id="append-fixture",
        producer="test",
        config_hash="fixture",
        row_count=2,
    )

    assert report.valid
    assert artifact.row_count == 2
    assert destination.is_dir()
    assert len(list(destination.glob("*.parquet"))) == 2
    assert validate_sightings_artifact(artifact).valid
