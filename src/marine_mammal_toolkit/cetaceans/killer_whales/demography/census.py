"""Validate and export Southern Resident killer whale annual census counts.

This module reads annual SRKW census counts from a configured Excel workbook,
validates and normalizes the source rows, and writes an app-ready JSON artifact.

The exported payload includes:

* annual J, K, and L pod counts;
* the reported all-pods total;
* a calculated pod total and row-level consistency flag;
* validation metadata describing any total mismatches;
* the latest available census record; and
* basic source provenance.

The module is designed for both command-line use and import from a larger data
pipeline. Writes are atomic through ``atomic_write_text`` so downstream readers
never observe a partially written JSON file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from marine_mammal_toolkit.tools.schemas.artifacts import atomic_write_text
from marine_mammal_toolkit.tools._core.config.data import load_data_config
from marine_mammal_toolkit.tools._core.config.paths import project_root
from marine_mammal_toolkit.tools._core.config.paths import resolve_config_path
from marine_mammal_toolkit.cetaceans.killer_whales.resources import (
    config_path as resource_config,
)

DEFAULT_DATA_CONFIG_PATH = resource_config("populations")
DEFAULT_SHEET_NAME = "Chart Data"
DEFAULT_ECOTYPE = "SRKW"

HEADER_ALIASES = {
    "census year": "census_year",
    "k pod": "k_pod",
    "j pod": "j_pod",
    "l pod": "l_pod",
    "all pods": "all_pods",
    "chart axis label": "chart_axis_label",
}
REQUIRED_COLUMNS = {"census_year", "j_pod", "k_pod", "l_pod", "all_pods"}
POPULATION_COLUMNS = ("j_pod", "k_pod", "l_pod", "all_pods")


class PopulationRow(TypedDict):
    """Normalized annual population record exported to JSON."""

    census_year: int
    j_pod: int
    k_pod: int
    l_pod: int
    all_pods: int
    pod_sum: int
    total_matches_pod_sum: bool


class PopulationMismatch(TypedDict):
    """Summary of a row where the reported total differs from the pod sum."""

    census_year: int
    pod_sum: int
    all_pods: int
    difference: int


class PopulationConfig(TypedDict):
    """Resolved population-export configuration."""

    config_path: Path
    base_directory: Path
    source_path: str | Path
    output_path: str | Path
    sheet_name: str
    ecotype: str


def _resolve(base_dir: Path, path: str | Path) -> Path:
    """Resolve ``path`` against ``base_dir`` unless it is already absolute."""
    candidate = Path(path).expanduser()
    resolved = candidate if candidate.is_absolute() else base_dir / candidate
    return resolved.resolve()


def _resolve_base_directory(config_path: Path, raw_base_dir: object) -> Path:
    """Resolve the configured base directory against the selected data workspace."""
    candidate = Path(str(raw_base_dir or ".")).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root() / candidate).resolve()


def load_population_config(
    config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
) -> PopulationConfig:
    """Load population settings from the top-level data configuration.

    Relative ``base_directory`` values are interpreted relative to the selected
    data workspace, matching the toolkit's configuration contract.
    """
    resolved_config_path = resolve_config_path(config_path)
    raw = load_data_config(resolved_config_path)

    source_path = raw.get("population_source_path")
    output_path = raw.get("population_output_path")

    if not source_path:
        raise ValueError(
            f"Data config {resolved_config_path} is missing required key "
            "'population_source_path'."
        )
    if not output_path:
        raise ValueError(
            f"Data config {resolved_config_path} is missing required key "
            "'population_output_path'."
        )

    sheet_name = str(raw.get("population_sheet_name", DEFAULT_SHEET_NAME)).strip()
    ecotype = str(raw.get("ecotype", DEFAULT_ECOTYPE)).strip()

    if not sheet_name:
        raise ValueError(
            f"Data config {resolved_config_path} contains an empty "
            "'population_sheet_name'."
        )
    if not ecotype:
        raise ValueError(
            f"Data config {resolved_config_path} contains an empty 'ecotype'."
        )

    return {
        "config_path": resolved_config_path,
        "base_directory": _resolve_base_directory(
            resolved_config_path,
            raw.get("base_directory", "."),
        ),
        "source_path": source_path,
        "output_path": output_path,
        "sheet_name": sheet_name,
        "ecotype": ecotype,
    }


def build_population_payload(
    rows: list[PopulationRow],
    *,
    source_path: str | Path,
    sheet_name: str,
    ecotype: str = DEFAULT_ECOTYPE,
) -> dict[str, Any]:
    """Build the versioned JSON payload consumed by the application."""
    if not rows:
        raise ValueError("Cannot build a population payload from an empty row list.")

    effective_ecotype = ecotype.strip()
    if not effective_ecotype:
        raise ValueError("Population ecotype cannot be empty.")

    latest = max(rows, key=lambda item: item["census_year"])
    total_mismatches: list[PopulationMismatch] = [
        {
            "census_year": row["census_year"],
            "pod_sum": row["pod_sum"],
            "all_pods": row["all_pods"],
            "difference": row["all_pods"] - row["pod_sum"],
        }
        for row in rows
        if not row["total_matches_pod_sum"]
    ]

    return {
        "schema_version": 1,
        "ecotype": effective_ecotype,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source_path),
            "sheet": sheet_name,
        },
        "validation": {
            "row_count": len(rows),
            "first_census_year": rows[0]["census_year"],
            "latest_census_year": latest["census_year"],
            "total_mismatch_count": len(total_mismatches),
            "total_mismatches": total_mismatches,
        },
        "latest": latest,
        "rows": rows,
    }


def _source_label(source: Path, base_dir: Path) -> Path:
    """Return portable source provenance without leaking unnecessary paths."""
    try:
        return source.relative_to(base_dir)
    except ValueError:
        return Path(source.name)


def prepare_population_numbers(
    *,
    config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
    ecotype: str | None = None,
    workspace_root: str | Path | None = None,
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
    sheet_name: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Validate a census workbook and build the JSON payload without writing it."""
    if workspace_root is not None:
        from marine_mammal_toolkit.tools._core.config.paths import workspace

        with workspace(workspace_root):
            return prepare_population_numbers(
                config_path=config_path,
                ecotype=ecotype,
                source_path=source_path,
                output_path=output_path,
                sheet_name=sheet_name,
            )
    cfg = load_population_config(config_path)
    base_dir = cfg["base_directory"]
    source = _resolve(
        base_dir, source_path if source_path is not None else cfg["source_path"]
    )
    output = _resolve(
        base_dir, output_path if output_path is not None else cfg["output_path"]
    )
    selected_sheet = str(sheet_name if sheet_name is not None else cfg["sheet_name"]).strip()
    if not selected_sheet:
        raise ValueError("Population sheet name cannot be empty.")
    effective_ecotype = str(ecotype if ecotype is not None else cfg["ecotype"]).strip()

    rows = load_population_rows(source, sheet_name=selected_sheet)
    payload = build_population_payload(
        rows,
        source_path=_source_label(source, base_dir),
        sheet_name=selected_sheet,
        ecotype=effective_ecotype,
    )
    return payload, output


def export_population_numbers(
    *,
    config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
    ecotype: str | None = None,
    fail_on_total_mismatch: bool = False,
    workspace_root: str | Path | None = None,
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
    sheet_name: str | None = None,
) -> Path:
    """Read the configured workbook and write app-ready population JSON.

    Args:
        config_path: Path to the top-level data project configuration.
        ecotype: Optional override for the exported ecotype label.
        fail_on_total_mismatch: Raise an error instead of writing output when
            any reported all-pods total differs from J + K + L.
        source_path: Optional source workbook override.
        output_path: Optional JSON destination override.
        sheet_name: Optional workbook sheet override.

    Returns:
        The absolute path of the JSON artifact written to disk.
    """
    payload, output = prepare_population_numbers(
        config_path=config_path,
        ecotype=ecotype,
        workspace_root=workspace_root,
        source_path=source_path,
        output_path=output_path,
        sheet_name=sheet_name,
    )

    mismatch_count = payload["validation"]["total_mismatch_count"]
    if fail_on_total_mismatch and mismatch_count:
        raise ValueError(
            f"Found {mismatch_count} row(s) where all_pods does not equal "
            "j_pod + k_pod + l_pod. No output was written."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        overwrite=True,
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export SRKW population counts from the configured metadata "
            "workbook to app-ready JSON."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_DATA_CONFIG_PATH),
        help="Path to the top-level data project configuration.",
    )
    parser.add_argument(
        "--ecotype",
        help="Override the population ecotype label in the output JSON.",
    )
    parser.add_argument(
        "--fail-on-total-mismatch",
        action="store_true",
        help=(
            "Fail without writing output when an all-pods total differs from "
            "the sum of J, K, and L pod counts."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the population export CLI."""
    args = parse_args()
    output = export_population_numbers(
        config_path=args.config,
        ecotype=args.ecotype,
        fail_on_total_mismatch=args.fail_on_total_mismatch,
    )
    print(f"[population] JSON written -> {output}")


def load_population_rows(
    source_path: str | Path, *, sheet_name: str
) -> list[PopulationRow]:
    """Apply the SRKW workbook profile to the reusable census reader."""
    from marine_mammal_toolkit.tools.populations.workbook import read_annual_counts

    return read_annual_counts(
        source_path,
        sheet_name=sheet_name,
        aliases=HEADER_ALIASES,
        count_columns=POPULATION_COLUMNS,
        component_columns=("j_pod", "k_pod", "l_pod"),
        total_column="all_pods",
    )
