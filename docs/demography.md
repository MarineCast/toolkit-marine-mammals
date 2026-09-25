# Killer-whale demography

The toolkit's demography component processes the annual Southern Resident killer whale (SRKW)
census workbook that the original OrcaCast workflow exported. It reads a selected Excel sheet,
validates annual J, K and L pod counts, and writes the existing app-compatible JSON shape. The
workbook is supplied by the caller; the package does not contain census data or download it.
This component does not infer births, deaths, survival, abundance outside the census population,
or whale occurrence from sightings.

## Run from a standalone workspace

Install the base package and choose an explicit workspace. The workbook and output paths may be
absolute or relative to that workspace. `--sheet` overrides the packaged default, `Chart Data`.

```bash
python -m pip install .
marine-mammals --workspace-root /absolute/workspace killer-whales demography census \
  --workbook /absolute/path/to/census.xlsx --sheet 'Chart Data' \
  --output data/processed/whales/srkw-census.json --dry-run
marine-mammals --workspace-root /absolute/workspace killer-whales demography census \
  --workbook /absolute/path/to/census.xlsx --sheet 'Chart Data' \
  --output data/processed/whales/srkw-census.json --fail-on-total-mismatch
```

The preview reads and validates the workbook, prints the target path, row count, year span,
latest row and any reported-total mismatches, and writes nothing. The export writes one JSON
file atomically. A mismatch is reported in `validation.total_mismatches`; with
`--fail-on-total-mismatch`, it stops before replacing an existing output. The mismatch is never
silently corrected.

The packaged configuration keeps the original OrcaCast source and output path defaults. If your
workbook differs, use `--workbook`, `--sheet`, and `--output`, or pass `--config` with
`population_source_path`, `population_output_path`, and `population_sheet_name`. The legacy
`killer-whales populations run` command is retained for existing callers.

## Python API

```python
from marine_mammal_toolkit.cetaceans.killer_whales.demography import (
    export_population_numbers,
    prepare_population_numbers,
)

options = dict(
    workspace_root="/absolute/workspace",
    source_path="/absolute/path/to/census.xlsx",
    sheet_name="Chart Data",
    output_path="data/processed/whales/srkw-census.json",
)
payload, destination = prepare_population_numbers(**options)  # No write.
output = export_population_numbers(**options, fail_on_total_mismatch=True)
```

`load_population_rows` and `build_population_payload` are also public for callers that need
validated rows without an output file. The former
`cetaceans.killer_whales.populations.prepare` import path re-exports these functions and
`export_population_numbers`; OrcaCast's current consumer can continue using it.

## Input and output contract

The selected sheet must have `Census Year`, `J Pod`, `K Pod`, `L Pod`, and `All Pods` columns
(case and underscores are normalized). Years must be unique whole numbers from 1900 through
the next calendar year. Counts must be nonnegative whole numbers; blank or unrecalculated
formula cells fail validation. Rows are sorted by year before export. Other sheet columns are
ignored.

The JSON has `schema_version: 1`, the `SRKW` ecotype label by default, annual `rows`, `latest`,
source path/sheet metadata, and validation metadata. Each row retains the reported `all_pods`,
a calculated `pod_sum`, and `total_matches_pod_sum`. The output is a reported annual census
series, separate from opportunistic sighting counts and from model estimates. Review source
authority, census-year meaning, and redistribution rights before publishing the workbook or
derived JSON. The source workbook is not bundled or copied by this workflow.
