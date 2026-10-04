# Census

Validate and export annual Southern Resident killer-whale population counts.
{ .marine-mammals-lead }

The census workflow reads a caller-supplied Excel sheet, validates annual J, K,
and L pod counts, and exports the toolkit's app-compatible JSON. It preserves
the reported `All Pods` value alongside the calculated pod sum and records any
mismatch in validation metadata. The toolkit does not bundle or download census data.

## Preview a workbook

```bash
marine-mammals --workspace-root /absolute/workspace killer-whales demography census \
  --workbook /absolute/path/to/census.xlsx --sheet 'Chart Data' \
  --output data/processed/whales/srkw-census.json --dry-run
```

The preview reads and validates the selected sheet without writing. Remove
`--dry-run` to export JSON; add `--fail-on-total-mismatch` to stop before replacing
an output when reported totals disagree with the sum of the pods.

[Read the complete workbook guide and Python API](../../../demography.md)

## Keep population meaning separate

Census rows describe the supplied annual population series. They do not estimate
births, deaths, survival, populations outside the census, or occurrence from
sightings. Review source authority, census-year meaning, and redistribution rights
before publishing a workbook or derived JSON. For observation workflows, visit
[Sightings](../sightings/index.md).
