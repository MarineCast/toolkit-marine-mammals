# Sources and coverage

The killer-whale workflow exposes six source adapters. The live `sources` command
lists the capabilities and GBIF dataset selection packaged with your installation:

```bash
marine-mammals --workspace-root ./marine-mammal-workspace \
  killer-whales observations sources
```

| Source | Collection route | Coverage considerations |
| --- | --- | --- |
| TWM | Caller-supplied local CSV files | Missing files warn and produce an unavailable-source manifest; missing coverage is not zero sightings |
| Acartia | Current endpoint and optional retained local history | Current observations do not supply arbitrary historical coverage; access can require credentials |
| Maplify/WASEAK | Configured sightings endpoint | Queries and canonical filtering remain bounded by source availability and the selected geography |
| iNaturalist | Observation API with taxon, date, and geographic selection | Records retain source rights and coordinate uncertainty; the access token is optional |
| CWR | Configured archive years and map IDs | Archive availability differs by year; canonical date filtering does not make unprovided years complete |
| GBIF | Occurrence search with selected dataset UUIDs | The search path rejects scopes above 100,000 records; dataset and record policy require review |

See [query examples and provider limitations](../../../public-usage.md#query-a-provider)
for exact flags. Repeat `--source` to combine providers and `--dataset` to select
GBIF datasets. The default query geography is North Pacific; choose appropriate
bounds for another region and split boxes crossing the antimeridian.

## Keep source policy visible

Source selection does not grant redistribution permission. Packaged policies
retain rights and use classes; new GBIF dataset selections default to internal
use. Encounter aggregation requires a reviewed dataset policy.

Event times, model-local filtering, coordinate uncertainty, quarantine, and
coverage belong in the interpretation of a result. Location-free or unsupported
records may be quarantined; inspect audits and manifests before using the table.

## Local checks and offline replay

`preflight` checks configuration and local prerequisites without probing live
provider health. `query --dry-run` previews scope. `query --offline` replays that
exact query's previously collected snapshots; it does not download missing inputs.
The synthetic demo checks the software path and is not live-provider acceptance.
