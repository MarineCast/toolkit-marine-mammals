# Choose a workflow

The current executable workflows are for orcas. Seals, gray whales, sea lions,
and humpbacks have their own documentation areas and package extension points;
they do not yet have source collectors, species workflows, or CLI commands.

| Goal | Entry point | Inputs and effects |
| --- | --- | --- |
| Try the toolkit offline | `killer-whales observations demo` | Writes synthetic records and query artifacts in the chosen workspace |
| Check source choices | `killer-whales observations sources` | Lists capabilities and configured GBIF datasets |
| Check local prerequisites | `killer-whales observations preflight` | Reads configuration and local inputs; no downloads or writes |
| Preview query scope | `killer-whales observations query --dry-run` | Prints selection; no acquisition |
| Query real observations | `killer-whales observations query` | Contacts selected providers or reads local TWM files; writes isolated query artifacts |
| Create a source-observation release | `killer-whales observations run --profile observations-only` | Collects enabled sources, normalizes, validates, and advances a local release pointer |
| Build a regional consumer product | `killer-whales observations product` | Requires modeling dependencies and regional inputs; writes an immutable generation and advances its product pointer |
| Rebuild a product report | `killer-whales observations report --force` | Uses the existing product; writes a new report sidecar without collection or fitting |
| Validate or export an annual census | `killer-whales demography census` | Reads your Excel workbook; `--dry-run` validates without writing; export writes JSON |

## Observation queries

Use [observation queries](../public-usage.md) for bounded selections with explicit
sources, dates, and geography. Outputs describe reported records. A query does
not promote a release or provide evidence of absence where no records are returned.

## Regional products and imputation

Read [processing and imputation](../species/orcas/sightings/methods.md) before choosing an
advanced profile. The `observed-only` profile includes counts and requires water
universes; it differs from `observations-only`, which stops after normalization.
See [releases and reports](../species/orcas/sightings/products.md) for pointer resolution
and retention.

## SRKW census

The [census guide](../demography.md) documents workbook columns, pod totals,
validation, and the public Python API. Census values remain separate from
sighting counts and model estimates.
