# Releases and reports

Observation queries, stage releases, and consumer-facing products use different
artifacts and pointers. Choose the contract your consumer needs.

| Operation | Output or pointer beneath the selected data root |
| --- | --- |
| `query` | `queries/<query-hash>/query-result.json` and scoped canonical observations, associations, audits, and snapshots |
| `run` | `processed/sightings/final/releases/latest.json`, resolving a validated immutable stage release |
| `product` | `processed/sightings/final/latest.json`, binding one dated consumer generation and its report |

## Source-observation release

After reviewing source access and preflight results, run the packaged configuration
in an explicit workspace:

```bash
marine-mammals --workspace-root /absolute/workspace \
  --data-root data/sightings \
  killer-whales observations preflight --profile observations-only
marine-mammals --workspace-root /absolute/workspace \
  --data-root data/sightings \
  killer-whales observations run \
  --profile observations-only --end-date 2025-06-08
```

The date is an example; choose the actual endpoint for your build. `run` collects
enabled providers and advances a local release pointer after validation. This
profile stops after normalization. Other profiles have additional
[regional prerequisites](methods.md).

## Consumer product

Install `.[imputation,report]` and provision model domains and regional water
support before running `product`:

```bash
marine-mammals --workspace-root /absolute/workspace \
  --data-root data/marine-mammals/killer-whales \
  killer-whales observations product --end-date 2025-06-08
```

The packaged configuration is the default. Pass `--config` explicitly for a custom
YAML. The product keeps `raw/` inputs and `processed/sightings/` normalization,
model work, and final generations within the selected product root. It materializes
the generation and HTML report before advancing the product pointer.

Resolve the pointer once to read a consistent generation:

```python
from marine_mammal_toolkit.cetaceans.killer_whales.observations import (
    resolve_sightings_product,
)

product = resolve_sightings_product("/absolute/product-root")
composite_path = product.composite_path
imputed_path = product.imputed_path
```

Flat filenames remain compatibility copies; independent reads can span different
generations during a write. The pointer and public resolver are the maintained
consumer contract. See [publication and retention](../../../public-usage.md#product-publication-and-retention)
for concurrency, pinning, and failure behavior.

## HTML report and retention

```bash
marine-mammals --workspace-root /absolute/workspace \
  --data-root data/marine-mammals/killer-whales \
  killer-whales observations report --force
```

This uses the existing product without collection or fitting and creates a new
report sidecar. Density maps and daily counts describe reported records; they
do not measure abundance, occupancy, reporting effort, or verified absence.
Map tiles load when the report is opened.

Historical generations are retained by default. Pruning requires `--prune`;
`--keep-generations` and `--pin-release` control retention. The row-count guard
limits deletion and does not qualify a new product scientifically. Raw snapshots
and source history are not pruned by this policy. Review source rights before any
external redistribution; creating a local product does not authorize publication.
