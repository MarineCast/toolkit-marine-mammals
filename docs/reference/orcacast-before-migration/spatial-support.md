> Historical pre-migration reference, retained for scientific context. Old paths, commands,
> schema-version descriptions and execution claims are not current toolkit instructions.
> See [current migration notes](../../migration.md).

# Whale spatial-support contracts

The sightings pipeline uses three different spatial concepts. They must not be
treated as interchangeable:

1. **Operational model extents** select the broad SRKW and Transient/Bigg's areas
   in which dense model rows may be emitted.
2. **Canonical full-counting water universes** supply the eligible H3 water cells
   at resolutions 4, 5, and 6.
3. **Canonical water graphs** define passable marine connectivity for imputation
   and intensity. Membership in a water universe does not itself prove graph
   reachability.

The current model-domain polygons are operational analysis extents. They are not
species distribution maps, critical-habitat designations, observations of
presence, or evidence that whales occupy every included cell.

## Current operational domains

`config/data/whale/model_domains.yaml` (`../../../../config/data/whale/model_domains.yaml`; historical path)
is the reviewed build contract. It materializes the two AOIs already maintained
in `config/common.yaml` (`../../../../config/common.yaml`; historical path):

| Model bucket | AOI key | WGS84 bounds |
|---|---|---|
| SRKW | `areas.srkw_range` | -132, 44, -121, 54 |
| TRANSIENT | `areas.transient_range` | -150, 40, -120, 62 |
| OTHER | union of the two effective cell sets | not a separate polygon |

The AOI rectangles are polygonized without geometric water clipping. The model
grid first loads the canonical full-counting H3 water universe and then retains
cells whose H3 representative points are covered by the applicable AOI polygon.
This ordering preserves the existing counting-water definition and avoids
creating a second, subtly different coast/water mask.

The build has an explicit parity gate against the reviewed effective cell counts:

| H3 resolution | SRKW cells | Transient cells | OTHER union cells |
|---:|---:|---:|---:|
| 4 | 182 | 421 | 421 |
| 5 | 964 | 2,174 | 2,174 |
| 6 | 5,442 | 12,480 | 12,480 |

Because the SRKW AOI is contained by the broader Transient AOI for this contract,
the current OTHER union count equals the Transient count. That is a geometric
property of these operational extents, not an assertion that the ecotypes have
identical distributions.

## Build and outputs

Build or replace the artifacts after the counting water universes exist:

```bash
python -m orcacast.cli \
  --run-id operational-model-domains-v1 \
  data build sightings model-domains \
  --config config/data/sightings.yaml \
  --domain-config config/data/whale/model_domains.yaml \
  --force
```

The command writes:

- `data/processed/domain/whale_layer/spatial_support/SRKW_MODEL_DOMAIN.parquet`
- `data/processed/domain/whale_layer/spatial_support/SRKW_MODEL_DOMAIN.metadata.json`
- `data/processed/domain/whale_layer/spatial_support/TRANSIENT_MODEL_DOMAIN.parquet`
- `data/processed/domain/whale_layer/spatial_support/TRANSIENT_MODEL_DOMAIN.metadata.json`
- `data/processed/domain/whale_layer/spatial_support/MODEL_DOMAINS_MANIFEST.json`

The manifest binds the sightings, model-domain, AOI, and water-universe config or
artifact checksums to the exact output geometry checksums and effective cell
counts. The model-grid loader rechecks geometry validity, polygon type, ecotype,
approval state, geometry checksum, and supported producer.

## Metadata contract

Each polygon has a sibling `.metadata.json` sidecar with:

- `ecotype` (`SRKW` or `TRANSIENT`);
- `domain_kind=operational_model_extent`;
- source authority, repository reference, and release;
- approval/retrieval timestamp and `review_status=approved_for_model_domain`;
- derivation and `water_membership` semantics;
- the originating AOI config and its checksum; and
- `geometry_sha256`, which must match the GeoParquet.

Missing, invalid, unapproved, ecotype-mismatched, or checksum-mismatched artifacts
fail closed. The old implicit bounding-box fallback is not available.

## Review and replacement policy

Changing an AOI, water universe, membership rule, or intended domain kind requires
a new reviewed release and updated expected cell counts. Review should inspect a
map at every supported H3 resolution and explain any changed cells before the new
manifest is promoted.

Official ecological-range, habitat-use, or critical-habitat polygons may be added
as separate evidence products. They must preserve their legal/scientific
authority, release, geometry, and intended-use semantics. They should not silently
replace these operational model extents or be described as direct observation,
occupancy, or absence evidence.
