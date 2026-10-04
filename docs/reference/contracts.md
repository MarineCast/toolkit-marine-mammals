# Scientific contracts

Keep source meaning and uncertainty attached to every product.

| Quantity | Interpretation |
| --- | --- |
| Reported observation | A source record with its own time, location, identity, record grain, rights, and quality flags |
| Imputed observation label | Species-workflow inference with probabilities, known-Other handling, and abstention |
| Sighting count or reported activity | A derived summary on explicit spatial and temporal support |
| Annual census | Reported population values from the supplied census workbook |
| Occurrence forecast | Application-owned modeling; not produced by this toolkit |

## Identity, support, and missingness

Preserve source and event identifiers, animal/population identity, observation time,
coordinate uncertainty, record grain, and provenance. Preserve units and spatial
and temporal support when aggregating. Unknown, unavailable, partial coverage,
not applicable, and observed zero have different meanings.

An empty query, missing local file, or unprovided archive year is not evidence of
zero sightings. Opportunistic sightings do not establish observer effort, detection
probability, abundance, occupancy, or verified absence. Acoustic detections and
telemetry fixes would require separate contracts if implemented.

## Inference and population data

The current orca imputer is binary SRKW/Transient observation-label inference.
Known-Other mass, abstention, encounter isolation, and purged certification gates
remain explicit. Loading a model cannot confer soft-count certification.

SRKW census export preserves reported totals even when they differ from summed
pod values, records mismatches in validation metadata, and can fail before export
when requested. It does not infer demographic rates or missing annual values.
See the [census contract](../demography.md#input-and-output-contract).

## Rights and validation

Software licensing does not grant upstream data redistribution rights. Source
policy, coverage, use class, and provenance belong with the artifacts. Operational
model domains are not ecological-range or legal-habitat claims.

Offline synthetic tests establish software behavior. They do not establish
live-provider availability, production-data equivalence, regional model acceptance,
or publication readiness. The [migration record](../migration.md) distinguishes
historical validation from later changes.
