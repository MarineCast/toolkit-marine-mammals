# Seals

This is the dedicated area for seal workflows in the Marine Mammal Toolkit.

!!! info "Current status: extension point"
    `marine_mammal_toolkit.pinnipeds.seals` currently contains a package marker.
    Seal source collection, processing, population products, and CLI commands
    are not implemented. No seal dataset ships with the toolkit.

## Package area

Seal-specific interpretation belongs in `pinnipeds/seals`. Reusable observation
and population engines live under `tools/`. The separate `pinnipeds/haulouts`
namespace is also an extension point; it does not currently provide a haulout
workflow or spatial product. See the [architecture](../../architecture.md).

## Requirements for future workflows

Before adding a seal product, identify the species or population and define its
source records, event/animal identity, observation time, coordinate uncertainty,
and quality flags. Keep individual observations, haulout counts, and population
estimates distinct. Any aggregation needs explicit spatial and temporal support,
coverage, and rights.

The existing orca ecotype imputer has species-specific assumptions and should
not be presented as a seal model. Apply the [shared scientific contracts](../../reference/contracts.md)
when designing a new workflow.

## Available now

For an executable example of the toolkit's current observation pipeline, use
the [orca offline demo](../../public-usage.md). Its records and interpretation
are specific to the orca implementation and are not seal observations.
