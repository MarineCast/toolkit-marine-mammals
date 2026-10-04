# Gray whales

This is the dedicated area for gray-whale workflows. “Grey whales” refers to
the same documentation area; the package uses the spelling `gray_whales`.

!!! info "Current status: extension point"
    `marine_mammal_toolkit.cetaceans.gray_whales` currently contains a package
    marker. Gray-whale source collection, processing, population products,
    and CLI commands are not implemented. No gray-whale dataset is bundled.

## Package area

Gray-whale interpretation belongs in `cetaceans/gray_whales`, composing reusable
engines under `tools/` when a reviewed workflow is implemented. The existing
`killer-whales` command and its configured source selections remain orca-specific.
See the [architecture](../../architecture.md) for shared responsibilities.

## Requirements for future workflows

A future workflow needs reviewed taxon and population selection, source adapters,
event and animal identifiers, date semantics, coordinate uncertainty, record grain,
rights, and coverage. Direct observations, telemetry fixes, acoustic detections,
and inferred presence require distinct contracts; sightings alone do not establish
migration timing, abundance, or verified absence.

The current SRKW/Transient imputer and SRKW census aliases do not define a
gray-whale model or census product. Use the [scientific contracts](../../reference/contracts.md)
to review any new species composition.

## Available now

The [orca observation guide](../../public-usage.md) demonstrates the implemented
query API. Changing an orca query's geography does not turn it into a gray-whale
workflow.
