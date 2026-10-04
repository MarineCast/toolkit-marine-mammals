# Humpbacks

This is the dedicated area for humpback-whale workflows in the Marine Mammal Toolkit.

!!! info "Current status: extension point"
    `marine_mammal_toolkit.cetaceans.humpbacks` currently contains a package
    marker. Humpback source collection, processing, population products,
    and CLI commands are not implemented. No humpback dataset is bundled.

## Package area

Humpback interpretation belongs in `cetaceans/humpbacks`, composing reusable
engines under `tools/` when implemented. Species selection and acceptance policy
must be reviewed independently of the current orca workflow. See the
[architecture](../../architecture.md).

## Requirements for future workflows

Define source grain and identity before combining sightings, encounters,
photo-identification records, telemetry, or acoustic detections. Preserve animal
and event identifiers where supplied, time support, coordinate uncertainty,
source rights, quality flags, and coverage. Repeated records do not automatically
represent different animals or an abundance estimate.

The current binary ecotype imputer and SRKW census export do not supply humpback
classification or population estimation. The [scientific contracts](../../reference/contracts.md)
describe the boundaries a new workflow must preserve.

## Available now

The [orca observation guide](../../public-usage.md) demonstrates the existing
collection and normalization API. Its species settings are not a supported
humpback configuration.
