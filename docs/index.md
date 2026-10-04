# Marine Mammal Toolkit

Observation and population processing, organized by species.
{ .marine-mammals-lead }

The toolkit provides an independently installable Python package and the
`marine-mammals` command. Start with a species area to find its workflows,
data contracts, and current implementation status.

## Explore a species

<div class="grid cards" markdown>

- **[Orcas](species/orcas/index.md)**

    Implemented: observation queries, source processing, SRKW/Transient label
    imputation, releases, reports, and SRKW annual census export.

- **[Seals](species/seals/index.md)**

    Extension point: a dedicated area for future seal observations and workflows.

- **[Gray whales](species/gray-whales/index.md)**

    Extension point: a dedicated area for future gray-whale observations and workflows.

- **[Sea lions](species/sea-lions/index.md)**

    Extension point: a dedicated area for future sea-lion observations and workflows.

- **[Humpbacks](species/humpbacks/index.md)**

    Extension point: a dedicated area for future humpback observations and workflows.

</div>

## Start using the toolkit

[Install the package](getting-started/installation.md), then
[choose a workflow](getting-started/workflows.md). The offline orca demo uses
synthetic records; real queries require source access. Data, census workbooks,
regional support files, and trained models are supplied separately.

Reported sightings, inferred labels, and census counts retain different meanings.
The toolkit does not produce occurrence forecasts. Read the
[scientific contracts](reference/contracts.md) before combining outputs with
other data or publishing derived products.

For developers, the [architecture and API map](architecture.md) explains shared
engines and species composition. The [documentation guide](development/documentation.md)
describes local preview and GitHub Pages hosting.
