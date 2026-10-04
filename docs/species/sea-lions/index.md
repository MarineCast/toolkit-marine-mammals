# Sea lions

This is the dedicated area for sea-lion workflows in the Marine Mammal Toolkit.

!!! info "Current status: extension point"
    `marine_mammal_toolkit.pinnipeds.sea_lions` currently contains a package
    marker. Sea-lion source collection, processing, population products,
    and CLI commands are not implemented. No sea-lion dataset is bundled.

## Package area

Sea-lion interpretation belongs in `pinnipeds/sea_lions`. Shared observation and
population engines live under `tools/`. The `pinnipeds/haulouts` namespace is a
separate extension point and does not currently provide reusable haulout products.
See the [architecture](../../architecture.md).

## Requirements for future workflows

Specify the species and population, distinguish individual observations from
site counts, and retain survey time, count meaning, age/sex categories when
provided, source identity, uncertainty, and coverage. A reported haulout count
does not automatically describe an entire population or establish absence from
unobserved sites.

Review source rights and spatial/temporal support before aggregation. The current
orca imputer and SRKW census rules are species-specific; reuse shared validation
only where its assumptions hold. Read the [scientific contracts](../../reference/contracts.md).

## Available now

The [workflow guide](../../getting-started/workflows.md) lists current executable
commands. Those workflows process killer-whale inputs; sea-lion commands will
need their own implementation and acceptance evidence.
