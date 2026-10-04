# Orcas

Killer-whale observations, label imputation, and Southern Resident annual census
processing are implemented in `marine_mammal_toolkit.cetaceans.killer_whales`.
The CLI uses the name `killer-whales`.

## Choose an area

<div class="grid cards" markdown>

- **[Sightings](sightings/index.md)**

    Query and explore reported observations, understand source coverage, and
    work with label imputation, immutable releases, and reports.

    [Explore the example map](sightings/example-map.md)

- **[Census](census/index.md)**

    Validate annual Southern Resident census workbooks and export J, K, and L
    pod counts with explicit reconciliation and source metadata.

    [Read the workbook guide](../../demography.md)

</div>

## What the outputs mean

Sightings describe reported observations. SRKW/Transient inference imputes
observation labels, preserving known-Other handling and abstention. Annual SRKW
census rows describe the supplied population workbook. These quantities do not
establish occurrence probabilities, observation effort, or verified absence.

The toolkit owns these producers. OrcaCast owns occurrence forecasting,
forecast-model evaluation, and application publishing. See the
[architecture](../../architecture.md) and [scientific contracts](../../reference/contracts.md).

## Implementation scope

Bounded observation queries use the base installation. Regional imputation,
counts, grids, intensity, and consumer products require their documented optional
dependencies and support inputs. The current binary imputer is not a general
classifier for every killer-whale population or other species.

Earlier OrcaCast descriptions remain available in the
[historical archive](../../reference/orcacast-before-migration/index.md). Use the
current guides above for executable commands.
