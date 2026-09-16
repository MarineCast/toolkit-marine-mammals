# Marine Mammal Toolkit

`toolkit-marine-mammals` is the MarineCast repository for reusable marine-mammal data
contracts and domain processing. The repository currently contains an installable namespace
scaffold; acquisition pipelines, datasets, and generated products have not been moved here.

## Package layout

```text
src/marine_mammal_toolkit/
├── observations/       # Encounter and observation records
├── populations/        # Population identities and demographic records
├── taxonomy/           # Taxonomic names and identifiers
├── telemetry/          # Animal-borne tracking records
├── acoustic/           # Marine-mammal acoustic observations
├── schemas/            # Shared package-level data contracts
├── quality/            # Validation and quality-control rules
├── cetaceans/
│   ├── killer_whales/
│   ├── humpbacks/
│   └── gray_whales/
└── pinnipeds/
    ├── haulouts/
    ├── seals/
    └── sea_lions/
```

These namespaces define ownership, not proof of implementation. Empty domain packages must not be
presented as working data pipelines.

## Development

The package requires Python 3.11 or newer. From this repository:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

No source data is bundled. Future data work must preserve source rights, provenance, observation
time, spatial support, identity, quality flags, and the distinction between unavailable data and
an observed zero.
