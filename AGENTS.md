# toolkit-marine-mammals agent guidance

## Scope and current state

This repository owns reusable marine-mammal observation, population, taxonomy, telemetry,
acoustic, schema, and quality-control code. Species and clade-specific namespaces live beneath
`cetaceans` and `pinnipeds`.

The repository is currently an installable namespace scaffold. A directory's presence does not
mean that its pipeline, source integration, or data product is implemented.

## Shared MarineCast context

For repository boundaries, shared product contracts, provenance, or application integration,
read `../../.github/INFRASTRUCTURE.md` when this checkout is under
`MarineCast/Toolkits/toolkit-marine-mammals`. In another checkout layout, use the corresponding
MarineCast infrastructure guide and report when it is unavailable.

Keep this toolkit independently installable. Do not import through sibling filesystem paths or
require an OrcaCast checkout at runtime.

## Scientific and data contracts

- Preserve source and event identifiers, animal/population identity, observation time, spatial
  support, coordinate uncertainty, record grain, provenance, rights, and quality flags.
- Keep direct observations, inferred presence, acoustic detections, telemetry fixes, population
  estimates, and modeled occurrence as distinct quantities.
- Keep unknown, unavailable, partial, not applicable, and observed zero distinct.
- Do not commit raw, private, restricted, or large generated datasets. Define configured external
  storage and redistribution rights before adding acquisition workflows.
- Species-specific transformations belong in their species namespace when they are not valid for
  marine mammals generally.

## Validation

Before edits, run `git status --short` and preserve unrelated changes. For scaffold and
documentation changes, run `python -m pytest` and `git diff --check`. Add focused tests and exact
commands when executable behavior is introduced. Report acquisition, regional builds, and
application integration as unverified unless they were actually run.
