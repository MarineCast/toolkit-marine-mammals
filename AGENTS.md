# toolkit-marine-mammals agent guidance

## Scope and current state

This repository owns reusable marine-mammal observation, population, taxonomy, telemetry,
acoustic, schema, and quality-control code. Species and clade-specific namespaces live beneath
`cetaceans` and `pinnipeds`.

Killer-whale observations and SRKW annual census processing are implemented. Shared engines live
in `tools/observations/{collect,process,impute,post_process}` and `tools/populations`;
species interpretation, feature construction, acceptance policy, release profiles, and census
presentation live in `cetaceans/killer_whales`. The maintained census API is
`cetaceans/killer_whales/demography`; `populations/prepare.py` preserves old consumer imports.
Other species remain extension points.
Read `docs/migration.md` before changing compatibility or ownership boundaries.

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
- Binary SRKW/Transient inference is observation-label imputation, not occurrence forecasting.
  Preserve known-Other mass, abstention, encounter isolation, and purged certification gates.
- Legacy Joblib models must fail before deserialization with a refit instruction. Model loading
  must not grant soft-count certification. Never rewrite existing trained models during migration.
- Require an explicit workspace for relative data/model/output paths. Includes resolve relative
  to their declaring file. Canonical configs are installed resources, not checkout paths.
- Seascape is a declared optional dependency of the `imputation` extra; use its public water-network API through the explicit-base
  integration helper. `SEASCAPE_WORKSPACE` selects its named-area configuration. Never copy producers.
- Dataset identifiers, `orca:v4:` identities, Arrow schemas, rights and missingness remain compatible.
  Source-content producer revisions intentionally change under the toolkit namespace.

## Validation

Before edits, run `git status --short` and preserve unrelated changes. Install `.[dev,imputation,report]` for the full suite. Run `python -m pytest tests`
and `python -m pytest notebooks/cetaceans/killer_whales/imputation/testing --import-mode=prepend`.
Run `git diff --check` and installed-wheel/CLI smoke checks after packaging changes. The integration
tests use only synthetic temporary storage. Do not run live acquisition, production fitting,
artifact rewriting, or release promotion as part of ordinary validation.
No local structural graph is present yet; use scoped source search. Any future Graphify cache
must stay disposable and local-only, and must be built from this checkout, not MarineCast root.
