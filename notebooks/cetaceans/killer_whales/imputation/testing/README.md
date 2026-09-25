# Ecotype imputation experiments

This directory is an isolated testing area for whale-sighting imputation work.
Set `MARINE_MAMMALS_WORKSPACE_ROOT` to an existing data workspace before running
the notebooks. As written, they read the immutable release selected by that
workspace's retained OrcaCast pointer at
`data/processed/domain/whale_layer/sightings/releases/latest.json` and write
only beneath this directory's `outputs/`. The release must contain observations,
imputation diagnostics, and a fitted model; an `observations-only` release or a
public query is insufficient. The helper `resolve_release_paths` accepts an
explicit `release_manifest` for other release layouts, but these notebooks use
the default. They never update production models, canonical Parquet files, or
release pointers.

From the toolkit checkout root, install it with `.[research,imputation,report]`
and make the standalone Seascape package available before running the imputation
notebooks. See the [research setup](../../README.md) for the other required
workspace paths.

The notebooks are intentionally split:

- `01_current_model_baseline.ipynb` reproduces the current fitted model's
  reconstruction, encounter, blocked, purged, and source-stratified metrics.
- `02_source_aware_open_set.ipynb` evaluates nested source/era calibration,
  natural-prevalence post-stratification, leave-one-source-out transport, and
  a hierarchical Other-versus-modeled prototype.
- `03_learned_spatiotemporal_transport.ipynb` rebuilds leakage-controlled
  anchor support over the canonical marine graph, learns distance/time/speed
  kernels, and applies source-level degradation gates to transport blends.
- `04_seasonal_static_seascape.ipynb` adds smooth seasonal harmonics,
  daylight/solstice features, season-conditioned transport, and curated static
  physical seascape covariates. It reports source, era, and H3 R4 regional
  degradation gates.
- `05_guarded_hierarchical_residual.ipynb` keeps the safe transport blend as a
  fixed offset, learns a bounded correction from a reduced seasonal/seascape
  panel, applies regularized group calibration, and shrinks unreliable rows
  toward the offset baseline.
- `06_model_strategy_bakeoff.ipynb` compares regularized logistic, spline/GAM,
  group-robust histogram boosting, water-graph diffusion stacking, group-DRO,
  evidence-regime experts, and a nested calibrated ensemble. It adds rolling,
  spatial-region, and source-held-out residual stress tests.
- `07_open_set_missingness_and_active_labeling.ipynb` compares Stage-A open-set
  models, estimates source-dependent label availability, runs an
  inverse-propensity sensitivity analysis, and writes an explicitly unreviewed
  internal audit sample of unknown encounters.

Run the notebooks from this directory:

```bash
MPLCONFIGDIR=/tmp/orcacast-mpl \
  python -m jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
  01_current_model_baseline.ipynb 02_source_aware_open_set.ipynb \
  03_learned_spatiotemporal_transport.ipynb \
  04_seasonal_static_seascape.ipynb \
  05_guarded_hierarchical_residual.ipynb \
  06_model_strategy_bakeoff.ipynb \
  07_open_set_missingness_and_active_labeling.ipynb
```

The improved notebook is research-only. Its open-set results cannot certify
production behavior because the repository does not yet contain a blinded,
double-reviewed unknown-label audit sample and the known-Other sample is small.
The static-seascape notebook additionally excludes kelp, seagrass, and
anthropogenic snapshots because those inputs are not reconstructed as-of every
historical sighting date.

`P_OTHER` is a biological class probability and is not an abstention flag.
Abstained rows retain unit expected unknown mass; a supported Other prediction
contributes to expected Other mass. The guarded residual experiment remains a
conditional SRKW-versus-Transient experiment and does not fabricate `P_OTHER`.
The strategy bake-off's rolling, region, and source tests operate on frozen OOF
base scores; promotion would still require refitting the full imputer and
transport anchors inside every outer split. The active-label CSV contains review
priorities only—never pseudo-labels—and every proposed row remains soft-count
ineligible.
