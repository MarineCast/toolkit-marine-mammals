# Documentation and GitHub Pages

This repository uses MkDocs with the Material theme. The site navigation has
separate areas for orcas, seals, gray whales, sea lions, and humpbacks. Current
orca guides are reused directly; the other species pages describe extension
points. Historical OrcaCast references remain in their own archive.
The Orcas area separates `species/orcas/sightings/` and `species/orcas/census/`.

## Local preview

From the repository root, create a documentation-only environment:

```bash
python -m venv .venv-docs
# macOS/Linux:
source .venv-docs/bin/activate
# Windows PowerShell:
# .venv-docs\Scripts\Activate.ps1
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Open <http://127.0.0.1:8000/toolkit-marine-mammals/>. The project URL prefix matches
GitHub Pages, so nested links and image paths can be checked locally.

The documentation requirements pin the MkDocs and Material versions exercised
by the docs workflow. The toolkit's `docs` extra also supports installing these
tools alongside the package; a docs-only build does not require scientific
dependencies, regional data, OrcaCast, or Seascape.

## Validate a change

```bash
mkdocs build --strict
git diff --check
```

Strict builds fail on missing pages, missing local links or anchors, and omitted
navigation pages. Review desktop and narrow-screen navigation and the species
areas before publishing. The generated `site/` directory is ignored.

## Host on GitHub Pages

The checked-in workflow at
[`.github/workflows/docs.yml`](https://github.com/MarineCast/toolkit-marine-mammals/blob/main/.github/workflows/docs.yml)
builds pull requests without deploying them. Pushes to `main` build the site and
deploy the resulting artifact. A manual workflow run deploys only when its
selected branch is `main`.

To activate hosting after the changes reach GitHub:

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, select **GitHub Actions** as the source.
3. Ensure Actions are enabled and the `github-pages` environment permits
   deployment from `main`; apply any desired review requirement there.
4. Merge the documentation changes into `main` or run **Documentation site**
   manually on `main`.
5. Confirm the build and deploy jobs pass, then open
   <https://marinecast.github.io/toolkit-marine-mammals/>.

The workflow uses read-only repository access for builds. The deployment job
has `pages: write` and `id-token: write`, and uses the `github-pages` environment.
No personal access token or `gh-pages` branch is needed. This follows
[GitHub's custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
Local site validation does not establish that remote hosting is enabled or a
deployment has succeeded.

## Maintain the species areas

Add species guides under `docs/species/<species>/` and list them beneath that
species in `mkdocs.yml`. Keep public guides tied to implemented APIs and distinguish
planned workflows from current capabilities. Seals and sea lions are documentation
groups that may gain multiple species-specific workflows over time.

The original `docs/public-usage.md` and `docs/demography.md` paths remain canonical
for current orca workflows, preserving existing repository links. Update the
archive only when maintaining its historical context, not to imply old application
commands are current.

## Banner and appearance

`docs/assets/marine-mammals-toolkit-banner.png` is the shared, unmodified
**2172 × 724** banner, exactly **3:1**. The Material template override at
`docs/overrides/main.html` includes it only on the site homepage. The `url`
filter resolves the asset correctly from the Pages project path.
CSS in `docs/assets/stylesheets/extra.css` displays it at full content width and
preserves its aspect ratio. The artwork is decorative; it is not a scientific
product or evidence of species distribution.

## Sightings example map

The [example map](../species/orcas/sightings/example-map.md) is an interactive
documentation demonstration with twelve explicitly synthetic records. The fixture
is `docs/assets/examples/orca-sightings-demo.json`; the map and record list use the
same records. It does not acquire, copy, or redistribute provider observations.
Its grain is one fictional example record, with date, example label, position,
location context, identifier, and synthetic provenance. Marker shape and label
letters supplement color; filters apply identically to the map and list.

Leaflet 1.9.4 is vendored under `docs/assets/vendor/leaflet/` with its BSD license,
so the library loads locally. The basemap uses OpenStreetMap's standard raster
tiles with visible attribution. Tiles need network access and are not bundled or
prefetched. If tiles fail, a visible status explains it; the synthetic records
and filters remain usable. JavaScript-disabled readers can inspect the linked
fixture. Keep the synthetic notice visible when modifying the example.

Navigation and configuration use
[MkDocs configuration](https://www.mkdocs.org/user-guide/configuration/) and
[Material navigation](https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/).
