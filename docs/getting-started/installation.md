# Installation

The toolkit supports Python 3.11 and newer. Install from this repository's
standalone checkout; an OrcaCast checkout is not required.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install .
marine-mammals --help
```

## Choose dependencies for your workflow

| Installation | Use |
| --- | --- |
| `python -m pip install .` | Orca observation queries and SRKW census workbook processing |
| `python -m pip install '.[report]'` | Optional Plotly reporting dependencies |
| `python -m pip install '.[imputation,report]'` | Regional label imputation, product materialization, and reporting |
| `python -m pip install -e '.[dev,imputation,report]'` | Full toolkit and research test suites |
| `python -m pip install -e '.[docs]'` | Toolkit development with MkDocs available |

The imputation extra declares Seascape as a dependency. It must be available from
your package index or installed from its independent repository. Regional marine
support and an explicit Seascape workspace are additional inputs; installing a
dependency does not supply those files. See the
[orca methods guide](../species/orcas/sightings/methods.md).

## Use an explicit data workspace

Commands that process data require `--workspace-root`. Relative data, artifact,
output, and input paths resolve there. Configuration includes resolve relative
to the YAML file that declares them. Canonical configurations are installed
package resources, so normal queries do not need checkout-local configuration.

```bash
marine-mammals --workspace-root ./marine-mammal-workspace \
  killer-whales observations demo
```

This writes a small **synthetic, offline** example into the selected workspace.
It does not download observations, fit a model, or qualify a live provider.
Continue with [workflow selection](workflows.md) or the full
[observation-query guide](../public-usage.md).

To build just this website without installing scientific dependencies, use the
[documentation-only environment](../development/documentation.md#local-preview).
