"""Read-only public seascape integration with explicit workspace resolution."""

from pathlib import Path
from contextlib import contextmanager
from seascape.spatial_support.water_network import load_water_graph as _load_water_graph
from seascape.spatial_support.water_network import (
    load_water_network_config as _load_water_network_config,
)


@contextmanager
def resolved_water_config(config_path):
    """Pass an explicit, fully resolved data base to the public seascape reader.

    Named areas are owned by seascape and require its explicit workspace setting;
    never let that dependency discover an unrelated current working directory.
    Only a temporary configuration is written, never a water-network product.
    """
    import os
    import tempfile
    import yaml
    from marine_mammal_toolkit.tools._core.config.data import load_data_config
    from marine_mammal_toolkit.tools._core.config.paths import project_root

    path = Path(config_path).expanduser().resolve()
    raw = load_data_config(path, domains="SEASCAPE_LAYER")
    base = Path(str(raw.get("base_directory", "."))).expanduser()
    raw["base_directory"] = str(
        base.resolve() if base.is_absolute() else (project_root() / base).resolve()
    )
    raw.pop("SEASCAPE_LAYER", None)
    raw.pop("extends", None)
    seascape_workspace = os.environ.get("SEASCAPE_WORKSPACE")
    if (
        not seascape_workspace
        or not Path(seascape_workspace).expanduser().is_absolute()
    ):
        raise ValueError(
            "Set SEASCAPE_WORKSPACE to the absolute data workspace containing "
            "config/common.yaml for the seascape named-area contract. "
            "Also supply imputation.inputs.water_network_config explicitly."
        )
    with tempfile.TemporaryDirectory(prefix="marine-mammals-seascape-") as temporary:
        resolved = Path(temporary) / "water-network.yaml"
        resolved.write_text(yaml.safe_dump(raw), encoding="utf-8")
        yield resolved


def load_water_graph(resolution, config_path):
    with resolved_water_config(config_path) as path:
        return _load_water_graph(resolution, path)


def load_water_network_config(config_path):
    with resolved_water_config(config_path) as path:
        return _load_water_network_config(path)
