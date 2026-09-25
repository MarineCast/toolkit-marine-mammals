"""Compatibility imports for the original toolkit population-export path.

New callers should use ``cetaceans.killer_whales.demography``. The old path
remains available to OrcaCast and existing research scripts.
"""

from ..demography.census import DEFAULT_DATA_CONFIG_PATH
from ..demography.census import DEFAULT_ECOTYPE
from ..demography.census import DEFAULT_SHEET_NAME
from ..demography.census import HEADER_ALIASES
from ..demography.census import POPULATION_COLUMNS
from ..demography.census import REQUIRED_COLUMNS
from ..demography.census import PopulationConfig
from ..demography.census import PopulationMismatch
from ..demography.census import PopulationRow
from ..demography.census import build_population_payload
from ..demography.census import export_population_numbers
from ..demography.census import load_population_config
from ..demography.census import load_population_rows
from ..demography.census import main
from ..demography.census import parse_args
from ..demography.census import prepare_population_numbers

if __name__ == "__main__":
    main()
