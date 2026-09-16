"""Mechanical relocation with import rewriting; no data/artifacts are touched."""

import ast
import importlib.util
import json
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(
    description="Compare deterministic synthetic behavior against the pre-extraction source revision; no production data is read."
)
parser.add_argument("--legacy-checkout", required=True, type=Path)
parser.add_argument("--revision", default="aba2a9f68452994fe07b72ada6d210f563e48871")
args = parser.parse_args()
OLD = args.legacy_checkout.expanduser().resolve()
NEW = Path(__file__).resolve().parents[1]
ROOT = OLD.parent.parent
REVISION = args.revision
PKG = "marine_mammal_toolkit"
SIGHT = "orcacast.domains.whale.sightings"
MAP = {
    SIGHT: PKG + ".cetaceans.killer_whales.observations",
    SIGHT + ".collection": PKG + ".tools.observations.collect.pipeline",
    SIGHT + ".adapters": PKG + ".tools.observations.process.adapters",
    SIGHT + ".source_records": PKG + ".tools.observations.process.records",
    SIGHT + ".normalization": PKG + ".tools.observations.process.pipeline",
    SIGHT + ".cwr": PKG + ".tools.observations.collect.sources.cwr",
    SIGHT + ".configuration": PKG + ".cetaceans.killer_whales.configuration",
    SIGHT + ".contracts": PKG + ".tools.schemas.observations",
    SIGHT + ".runtime": PKG + ".tools.observations.runtime",
    SIGHT + ".counts": PKG + ".tools.observations.post_process.counts",
    SIGHT + ".aggregation": PKG + ".tools.observations.post_process.aggregation",
    SIGHT + ".spatial": PKG + ".tools.observations.post_process.spatial",
    SIGHT + ".model_domains": PKG + ".cetaceans.killer_whales.observations.domains",
    SIGHT + ".migration": PKG + ".tools.observations.migration",
    SIGHT + ".release": PKG + ".cetaceans.killer_whales.observations.release",
    SIGHT + ".release_pipeline": PKG + ".cetaceans.killer_whales.pipeline",
    SIGHT + ".service": PKG + ".cetaceans.killer_whales.observations.service",
    SIGHT + ".validation": PKG + ".tools.quality.observations",
    "orcacast.domains.whale.demography.prepare": PKG
    + ".cetaceans.killer_whales.populations.prepare",
    "orcacast.core.artifacts.contracts": PKG + ".tools.schemas.artifacts",
    "orcacast.core.artifacts.checksums": PKG + ".tools._core.checksums",
    "orcacast.core.artifacts": PKG + ".tools.schemas.artifacts",
    "orcacast.core.data.contracts": PKG + ".tools.schemas.stages",
    "orcacast.core.data.persistence": PKG + ".tools._core.persistence",
    "orcacast.core.data.registry": PKG + ".tools._core.registry",
    "orcacast.core.data.catalog": PKG + ".cetaceans.killer_whales.catalog",
    "orcacast.core.data.validation": PKG + ".tools.quality.tables",
    "orcacast.core.data": PKG + ".tools._core.data",
    "orcacast.core.config": PKG + ".tools._core.config",
    "orcacast.domains.environment.seascape": "seascape",
}
for p in (OLD / "src/orcacast/domains/whale/sightings/imputation").glob("*.py"):
    suffix = "" if p.stem == "__init__" else "." + p.stem
    MAP[SIGHT + ".imputation" + suffix] = PKG + ".tools.observations.impute" + suffix


def mapped(module):
    for before, after in sorted(MAP.items(), key=lambda x: -len(x[0])):
        if module == before or module.startswith(before + "."):
            return after + module[len(before) :]
    return module


def rewrite(text, module=None, is_init=False, shared=True):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    edits = []
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    if tree:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level and module:
                    package = module if is_init else module.rpartition(".")[0]
                    base = importlib.util.resolve_name("." * node.level + base, package)
                if not base.startswith("orcacast."):
                    continue
                if not shared and base.startswith("orcacast.core."):
                    continue
                fragments = []
                for item in node.names:
                    fullname = base + "." + item.name
                    if fullname in MAP:
                        target = mapped(fullname)
                        parent, _, name = target.rpartition(".")
                        alias = item.asname or item.name
                        fragments.append(
                            f"from {parent} import {name}"
                            + (f" as {alias}" if alias != name else "")
                        )
                    else:
                        fragments.append(
                            f"from {mapped(base)} import {item.name}"
                            + (f" as {item.asname}" if item.asname else "")
                        )
                indent = " " * node.col_offset
                replacement = ("\n" + indent).join(fragments)
                edits.append(
                    (
                        offsets[node.lineno - 1] + node.col_offset,
                        offsets[node.end_lineno - 1] + node.end_col_offset,
                        replacement,
                    )
                )
    for start, end, value in sorted(edits, reverse=True):
        text = text[:start] + value + text[end:]
    for before, after in sorted(MAP.items(), key=lambda x: -len(x[0])):
        if not shared and before.startswith("orcacast.core."):
            continue
        text = text.replace(before, after)
    return text


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def module_path(module, is_init=False):
    p = NEW / "src" / Path(*module.split("."))
    return p / "__init__.py" if is_init else p.with_suffix(".py")


"""Deterministic extraction parity against the pre-migration Git revision."""
import json, subprocess, sys, types, tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import h3
from openpyxl import Workbook

prefix = "orcacast.domains.whale.sightings.imputation"
for path in subprocess.check_output(
    [
        "git",
        "-C",
        str(OLD),
        "ls-tree",
        "-r",
        "--name-only",
        REVISION,
        "src/orcacast/domains/whale/sightings/imputation",
    ],
    text=True,
).splitlines():
    stem = Path(path).stem
    suffix = "" if stem == "__init__" else "." + stem
    MAP[prefix + suffix] = "marine_mammal_toolkit.tools.observations.impute" + suffix
MAP[prefix + ".features"] = (
    "marine_mammal_toolkit.cetaceans.killer_whales.observations.features"
)


def baseline(relative):
    raw = subprocess.check_output(
        ["git", "-C", str(OLD), "show", REVISION + ":" + relative], text=True
    )
    module = ".".join(Path(relative).with_suffix("").parts[1:])
    name = "parity_" + Path(relative).stem
    m = types.ModuleType(name)
    m.__file__ = str(OLD / relative)
    sys.modules[name] = m
    exec(compile(rewrite(raw, module), m.__file__, "exec"), m.__dict__)
    return m


from marine_mammal_toolkit.cetaceans.killer_whales.configuration import (
    load_sightings_config,
)
from marine_mammal_toolkit.cetaceans.killer_whales.resources import config_path
from marine_mammal_toolkit.tools._core.config import workspace
from marine_mammal_toolkit.tools.observations.process import pipeline as current
from marine_mammal_toolkit.tools.observations.process.adapters import _source_record
from marine_mammal_toolkit.tools.observations.post_process import counts
from marine_mammal_toolkit.cetaceans.killer_whales.populations.prepare import (
    load_population_rows,
)
from marine_mammal_toolkit.cetaceans.killer_whales.observations.features import (
    DateContextFeatureBuilder,
    prepare_model_frame,
)
from marine_mammal_toolkit.tools.observations.impute.config import FeatureConfig

before = baseline("src/orcacast/domains/whale/sightings/normalization.py")
before_counts = baseline("src/orcacast/domains/whale/sightings/counts.py")
before_pop = baseline("src/orcacast/domains/whale/demography/prepare.py")
before_features = baseline(
    "src/orcacast/domains/whale/sightings/imputation/features.py"
)
with tempfile.TemporaryDirectory(prefix="mammal-parity-") as temp, workspace(temp):
    root = Path(temp)
    _, config = load_sightings_config(config_path())
    rows = []
    for i, (source, label, lon) in enumerate(
        [
            ("TWM", "SRKW", -123.0),
            ("INATURALIST", "SRKW", -123.0001),
            ("ACARTIA", "Transient", -123.1),
            ("MAPLIFY", "Unknown", -123.2),
            ("TWM", "NRKW", -123.3),
        ]
    ):
        row = _source_record(
            source,
            str(i),
            {"id": i},
            observed_date="2025-06-03",
            latitude=48.5,
            longitude=lon,
            species="Orcinus orca",
            pod_ecotype=label,
            coordinate_uncertainty_m=50,
            source_license="UNKNOWN",
            source_use_class="INTERNAL_ONLY",
            source_qc_status="ACCEPTED",
        )
        row.update(
            SOURCE_RETRIEVED_AT_UTC=pd.Timestamp("2025-06-05T00:00Z"),
            SOURCE_PAYLOAD_CORRECTED=False,
            LAST_CORRECTED_AT_UTC=pd.Timestamp("2025-06-05T00:00Z"),
        )
        rows.append(row)
    a, aa = before._normalize_records(pd.DataFrame(rows), config)
    b, ba = current._normalize_records(pd.DataFrame(rows), config)
    assert a == b and aa == ba
    ga = before._cluster(a, config, aa)
    gb = current._cluster(b, config, ba)
    assert ga == gb
    for name in ("before", "after"):
        (root / name).mkdir()
    ia = before._resolve_identities(
        ga, *[root / "before" / x for x in ("identity", "alias", "lineage")], "parity"
    )
    ib = current._resolve_identities(
        gb,
        *[root / "after" / x for x in ("identity", "alias", "lineage")],
        "parity",
        policy=config.observation_policy,
    )
    assert ia[0] == ib[0]
    for x, y in zip(ia[1:], ib[1:]):
        assert x.equals(y)
    oa, assoca = before._materialize(ga, aa, ia[0])
    ob, assocb = current._materialize(gb, ba, ib[0], policy=config.observation_policy)
    assert oa == ob and assoca == assocb
    frame = pd.DataFrame(oa)
    assoc = pd.DataFrame(assoca)
    cells = {h3.latlng_to_cell(row["LATITUDE"], row["LONGITUDE"], 6) for row in oa}
    ca = before_counts._daily_counts(frame, assoc, cells, 6)
    cb = counts._daily_counts(frame, assoc, cells, 6, policy=config.count_policy)
    for x, y in zip(ca, cb):
        pd.testing.assert_frame_equal(x, y)
    book = Workbook()
    book.active.title = "Chart Data"
    book.active.append(["Census Year", "J Pod", "K Pod", "L Pod", "All Pods"])
    book.active.append([2021, 2, 3, 4, 9])
    book.active.append([2020, 1, 2, 3, 6])
    book.save(root / "census.xlsx")
    assert before_pop.load_population_rows(
        root / "census.xlsx", sheet_name="Chart Data"
    ) == load_population_rows(root / "census.xlsx", sheet_name="Chart Data")
    features = FeatureConfig(
        marine_fallback_to_haversine=True,
        water_network_config_path=str(root / "missing.yaml"),
    )
    prepared = prepare_model_frame(frame, features)
    from marine_mammal_toolkit.tools.observations.impute import marine
    from seascape.spatial_support.water_network import WaterGraph

    feature_cells = sorted(
        {h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], 6) for r in oa}
    )
    graph = WaterGraph(
        resolution=6,
        cells=np.array(feature_cells),
        offsets=np.zeros(len(feature_cells) + 1, dtype=int),
        neighbors=np.array([], dtype=int),
        weights_m=np.array([]),
        support=pd.DataFrame({"H3_INDEX": feature_cells}),
        cell_to_position={c: i for i, c in enumerate(feature_cells)},
        water_mask_version="synthetic",
        spatial_support_version="synthetic",
    )
    marine.load_water_graph = lambda *args, **kwargs: graph
    # Build identical explicit context features; unavailable marine network is an intentional synthetic fallback.
    fa = before_features.DateContextFeatureBuilder(features).transform(
        prepared, prepared
    )
    fb = DateContextFeatureBuilder(features).transform(prepared, prepared)
    pd.testing.assert_frame_equal(fa.X, fb.X)
    pd.testing.assert_frame_equal(fa.meta, fb.meta)
    result = {
        "source_revision": subprocess.check_output(
            ["git", "-C", str(OLD), "rev-parse", REVISION], text=True
        ).strip(),
        "checks": [
            "canonical rows",
            "audit",
            "clustering",
            "observation ids",
            "identity aliases and lineage",
            "associations",
            "five count tables",
            "annual census ordering and reconciliation",
            "feature matrix",
            "feature metadata",
        ],
        "observations": len(oa),
        "status": "passed",
    }
    print(json.dumps(result, indent=2))
