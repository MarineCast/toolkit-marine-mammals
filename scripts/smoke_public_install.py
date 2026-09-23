"""Offline acceptance of an installed public package, run outside the checkout."""

from __future__ import annotations

import importlib.abc
import json
import socket
import sys
import tempfile
from pathlib import Path


class NoOptionalStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "seascape",
            "orcacast",
            "sklearn",
            "matplotlib",
            "plotly",
            "geopandas",
        }:
            raise AssertionError(f"Basic queries must not import {fullname}")
        return None


sys.meta_path.insert(0, NoOptionalStack())


def no_network(*args, **kwargs):
    raise AssertionError("The public demo must work offline")


socket.create_connection = no_network
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
from marine_mammal_toolkit.cetaceans.killer_whales.query import run_demo
from marine_mammal_toolkit.cli import cli
from click.testing import CliRunner

with tempfile.TemporaryDirectory(prefix="marine-mammals-smoke-") as temporary:
    root = Path(temporary)
    result = run_demo(root)
    assert result.observations.row_count == 2
    assert not result.read().PUBLIC_RELEASE_ELIGIBLE.any()
    response = CliRunner().invoke(
        cli, ["--workspace-root", str(root), "killer-whales", "observations", "sources"]
    )
    assert response.exit_code == 0, response.exception
    assert "gbif" in json.loads(response.output)["sources"]
    release = CliRunner().invoke(
        cli,
        [
            "--workspace-root",
            str(root),
            "killer-whales",
            "observations",
            "run",
            "--profile",
            "observations-only",
            "--config",
            str(result.query_root / "query.yaml"),
            "--twm-file",
            str(root / "demo-inputs/synthetic-twm.csv"),
            "--end-date",
            "2025-06-08",
        ],
    )
    assert release.exit_code == 0, release.exception
    assert Path(release.output.strip()).is_file()
    print(
        "PASS: installed API, source discovery, offline query/release, and optional dependency isolation"
    )
