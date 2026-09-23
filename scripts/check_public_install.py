"""Install a built wheel in a clean venv and run the offline public acceptance demo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv

parser = argparse.ArgumentParser()
parser.add_argument("--wheel-dir", required=True, type=Path)
args = parser.parse_args()
wheels = list(args.wheel_dir.resolve().glob("marine_mammal_toolkit-*.whl"))
if len(wheels) != 1:
    raise SystemExit("Expected exactly one marine-mammal toolkit wheel")
with tempfile.TemporaryDirectory(prefix="marine-mammals-install-") as temporary:
    root = Path(temporary)
    environment = root / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    outside = root / "outside"
    outside.mkdir()
    smoke = outside / "smoke_public_install.py"
    shutil.copy2(Path(__file__).with_name("smoke_public_install.py"), smoke)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    for command in (
        [str(python), "-m", "pip", "install", str(wheels[0])],
        [str(python), "-m", "pip", "check"],
        [str(python), str(smoke)],
    ):
        subprocess.run(command, cwd=outside, env=env, check=True)
