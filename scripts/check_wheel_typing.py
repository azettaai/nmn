"""Type-check a consumer from outside the source tree in an installed-wheel venv."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GOOD = """from nmn.torch import YatNMN as TorchDense
from nmn.keras import YatNMN as KerasDense
from nmn.tf import YatNMN as TFDense
TorchDense(in_features=3, out_features=4)
KerasDense(units=4, use_bias=True)
TFDense(features=4)
"""
BAD = """from nmn.torch import YatNMN as TorchDense
from nmn.keras import YatNMN as KerasDense
from nmn.tf import YatNMN as TFDense
TorchDense(in_features="wrong", out_features=4)
KerasDense(units="wrong")
TFDense(features="wrong")
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nmn-consumer-") as directory:
        root = Path(directory)
        subprocess.run([sys.executable, "-m", "venv", str(root / "venv")], check=True)
        python = (
            root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "mypy==2.3.1",
                str(args.wheel.resolve()),
            ],
            check=True,
        )
        env = {
            k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MYPYPATH")
        }
        for name, contents in [("good", GOOD), ("bad", BAD)]:
            path = root / (name + ".py")
            path.write_text(contents)
            result = subprocess.run(
                [
                    str(python),
                    "-m",
                    "mypy",
                    "--follow-imports=silent",
                    "--ignore-missing-imports",
                    "--no-incremental",
                    str(path),
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            if name == "good" and result.returncode != 0:
                raise SystemExit(result.stdout + result.stderr)
            if name == "bad" and (
                result.returncode != 1 or result.stdout.count("[arg-type]") != 3
            ):
                raise SystemExit(
                    "Installed signatures failed to reject three invalid scalar arguments: "
                    + result.stdout
                    + result.stderr
                )


if __name__ == "__main__":
    main()
