"""Execute the curated documentation rather than a duplicated test snippet."""

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTENT = (ROOT / "docs/smoke-examples.md").read_text()
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


@pytest.mark.parametrize(
    "backend,code",
    re.findall(r"<!-- backend: (\w+) -->\n```python\n(.*?)```", CONTENT, re.S),
)
def test_documented_python_example(backend, code):
    dependency = {"torch": "torch", "nnx": "flax", "tf": "tensorflow"}[backend]
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"Optional {dependency} is not installed")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_documented_shell_commands():
    for block in re.findall(r"```sh\n(.*?)```", CONTENT, re.S):
        for line in block.strip().splitlines():
            executable, *args = shlex.split(line)
            assert executable == "python"
            result = subprocess.run(
                [sys.executable, *args],
                env=ENV,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stderr
