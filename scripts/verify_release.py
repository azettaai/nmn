"""Verify release archives before granting a publishing job OIDC access (#156)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

# Stable releases only. Prereleases require an explicit policy change.
TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
REQUIRED = {
    "nmn/__init__.py",
    "nmn/__main__.py",
    "nmn/cli.py",
    "nmn/py.typed",
    "nmn/conformance_manifest.json",
}


def validate_tag(tag: str) -> str:
    """Return the exact stable version or reject a malformed release tag."""
    if not TAG.fullmatch(tag):
        raise ValueError(
            f"Unsupported release tag: {tag!r}; expected vMAJOR.MINOR.PATCH"
        )
    return tag[1:]


def verify_archives(directory: Path, tag: str | None = None) -> dict[str, str]:
    """Validate matching wheel/sdist metadata and required installed contents."""
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    expected = validate_tag(tag) if tag else None
    versions = []
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        if not REQUIRED <= names:
            raise ValueError(f"Wheel missing required files: {REQUIRED - names}")
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected one wheel METADATA")
        messages = [BytesParser().parsebytes(wheel.read(metadata[0]))]
    with tarfile.open(sdists[0]) as sdist:
        members = sdist.getmembers()
        metadata = [
            m
            for m in members
            if m.name.endswith("/PKG-INFO") and m.name.count("/") == 1
        ]
        if len(metadata) != 1:
            raise ValueError("Expected one root sdist PKG-INFO")
        contents = {m.name.split("/", 1)[-1] for m in members}
        if not {"src/" + name for name in REQUIRED} <= contents:
            raise ValueError("Sdist missing required package contents")
        stream = sdist.extractfile(metadata[0])
        if stream is None:
            raise ValueError("Invalid sdist metadata")
        messages.append(BytesParser().parsebytes(stream.read()))
    for message in messages:
        if message["Name"] != "nmn" or not message["Version"]:
            raise ValueError("Invalid package name/version")
        versions.append(message["Version"])
    if versions[0] != versions[1] or (expected and versions[0] != expected):
        raise ValueError(f"Version mismatch: tag={expected}, archives={versions}")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels + sdists}


def main() -> None:
    """Write or verify a content manifest supplied by the build job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    hashes = verify_archives(args.directory, args.tag)
    manifest = args.directory / "SHA256SUMS.json"
    if args.check:
        if json.loads(manifest.read_text()) != hashes:
            raise ValueError("Distribution digest mismatch")
    else:
        manifest.write_text(json.dumps(hashes, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
