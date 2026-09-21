"""Release integrity regressions for #156; no index credentials are needed."""

import hashlib
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_release", ROOT / "scripts/verify_release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


@pytest.mark.parametrize(
    "tag", ["v1.2", "v1.2.3junk", "v01.2.3", "v1.2.3rc1", "v1.2.3+local", "v1.2.3\n"]
)
def test_malformed_tags_fail(tag):
    with pytest.raises(ValueError, match="Unsupported release tag"):
        release.validate_tag(tag)


def archives(root, wheel_version="1.2.3", sdist_version="1.2.3", missing=None):
    with zipfile.ZipFile(root / "nmn.whl", "w") as wheel:
        for name in release.REQUIRED - {missing}:
            wheel.writestr(name, "")
        wheel.writestr(
            "nmn.dist-info/METADATA", f"Name: nmn\nVersion: {wheel_version}\n"
        )
    with tarfile.open(root / "nmn.tar.gz", "w:gz") as sdist:
        files = {"nmn/src/" + name: b"" for name in release.REQUIRED}
        files["nmn/PKG-INFO"] = f"Name: nmn\nVersion: {sdist_version}\n".encode()
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            sdist.addfile(info, io.BytesIO(data))


def test_matching_archives_produce_content_digests(tmp_path):
    archives(tmp_path)
    hashes = release.verify_archives(tmp_path, "v1.2.3")
    assert hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()
    }


@pytest.mark.parametrize(
    "kwargs",
    [{"wheel_version": "1.2.4"}, {"sdist_version": "1.2.4"}, {"missing": "nmn/cli.py"}],
)
def test_version_mismatch_and_broken_wheel_fail(tmp_path, kwargs):
    archives(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        release.verify_archives(tmp_path, "v1.2.3")


def test_oidc_jobs_only_promote_verified_artifacts():
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    build, testpypi, pypi = workflow.split("  publish-to-testpypi:")[
        0
    ], *workflow.split("  publish-to-testpypi:")[1].split("  publish-to-pypi:")
    assert "python scripts/check_release_ci.py" in build
    assert "twine check --strict" in build
    assert "--no-isolation" in build
    assert "-I -m nmn doctor" in build
    assert build.index("verify_release.py dist") < build.index(
        "actions/upload-artifact"
    )
    for job in (testpypi, pypi):
        assert job.index("Distribution digest mismatch") < job.index(
            "pypa/gh-action-pypi-publish"
        )
        assert "skip-existing: false" in job
        assert "python -m build" not in job
    assert "needs: [build, publish-to-testpypi]" in pypi


def test_rehashed_or_modified_artifact_is_rejected(tmp_path, monkeypatch):
    import sys

    archives(tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_release.py", str(tmp_path)])
    release.main()
    with zipfile.ZipFile(tmp_path / "nmn.whl", "a") as wheel:
        wheel.writestr("unexpected.txt", "changed after validation")
    monkeypatch.setattr(sys, "argv", ["verify_release.py", str(tmp_path), "--check"])
    with pytest.raises(ValueError, match="digest mismatch"):
        release.main()


@pytest.mark.parametrize(
    "runs,passes",
    [
        ([], False),
        (
            [
                {
                    "id": 1,
                    "head_sha": "abc",
                    "head_branch": "master",
                    "conclusion": "success",
                }
            ],
            True,
        ),
        (
            [
                {
                    "id": 1,
                    "head_sha": "different",
                    "head_branch": "master",
                    "conclusion": "success",
                }
            ],
            False,
        ),
        (
            [
                {
                    "id": 1,
                    "head_sha": "abc",
                    "head_branch": "feature",
                    "conclusion": "success",
                }
            ],
            False,
        ),
        (
            [
                {
                    "id": 1,
                    "head_sha": "abc",
                    "head_branch": "master",
                    "conclusion": "success",
                },
                {
                    "id": 2,
                    "head_sha": "abc",
                    "head_branch": "master",
                    "conclusion": "failure",
                },
            ],
            False,
        ),
    ],
)
def test_release_requires_latest_success_for_exact_master_commit(
    monkeypatch, runs, passes
):
    import json

    spec = importlib.util.spec_from_file_location(
        "check_ci", ROOT / "scripts/check_release_ci.py"
    )
    check_ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_ci)
    monkeypatch.setenv("GITHUB_REPOSITORY", "azettaai/nmn")
    monkeypatch.setattr(
        check_ci.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "abc\n" if args[0] == "git" else json.dumps({"workflow_runs": runs})
        ),
    )
    if passes:
        check_ci.main()
    else:
        with pytest.raises(SystemExit, match="latest Test Suite"):
            check_ci.main()
