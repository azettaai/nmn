"""Require a successful Test Suite push run for the exact release commit."""

import json
import os
import subprocess


def main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    # Annotated tag event SHAs are resolved to the commit rather than the tag object.
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    endpoint = f"repos/{repository}/actions/workflows/test.yml/runs?head_sha={sha}&event=push&per_page=100"
    report = json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))
    runs = [
        run
        for run in report["workflow_runs"]
        if run["head_sha"] == sha and run["head_branch"] == "master"
    ]
    if not runs or max(runs, key=lambda run: run["id"])["conclusion"] != "success":
        raise SystemExit(
            "Release requires the latest Test Suite push run on this master commit to succeed"
        )


if __name__ == "__main__":
    main()
