"""Regression tests for the standalone IQ generation entry point."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_iq.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_generate_script_can_show_help_without_package_install() -> None:
    result = _run("--help")

    assert result.returncode == 0, result.stderr
    assert "Generate normal or RF-level spoofed GPS L1 C/A software IQ" in result.stdout


def test_generate_script_requires_explicit_notebook_created_config() -> None:
    result = _run()

    assert result.returncode != 0
    assert "required" in result.stderr
    assert "configs/gps_l1ca_static.example.yaml" not in result.stderr
