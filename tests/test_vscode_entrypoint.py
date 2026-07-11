"""Regression tests for the VS Code-friendly script entry point."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_iq.py"


def test_generate_script_can_show_help_without_package_install() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Generate normal GPS L1 C/A software IQ" in result.stdout


def test_visualize_script_can_show_help_without_package_install() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    script = REPO_ROOT / "scripts" / "visualize_iq.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "GPS L1 C/A IQ" in result.stdout
