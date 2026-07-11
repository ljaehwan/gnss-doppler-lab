"""Shared helpers for the ordered, notebook-driven research workflow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _complete_rf_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path.parent
        for path in root.glob("*/manifest.json")
        if (path.parent / "gps_l1ca_s8_iq.bin").is_file()
    ]


def latest_run(root: str | Path) -> Path:
    """Return the newest RF run containing both manifest and IQ artifacts."""
    runs = _complete_rf_runs(Path(root))
    if not runs:
        raise FileNotFoundError(f"No complete RF run found under {Path(root)}")
    return max(runs, key=lambda run: (run / "manifest.json").stat().st_mtime_ns)


def load_run_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Load and validate a run's JSON manifest."""
    path = Path(run_dir) / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return data


def _newest_parent(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max((path.parent for path in paths), key=lambda path: path.stat().st_mtime_ns)


def _entry(path: Path | None, detail: str) -> dict[str, Any]:
    return {"ready": path is not None, "path": str(path) if path else None, "detail": detail}


def sequence_status(artifacts_root: str | Path) -> dict[str, dict[str, Any]]:
    """Report artifact readiness for each research sequence stage."""
    root = Path(artifacts_root)
    normal_runs = _complete_rf_runs(root / "rf_runs")
    normal = max(normal_runs, key=lambda p: (p / "manifest.json").stat().st_mtime_ns) if normal_runs else None
    receiver_candidates = list((root / "receiver_runs").glob("*/tracking.csv"))
    receiver_candidates += list((root / "receiver_runs").glob("*/observables.csv"))
    receiver = _newest_parent(receiver_candidates)
    spoof = _newest_parent(list((root / "spoofing_runs").glob("*/gps_l1ca_s8_iq.bin")))
    comparison = _newest_parent(list((root / "comparisons").glob("*/comparison.csv")))
    dataset_files = list((root / "datasets").glob("*.csv")) + list((root / "datasets").glob("*.parquet"))
    dataset = max(dataset_files, key=lambda p: p.stat().st_mtime_ns) if dataset_files else None
    return {
        "01_normal_iq": _entry(normal, "normal IQ + manifest"),
        "02_receiver_processing": _entry(receiver, "GNSS-SDR observables"),
        "03_spoofing_comparison": _entry(spoof, "spoofing IQ"),
        "03_comparison_table": _entry(comparison, "normal/spoofing aligned comparison"),
        "04_detection_dataset": _entry(dataset, "windowed feature dataset"),
    }
