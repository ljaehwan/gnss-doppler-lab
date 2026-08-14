"""Synthetic regression for fork-safe explicit A5 execution."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_one_worker_a5_scoring_runs_in_launch_process_without_pool(monkeypatch):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "gcspo_clean_a5_runner", root / "scripts/run_gcspo_clean_a5.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_pool", lambda _workers: (_ for _ in ()).throw(
        AssertionError("one-worker path must not fork")))
    monkeypatch.setattr(module, "_score_index", lambda pair: {"index": pair[0], "lambda": pair[1]})
    rows = [{"synthetic": 1}, {"synthetic": 2}]
    assert module.score_parallel(rows, 100.0, 1) == [
        {"index": 0, "lambda": 100.0},
        {"index": 1, "lambda": 100.0},
    ]

