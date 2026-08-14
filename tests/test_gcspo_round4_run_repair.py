"""Synthetic regression for fork-safe explicit A5 execution."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


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


def test_cuda_backend_is_initialized_and_attested_before_workload(monkeypatch):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "gcspo_clean_a5_runner_backend", root / "scripts/run_gcspo_clean_a5.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    initialized = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        init=lambda: initialized.append(True),
        get_device_name=lambda _index: "Synthetic GPU",
    )
    fake_torch = SimpleNamespace(cuda=cuda, __version__="test", version=SimpleNamespace(cuda="test"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert module._backend_truth("cuda") == {
        "requested": "cuda", "resolved": "cuda", "cuda_available": True,
        "device": "Synthetic GPU", "torch_version": "test", "cuda_version": "test",
    }
    assert initialized == [True]

def test_causal_launcher_preserves_virtualenv_symlink_path(tmp_path):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "gcspo_causal_runner", root / "scripts/run_gcspo_a5_causal.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    launcher = tmp_path / "venv-python"
    launcher.symlink_to(Path(sys.executable).resolve())
    observed = module._python_command_path(launcher)
    assert observed == str(launcher.absolute())
    assert observed != str(launcher.resolve())
