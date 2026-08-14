"""Synthetic child-path tests; no clean/protected dataset is opened."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROUND5_SOURCE_COMMIT = "a9a6f03a8fe984ee75c15fbcf81f7c04c5ab2e46"
SCIENTIFIC_OUTPUT_NAMES = (
    "clean_a5_report.json",
    "thresholds.json",
    "a5_numeric_trace.json",
    "a5_backend_truth.json",
)


def _module(source_path=None):
    root = Path(__file__).parents[1]
    path = source_path or root / "scripts/run_gcspo_clean_a5.py"
    spec = importlib.util.spec_from_file_location(
        f"gcspo_clean_a5_child_{id(path)}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_synthetic(module, monkeypatch, artifact, challenge, receipt=None):
    artifact.mkdir()
    (artifact / "config.json").write_text(json.dumps({
        "h0_predictor": {"ridge_grid": [1.0]},
        "lambda_selection": {"grid": [100.0]},
    }) + "\n")
    (artifact / "thresholds.json").write_text("{}\n")
    monkeypatch.setattr(module, "_backend_truth", lambda requested: {
        "requested": requested, "resolved": requested, "cuda_available": True,
        "device": "Synthetic GPU", "torch_version": "synthetic", "cuda_version": "synthetic",
    })
    monkeypatch.setattr(module, "run_clean_a1", lambda *_args, **_kwargs: {
        "data": object(), "model": object(), "whitener": object(), "gamma": object(),
    })
    calls = iter(([{"synthetic": index} for index in range(100)],
                  [{"synthetic": "calibration"}], [{"synthetic": "holdout"}]))
    monkeypatch.setattr(module, "role_a5_terms", lambda *_args, **_kwargs: next(calls))
    monkeypatch.setattr(module, "_grid_objective", lambda pair: [float(pair[1][0])])
    trace = {"window_start_s": 1.0, "availability_s": 2.0, "prns": [1],
             "epoch_ids": [1], "epoch_prn_support": [1], "score": 1.0,
             "rss": 2.0, "gcv": 3.0, "effective_dof": 4.0, "rank": 1,
             "state": [5.0]}
    monkeypatch.setattr(module, "score_parallel", lambda *_args, **_kwargs: [dict(trace)])
    monkeypatch.setattr(module, "_source_commit", lambda: "a" * 40)
    argv = [str(Path(module.__file__).resolve()), "--artifact-dir", str(artifact),
            "--clean-root", str(artifact / "unused-clean"), "--workers", "1",
            "--backend", "cuda", "--numeric-trace"]
    if receipt is not None:
        argv.extend(["--run-id", "round5-cuda-1", "--nonce", "1" * 64,
                     "--execution-receipt", str(receipt),
                     "--challenge-file", str(challenge)])
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    return [sys.executable, *argv]


def test_witness_arguments_enter_actual_child_receipt_without_changing_scientific_bytes(
        tmp_path, monkeypatch):
    assert str(tmp_path).startswith("/tmp/")
    challenge = tmp_path / "challenge.json"
    challenge.write_text("{}\n")
    module = _module()
    plain = tmp_path / "plain"
    _run_synthetic(module, monkeypatch, plain, challenge)

    module = _module()
    witnessed = tmp_path / "witnessed"
    receipt = tmp_path / "execution_receipt.json"
    actual_argv = _run_synthetic(module, monkeypatch, witnessed, challenge, receipt)
    document = json.loads(receipt.read_text())
    assert document["run_id"] == "round5-cuda-1"
    assert document["nonce"] == "1" * 64
    assert document["argv"] == actual_argv
    assert document["process_identity"]["pid"] > 0
    assert document["process_identity"]["proc_start_ticks"] > 0
    assert document["transcript_state"]["event_count"] == 4
    assert set(document["scientific_outputs"]) == {
        "clean_a5_report.json", "thresholds.json", "a5_numeric_trace.json",
        "a5_backend_truth.json",
    }
    for name in document["scientific_outputs"]:
        assert (plain / name).read_bytes() == (witnessed / name).read_bytes()


def test_real_child_receipt_is_exact_compact_canonical_json_plus_lf(
        tmp_path, monkeypatch):
    """The verifier must accept the exact receipt emitted by the child path."""
    assert str(tmp_path).startswith("/tmp/")
    challenge = tmp_path / "challenge.json"
    challenge.write_text("{}\n")
    receipt = tmp_path / "execution_receipt.json"
    module = _module()
    _run_synthetic(module, monkeypatch, tmp_path / "witnessed", challenge, receipt)

    from gnss_doppler_lab.gcspo_witness import _canonical_document

    document = json.loads(receipt.read_text())
    assert _canonical_document(receipt, "execution receipt") == document
    expected = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode() + b"\n"
    assert receipt.read_bytes() == expected


def test_all_scientific_output_bytes_match_exact_round5_source_commit(
        tmp_path, monkeypatch):
    """The receipt-only repair cannot alter any scientific serialization path."""
    assert str(tmp_path).startswith("/tmp/")
    root = Path(__file__).parents[1]
    baseline_source = tmp_path / "run_gcspo_clean_a5_round5.py"
    baseline_source.write_bytes(subprocess.run(
        ["git", "show", f"{ROUND5_SOURCE_COMMIT}:scripts/run_gcspo_clean_a5.py"],
        cwd=root, check=True, capture_output=True,
    ).stdout)
    challenge = tmp_path / "challenge.json"
    challenge.write_text("{}\n")

    baseline = tmp_path / "baseline"
    _run_synthetic(_module(baseline_source), monkeypatch, baseline, challenge)
    current = tmp_path / "current"
    receipt = tmp_path / "execution_receipt.json"
    _run_synthetic(_module(), monkeypatch, current, challenge, receipt)

    document = json.loads(receipt.read_text())
    assert set(document["scientific_outputs"]) == set(SCIENTIFIC_OUTPUT_NAMES)
    assert receipt.name not in document["scientific_outputs"]
    for name in SCIENTIFIC_OUTPUT_NAMES:
        assert (current / name).read_bytes() == (baseline / name).read_bytes()
