from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_peak_floor_contrastive_predictive.py"


def load_module():
    spec = importlib.util.spec_from_file_location("peak_floor_cpc", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_frames(epochs: int = 48, prns=("G01", "G02", "G03")):
    m = load_module()
    morph_rows = []
    for ti in range(epochs):
        t = 0.5 * (ti + 1)
        for pi, prn in enumerate(prns):
            row = {
                "run_id": "clean-run",
                "source_fingerprint": "synthetic-clean-source",
                "label": "clean",
                "tap_count": 9,
                "prn": prn,
                "window_bin_s": t,
                "window_start_s": t - 0.48,
                "window_end_s": t,
            }
            for fi, col in enumerate(m.DEFAULT_MORPH_FEATURES):
                row[col] = np.sin(t / 8.0 + fi / 17.0) + pi * 0.03
            morph_rows.append(row)
    floor_rows = []
    for ti in range(epochs + 1):
        t = 0.5 * ti
        row = {"scenario": "clean", "window_start_s": t, "window_end_s": t + 0.01}
        for fi, col in enumerate(m.DEFAULT_FLOOR_FEATURES):
            row[col] = np.cos(t / 9.0 + fi / 13.0)
        floor_rows.append(row)
    return pd.DataFrame(morph_rows), pd.DataFrame(floor_rows)


def test_predictive_pairs_use_only_past_context_and_future_endpoint():
    m = load_module()
    morph, floor = synthetic_frames(epochs=16)
    pairs = m.make_predictive_pairs(m.align_modalities(morph, floor), context_len=4, horizon=1, stride=1)
    assert pairs.context_morph.shape[1] == 4
    assert pairs.target_morph.shape[1] == m.MAX_PRNS
    assert np.all(pairs.target_times > pairs.context_times[:, -1])
    assert np.allclose(pairs.target_times - pairs.context_times[:, -1], 0.5)
    assert np.all(pairs.available_times >= pairs.target_times)


def test_info_nce_prefers_correct_future_pairing():
    m = load_module()
    actual = torch.eye(6)
    predicted = actual.clone()
    good = m.symmetric_info_nce(predicted, actual, temperature=0.1)
    bad = m.symmetric_info_nce(predicted, actual.roll(1, dims=0), temperature=0.1)
    assert good.item() < bad.item()


def test_model_predicts_normalized_future_embedding_from_joint_peak_floor_context():
    m = load_module()
    cfg = m.ModelConfig(morph_dim=5, floor_dim=7, context_len=4, hidden_dim=24,
                        embedding_dim=12, token_layers=1, token_heads=4, dropout=0.0)
    model = m.PeakFloorCPC(cfg)
    context_morph = torch.randn(3, 4, m.MAX_PRNS, 5)
    context_floor = torch.randn(3, 4, 7)
    context_mask = torch.zeros(3, 4, m.MAX_PRNS, dtype=torch.bool); context_mask[:, :, :3] = True
    target_morph = torch.randn(3, m.MAX_PRNS, 5)
    target_floor = torch.randn(3, 7)
    target_mask = torch.zeros(3, m.MAX_PRNS, dtype=torch.bool); target_mask[:, :3] = True
    out = model(context_morph, context_floor, context_mask, target_morph, target_floor, target_mask)
    assert out["predicted"].shape == (3, 12)
    assert out["actual"].shape == (3, 12)
    assert torch.allclose(out["predicted"].norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(out["actual"].norm(dim=1), torch.ones(3), atol=1e-5)


def test_scoring_timestamp_is_future_endpoint_and_is_causal():
    m = load_module()
    morph, floor = synthetic_frames(epochs=16)
    aligned = m.align_modalities(morph, floor)
    scalers = m.fit_robust_scalers(aligned)
    pairs = m.make_predictive_pairs(m.apply_scalers(aligned, scalers), context_len=4, horizon=1)
    cfg = m.ModelConfig(morph_dim=len(m.DEFAULT_MORPH_FEATURES), floor_dim=len(m.DEFAULT_FLOOR_FEATURES),
                        context_len=4, hidden_dim=24, embedding_dim=12, token_layers=1, token_heads=4, dropout=0.0)
    scores = m.score_pairs(m.PeakFloorCPC(cfg), pairs, batch_size=4, device=torch.device("cpu"))
    assert scores.window_start_s.tolist() == pytest.approx(pairs.target_times.tolist())
    assert np.all(scores.available_time_s.to_numpy() >= scores.window_start_s.to_numpy())
    assert np.all((scores.pf_cpc_surprisal >= 0.0) & (scores.pf_cpc_surprisal <= 2.0))


def test_tiny_normal_only_campaign_writes_cpc_artifacts(tmp_path):
    m = load_module()
    morph, floor = synthetic_frames(epochs=64)
    morph_csv = tmp_path / "clean_morph.csv"; floor_csv = tmp_path / "clean_floor.csv"
    morph.to_csv(morph_csv, index=False); floor.to_csv(floor_csv, index=False)
    out = tmp_path / "artifact"
    result = m.run_campaign(
        morph_csv=morph_csv, floor_csv=floor_csv, output_dir=out, epochs=1, batch_size=8,
        context_len=4, hidden_dim=24, embedding_dim=12, token_layers=1, token_heads=4,
        split_rules={"train": (None, 12.0), "validation": (13.0, 18.0),
                     "calibration": (19.0, 24.0), "held_clean": (25.0, None)},
        device="cpu", seed=7,
    )
    required = {"model.pt", "model_metadata.json", "scalers.json", "split_manifest.json",
                "training_history.csv", "calibration_scores.csv", "calibration.json",
                "held_clean_scores.csv", "held_clean_summary.json", "campaign_manifest.json"}
    assert required.issubset({p.name for p in out.iterdir() if p.is_file()})
    meta = json.loads((out / "model_metadata.json").read_text())
    assert meta["architecture"] == "PeakFloorCPC"
    assert meta["normal_only_training"] is True
    assert meta["objective"] == "symmetric_info_nce_future_prediction"
    assert 0.0 <= result["held_clean"]["mean_p_value"] <= 1.0
