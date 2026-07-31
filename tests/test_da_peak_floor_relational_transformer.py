from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_da_peak_floor_relational_transformer.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_imports_without_temporal_autoencoder_files(tmp_path):
    isolated_script = tmp_path / "scripts" / SCRIPT.name
    isolated_script.parent.mkdir()
    isolated_script.write_text(SCRIPT.read_text())
    module = load(isolated_script, "da_pf_isolated")
    assert module.MAX_PRNS == 32
    assert len(module.DEFAULT_MORPH_FEATURES) == 35
    assert len(module.DEFAULT_FLOOR_FEATURES) == 57
    morph, floor = synthetic_frames(module, epochs=8)
    aligned = module.align_modalities(morph, floor)
    scaled = module.apply_scalers(aligned, module.fit_robust_scalers(aligned))
    sequences = module.make_sequences(scaled, seq_len=4)
    assert len(sequences) == 5
    assert sequences.morph.shape == (5, 4, 32, 35)


def synthetic_frames(module, epochs: int = 36, prns=("G01", "G02", "G03")):
    import pandas as pd
    morph_rows = []
    for time_index in range(epochs):
        time = 0.5 * (time_index + 1)
        for prn_index, prn in enumerate(prns):
            row = {"run_id": "clean-run", "label": "clean", "prn": prn,
                   "window_bin_s": time, "window_start_s": time - 0.48 + prn_index * 0.001,
                   "window_end_s": time + 0.02 + prn_index * 0.001}
            for feature_index, column in enumerate(module.DEFAULT_MORPH_FEATURES):
                row[column] = np.sin(time / 8.0 + feature_index / 17.0) + prn_index * 0.03
            morph_rows.append(row)
    floor_rows = []
    for time_index in range(epochs + 1):
        time = 0.5 * time_index
        row = {"scenario": "clean", "window_index": time_index, "window_start_s": time,
               "window_mid_s": time + 0.005, "window_end_s": time + 0.01,
               "block_ms": 10.0, "stride_s": 0.5}
        for feature_index, column in enumerate(module.DEFAULT_FLOOR_FEATURES):
            row[column] = np.cos(time / 9.0 + feature_index / 13.0)
        floor_rows.append(row)
    return pd.DataFrame(morph_rows), pd.DataFrame(floor_rows)


def test_gradient_reversal_changes_only_backward_sign():
    m = load(SCRIPT, "da_pf_grl")
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    y = m.gradient_reverse(x, 0.7)
    assert torch.equal(x.detach(), y.detach())
    y.sum().backward()
    assert x.grad.tolist() == pytest.approx([-0.7, -0.7])


def test_labeled_sequences_use_endpoint_labels_and_ranges_only():
    m = load(SCRIPT, "da_pf_sequences")
    morph, floor = synthetic_frames(m, epochs=40)
    aligned = m.align_modalities(morph, floor, m.DEFAULT_MORPH_FEATURES, m.DEFAULT_FLOOR_FEATURES)
    scalers = m.fit_robust_scalers(aligned)
    scaled = m.apply_scalers(aligned, scalers)
    data = m.make_labeled_sequences(
        scaled, seq_len=4, domain_id=2, onset_s=12.0,
        endpoint_ranges=[(2.0, 8.0), (14.0, 18.0)],
    )
    assert len(data) > 0
    endpoints = data.times[:, -1]
    assert np.all(((endpoints >= 2.0) & (endpoints <= 8.0)) |
                  ((endpoints >= 14.0) & (endpoints <= 18.0)))
    assert np.array_equal(data.labels, (endpoints >= 12.0).astype(np.float32))
    assert np.all(data.domains == 2)
    assert np.all(data.available_times[:, -1] >= endpoints)


def test_model_outputs_spoof_domain_branch_and_matching_heads():
    m = load(SCRIPT, "da_pf_model")
    cfg = m.DAPFRTConfig(morph_dim=9, floor_dim=12, seq_len=5, hidden_dim=24,
                         token_layers=1, token_heads=4, domain_count=3, dropout=0.0)
    model = m.DomainAdversarialPeakFloorRelationalTransformer(cfg)
    morph = torch.randn(4, 5, 32, 9)
    floor = torch.randn(4, 5, 12)
    mask = torch.zeros(4, 5, 32, dtype=torch.bool)
    mask[:, :, :4] = True
    negative_floor = torch.roll(floor, shifts=1, dims=0)
    out = model(morph, floor, mask, grl_alpha=1.0, corrupt_modalities=False,
                negative_floor=negative_floor)
    expected = {"spoof_logit", "snapshot_logit", "peak_logit", "floor_logit",
                "domain_logits", "match_positive_logit", "match_negative_logit",
                "embedding"}
    assert expected.issubset(out)
    assert out["spoof_logit"].shape == (4,)
    assert out["domain_logits"].shape == (4, 3)
    assert out["match_positive_logit"].shape == (4,)
    assert out["embedding"].shape[0] == 4


def test_fixed_threshold_uses_validation_normal_scores_only():
    m = load(SCRIPT, "da_pf_threshold")
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    threshold = m.select_fixed_threshold(probabilities, labels, target_fpr=0.5)
    assert threshold == pytest.approx(0.2)
    with pytest.raises(ValueError):
        m.select_fixed_threshold(np.array([0.8, 0.9]), np.array([1, 1]), target_fpr=0.01)


def test_hard_negatives_preserve_domain_and_label_and_change_sample():
    m = load(SCRIPT, "da_pf_negatives")
    domains = torch.tensor([0, 0, 0, 0, 1])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    times = torch.tensor([10.0, 11.0, 20.0, 21.0, 10.0])
    indices, valid = m.build_hard_negative_indices(domains, labels, times)
    for i in torch.where(valid)[0].tolist():
        j = int(indices[i])
        assert j != i
        assert int(domains[j]) == int(domains[i])
        assert float(labels[j]) == float(labels[i])
    assert not bool(valid[4])


def test_three_consecutive_alarm_is_timestamped_at_third_available_score():
    m = load(SCRIPT, "da_pf_latency")
    import pandas as pd
    frame = pd.DataFrame({
        "spoof_probability": [0.9, 0.9, 0.9],
        "available_time_s": [120.7, 121.2, 121.7],
        "window_start_s": [120.0, 120.5, 121.0],
    })
    delay, alarm = m._first_three_delay(frame, threshold=0.5, onset_s=120.0)
    assert alarm == pytest.approx(121.7)
    assert delay == pytest.approx(1.7)


def test_fold_contract_excludes_held_scenario_from_training_and_threshold():
    m = load(SCRIPT, "da_pf_contract")
    contract = m.build_fold_contract(["cleanStatic", "os2", "os3", "os4"], held_out="os4")
    assert contract["held_out"] == "os4"
    assert "os4" not in contract["training_scenarios"]
    assert "os4" not in contract["validation_scenarios"]
    assert set(contract["training_scenarios"]) == {"cleanStatic", "os2", "os3"}
    with pytest.raises(ValueError):
        m.build_fold_contract(["cleanStatic", "os2"], held_out="cleanStatic")



def test_alignment_requires_finite_causal_end_times_and_sequence_uses_max_availability():
    m = load(SCRIPT, "da_pf_availability_integrity")
    morph, floor = synthetic_frames(m, epochs=8)
    for frame, name in ((morph, "morphology"), (floor, "floor")):
        missing = frame.drop(columns="window_end_s")
        with pytest.raises(ValueError, match=rf"{name}.*window_end_s"):
            m.align_modalities(missing if name == "morphology" else morph,
                               missing if name == "floor" else floor)
        nonfinite = frame.copy(); nonfinite.loc[0, "window_end_s"] = np.nan
        with pytest.raises(ValueError, match=rf"{name}.*finite"):
            m.align_modalities(nonfinite if name == "morphology" else morph,
                               nonfinite if name == "floor" else floor)
        noncausal = frame.copy()
        clock = "window_bin_s" if name == "morphology" else "window_start_s"
        noncausal.loc[0, "window_end_s"] = float(noncausal.loc[0, clock]) - 1.0
        with pytest.raises(ValueError, match=rf"{name}.*causal"):
            m.align_modalities(noncausal if name == "morphology" else morph,
                               noncausal if name == "floor" else floor)
    aligned = m.align_modalities(morph, floor); aligned.available_times[0] = 99.0
    sequences = m.make_sequences(aligned, seq_len=4)
    assert m.sequence_score_available_times(sequences).tolist()[0] == pytest.approx(99.0)


def test_manifest_is_exact_pinned_and_rejects_swaps_and_identity_mismatches(tmp_path, monkeypatch):
    import hashlib, json
    import pandas as pd
    m = load(SCRIPT, "da_pf_manifest_integrity"); datasets = {}
    for scenario in m.EXPECTED_SCENARIOS:
        morph, floor = synthetic_frames(m, epochs=4)
        morph["label"] = f"oakbat_{scenario}_9tap"; morph["run_id"] = f"oakbat-{scenario}-method-a-9tap"
        morph["source_fingerprint"] = hashlib.sha256(f"source-{scenario}".encode()).hexdigest()
        floor["scenario"] = f"oakbat_{scenario}"
        mp, fp = tmp_path / f"{scenario}-morph.csv", tmp_path / f"{scenario}-floor.csv"
        morph.to_csv(mp, index=False); floor.to_csv(fp, index=False)
        datasets[scenario] = {"morph_csv": str(mp), "morph_sha256": m.sha256(mp),
            "floor_csv": str(fp), "floor_sha256": m.sha256(fp),
            "onset_s": None if scenario == "cleanStatic" else 120.0,
            "identity": {"scenario": scenario, "morph_label": f"oakbat_{scenario}_9tap",
                "morph_run_id": f"oakbat-{scenario}-method-a-9tap",
                "morph_source_fingerprint": hashlib.sha256(f"source-{scenario}".encode()).hexdigest(),
                "floor_scenario": f"oakbat_{scenario}"}}
    manifest = {"schema": m.DATASET_MANIFEST_SCHEMA, "datasets": datasets}
    assert set(m._load_datasets(manifest)[0]) == set(m.EXPECTED_SCENARIOS)
    with pytest.raises(ValueError, match="schema"): m._load_datasets(dict(manifest, schema="wrong"))
    missing = {**manifest, "datasets": dict(datasets)}; missing["datasets"].pop("os4")
    with pytest.raises(ValueError, match="roster"): m._load_datasets(missing)
    swapped = json.loads(json.dumps(manifest))
    swapped["datasets"]["os2"]["floor_csv"], swapped["datasets"]["os3"]["floor_csv"] = swapped["datasets"]["os3"]["floor_csv"], swapped["datasets"]["os2"]["floor_csv"]
    reads = []; real_read_csv = pd.read_csv
    monkeypatch.setattr(m.pd, "read_csv", lambda *a, **k: (reads.append(a[0]), real_read_csv(*a, **k))[1])
    with pytest.raises(ValueError, match="SHA256"): m._load_datasets(swapped)
    assert reads == [], "all pins must be verified before any dataset is read"
    mismatch = json.loads(json.dumps(manifest)); mismatch["datasets"]["os2"]["identity"]["morph_run_id"] = "oakbat-os3-method-a-9tap"
    with pytest.raises(ValueError, match="identity|linkage"): m._load_datasets(mismatch)
