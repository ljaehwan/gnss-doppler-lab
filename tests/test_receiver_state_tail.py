import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_doppler_lab.receiver_state_tail import (
    GLOBAL_SCORE,
    STATE_SCORE,
    annotate_receiver_state,
    build_conformal_event_scores,
    calibrate_receiver_state_detectors,
    chronological_clean_split,
    fit_receiver_state_reference,
    right_tail_conformal_p,
    select_reference_pool,
    validate_contract_scores,
)


def _scores(run_id="clean", bins=range(10), prns=("G01", "G02"), offset=0.0):
    rows = []
    for index in bins:
        time_s = index * 0.5
        for prn_index, prn in enumerate(prns):
            rows.append({
                "run_id": run_id,
                "prn": prn,
                "window_bin_s": time_s,
                "window_start_s": time_s,
                "window_mid_s": time_s + 0.5,
                "window_end_s": time_s + 1.0,
                "prn_node_rmse": offset + index / 100.0 + prn_index / 1000.0,
                "channel": prn_index,
                "segment_index": 0,
                "continuity_block_index": 0,
                "tracking_age_s": time_s,
                "reacquisition_flag": 0,
                "history_same_segment_flag": 1,
            })
    return pd.DataFrame(rows)


def test_receiver_state_uses_contract_age_and_restart_origin():
    frame = _scores(bins=range(3), prns=("G01",))
    frame.loc[1, "continuity_block_index"] = 1
    frame.loc[1, "tracking_age_s"] = 0.0
    frame.loc[2, "segment_index"] = 1
    frame.loc[2, "reacquisition_flag"] = 1
    frame.loc[2, "tracking_age_s"] = 0.0

    got = annotate_receiver_state(frame, age_cutoffs_s=(1.0, 2.0))

    assert got.receiver_origin.tolist() == ["initial", "gap_restart", "reacquired"]
    assert got.receiver_age_bin.tolist() == ["age_lt_1s"] * 3


def test_contract_validation_rejects_unsafe_history_and_missing_metadata():
    unsafe = _scores()
    unsafe.loc[0, "history_same_segment_flag"] = 0
    with pytest.raises(ValueError, match="within one receiver segment"):
        validate_contract_scores(unsafe)
    with pytest.raises(ValueError, match="missing columns"):
        validate_contract_scores(_scores().drop(columns="tracking_age_s"))


def test_chronological_split_keeps_complete_events_and_has_no_overlap():
    clean = pd.concat([
        _scores("static", range(10)),
        _scores("dynamic", range(10), offset=1.0),
    ], ignore_index=True)

    split = chronological_clean_split(
        clean, reference_fraction=0.6, event_calibration_fraction=0.2
    )

    for run_id in ("static", "dynamic"):
        reference_bins = set(split.reference.loc[split.reference.run_id == run_id, "window_bin_s"])
        calibration_bins = set(
            split.event_calibration.loc[split.event_calibration.run_id == run_id, "window_bin_s"]
        )
        held_bins = set(split.held_clean.loc[split.held_clean.run_id == run_id, "window_bin_s"])
        assert len(reference_bins) == 6
        assert len(calibration_bins) == 2
        assert len(held_bins) == 2
        assert max(reference_bins) < min(calibration_bins) < min(held_bins)
        assert not reference_bins & calibration_bins
        assert not calibration_bins & held_bins
    assert split.reference.groupby(["run_id", "window_bin_s"]).size().eq(2).all()


def test_right_tail_conformal_p_is_finite_sample_and_includes_ties():
    reference = np.asarray([1.0, 2.0, 2.0, 4.0])

    got = right_tail_conformal_p(np.asarray([0.0, 2.0, 3.0, 5.0]), reference)

    assert got.tolist() == pytest.approx([1.0, 0.8, 0.4, 0.2])


def test_reference_pool_falls_back_exact_then_age_then_global():
    clean = _scores(bins=range(8), prns=("G01", "G02"))
    reference = fit_receiver_state_reference(
        clean, age_cutoffs_s=(1.0, 2.0), min_pool_rows=3
    )

    exact, exact_level = select_reference_pool(reference, "initial|age_ge_2s", "age_ge_2s")
    age, age_level = select_reference_pool(reference, "reacquired|age_ge_2s", "age_ge_2s")
    global_pool, global_level = select_reference_pool(
        reference, "reacquired|unseen", "unseen"
    )

    assert exact_level == "exact_state" and len(exact) == 8
    assert age_level == "age_bin" and len(age) == 8
    assert global_level == "global" and len(global_pool) == 16


def test_event_scores_emit_matched_global_and_state_evidence():
    reference_scores = _scores(bins=range(20), prns=("G01", "G02", "G03"))
    reference = fit_receiver_state_reference(
        reference_scores, age_cutoffs_s=(2.0, 5.0), min_pool_rows=3
    )
    evaluated = _scores("attack", bins=range(3), prns=("G01", "G02", "G03"), offset=10.0)
    evaluated.loc[evaluated.prn.eq("G03"), "reacquisition_flag"] = 1
    evaluated.loc[evaluated.prn.eq("G03"), "segment_index"] = 1

    events, nodes = build_conformal_event_scores(evaluated, reference=reference)

    assert {GLOBAL_SCORE, STATE_SCORE}.issubset(events.columns)
    assert events[[GLOBAL_SCORE, STATE_SCORE]].to_numpy().max() > 0.0
    assert nodes.loc[nodes.prn.eq("G03"), "calibration_pool_level"].eq("age_bin").all()


def test_attack_extension_cannot_change_frozen_clean_calibration():
    clean = _scores(bins=range(30), prns=("G01", "G02", "G03"))
    split = chronological_clean_split(clean)
    first, _, _ = calibrate_receiver_state_detectors(
        split.reference, split.event_calibration, age_cutoffs_s=(2.0, 5.0), min_pool_rows=3
    )
    attack = _scores("attack", bins=range(20), offset=100.0)
    build_conformal_event_scores(attack, reference=first.reference)
    second, _, _ = calibrate_receiver_state_detectors(
        split.reference, split.event_calibration, age_cutoffs_s=(2.0, 5.0), min_pool_rows=3
    )

    assert second.global_event_threshold == pytest.approx(first.global_event_threshold)
    assert second.state_event_threshold == pytest.approx(first.state_event_threshold)
    assert np.array_equal(second.reference.global_scores, first.reference.global_scores)


def test_runner_writes_auditable_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "eval_receiver_state_tail.py"
    spec = importlib.util.spec_from_file_location("eval_receiver_state_tail", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    clean_path = tmp_path / "clean.csv"
    attack_path = tmp_path / "attack.csv"
    _scores(bins=range(20), prns=("G01", "G02", "G03")).to_csv(clean_path, index=False)
    attack = _scores("attack", bins=range(24), prns=("G01", "G02", "G03"))
    attack.loc[attack.window_start_s.ge(5.0), "prn_node_rmse"] += 10.0
    attack.to_csv(attack_path, index=False)
    manifest = {
        "schema": module.MANIFEST_SCHEMA,
        "clean_scores": [str(clean_path)],
        "clean_split": {"reference_fraction": 0.6, "event_calibration_fraction": 0.2},
        "age_cutoffs_s": [2.0, 5.0],
        "min_pool_rows": 3,
        "event_quantile": 0.99,
        "guard_s": 1.0,
        "attacks": {"synthetic": {"score_csv": str(attack_path), "onset_s": 5.0}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out_dir = tmp_path / "out"

    summary = module.run(manifest, out_dir, manifest_path)

    assert summary["schema"] == module.SUMMARY_SCHEMA
    assert summary["calibration"]["normal_only"] is True
    assert summary["calibration"]["attack_score_csv_opened_before_freeze"] is False
    assert set(summary["attacks"]["synthetic"]["detectors"]) == {
        "matched_global_conformal", "receiver_state_conformal"
    }
    assert (out_dir / "comparison.csv").is_file()
    assert (out_dir / "summary.json").is_file()
