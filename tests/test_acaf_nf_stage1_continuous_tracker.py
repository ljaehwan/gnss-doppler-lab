import json
from pathlib import Path

import h5py
import numpy as np

from gnss_doppler_lab.acaf_nf_stage1_continuous_tracker import audit_tracker_cadence, build_audit


def _write_simple_mat(path: Path) -> None:
    n = 25
    stamps = np.arange(0, n) * 25_000
    with h5py.File(path, "w") as handle:
        handle["PRN_start_sample_count"] = stamps.reshape(-1, 1)
        handle["PRN"] = np.full(n, 1, dtype=np.uint32).reshape(-1, 1)
        handle["carrier_doppler_hz"] = np.full(n, -1500.0).reshape(-1, 1)
        handle["code_freq_chips"] = np.full(n, 1.023e6).reshape(-1, 1)
        handle["aux1"] = np.zeros(n).reshape(-1, 1)
        handle["Prompt_I"] = np.arange(n, dtype=np.float32).reshape(-1, 1)
        handle["Prompt_Q"] = np.arange(n, dtype=np.float32).reshape(-1, 1)
        handle["CN0_SNV_dB_Hz"] = np.full(n, 30.0).reshape(-1, 1)
        handle["carrier_lock_test"] = np.full(n, 0.9).reshape(-1, 1)


def test_audit_generates_rows_and_csv(tmp_path):
    tracker = tmp_path / "tracker"
    tracker.mkdir()
    mat = tracker / "epl_tracking_ch_0.mat"
    dat = tracker / "epl_tracking_ch_0.dat"
    _write_simple_mat(mat)
    # dat payload bytes match MAT row count to exercise cadence check
    dat.write_bytes(b"\x00" * (25 * len(np.arange(25)) * 148))

    summary, rows = audit_tracker_cadence("cleanStatic", tracker)
    assert summary["scenario"] == "cleanStatic"
    assert summary["total_l20_windows"] >= 1
    assert rows
    assert rows[0].row_delta_in_range_ratio == 1.0


def test_build_audit(tmp_path, monkeypatch):
    cfg = {
        "scenarios": {
            "cleanStatic": {
                "tracker_path": str(tmp_path / "tracker"),
            }
        }
    }
    tracker = tmp_path / "tracker"
    tracker.mkdir()
    _write_simple_mat(tracker / "epl_tracking_ch_0.mat")
    (tracker / "epl_tracking_ch_0.dat").write_bytes(b"\x00" * (25 * 25 * 148))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    output = build_audit(cfg_path, tmp_path / "artifact")
    assert (output / "tracker_cadence_audit.json").exists()
    assert (output / "tracker_cadence_by_channel.csv").exists()
