import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from gnss_doppler_lab.simulation_v4 import OutageEvent, SimulationScenario, SpoofEvent


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_simulation_v4_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_simulation_v4_pipeline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_event_labels_keep_outage_normal_and_switch_spoof_only_at_onset():
    recovery = SimulationScenario("recovery", "recovery_normal", outage=OutageEvent(2, 4, -80, 1))
    spoof = SimulationScenario("spoof", "carryoff_spoof", spoofing=SpoofEvent(2, 2, (10, 0, 0), -20, 3, 2))

    assert MODULE._event_label(recovery, 3.0) == ("normal", "normal_outage", 0)
    assert MODULE._event_label(recovery, 4.5) == ("normal", "normal_recovery_ramp", 0)
    assert MODULE._event_label(spoof, 1.999) == ("normal", "pre_event_normal", 0)
    assert MODULE._event_label(spoof, 2.0) == ("spoofing", "carryoff_transition", 1)
    assert MODULE._event_label(spoof, 4.0) == ("spoofing", "carryoff_final", 1)


def test_lock_metrics_distinguish_dump_epochs_from_valid_carrier_lock(tmp_path):
    receiver = tmp_path / "receiver"
    raw = receiver / "raw"
    raw.mkdir(parents=True)
    (receiver / "manifest.json").write_text(
        '{"source":{"sample_rate_hz":10},"tracking":{"raw_directory":"raw"}}\n'
    )
    with h5py.File(raw / "epl_tracking_ch_0.mat", "w") as handle:
        handle.create_dataset("PRN", data=np.array([1, 1, 1, 1, 0]))
        handle.create_dataset("PRN_start_sample_count", data=np.array([10, 20, 30, 40, 20]))
        handle.create_dataset("carrier_lock_test", data=np.array([0.9, 0.8, 0.1, 0.0, 1.0]))
        handle.create_dataset("CN0_SNV_dB_Hz", data=np.array([42.0, 41.0, 29.0, 28.0, 99.0]))

    locked = MODULE._lock_metrics(receiver, 1.0, 3.0)
    lost = MODULE._lock_metrics(receiver, 3.0, 5.0)

    assert locked["epoch_count"] == 2
    assert locked["carrier_lock_median"] == pytest.approx(0.85)
    assert locked["carrier_lock_above_0_5_fraction"] == 1.0
    assert lost["epoch_count"] == 2
    assert lost["carrier_lock_median"] == pytest.approx(0.05)
    assert lost["carrier_lock_above_0_5_fraction"] == 0.0


def test_combined_dataset_carries_atomic_paired_group_id(tmp_path):
    source = tmp_path / "steady.csv"
    source.write_text("label,window_mid_s\nsteady_normal,1.0\n")
    output = tmp_path / "combined.csv"

    count, states = MODULE._combine_labeled_features(
        {"steady": source},
        (SimulationScenario("steady", "steady_normal"),),
        "pair-location-epoch-seed-001",
        output,
    )

    import csv
    with output.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert count == 1
    assert states == {"steady_normal": 1}
    assert row["paired_group_id"] == "pair-location-epoch-seed-001"
    assert row["dataset_role"] == "simulation_only_pilot"
