from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from gnss_doppler_lab.gnss_sdr import (
    export_tracking_csv,
    parse_acquired_prns,
    render_receiver_config,
    run_receiver,
)


def _tracking_mat(path: Path, *, prn: int, samples: list[int], doppler: list[float]) -> None:
    with h5py.File(path, "w") as handle:
        values = {
            "PRN": np.array(samples, dtype=np.uint32) * 0 + prn,
            "PRN_start_sample_count": np.array(samples, dtype=np.uint64),
            "carrier_doppler_hz": np.array(doppler, dtype=np.float32),
            "carrier_doppler_rate_hz": np.array([0.0, 1.0], dtype=np.float32),
            "CN0_SNV_dB_Hz": np.array([44.0, 46.0], dtype=np.float32),
            "Prompt_I": np.array([3.0, 4.0], dtype=np.float32),
            "Prompt_Q": np.array([1.0, 2.0], dtype=np.float32),
            "carrier_lock_test": np.array([0.8, 0.9], dtype=np.float32),
            "carr_error_hz": np.array([2.0, 1.0], dtype=np.float32),
            "code_error_chips": np.array([0.2, 0.1], dtype=np.float32),
        }
        for key, value in values.items():
            handle.create_dataset(key, data=value.reshape(-1, 1))


def test_render_receiver_config_matches_s8_complex_baseband(tmp_path: Path) -> None:
    iq = tmp_path / "normal iq.bin"
    output = tmp_path / "receiver"

    config = render_receiver_config(iq, output, sample_rate_hz=2_600_000, channel_count=11)

    assert f"SignalSource.filename={iq.resolve()}" in config
    assert "SignalSource.item_type=byte" in config
    assert "DataTypeAdapter.implementation=Ibyte_To_Complex" in config
    assert "GNSS-SDR.internal_fs_sps=2600000" in config
    assert "Channels_1C.count=11" in config
    assert "Acquisition_1C.doppler_max=6000" in config
    assert "Tracking_1C.dump=true" in config
    assert str(output.resolve() / "raw" / "epl_tracking_ch_") in config


def test_parse_acquired_prns_returns_sorted_unique_gps_ids() -> None:
    log = """
Tracking of GPS L1 C/A signal started on channel 2 for satellite GPS PRN 18 (Block III)
Tracking of GPS L1 C/A signal started on channel 4 for satellite GPS PRN 05 (Block IIR-M)
Tracking of GPS L1 C/A signal started on channel 9 for satellite GPS PRN 18 (Block III)
"""

    assert parse_acquired_prns(log) == ["G05", "G18"]


def test_export_tracking_csv_preserves_prn_time_doppler_and_summary(tmp_path: Path) -> None:
    first = tmp_path / "epl_tracking_ch_0.mat"
    second = tmp_path / "epl_tracking_ch_1.mat"
    _tracking_mat(first, prn=5, samples=[2600, 5200], doppler=[1000.0, 1010.0])
    _tracking_mat(second, prn=18, samples=[1300, 3900], doppler=[-2000.0, -1980.0])
    output = tmp_path / "tracking.csv"
    summary = tmp_path / "tracking_summary.csv"

    report = export_tracking_csv([second, first], output, summary, sample_rate_hz=2_600_000)

    rows = list(csv.DictReader(output.open()))
    summaries = list(csv.DictReader(summary.open()))
    assert report == {"row_count": 4, "prns": ["G05", "G18"], "channel_count": 2}
    assert rows[0]["prn"] == "G05"
    assert float(rows[0]["time_s"]) == 0.001
    assert float(rows[0]["carrier_doppler_hz"]) == 1000.0
    assert {row["prn"] for row in summaries} == {"G05", "G18"}
    g05 = next(row for row in summaries if row["prn"] == "G05")
    assert int(g05["epoch_count"]) == 2
    assert float(g05["median_cn0_db_hz"]) == 45.0


def test_run_receiver_creates_reproducible_run_artifacts(tmp_path: Path, monkeypatch) -> None:
    rf_run = tmp_path / "rf_runs" / "normal_20220101T000000Z"
    rf_run.mkdir(parents=True)
    iq = rf_run / "gps_l1ca_s8_iq.bin"
    iq.write_bytes(b"\x01\x02" * 5200)
    rf_manifest = rf_run / "manifest.json"
    rf_manifest.write_text(json.dumps({
        "run_id": rf_run.name,
        "iq": {"path": iq.name, "rf_sample_rate_hz": 2_600_000, "sha256": hashlib.sha256(iq.read_bytes()).hexdigest()},
    }))
    executable = tmp_path / "gnss-sdr"
    executable.write_text("fake")

    def fake_run(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "gnss-sdr version 0.0.test\n", "")
        config_arg = next(arg for arg in command if arg.startswith("--config_file="))
        config = Path(config_arg.split("=", 1)[1]).read_text()
        prefix = next(line.split("=", 1)[1] for line in config.splitlines() if line.startswith("Tracking_1C.dump_filename="))
        mat = Path(prefix + "0.mat")
        mat.parent.mkdir(parents=True, exist_ok=True)
        _tracking_mat(mat, prn=5, samples=[2600, 5200], doppler=[1000.0, 1010.0])
        stdout = "Tracking of GPS L1 C/A signal started on channel 0 for satellite GPS PRN 05 (Block IIR-M)\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr("gnss_doppler_lab.gnss_sdr.subprocess.run", fake_run)

    manifest_path = run_receiver(
        rf_manifest,
        tmp_path / "receiver_runs",
        executable=executable,
        channel_count=1,
    )

    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    assert manifest["source_rf_run_id"] == rf_run.name
    assert manifest["receiver"]["version"] == "gnss-sdr version 0.0.test"
    assert manifest["acquisition"]["tracked_prns"] == ["G05"]
    assert manifest["tracking"]["row_count"] == 2
    assert (run_dir / "receiver.conf").is_file()
    assert (run_dir / "receiver.log").is_file()
    assert (run_dir / "tracking.csv").is_file()
    assert (run_dir / "tracking_summary.csv").is_file()
