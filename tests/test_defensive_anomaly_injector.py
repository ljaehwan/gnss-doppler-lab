from __future__ import annotations

import csv
import json
from pathlib import Path

from gnss_doppler_lab.defensive_anomaly_injector import (
    TrackingAnomalyScenario,
    TimeWindow,
    inject_tracking_rows,
    write_injected_tracking_csv,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"time_s": 0.0, "prn": "G05", "carrier_doppler_hz": 1000.0, "CN0_SNV_dB_Hz": 45.0},
        {"time_s": 1.0, "prn": "G05", "carrier_doppler_hz": 1010.0, "CN0_SNV_dB_Hz": 45.0},
        {"time_s": 1.0, "prn": "G18", "carrier_doppler_hz": -2000.0, "CN0_SNV_dB_Hz": 44.0},
        {"time_s": 3.0, "prn": "G05", "carrier_doppler_hz": 1030.0, "CN0_SNV_dB_Hz": 45.0},
    ]


def test_inject_tracking_rows_applies_windowed_common_and_prn_specific_bias() -> None:
    scenario = TrackingAnomalyScenario(
        name="defensive-windowed-bias",
        window=TimeWindow(start_s=1.0, end_s=2.0),
        prn_subset=("G05",),
        common_bias_hz=25.0,
        ramp_hz_per_s=5.0,
        per_prn_bias_hz={"G05": 2.5},
        cn0_drop_db_hz=3.0,
    )

    injected = inject_tracking_rows(_rows(), scenario)

    assert injected[0]["carrier_doppler_hz"] == 1000.0
    assert injected[1]["carrier_doppler_hz"] == 1037.5
    assert injected[1]["CN0_SNV_dB_Hz"] == 42.0
    assert injected[2]["carrier_doppler_hz"] == -2000.0
    assert injected[2]["CN0_SNV_dB_Hz"] == 44.0
    assert injected[3]["carrier_doppler_hz"] == 1030.0


def test_write_injected_tracking_csv_publishes_manifest_and_preserves_columns(tmp_path: Path) -> None:
    source = tmp_path / "tracking.csv"
    output = tmp_path / "tracking_injected.csv"
    manifest = tmp_path / "tracking_injected_manifest.json"

    rows = _rows()
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scenario = TrackingAnomalyScenario(
        name="defensive-common-drift",
        window=TimeWindow(start_s=0.5, end_s=3.0),
        common_bias_hz=-10.0,
        ramp_hz_per_s=1.0,
        cn0_drop_db_hz=1.5,
    )

    summary = write_injected_tracking_csv(source, output, manifest, scenario)

    assert summary["row_count"] == 4
    assert summary["changed_row_count"] == 3

    with output.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["prn"] == "G05"
    assert float(written[1]["carrier_doppler_hz"]) == 1000.5
    assert float(written[2]["carrier_doppler_hz"]) == -2009.5

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["scenario"]["name"] == "defensive-common-drift"
    assert manifest_data["changed_row_count"] == 3
    assert manifest_data["columns"] == ["time_s", "prn", "carrier_doppler_hz", "CN0_SNV_dB_Hz"]
