"""Safe defensive anomaly injection on normalized tracking CSVs.

This module intentionally operates on derived receiver observables/tracking exports,
not on RF/IQ waveforms. It is meant for detector stress-testing without creating
usable counterfeit GNSS signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float

    def contains(self, time_s: float) -> bool:
        return self.start_s <= time_s <= self.end_s


@dataclass(frozen=True)
class TrackingAnomalyScenario:
    name: str
    window: TimeWindow
    prn_subset: tuple[str, ...] | None = None
    common_bias_hz: float = 0.0
    ramp_hz_per_s: float = 0.0
    per_prn_bias_hz: Mapping[str, float] = field(default_factory=dict)
    cn0_drop_db_hz: float = 0.0

    def applies_to(self, prn: str, time_s: float) -> bool:
        if not self.window.contains(time_s):
            return False
        if self.prn_subset is None:
            return True
        return prn in self.prn_subset


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def inject_tracking_rows(
    rows: Iterable[Mapping[str, Any]],
    scenario: TrackingAnomalyScenario,
) -> list[dict[str, object]]:
    injected: list[dict[str, object]] = []
    for row in rows:
        out = dict(row)
        prn = str(out["prn"])
        time_s = _float(out, "time_s")
        out["time_s"] = time_s
        if scenario.applies_to(prn, time_s):
            time_offset = time_s - scenario.window.start_s
            bias = scenario.common_bias_hz + scenario.ramp_hz_per_s * time_offset
            bias += float(scenario.per_prn_bias_hz.get(prn, 0.0))
            out["carrier_doppler_hz"] = _float(out, "carrier_doppler_hz") + bias
            if "CN0_SNV_dB_Hz" in out:
                out["CN0_SNV_dB_Hz"] = _float(out, "CN0_SNV_dB_Hz") - scenario.cn0_drop_db_hz
        else:
            if "carrier_doppler_hz" in out:
                out["carrier_doppler_hz"] = _float(out, "carrier_doppler_hz")
            if "CN0_SNV_dB_Hz" in out:
                out["CN0_SNV_dB_Hz"] = _float(out, "CN0_SNV_dB_Hz")
        injected.append(out)
    return injected


def write_injected_tracking_csv(
    source_csv: str | Path,
    output_csv: str | Path,
    manifest_path: str | Path,
    scenario: TrackingAnomalyScenario,
) -> dict[str, object]:
    source = Path(source_csv)
    output = Path(output_csv)
    manifest = Path(manifest_path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    injected = inject_tracking_rows(rows, scenario)
    changed_row_count = sum(
        1
        for original, updated in zip(rows, injected)
        if float(original["carrier_doppler_hz"]) != float(updated["carrier_doppler_hz"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(injected)
    manifest.write_text(
        json.dumps(
            {
                "scenario": {
                    "name": scenario.name,
                    "window": {
                        "start_s": scenario.window.start_s,
                        "end_s": scenario.window.end_s,
                    },
                    "prn_subset": list(scenario.prn_subset) if scenario.prn_subset is not None else None,
                    "common_bias_hz": scenario.common_bias_hz,
                    "ramp_hz_per_s": scenario.ramp_hz_per_s,
                    "per_prn_bias_hz": dict(scenario.per_prn_bias_hz),
                    "cn0_drop_db_hz": scenario.cn0_drop_db_hz,
                },
                "row_count": len(injected),
                "changed_row_count": changed_row_count,
                "columns": fieldnames,
                "source_csv": str(source),
                "output_csv": str(output),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "row_count": len(injected),
        "changed_row_count": changed_row_count,
        "output_csv": str(output),
        "manifest_path": str(manifest),
    }
