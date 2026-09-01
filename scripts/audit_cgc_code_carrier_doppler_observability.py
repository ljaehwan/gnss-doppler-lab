#!/usr/bin/env python3
"""Audit dual-Doppler observability in the static code/carrier pilot IQ."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.acquisition_surface import (  # noqa: E402
    compute_acquisition_surface,
    read_s8_iq,
)
from gnss_doppler_lab.doppler_observability import (  # noqa: E402
    dominant_doppler_peaks,
    local_probe,
    normalized_doppler_envelope,
)


DEFAULT_PILOT_ROOT = Path("/home/ubuntu/hdd_data/cgc_code_carrier_decoupling_pilot_v1")
SOURCE_RATE_HZ = 25_000_000
ANALYSIS_RATE_HZ = 5_000_000
COHERENT_MS = 20
DOPPLER_STEP_HZ = 10
SEARCH_MARGIN_HZ = 160
PROBE_HALF_WIDTH_HZ = 25
TIMES_S = (8.3, 8.4, 8.5, 8.6, 8.7)
PRNS = (12, 24, 26)
MODES = ("carrier-coupled", "doppler-locked")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_truth(path: Path) -> dict[tuple[float, int], dict[str, float]]:
    output: dict[tuple[float, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (round(float(row["time_s"]), 1), int(row["prn"]))
            output[key] = {name: float(value) for name, value in row.items() if name not in {"time_s", "prn"}}
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def receiver_diagnostics(pilot_root: Path) -> dict[str, dict[str, float | int]]:
    diagnostics: dict[str, dict[str, float | int]] = {}
    for mode in MODES:
        source = pilot_root / "conditions" / mode / "receiver" / f"cgc-cc-pilot-{mode}" / "tracking.csv"
        selected = []
        with source.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                time_s = float(row["time_s"])
                prn = int(row["prn"][1:])
                if 8.3 <= time_s < 8.72 and prn in PRNS:
                    selected.append(row)
        if not selected:
            raise ValueError(f"no receiver diagnostics selected from {source}")
        diagnostics[mode] = {
            "rows": len(selected),
            "median_carrier_lock_test": float(np.median([float(row["carrier_lock_test"]) for row in selected])),
            "median_cn0_db_hz": float(np.median([float(row["CN0_SNV_dB_Hz"]) for row in selected])),
            "median_absolute_carrier_error_hz": float(np.median([abs(float(row["carr_error_hz"])) for row in selected])),
        }
    return diagnostics


def plot_example(rows: list[dict[str, Any]], output_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    chosen = [row for row in rows if row["prn"] == "G24" and row["time_s"] == 8.5]
    if len(chosen) != 2:
        raise ValueError("expected one G24/8.5 row per condition")
    fig, ax = plt.subplots(figsize=(6.7, 3.2), constrained_layout=True)
    styles = {"carrier-coupled": ("#d55e00", "o"), "doppler-locked": ("#0072b2", "s")}
    for row in chosen:
        color, marker = styles[row["condition"]]
        bins = np.asarray(row.pop("_doppler_bins_hz"), dtype=float)
        envelope = np.asarray(row.pop("_normalized_envelope"), dtype=float)
        relative = bins - float(row["expected_authentic_doppler_hz"])
        ax.plot(relative, envelope, color=color, marker=marker, markevery=4, linewidth=1.5, markersize=3.5, label=row["condition"])
    coupled = next(row for row in chosen if row["condition"] == "carrier-coupled")
    separation = float(coupled["expected_coupled_separation_hz"])
    direction = np.sign(float(coupled["expected_coupled_spoof_doppler_hz"]) - float(coupled["expected_authentic_doppler_hz"]))
    ax.axvline(0, color="0.25", linestyle="--", linewidth=1, label="authentic Doppler")
    ax.axvline(direction * separation, color="#d55e00", linestyle=":", linewidth=1.2, label="coupled spoof Doppler")
    ax.set(xlabel="Doppler relative to authentic (Hz)", ylabel="Normalized PRN correlation", ylim=(0, 1.06), title="PRN G24 at 8.5 s: dual-Doppler observability")
    ax.grid(True, alpha=0.25); ax.legend(frameon=False, ncol=2, fontsize=8)
    for suffix in ("pdf", "svg"):
        fig.savefig(output_root / f"g24_dual_doppler_observability.{suffix}")
    plt.close(fig)


def run(pilot_root: Path) -> Path:
    output_root = pilot_root / "doppler_observability"
    output_root.mkdir(parents=True, exist_ok=True)
    truths = {
        "authentic": read_truth(pilot_root / "truth_preflight" / "authentic.csv"),
        "carrier-coupled": read_truth(pilot_root / "truth_preflight" / "carrier_coupled.csv"),
        "doppler-locked": read_truth(pilot_root / "truth_preflight" / "doppler_locked.csv"),
    }
    rows: list[dict[str, Any]] = []
    frontend_offset = 45.0
    frontend_drift = 0.15
    samples = int(round(SOURCE_RATE_HZ * COHERENT_MS / 1000.0))
    samples_per_code = int(round(ANALYSIS_RATE_HZ / 1000.0))
    for time_s in TIMES_S:
        iq_by_mode = {}
        for mode in MODES:
            iq_path = pilot_root / "conditions" / mode / "rf" / "gps_l1ca_s8_iq.bin"
            raw = read_s8_iq(iq_path, samples, int(round(time_s * SOURCE_RATE_HZ)))
            iq_by_mode[mode] = resample_poly(raw, 1, SOURCE_RATE_HZ // ANALYSIS_RATE_HZ).astype(np.complex64)
        for prn in PRNS:
            key = (round(time_s, 1), prn)
            offset = frontend_offset + frontend_drift * time_s
            authentic_hz = truths["authentic"][key]["carrier_doppler_hz"] + offset
            coupled_hz = truths["carrier-coupled"][key]["carrier_doppler_hz"] + offset
            locked_hz = truths["doppler-locked"][key]["carrier_doppler_hz"] + offset
            lower = int(np.floor(min(authentic_hz, coupled_hz) - SEARCH_MARGIN_HZ))
            upper = int(np.ceil(max(authentic_hz, coupled_hz) + SEARCH_MARGIN_HZ))
            lower = DOPPLER_STEP_HZ * int(np.floor(lower / DOPPLER_STEP_HZ))
            upper = DOPPLER_STEP_HZ * int(np.ceil(upper / DOPPLER_STEP_HZ))
            for mode in MODES:
                surface = compute_acquisition_surface(
                    iq_by_mode[mode], prn, ANALYSIS_RATE_HZ,
                    coherent_ms=COHERENT_MS,
                    doppler_min_hz=lower,
                    doppler_max_hz=upper,
                    doppler_step_hz=DOPPLER_STEP_HZ,
                )
                envelope = normalized_doppler_envelope(surface.magnitude, samples_per_code=samples_per_code)
                peaks = dominant_doppler_peaks(surface.doppler_bins_hz, envelope)
                auth_probe_hz, auth_probe = local_probe(surface.doppler_bins_hz, envelope, authentic_hz, half_width_hz=PROBE_HALF_WIDTH_HZ)
                coupled_probe_hz, coupled_probe = local_probe(surface.doppler_bins_hz, envelope, coupled_hz, half_width_hz=PROBE_HALF_WIDTH_HZ)
                rows.append({
                    "condition": mode,
                    "time_s": time_s,
                    "prn": f"G{prn:02d}",
                    "expected_authentic_doppler_hz": authentic_hz,
                    "expected_coupled_spoof_doppler_hz": coupled_hz,
                    "expected_condition_spoof_doppler_hz": coupled_hz if mode == "carrier-coupled" else locked_hz,
                    "expected_coupled_separation_hz": abs(coupled_hz - authentic_hz),
                    "expected_condition_separation_hz": abs((coupled_hz if mode == "carrier-coupled" else locked_hz) - authentic_hz),
                    "dominant_peak_count": len(peaks.frequencies_hz),
                    "strongest_peak_hz": peaks.frequencies_hz[0] if peaks.frequencies_hz else surface.peak_doppler_hz,
                    "second_peak_hz": peaks.frequencies_hz[1] if len(peaks.frequencies_hz) >= 2 else "",
                    "second_peak_height": peaks.normalized_heights[1] if len(peaks.normalized_heights) >= 2 else "",
                    "authentic_probe_hz": auth_probe_hz,
                    "authentic_probe_height": auth_probe,
                    "coupled_spoof_probe_hz": coupled_probe_hz,
                    "coupled_spoof_probe_height": coupled_probe,
                    "_doppler_bins_hz": surface.doppler_bins_hz,
                    "_normalized_envelope": envelope,
                })

    plot_example(rows, output_root)
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    write_csv(output_root / "window_metrics.csv", public_rows)
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        selected = [row for row in public_rows if row["condition"] == mode]
        expected = 2 if mode == "carrier-coupled" else 1
        by_mode[mode] = {
            "windows": len(selected),
            "expected_dominant_peak_count": expected,
            "windows_matching_expected_peak_count": sum(int(row["dominant_peak_count"]) == expected for row in selected),
            "fraction_matching_expected_peak_count": float(np.mean([int(row["dominant_peak_count"]) == expected for row in selected])),
            "windows_with_authentic_frequency_peak": sum(float(row["authentic_probe_height"]) >= 0.5 for row in selected),
            "windows_with_coupled_spoof_frequency_peak": sum(float(row["coupled_spoof_probe_height"]) >= 0.5 for row in selected),
            "median_dominant_peak_count": float(np.median([int(row["dominant_peak_count"]) for row in selected])),
            "median_authentic_probe_height": float(np.median([float(row["authentic_probe_height"]) for row in selected])),
            "median_coupled_spoof_probe_height": float(np.median([float(row["coupled_spoof_probe_height"]) for row in selected])),
        }
    truth_separations = [abs(truths["carrier-coupled"][(time_s, prn)]["carrier_doppler_hz"] - truths["authentic"][(time_s, prn)]["carrier_doppler_hz"]) for time_s in TIMES_S for prn in PRNS]
    locked_separations = [abs(truths["doppler-locked"][(time_s, prn)]["carrier_doppler_hz"] - truths["authentic"][(time_s, prn)]["carrier_doppler_hz"]) for time_s in TIMES_S for prn in PRNS]
    inputs = {}
    for label, source in {
        "carrier_coupled_rf": pilot_root / "conditions/carrier-coupled/rf/gps_l1ca_s8_iq.bin",
        "doppler_locked_rf": pilot_root / "conditions/doppler-locked/rf/gps_l1ca_s8_iq.bin",
        "authentic_truth": pilot_root / "truth_preflight/authentic.csv",
        "carrier_coupled_truth": pilot_root / "truth_preflight/carrier_coupled.csv",
        "doppler_locked_truth": pilot_root / "truth_preflight/doppler_locked.csv",
    }.items():
        inputs[label] = {"path": str(source.resolve()), "sha256": sha256(source), "bytes": source.stat().st_size}
    summary = {
        "schema": "gnss-doppler-lab.cgc-code-carrier-doppler-observability-audit",
        "schema_version": 1,
        "role": "development mechanism audit; not a fresh confirmatory result",
        "analysis": {
            "times_s": TIMES_S,
            "prns": [f"G{prn:02d}" for prn in PRNS],
            "source_rate_hz": SOURCE_RATE_HZ,
            "analysis_rate_hz": ANALYSIS_RATE_HZ,
            "coherent_ms": COHERENT_MS,
            "doppler_step_hz": DOPPLER_STEP_HZ,
            "dominant_peak_rule": "normalized height >= 0.5, prominence >= 0.1, separation >= 60 Hz",
        },
        "truth": {
            "carrier_coupled_minimum_selected_separation_hz": float(min(truth_separations)),
            "carrier_coupled_median_selected_separation_hz": float(np.median(truth_separations)),
            "doppler_locked_maximum_selected_separation_hz": float(max(locked_separations)),
        },
        "correlation_profiles": by_mode,
        "receiver_tracking": receiver_diagnostics(pilot_root),
        "inputs": inputs,
        "artifacts": {
            "window_metrics_csv": str((output_root / "window_metrics.csv").resolve()),
            "example_pdf": str((output_root / "g24_dual_doppler_observability.pdf").resolve()),
            "example_svg": str((output_root / "g24_dual_doppler_observability.svg").resolve()),
        },
        "interpretation_boundary": "The audit establishes that the simulated target condition removes the distinct counterfeit carrier-Doppler peak while preserving code carry-off. It does not establish superiority to every Doppler- or code-carrier-based detector.",
    }
    destination = output_root / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    args = parser.parse_args()
    summary = run(args.pilot_root.resolve())
    print(summary.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
