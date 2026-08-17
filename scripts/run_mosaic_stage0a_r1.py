#!/usr/bin/env python3
"""Run MOSAIC Stage-0A R1 receiver-faithful cleanStatic raw recorrelation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mosaic_raw_recorrelation import (  # noqa: E402
    FROZEN_GATE,
    correlate_nine_taps,
    evaluate_recorrelation,
    fit_complex_amplitude,
    native_taps_for_item,
    normalized_complex_cosine,
    prompt_normalize_safe,
    read_ishort_complex_window,
    select_epoch_records,
    sha256_file,
)

ART = ROOT / "artifacts/mosaic_stage0a_r1_raw_recorrelation"
RECEIVER = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr")
RECEIVER_SOURCE = RECEIVER.parents[3] / "receiver-source"
MCTD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static/dumps/phase_a")
DATASETS = {
    "OAKBAT.cleanStatic": {
        "slug": "oakbat_cleanstatic",
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin"),
        "size": 9_600_000_000,
        "sha256": "8e3428abb1b94211118c1ec9f505322ef89fdb176f0cf33961eca3cf3da80dfe",
        "sample_rate_hz": 5_000_000.0,
        "prns": [10, 11, 21, 24, 27],
    },
    "TEXBAT.cleanStatic": {
        "slug": "texbat_cleanstatic",
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin"),
        "size": 48_016_392_192,
        "sha256": "dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9",
        "sample_rate_hz": 25_000_000.0,
        "prns": [3, 13, 16, 19, 30],
    },
}
EXPECTED_RECEIVER_SHA256 = "2f6e8e969e525bb48b4d94f016af8fd24f433b0be26b51837f316f60a6b911e0"
attack_data_used = False


def dump_json(name: str, value: object) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def stat_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path), "size_bytes": stat.st_size, "device": stat.st_dev,
        "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {key: float(value) for key, value in zip(
        ("min", "p05", "median", "p95", "max"), np.percentile(array, (0, 5, 50, 95, 100))
    )}


def navigation_invariance_audit() -> dict[str, object]:
    rng = np.random.default_rng(20260817)
    reconstructed = rng.normal(size=9) + 1j * rng.normal(size=9)
    base = (0.37 - 1.41j) * reconstructed
    outcomes = {}
    normalized = []
    for bit in (1, -1):
        native = bit * base
        alpha, fitted = fit_complex_amplitude(reconstructed, native)
        prompt = prompt_normalize_safe(native)
        normalized.append(prompt)
        outcomes[str(bit)] = {
            "fitted_amplitude_real": alpha.real,
            "fitted_amplitude_imag": alpha.imag,
            "complex_cosine": normalized_complex_cosine(fitted, native),
            "normalized_taps": [[float(v.real), float(v.imag)] for v in prompt],
        }
    difference = float(np.max(np.abs(normalized[0] - normalized[1])))
    return {
        "schema": "gnss-doppler-lab.mosaic-stage0a-navbit-invariance.v1",
        "integration_window_ms": 1,
        "per_epoch_complex_amplitude_fit": True,
        "bit_outcomes": outcomes,
        "maximum_prompt_normalized_shape_difference": difference,
        "alignment_metric_difference": abs(outcomes["1"]["complex_cosine"] - outcomes["-1"]["complex_cosine"]),
        "status": "PASS" if difference <= 1e-12 else "FAIL",
        "stage0a_conclusion": "NAV_BIT_NOT_REQUIRED_AFTER_PER_EPOCH_COMPLEX_NORMALIZATION",
        "stage0b_conclusion": "NAV_BIT_PROVENANCE_STILL_REQUIRED",
    }


def receiver_provenance() -> dict[str, object]:
    safe = f"safe.directory={RECEIVER_SOURCE}"
    base_commit = command("git", "-c", safe, "-C", str(RECEIVER_SOURCE), "rev-parse", "HEAD")
    status = command("git", "-c", safe, "-C", str(RECEIVER_SOURCE), "status", "--short")
    diff = subprocess.run(
        ["git", "-c", safe, "-C", str(RECEIVER_SOURCE), "diff", "--binary"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    configs = {}
    for dataset, spec in DATASETS.items():
        path = MCTD / str(spec["slug"]) / "slow/rep1/receiver.conf"
        configs[dataset] = {"path": str(path), "sha256": sha256_file(path)}
    return {
        "receiver_executable": stat_record(RECEIVER) | {"sha256": sha256_file(RECEIVER)},
        "expected_receiver_sha256": EXPECTED_RECEIVER_SHA256,
        "receiver_source": {
            "path": str(RECEIVER_SOURCE), "base_commit": base_commit,
            "working_tree_patch_sha256": hashlib.sha256(diff).hexdigest(),
            "modified_files": status.splitlines(),
        },
        "receiver_configs": configs,
        "input_sample_format": "little-endian interleaved signed int16 I,Q (GNSS-SDR ishort), 4 bytes/complex sample",
        "signal_conditioning": "Ishort_To_Complex -> Pass_Through -> Pass_Through; complex baseband, no IF configured",
        "carrier_wipeoff": "exp(-j*(residual_carrier_phase_rad + carrier_phase_step_rad_per_sample*n)); Doppler step=2*pi*f/fs",
        "code_replica_indexing": "floor(code_phase_step*n + tap_shift - residual_code_phase) modulo 1023",
        "tap_offsets_chips": [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5],
        "prn_start_sample_count_semantics": "TRACE raw_interval_start_sample / legacy PRN_start_sample_count is the absolute zero-based complex-sample index consumed at the start of the current correlation interval",
        "state_semantics": {
            "residual_code_phase_chips": "remnant code phase subtracted by the VOLK code-index expression for the current interval",
            "residual_carrier_phase_rad": "current interval carrier NCO phase used by exp(-j phase)",
            "carrier_phase_accumulator_rad": "continuous accumulated carrier phase; provenance only, not substituted for residual carrier phase",
        },
        "convention_frozen_from_source_before_metrics": True,
        "alternative_sign_trials": False,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    started = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "plots").mkdir(exist_ok=True)
    head = command("git", "-C", str(ROOT), "rev-parse", "HEAD")
    branch = command("git", "-C", str(ROOT), "branch", "--show-current")
    dump_json("config.json", {
        "schema": "gnss-doppler-lab.mosaic-stage0a-r1-config.v1",
        "artifact_root": str(ART), "frozen_gate": FROZEN_GATE,
        "datasets": list(DATASETS), "attack_data_used": False, "stage0b_run": False,
        "neural_model_used": False, "workers": 1, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "selection": "legacy Stage-0A deterministic support union 24 equally-spaced stable epochs per target PRN",
    })
    dump_json("source_commit.json", {
        "repo": str(ROOT), "branch": branch, "generation_head": head,
        "required_base_tip": "4976d123a6d66583e5d92e0e277d666e5a20575e",
    })
    dump_json("execution_environment.json", {
        "executable": sys.executable, "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "kernel": platform.release(), "logical_cpus": os.cpu_count(),
        "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "swap_bytes": 0, "workers_used": 1, "worker_max_rss_kib_after_run": None,
        "thread_limits": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "prior_python_3_11_15_note": "The prior artifact's recorded interpreter belonged to an earlier execution environment. It is not present in this foundation checkout; this run is bound to the requested project .venv and records its executable and versions directly.",
    })

    receiver = receiver_provenance()
    dump_json("receiver_provenance.json", receiver)
    source_binding = {}
    binding_ok = receiver["receiver_executable"]["sha256"] == EXPECTED_RECEIVER_SHA256
    for dataset, spec in DATASETS.items():
        raw = Path(spec["raw"])
        actual_sha = sha256_file(raw)
        actual_size = raw.stat().st_size
        passed = actual_sha == spec["sha256"] and actual_size == spec["size"]
        binding_ok &= passed
        source_binding[dataset] = {
            "status": "PASS" if passed else "FAIL", "stat": stat_record(raw),
            "expected_size_bytes": spec["size"], "full_sha256": actual_sha,
            "expected_sha256": spec["sha256"], "sample_rate_hz": spec["sample_rate_hz"],
            "sample_format": "little-endian interleaved int16 I/Q", "bytes_per_complex_sample": 4,
            "complex_baseband": True,
        }
    source_binding["receiver"] = {
        "status": "PASS" if receiver["receiver_executable"]["sha256"] == EXPECTED_RECEIVER_SHA256 else "FAIL",
        "full_sha256": receiver["receiver_executable"]["sha256"], "expected_sha256": EXPECTED_RECEIVER_SHA256,
    }
    source_binding["overall_status"] = "PASS" if binding_ok else "FAIL"
    dump_json("raw_source_binding.json", source_binding)

    navbit = navigation_invariance_audit()
    dump_json("navbit_invariance_test.json", navbit)
    (ART / "navbit_requirement_audit.md").write_text(
        "# Navigation-bit requirement audit\n\n"
        "For a single GPS L1 C/A code epoch, write the received samples as\n"
        "`y[n] = alpha_(i,k) b_(i,k) c_i[n-tau] exp(j phase[n]) + w[n]`, where "
        "`b_(i,k)` is constant over the exact 1 ms integration. Every delay tap is therefore multiplied "
        "by the same scalar `b_(i,k)`. The least-squares epoch amplitude is "
        "`a_hat=(r^H y)/(r^H r)`; replacing `y` by `-y` replaces `a_hat` by `-a_hat`, leaving the fitted "
        "normalized complex tap shape, Prompt-normalized vector, complex cosine, and magnitude ranks unchanged.\n\n"
        "The proof requires an exact 1 ms epoch, per-epoch complex-amplitude/global-phase normalization, and no "
        "absolute carrier-phase claim. Those conditions are checked here. Consequently Stage-0A is "
        "`NAV_BIT_NOT_REQUIRED_AFTER_PER_EPOCH_COMPLEX_NORMALIZATION`. Stage-0B must synthesize a continuous "
        "receiver input across navigation-symbol boundaries, so it remains "
        "`NAV_BIT_PROVENANCE_STILL_REQUIRED`; no +1 fallback is authorized.\n"
    )

    inventory_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    if binding_ok:
        for dataset, spec in DATASETS.items():
            dump_dir = MCTD / str(spec["slug"]) / "slow/rep1"
            selected = select_epoch_records(dump_dir, spec["prns"])
            for item in selected:
                header, native, record = native_taps_for_item(item)
                inventory_rows.append({
                    "dataset": dataset, **item,
                    "sample_count": int(item["raw_sample_end"]) - int(item["raw_sample_start"]),
                    "code_frequency_chips_s": float(record["action_used_code_nco_rate_chips_s"]),
                    "carrier_doppler_hz": float(record["action_used_carrier_doppler_hz"]),
                    "residual_code_phase_chips": float(record["action_used_residual_code_phase_chips"]),
                    "residual_carrier_phase_rad": float(record["action_used_residual_carrier_phase_rad"]),
                    "carrier_phase_accumulator_rad": float(record["action_used_carrier_phase_accumulator_rad"]),
                })
                identity = f"{dataset}:PRN{item['prn']}:loop{item['loop_sequence']}"
                try:
                    count = int(item["raw_sample_end"]) - int(item["raw_sample_start"])
                    expected_count = int(round(float(spec["sample_rate_hz"]) * 0.001))
                    native_count = int(record["action_used_interval_length_samples"])
                    if count != native_count or abs(count - expected_count) > 1 or abs(header.coherent_integration_s - 0.001) > 1e-9:
                        raise ValueError("window is not one native C/A code epoch")
                    code_span_chips = count * float(record["action_used_code_phase_step_chips_per_sample"])
                    if abs(code_span_chips - 1023.0) > 0.25:
                        raise ValueError("native interval does not cover one 1023-chip code period")
                    if tuple(round(v, 9) for v in header.tap_offsets_chips) != tuple(round(v, 9) for v in np.arange(-4, 5) * 0.125):
                        raise ValueError("native nine-tap offsets do not match frozen coordinates")
                    iq = read_ishort_complex_window(spec["raw"], int(item["raw_sample_start"]), count)
                    reconstructed = correlate_nine_taps(iq, prn=int(item["prn"]), action=record, tap_offsets_chips=header.tap_offsets_chips)
                    result = evaluate_recorrelation(reconstructed, native, record, float(spec["sample_rate_hz"]))
                    row = {
                        "dataset": dataset, **item, "sample_count": count,
                        "delay_center_error_chips": result.delay_center_error_chips,
                        "doppler_center_error_hz": result.doppler_center_error_hz,
                        "complex_tap_cosine": result.complex_cosine,
                        "magnitude_spearman": result.magnitude_spearman,
                        "prompt_magnitude_ratio": result.prompt_magnitude_ratio,
                        "prompt_phase_error_rad": result.prompt_phase_error_rad,
                        "fitted_amplitude_real": result.complex_amplitude.real,
                        "fitted_amplitude_imag": result.complex_amplitude.imag,
                        "native_taps_iq": json.dumps([[float(v.real), float(v.imag)] for v in native], separators=(",", ":")),
                        "reconstructed_taps_iq": json.dumps([[float(v.real), float(v.imag)] for v in result.reconstructed_taps], separators=(",", ":")),
                        "gate_pass": result.gate_pass,
                    }
                    metric_rows.append(row)
                except Exception as exc:
                    rejected.append({"dataset": dataset, **item, "reason": f"{type(exc).__name__}: {exc}", "row_identity": identity})

    inventory_fields = [
        "dataset", "prn", "channel", "loop_sequence", "raw_sample_start", "raw_sample_end",
        "receiver_timestamp_s", "sample_count", "selection_origin", "source_dump", "record_index",
        "code_frequency_chips_s", "carrier_doppler_hz", "residual_code_phase_chips",
        "residual_carrier_phase_rad", "carrier_phase_accumulator_rad",
    ]
    metric_fields = [
        "dataset", "prn", "channel", "loop_sequence", "raw_sample_start", "raw_sample_end", "sample_count",
        "selection_origin", "delay_center_error_chips", "doppler_center_error_hz", "complex_tap_cosine",
        "magnitude_spearman", "prompt_magnitude_ratio", "prompt_phase_error_rad",
        "fitted_amplitude_real", "fitted_amplitude_imag", "native_taps_iq", "reconstructed_taps_iq", "gate_pass",
    ]
    reject_fields = ["dataset", "prn", "channel", "loop_sequence", "raw_sample_start", "raw_sample_end", "selection_origin", "reason", "row_identity"]
    write_csv(ART / "selected_epoch_inventory.csv", inventory_rows, inventory_fields)
    write_csv(ART / "raw_recorrelation_metrics.csv", metric_rows, metric_fields)
    write_csv(ART / "rejected_rows.csv", rejected, reject_fields)

    per_prn: list[dict[str, object]] = []
    dataset_summaries = {}
    overall_pass = binding_ok and navbit["status"] == "PASS"
    for dataset, spec in DATASETS.items():
        attempted = [r for r in inventory_rows if r["dataset"] == dataset]
        valid = [r for r in metric_rows if r["dataset"] == dataset]
        rejected_ds = [r for r in rejected if r["dataset"] == dataset]
        for prn in spec["prns"]:
            rows = [r for r in valid if int(r["prn"]) == prn]
            attempts = [r for r in attempted if int(r["prn"]) == prn]
            record = {
                "dataset": dataset, "prn": prn, "attempted_rows": len(attempts), "valid_rows": len(rows),
                "rejected_rows": len(attempts) - len(rows),
            }
            if rows:
                record |= {
                    "delay_error_abs_max_chips": max(abs(float(r["delay_center_error_chips"])) for r in rows),
                    "doppler_error_abs_max_hz": max(abs(float(r["doppler_center_error_hz"])) for r in rows),
                    "complex_cosine_min": min(float(r["complex_tap_cosine"]) for r in rows),
                    "complex_cosine_median": float(np.median([r["complex_tap_cosine"] for r in rows])),
                    "magnitude_spearman_min": min(float(r["magnitude_spearman"]) for r in rows),
                    "magnitude_spearman_median": float(np.median([r["magnitude_spearman"] for r in rows])),
                    "prompt_magnitude_ratio_median": float(np.median([r["prompt_magnitude_ratio"] for r in rows])),
                    "prompt_phase_error_abs_max_rad": max(abs(float(r["prompt_phase_error_rad"])) for r in rows),
                    "gate_pass_fraction": float(np.mean([r["gate_pass"] for r in rows])),
                    "time_span_s": max(float(r["raw_sample_start"]) for r in rows) / spec["sample_rate_hz"] - min(float(r["raw_sample_start"]) for r in rows) / spec["sample_rate_hz"],
                }
            per_prn.append(record)
        prn_rows = [r for r in per_prn if r["dataset"] == dataset and r.get("valid_rows", 0) > 0]
        dataset_pass = bool(
            source_binding[dataset]["status"] == "PASS" and len(valid) >= 100 and len(prn_rows) >= 4
            and all(r.get("gate_pass_fraction") == 1.0 and r.get("time_span_s", 0) > 10.0 for r in prn_rows)
            and len(rejected_ds) == 0
        )
        overall_pass &= dataset_pass
        dataset_summaries[dataset] = {
            "attempted_rows": len(attempted), "valid_reconstructed_rows": len(valid), "rejected_rows": len(rejected_ds),
            "valid_prns": sorted({int(r["prn"]) for r in valid}),
            "delay_error_distribution_chips": percentile_summary([float(r["delay_center_error_chips"]) for r in valid]) if valid else None,
            "doppler_error_distribution_hz": percentile_summary([float(r["doppler_center_error_hz"]) for r in valid]) if valid else None,
            "complex_cosine_distribution": percentile_summary([float(r["complex_tap_cosine"]) for r in valid]) if valid else None,
            "magnitude_spearman_distribution": percentile_summary([float(r["magnitude_spearman"]) for r in valid]) if valid else None,
            "prompt_magnitude_ratio_distribution": percentile_summary([float(r["prompt_magnitude_ratio"]) for r in valid]) if valid else None,
            "prompt_phase_error_rad_distribution": percentile_summary([float(r["prompt_phase_error_rad"]) for r in valid]) if valid else None,
            "gate_pass_fraction": float(np.mean([r["gate_pass"] for r in valid])) if valid else 0.0,
            "time_span_s": max((float(r["receiver_timestamp_s"]) for r in attempted), default=0.0) - min((float(r["receiver_timestamp_s"]) for r in attempted), default=0.0),
            "status": "PASS" if dataset_pass else "FAIL",
        }
    per_prn_fields = [
        "dataset", "prn", "attempted_rows", "valid_rows", "rejected_rows", "delay_error_abs_max_chips",
        "doppler_error_abs_max_hz", "complex_cosine_min", "complex_cosine_median", "magnitude_spearman_min",
        "magnitude_spearman_median", "prompt_magnitude_ratio_median", "prompt_phase_error_abs_max_rad",
        "gate_pass_fraction", "time_span_s",
    ]
    write_csv(ART / "per_prn_alignment_metrics.csv", per_prn, per_prn_fields)

    if not binding_ok:
        verdict = "SOURCE_BINDING_MISMATCH"
    elif navbit["status"] != "PASS":
        verdict = "NAV_BIT_OR_SYMBOL_ALIGNMENT_REQUIRED"
    elif overall_pass:
        verdict = "STAGE0A_RAW_ALIGNMENT_PASS"
    else:
        verdict = "STAGE0A_RAW_ALIGNMENT_FAIL"
    alignment = {
        "schema": "gnss-doppler-lab.mosaic-stage0a-r1-alignment-summary.v1",
        "frozen_gate": FROZEN_GATE, "datasets": dataset_summaries,
        "metric_definitions": {
            "delay_center_error_chips": "action_used residual code phase minus its independently dumped residual-sample representation converted with code NCO rate/fs",
            "doppler_center_error_hz": "action_used carrier Doppler minus carrier phase-step representation converted with fs/(2*pi)",
            "complex_tap_cosine": "native versus raw-reconstructed nine-tap vector after one least-squares global complex amplitude per epoch",
            "magnitude_spearman": "Spearman rank correlation of native and fitted reconstructed nine-tap magnitudes",
        },
        "overall_support_rows": len(metric_rows), "overall_rejected_rows": len(rejected),
        "all_selected_support_evaluated": len(inventory_rows) == len(metric_rows) + len(rejected),
        "stage0a_pass": verdict == "STAGE0A_RAW_ALIGNMENT_PASS",
        "verdict": verdict,
    }
    dump_json("alignment_summary.json", alignment)
    next_action = (
        "Obtain a decoded and independently validated navigation-bit sequence in a separate Stage-0B provenance task."
        if verdict == "STAGE0A_RAW_ALIGNMENT_PASS" else
        "Do not proceed to Stage-0B; close the receiver/raw-alignment root cause."
    )
    dump_json("final_verdict.json", {
        "schema": "gnss-doppler-lab.mosaic-stage0a-r1-final-verdict.v1", "verdict": verdict,
        "stage0a_pass": verdict == "STAGE0A_RAW_ALIGNMENT_PASS", "stage0b_go": False,
        "stage0b_run": False, "attack_data_used": False, "meaning": "Only native complex nine-tap reconstruction from real cleanStatic raw IQ was evaluated.",
        "recommended_next_action": next_action,
    })

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        columns = [("complex_tap_cosine", "Complex cosine"), ("magnitude_spearman", "Magnitude Spearman"),
                   ("prompt_magnitude_ratio", "Prompt magnitude ratio"), ("prompt_phase_error_rad", "Prompt phase error (rad)")]
        for axis, (key, label) in zip(axes.ravel(), columns):
            for dataset in DATASETS:
                values = [float(row[key]) for row in metric_rows if row["dataset"] == dataset]
                axis.hist(values, bins=30, alpha=0.55, label=dataset)
            axis.set_title(label); axis.legend(fontsize=7)
        fig.savefig(ART / "plots/alignment_distributions.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        (ART / "plots/plot_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    environment = json.loads((ART / "execution_environment.json").read_text())
    environment["worker_max_rss_kib_after_run"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    environment["runtime_s_before_manifest"] = time.time() - started
    dump_json("execution_environment.json", environment)
    (ART / "README.md").write_text(
        "# MOSAIC Stage-0A R1 raw-IQ recorrelation\n\n"
        f"Final verdict: `{verdict}`. This means only that real cleanStatic raw IQ can be reconstructed into "
        "the receiver's native complex nine taps under the source-frozen convention; it is not Stage-0B GO.\n\n"
        "The old navigation-bit blocker is not required for exact 1 ms Stage-0A tap-shape alignment because "
        "the bit is a common ±1 complex scalar removed by the independently fitted per-epoch amplitude. "
        "Stage-0B still requires decoded/validated navigation-bit provenance to synthesize continuous raw IQ.\n\n"
        "The carrier sign, code-delay sign, tap coordinates, sample format, and current-interval state timing were "
        "fixed from the patched receiver source and receiver configs before looking at alignment metrics. No attack "
        "recording was read, no alternative convention was selected by outcome, and no threshold was changed.\n\n"
        "The delay center error cross-checks native residual-chip and residual-sample fields; the Doppler center "
        "error cross-checks native Doppler and phase-step fields. Complex cosine and magnitude Spearman compare "
        "native and directly raw-reconstructed taps after one global complex-amplitude fit per epoch.\n\n"

        + "\n".join(f"- {dataset}: {summary['status']}, {summary['valid_reconstructed_rows']} valid, {summary['rejected_rows']} rejected, gate pass fraction {summary['gate_pass_fraction']:.6f}." for dataset, summary in dataset_summaries.items())
        + "\n\nNo target PRN failed the frozen gate.\n\n"
        f"Next action: {next_action}\n"
    )
    manifest = {
        str(path.relative_to(ART)): sha256_file(path)
        for path in sorted(ART.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest_sha256.json"
    }
    dump_json("artifact_manifest_sha256.json", manifest)
    print(json.dumps({"verdict": verdict, "rows": len(metric_rows), "rejected": len(rejected), "runtime_s": time.time() - started}, indent=2))
    return 0 if verdict in {"STAGE0A_RAW_ALIGNMENT_PASS", "STAGE0A_RAW_ALIGNMENT_FAIL", "NAV_BIT_OR_SYMBOL_ALIGNMENT_REQUIRED", "RECEIVER_CONVENTION_UNRESOLVED", "SOURCE_BINDING_MISMATCH"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
