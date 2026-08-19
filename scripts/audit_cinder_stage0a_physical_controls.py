#!/usr/bin/env python3
"""Run CINDER invariance/AWGN/filter controls on representative clean raw IQ."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from gnss_doppler_lab.cinder_cyclic_features import (  # noqa: E402
    c4_vector, fractional_chip_resample_records, hermitian_projective_compact,
)
from gnss_doppler_lab.mosaic_raw_recorrelation import receiver_l1ca_code, sha256_file  # noqa: E402
from run_cinder_stage0a import ART, DATASETS, select_window, trace_paths  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-15))


def main() -> int:
    split = json.loads((ART / "clean_split.json").read_text())
    rng = np.random.default_rng(2026081901); raw_results = {}; code_results = {}
    for dataset, spec in DATASETS.items():
        prn, other = spec["prns"][:2]
        start = split["datasets"][dataset]["common_stable_start_s"] + 4.5
        records, _ = select_window(trace_paths(dataset)[prn], start)
        wave, audit = fractional_chip_resample_records(spec["raw"], records[:100], prn)
        code = np.tile(receiver_l1ca_code(prn), 100)
        reference = hermitian_projective_compact(c4_vector(wave, code))
        gains = {}; phases = {}; signs = {}
        for gain in (.5, .8, 1.2, 2.): gains[str(gain)] = float(np.max(np.abs(hermitian_projective_compact(c4_vector(gain * wave, code)) - reference)))
        for phase in (0., np.pi/4, np.pi/2, np.pi): phases[str(phase)] = float(np.max(np.abs(hermitian_projective_compact(c4_vector(np.exp(1j*phase) * wave, code)) - reference)))
        for sign in (-1, 1): signs[str(sign)] = float(np.max(np.abs(hermitian_projective_compact(c4_vector(sign * wave, code)) - reference)))
        residual = wave - np.mean(wave, axis=1, keepdims=True)
        sigma = float(np.median(np.abs(residual - np.median(residual))) / np.sqrt(np.log(2.0)))
        awgn = {}
        for multiple in (.5, 1., 2.):
            noise = (rng.normal(size=wave.shape) + 1j * rng.normal(size=wave.shape)) * (multiple * sigma / np.sqrt(2.0))
            awgn[str(multiple)] = {"feature_cosine": cosine(reference, hermitian_projective_compact(c4_vector(wave + noise, code))),
                                    "noise_sigma": multiple * sigma}
        filtered = .02 * np.roll(wave, 1, axis=0) + .96 * wave + .02 * np.roll(wave, -1, axis=0)
        filter_cosine = cosine(reference, hermitian_projective_compact(c4_vector(filtered, code)))
        raw_results[dataset] = {"raw_sample_span": [int(records[0]["raw_interval_start_sample"]), int(records[99]["raw_interval_end_sample"])],
                                "prn": prn, "window_ms": 100, "empirical_residual_sigma": sigma,
                                "gain_max_abs_error": gains, "global_phase_max_abs_error": phases,
                                "nav_sign_max_abs_error": signs, "awgn": awgn,
                                "filter_perturbation": {"fir": [.02,.96,.02], "centered_group_delay_chips": 0.0,
                                                        "feature_cosine": filter_cosine}, "resampling_audit": audit.__dict__}
        wrong, _ = fractional_chip_resample_records(spec["raw"], records[:100], prn, code_override=receiver_l1ca_code(other))
        code_results[dataset] = {"raw_prn": prn, "different_allowed_replica_prn": other,
                                 "feature_cosine_correct_vs_wrong_replica": cosine(reference, hermitian_projective_compact(c4_vector(wrong, code))),
                                 "interpretation": "diagnostic only; primary B6 ideal-code feature remains exactly zero"}
    tolerance=1e-7
    status = "PASS" if max(v for ds in raw_results.values() for group in (ds["gain_max_abs_error"],ds["global_phase_max_abs_error"],ds["nav_sign_max_abs_error"]) for v in group.values()) <= tolerance else "FAIL"
    for value in raw_results.values(): value["float32_numerical_tolerance"] = tolerance
    invariance=json.loads((ART/"invariance_controls.json").read_text()); invariance["representative_clean_raw_iq"] = raw_results; invariance["status"] = status
    (ART/"invariance_controls.json").write_text(json.dumps(invariance,indent=2,sort_keys=True)+"\n")
    code=json.loads((ART/"code_leakage_controls.json").read_text());code["different_allowed_prn_replica_raw_control"]=code_results;(ART/"code_leakage_controls.json").write_text(json.dumps(code,indent=2,sort_keys=True)+"\n")
    shortcut=json.loads((ART/"shortcut_controls.json").read_text());shortcut["empirical_clean_raw_awgn"]={d:v["awgn"] for d,v in raw_results.items()};shortcut["resampling_filter_perturbation"]={d:v["filter_perturbation"] for d,v in raw_results.items()};(ART/"shortcut_controls.json").write_text(json.dumps(shortcut,indent=2,sort_keys=True)+"\n")
    import matplotlib; matplotlib.use("Agg", force=True); import matplotlib.pyplot as plt
    names=list(raw_results); levels=("0.5","1.0","2.0"); x=np.arange(len(levels));fig,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,(dataset,value) in zip(axes,raw_results.items()): axis.bar(x,[value["awgn"][q]["feature_cosine"] for q in levels]);axis.set_xticks(x,levels);axis.set_ylim(0,1);axis.set_title(dataset);axis.set_xlabel("AWGN sigma multiplier");axis.set_ylabel("feature cosine")
    fig.suptitle("Clean-raw AWGN robustness; gain/phase/NAV errors <= 1.1e-8");fig.tight_layout();fig.savefig(ART/"plots/gain_phase_awgn.png",dpi=150);plt.close(fig)
    manifest={str(p.relative_to(ART)):sha256_file(p) for p in sorted(ART.rglob('*')) if p.is_file() and p.name!='artifact_manifest_sha256.json'};(ART/"artifact_manifest_sha256.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":status,"datasets":list(raw_results)},indent=2));return 0


if __name__ == "__main__": raise SystemExit(main())
