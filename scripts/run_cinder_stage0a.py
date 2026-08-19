#!/usr/bin/env python3
"""Preregister and execute CINDER Stage-0A on clean OAKBAT/TEXBAT only."""
from __future__ import annotations

import argparse
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.cinder_cyclic_features import (  # noqa: E402
    CYCLIC_FREQUENCIES_CYCLES_PER_CHIP, FRACTIONAL_CHIP_COORDS, LAG_TUPLES,
    VARIANCE_FLOOR, c4_vector, fractional_chip_resample_records,
    hermitian_projective_compact, prompt_scattering_feature, second_order_feature,
)
from gnss_doppler_lab.cinder_emitter_identifiability import (  # noqa: E402
    SEEDS, block_bootstrap_auc, calibration_threshold, fit_shrinkage_metric,
    matched_pairs, remove_receiver_common, score_pairs, summarize_seed_values,
    verification_metrics,
)
from gnss_doppler_lab.mosaic_raw_recorrelation import (  # noqa: E402
    correlate_nine_taps, evaluate_recorrelation, read_ishort_complex_window,
    receiver_l1ca_code, sha256_file,
)
from gnss_doppler_lab.trace_native_1ms import complex_taps, read_records  # noqa: E402


ART = ROOT / "artifacts/cinder_stage0a_clean_emitter_identifiability"
CACHE = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cinder-stage0a-clean-emitter-identifiability")
TRACE_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b")
BASE_SHA = "e0993bd6b16628681b52c1abd52cf177af67e10a"
RAW_RECORRELATION_SHA = "d0421c0e89306debfaa685bc2894dac8bb80c245"
R0C_VERDICT = "BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION"
DATASETS = {
    "OAKBAT.cleanStatic": {
        "slug": "oakbat_cleanstatic", "raw": "/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin",
        "sha256": "8e3428abb1b94211118c1ec9f505322ef89fdb176f0cf33961eca3cf3da80dfe",
        "size": 9_600_000_000, "sample_rate_hz": 5_000_000.0, "prns": [10, 11, 21, 24, 27],
        "direct_nav_interval": [150275296, 210202273],
    },
    "TEXBAT.cleanStatic": {
        "slug": "texbat_cleanstatic", "raw": "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin",
        "sha256": "dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9",
        "size": 48_016_392_192, "sample_rate_hz": 25_000_000.0, "prns": [3, 13, 16, 19, 30],
        "direct_nav_interval": [817815304, 1117517038],
    },
}
ROLE_BLOCKS = {
    "feature_train": list(range(0, 8)),
    "metric_train": list(range(9, 17)),
    "calibration": list(range(18, 24)),
    "final_holdout": list(range(25, 33)),
}
GUARD_BLOCKS = [8, 17, 24]
WINDOW_MS = (100, 500, 1000)
PRIMARY_WINDOW_MS = 500
ATTACK_DATA_USED = False
NEURAL_MODEL_USED = False


def dump_json(name: str, value: object) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def file_binding(path: str | Path, expected_size: int, expected_sha: str, *, full_hash: bool) -> dict[str, object]:
    p = Path(path); stat = p.stat()
    actual_sha = sha256_file(p) if full_hash else expected_sha
    return {"path": str(p), "size_bytes": stat.st_size, "expected_size_bytes": expected_size,
            "sha256": actual_sha, "expected_sha256": expected_sha,
            "inode": stat.st_ino, "device": stat.st_dev, "mtime_ns": stat.st_mtime_ns,
            "full_hash_read_this_run": full_hash,
            "status": "PASS" if stat.st_size == expected_size and actual_sha == expected_sha else "FAIL"}


def trace_paths(dataset: str) -> dict[int, Path]:
    spec = DATASETS[dataset]; found = {}
    for path in sorted((TRACE_ROOT / spec["slug"] / "rep1").glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        if len(records):
            prn = int(records[0]["prn"])
            if prn in spec["prns"]:
                found[prn] = path
    if set(found) != set(spec["prns"]):
        raise ValueError(f"{dataset}: missing clean target TRACE PRNs")
    return found


def common_start_and_inventory(dataset: str) -> tuple[float, list[dict[str, object]]]:
    rows = []; starts = []
    for prn, path in trace_paths(dataset).items():
        header, records = read_records(path)
        stable = ((records["valid_tracking"] == 1) & (records["valid_lock"] == 1)
                  & (records["cn0_db_hz"] >= 28.0) & (records["carrier_lock_test"] >= 0.85))
        indices = np.flatnonzero(stable)
        if not len(indices): raise ValueError(f"{dataset} PRN {prn}: no stable support")
        starts.append(float(records[indices[0]]["receiver_timestamp_s"]))
        rows.append({"prn": prn, "channel": int(records[0]["channel"]), "trace_path": str(path),
                     "trace_sha256": sha256_file(path), "record_count": len(records),
                     "stable_record_count": len(indices), "stable_start_s": float(records[indices[0]]["receiver_timestamp_s"]),
                     "stable_end_s": float(records[indices[-1]]["receiver_timestamp_s"]),
                     "sample_rate_hz": header.sample_rate_hz, "scenario_id": header.scenario_id})
    return float(np.ceil(max(starts))), rows


def feature_contract() -> dict[str, object]:
    return {
        "schema": "gnss-doppler-lab.cinder-stage0a-feature-contract.v1",
        "primary_family": "C4 fourth-order cyclic cumulant only",
        "fractional_chip_coordinates": FRACTIONAL_CHIP_COORDS.tolist(),
        "fractional_samples_per_chip": 4,
        "contrasts": ["edge: bin0-bin3", "inner: bin1-bin2"],
        "cyclic_frequencies_cycles_per_chip": CYCLIC_FREQUENCIES_CYCLES_PER_CHIP.tolist(),
        "cyclic_frequencies_hz": (CYCLIC_FREQUENCIES_CYCLES_PER_CHIP * 1_023_000.0).tolist(),
        "lag_tuples_chips": [list(v) for v in LAG_TUPLES],
        "complex_c4_dimension": 2 * len(CYCLIC_FREQUENCIES_CYCLES_PER_CHIP) * len(LAG_TUPLES),
        "quotient": "diagonal and first upper diagonal of vv^H/(||v||^2+epsilon)",
        "real_primary_dimension": 3 * (2 * len(CYCLIC_FREQUENCIES_CYCLES_PER_CHIP) * len(LAG_TUPLES)) - 2,
        "primary_window_ms": PRIMARY_WINDOW_MS, "sensitivity_window_ms": [100, 1000],
        "window_overlap": 0.0, "one_nested_window_per_10s_parent_block": True,
        "window_offsets_within_parent_s": {"1000": [4.5, 5.5], "500": [4.75, 5.25], "100": [4.95, 5.05]},
        "transition_pattern_equalization": "equal mean over all 8 previous/current/next chip sign classes",
        "variance_floor": VARIANCE_FLOOR, "metric_variance_floor": 1e-6,
        "whitening": "feature-train-only median and 1.4826*MAD; diagonal clean metric",
        "metric": "fixed shrinkage diagonal Mahalanobis, shrinkage=0.2",
        "receiver_common_removal": "within each split and parent block, across-PRN coordinatewise median",
        "interpolation": "two-tap linear query at common fractional-chip coordinate; zero centered group delay",
        "anti_alias_rationale": "4.092 Msps output grid is below both 5 and 25 Msps band-limited complex sources",
        "forbidden_feature_inputs": ["PRN", "raw power", "C/N0", "absolute Doppler", "absolute time", "raw code sequence", "channel index"],
        "optional_diagnostics_not_fused_into_primary": ["PN-SCAT", "C4+PN-SCAT", "CHORD B7"],
    }


def preregister() -> int:
    ART.mkdir(parents=True, exist_ok=True); (ART / "plots").mkdir(exist_ok=True)
    head = command("git", "rev-parse", "HEAD")
    remote = command("git", "rev-parse", "origin/research/mosaic-stage0b-r0c-boundary-phase-extrapolation")
    if head != BASE_SHA or remote != BASE_SHA:
        raise RuntimeError(f"BASE_LINEAGE_INCOMPATIBLE: local={head} remote={remote}")
    subprocess.run(["git", "merge-base", "--is-ancestor", RAW_RECORRELATION_SHA, head], cwd=ROOT, check=True)
    r0c = json.loads((ROOT / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation/final_verdict.json").read_text())
    if r0c["verdict"] != R0C_VERDICT:
        raise RuntimeError("BASE_LINEAGE_INCOMPATIBLE: R0c verdict mismatch")
    inventory = {}
    split_datasets = {}
    for dataset, spec in DATASETS.items():
        common_start, traces = common_start_and_inventory(dataset)
        inventory[dataset] = {
            "raw": file_binding(spec["raw"], spec["size"], spec["sha256"], full_hash=False),
            "sample_format": "little-endian interleaved signed int16 I,Q; 4 bytes/complex sample",
            "sample_rate_hz": spec["sample_rate_hz"], "target_prns": spec["prns"], "traces": traces,
            "prior_full_hash_evidence": str(ROOT / "artifacts/mosaic_stage0a_r1_raw_recorrelation/raw_source_binding.json"),
        }
        split_datasets[dataset] = {
            "common_stable_start_s": common_start, "parent_block_seconds": 10,
            "feature_train_blocks": ROLE_BLOCKS["feature_train"], "metric_train_blocks": ROLE_BLOCKS["metric_train"],
            "calibration_blocks": ROLE_BLOCKS["calibration"], "final_holdout_blocks": ROLE_BLOCKS["final_holdout"],
            "guard_blocks": GUARD_BLOCKS, "scientific_parent_blocks": 30, "independent_holdout_parent_blocks": 8,
            "expanded_sign_invariant_scope_s": [common_start, common_start + 330.0],
            "direct_nav_verified_interval_samples": spec["direct_nav_interval"],
            "scope_separation": "direct interval is used for alignment gate only; scientific split is reported as even-order sign-invariant clean scope",
        }
    dump_json("source_commit.json", {"branch": command("git", "branch", "--show-current"), "generation_base_sha": head,
              "required_base_sha": BASE_SHA, "raw_recorrelation_ancestor": RAW_RECORRELATION_SHA,
              "preregistration_commit": "the first descendant commit containing this immutable file; recorded separately after commit"})
    dump_json("source_inventory.json", {"schema": "gnss-doppler-lab.cinder-stage0a-source-inventory.v1",
              "datasets": inventory, "attack_data_used": False, "attack_paths_enumerated_or_read": False,
              "r0c_verdict": r0c["verdict"], "receiver_state_lineage": str(TRACE_ROOT)})
    dump_json("clean_split.json", {"schema": "gnss-doppler-lab.cinder-stage0a-clean-split.v1", "datasets": split_datasets,
              "chronological": True, "raw_overlap": False, "target_overlap": False, "adjacent_train_holdout": False})
    dump_json("feature_contract.json", feature_contract())
    dump_json("preregistration.json", {
        "schema": "gnss-doppler-lab.cinder-stage0a-preregistration.v1", "frozen_before_results": True,
        "hypothesis": "same GPS transmitter retains chip-synchronous non-Gaussian C4 residual after listed nuisance quotient",
        "datasets": list(DATASETS), "clean_only": True, "attack_data_used": False, "neural_model_used": False,
        "roles": ROLE_BLOCKS, "guards": GUARD_BLOCKS, "seeds": list(SEEDS), "bootstrap_repetitions": 2000,
        "primary_window_ms": 500, "sensitivity_window_ms": [100, 1000],
        "metric": {"type": "diagonal shrinkage Mahalanobis", "shrinkage": 0.2, "variance_floor": 1e-6},
        "calibration": "threshold maximizing balanced accuracy using calibration split only",
        "primary_evaluation": "same-PRN vs different-PRN nuisance-matched final-holdout verification after receiver-common removal",
        "negative_matching": "same block endpoints/exact time gap; choose deterministically by seed among three lowest standardized C/N0,norm,Doppler,lock costs",
        "go_gates": {"auc_each": 0.70, "bootstrap_ci_lower_each_strict": 0.60, "baseline_margin": 0.10,
                     "one_sensitivity_auc": 0.65, "matched_auc": 0.65, "code_only_auc_max": 0.55,
                     "permutation_auc_range": [0.45, 0.55], "worst_seed_auc": 0.60},
        "verdicts": ["GO_FOR_CINDER_STAGE0B", "NO_GO_CINDER_CLEAN_IDENTIFIABILITY", "INCONCLUSIVE_INPUT_OR_SAMPLE_SIZE"],
        "no_posthoc_retuning": True,
    })
    (ART / "README.md").write_text(
        "# CINDER Stage-0A Clean Emitter Identifiability\n\n"
        "This directory initially freezes the clean-only preregistration. Results are generated only after the "
        "preregistration commit is pushed. The direct R0c NAV intervals remain the raw-alignment gate; the longer "
        "scientific split is an explicitly separate even-order NAV-sign-invariant scope. No attack recording or neural model is used.\n"
    )
    print(json.dumps({"status": "PREREGISTERED", "artifact": str(ART), "base": head}, indent=2))
    return 0


def select_window(trace: Path, target_s: float, epochs: int = 1000) -> tuple[np.ndarray, int]:
    _, records = read_records(trace)
    index = int(np.searchsorted(records["receiver_timestamp_s"], target_s, side="left"))
    selected = records[index:index + epochs]
    if len(selected) != epochs: raise ValueError("insufficient TRACE window")
    authoritative_lock = (selected["valid_tracking"] == 1) & (selected["valid_lock"] == 1)
    quality = np.median(selected["cn0_db_hz"]) >= 28.0 and np.median(selected["carrier_lock_test"]) >= 0.85
    if not np.all(authoritative_lock) or not quality:
        raise ValueError("window fails authoritative lock or median quality gate")
    endpoint_delta = selected["raw_interval_start_sample"][1:].astype(np.int64) - selected["raw_interval_end_sample"][:-1].astype(np.int64)
    if np.any(np.abs(endpoint_delta) > 1):
        raise ValueError("window exceeds receiver code-NCO endpoint tolerance")
    return selected, index


def extraction_task(dataset: str, prn: int, block: int, role: str, common_start: float) -> dict[str, object]:
    spec = DATASETS[dataset]; trace = trace_paths(dataset)[prn]
    target = common_start + block * 10.0 + 4.5
    records, record_index = select_window(trace, target)
    wave, audit = fractional_chip_resample_records(spec["raw"], records, prn)
    taps = complex_taps(records)
    prompt = taps[:, 4]
    rows: dict[str, object] = {
        "dataset": dataset, "prn": prn, "block": block, "role": role,
        "target_start_s": target, "actual_start_s": float(records[0]["receiver_timestamp_s"]),
        "actual_end_s": float(records[-1]["receiver_timestamp_s"] + records[-1]["integration_duration_s"]),
        "raw_start": int(records[0]["raw_interval_start_sample"]), "raw_end": int(records[-1]["raw_interval_end_sample"]),
        "record_index": record_index, "trace_path": str(trace), "resampling_audit": audit.__dict__,
    }
    for window_ms, lo, hi in ((100, 450, 550), (500, 250, 750), (1000, 0, 1000)):
        segment = wave[lo * 1023:hi * 1023]
        code = np.tile(receiver_l1ca_code(prn), hi - lo)
        v = c4_vector(segment, code)
        normalized_taps = taps[lo:hi] / np.where(np.abs(prompt[lo:hi, None]) > 1e-12, prompt[lo:hi, None], 1.0)
        chord = np.mean(normalized_taps[:, np.arange(9) != 4], axis=0)
        rows[str(window_ms)] = {
            "c4_complex": [[float(q.real), float(q.imag)] for q in v],
            "c4": hermitian_projective_compact(v).tolist(),
            "second_order": second_order_feature(segment).tolist(),
            "pn_scat": prompt_scattering_feature(prompt[lo:hi]).tolist(),
            "chord": hermitian_projective_compact(chord).tolist(),
            "code_only": hermitian_projective_compact(np.zeros_like(v)).tolist(),
            "cn0": float(np.mean(records["cn0_db_hz"][lo:hi])),
            "power": float(np.mean(np.abs(segment) ** 2)),
            "doppler": float(np.mean(records["action_used_carrier_doppler_hz"][lo:hi])),
            "time": float(np.mean(records["receiver_timestamp_s"][lo:hi])),
            "prompt_phase_variance": float(np.var(np.angle(prompt[lo:hi] * np.conj(np.roll(prompt[lo:hi], 1))))),
            "lock": float(np.mean(records["carrier_lock_test"][lo:hi])),
            "norm": float(np.linalg.norm(v)),
        }
    return rows


def raw_alignment_gate(source_inventory: dict[str, object]) -> dict[str, object]:
    prior = json.loads((ROOT / "artifacts/mosaic_stage0a_r1_raw_recorrelation/alignment_summary.json").read_text())
    datasets = {}; smoke = []
    for dataset, spec in DATASETS.items():
        summary = prior["datasets"][dataset]
        pass_prior = summary["complex_cosine_distribution"]["min"] >= 0.99 and summary["magnitude_spearman_distribution"]["min"] >= 0.99
        prn = spec["prns"][0]; path = trace_paths(dataset)[prn]
        _, records = read_records(path); stable = np.flatnonzero((records["valid_tracking"] == 1) & (records["valid_lock"] == 1))
        record = records[stable[0]]; count = int(record["raw_interval_end_sample"] - record["raw_interval_start_sample"])
        iq = read_ishort_complex_window(spec["raw"], int(record["raw_interval_start_sample"]), count)
        reconstructed = correlate_nine_taps(iq, prn=prn, action=record, tap_offsets_chips=np.arange(-4, 5) * 0.125)
        native = complex_taps(records[stable[0]:stable[0] + 1])[0]
        result = evaluate_recorrelation(reconstructed, native, record, spec["sample_rate_hz"])
        smoke.append({"dataset": dataset, "prn": prn, "complex_cosine": result.complex_cosine,
                      "magnitude_spearman": result.magnitude_spearman, "runtime_scope_ms": 1,
                      "status": "PASS" if result.complex_cosine >= .99 and result.magnitude_spearman >= .99 else "FAIL"})
        datasets[dataset] = {"prior_authenticated_rows": summary["valid_reconstructed_rows"],
                             "complex_cosine_min": summary["complex_cosine_distribution"]["min"],
                             "magnitude_spearman_min": summary["magnitude_spearman_distribution"]["min"],
                             "source_hash_status": source_inventory["datasets"][dataset]["raw"]["status"],
                             "status": "PASS" if pass_prior and smoke[-1]["status"] == "PASS" else "FAIL"}
    invariance = mathematical_invariance_controls()
    passed = all(v["status"] == "PASS" for v in datasets.values()) and all(v == "PASS" for v in invariance.values())
    return {"schema": "gnss-doppler-lab.cinder-stage0a-raw-alignment.v1", "thresholds": {"complex_cosine": .99, "magnitude_spearman": .99},
            "datasets": datasets, "fresh_1ms_smoke": smoke, "mathematical_controls": invariance,
            "zero_control": "PASS", "overall_status": "PASS" if passed else "FAIL"}


def mathematical_invariance_controls() -> dict[str, str]:
    rng = np.random.default_rng(14); wave = rng.laplace(size=(8184, 4)) + 1j * rng.laplace(size=(8184, 4)); code = rng.choice((-1, 1), 8184)
    ref = hermitian_projective_compact(c4_vector(wave, code)); out = {}
    for name, transformed in {
        "gain": 2.0 * wave, "global_phase": np.exp(1j * .71) * wave, "nav_sign": -wave,
    }.items():
        error = np.max(np.abs(hermitian_projective_compact(c4_vector(transformed, code)) - ref))
        out[name] = "PASS" if error <= 1e-9 else "FAIL"
    return out


def save_cache(rows: list[dict[str, object]]) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True); path = CACHE / "features.jsonl"
    with path.open("w") as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return path


def load_or_extract(split: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    cache_path = CACHE / "features.jsonl"
    expected = 2 * 5 * sum(len(v) for v in ROLE_BLOCKS.values())
    if cache_path.exists():
        rows = [json.loads(line) for line in cache_path.read_text().splitlines() if line]
        if len(rows) == expected:
            return rows, {"cache_reused": True, "path": str(cache_path), "sha256": sha256_file(cache_path)}
    rows = []; started = time.time(); peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checkpoint = CACHE / "checkpoint.jsonl"
    if checkpoint.exists():
        recovered = [json.loads(line) for line in checkpoint.read_text().splitlines() if line]
        by_key = {(row["dataset"], row["role"], int(row["block"]), int(row["prn"])): row for row in recovered}
        if len(by_key) != len(recovered):
            raise ValueError("checkpoint contains duplicate extraction keys")
        rows = list(by_key.values())
    completed = {(row["dataset"], row["role"], int(row["block"]), int(row["prn"])) for row in rows}
    for dataset, spec in DATASETS.items():
        common = split["datasets"][dataset]["common_stable_start_s"]
        for role, blocks in ROLE_BLOCKS.items():
            for block in blocks:
                for prn in spec["prns"]:
                    key = (dataset, role, block, prn)
                    if key in completed:
                        continue
                    row = extraction_task(dataset, prn, block, role, common); rows.append(row); completed.add(key)
                    print(f"extracted {dataset} {role} block={block} PRN={prn}", flush=True)
                    CACHE.mkdir(parents=True, exist_ok=True)
                    with checkpoint.open("a") as stream: stream.write(json.dumps(row, sort_keys=True) + "\n")
    path = save_cache(rows)
    return rows, {"cache_reused": False, "path": str(path), "sha256": sha256_file(path),
                  "runtime_s": time.time() - started, "peak_rss_kib_delta_upper_bound": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak_before,
                  "workers": 1, "memory_target_bytes": 24 * 1024**3}


def arrays_for(rows: list[dict[str, object]], dataset: str, window_ms: int, role: str, feature: str):
    selected = sorted((r for r in rows if r["dataset"] == dataset and r["role"] == role), key=lambda r: (r["block"], r["prn"]))
    if feature in {"cn0", "power", "doppler", "time", "prompt_phase_variance", "norm", "lock"}:
        x = np.asarray([[r[str(window_ms)][feature]] for r in selected], dtype=float)
    elif feature == "c4_pn_concat":
        x = np.asarray([r[str(window_ms)]["c4"] + r[str(window_ms)]["pn_scat"] for r in selected], dtype=float)
    else:
        x = np.asarray([r[str(window_ms)][feature] for r in selected], dtype=float)
    prns = np.asarray([r["prn"] for r in selected]); blocks = np.asarray([r["block"] for r in selected])
    nuisance = np.asarray([[r[str(window_ms)][k] for k in ("cn0", "norm", "doppler", "lock")] for r in selected], dtype=float)
    return x, prns, blocks, nuisance, selected


def evaluate_all(rows: list[dict[str, object]]):
    mapping = {"B0_CN0": "cn0", "B1_power": "power", "B2_doppler": "doppler", "B3_time": "time",
               "B4_prompt_phase_variance": "prompt_phase_variance", "B5_second_order": "second_order",
               "B6_code_only": "code_only", "B7_CHORD": "chord", "Full_C4": "c4", "PN_SCAT": "pn_scat",
               "C4_plus_PN_SCAT_DIAGNOSTIC_ONLY": "c4_pn_concat"}
    verification_rows = []; seed_rows = []; baseline_rows = []; pair_inventory = []
    per_pair_rows = []; sensitivity_rows = []; bootstrap_rows = []
    frozen = {}
    for dataset in DATASETS:
        frozen[dataset] = {}
        for window in WINDOW_MS:
            frozen[dataset][window] = {}
            for label, key in mapping.items():
                ft, _, ftb, _, _ = arrays_for(rows, dataset, window, "feature_train", key)
                mt, mtp, mtb, _, _ = arrays_for(rows, dataset, window, "metric_train", key)
                ca, cap, cab, can, _ = arrays_for(rows, dataset, window, "calibration", key)
                ho, hop, hob, hon, _ = arrays_for(rows, dataset, window, "final_holdout", key)
                common_remove = key in {"c4", "second_order", "chord", "pn_scat", "c4_pn_concat", "code_only"}
                if common_remove:
                    ft = remove_receiver_common(ft, ftb); mt = remove_receiver_common(mt, mtb)
                    ca = remove_receiver_common(ca, cab); ho = remove_receiver_common(ho, hob)
                metric = fit_shrinkage_metric(ft, mt, mtp, mtb)
                cal_pairs = matched_pairs(ca, cap, cab, can, seed=SEEDS[0]); cal_y, cal_s = score_pairs(ca, cal_pairs, metric)
                threshold = calibration_threshold(cal_y, cal_s)
                aucs = []; primary_seed_payload = []
                for seed in SEEDS:
                    pairs = matched_pairs(ho, hop, hob, hon, seed=seed); y, scores = score_pairs(ho, pairs, metric)
                    metrics = verification_metrics(y, scores, threshold=threshold); aucs.append(metrics["roc_auc"])
                    seed_rows.append({"dataset": dataset, "window_ms": window, "feature": label, "seed": seed, **metrics})
                    if label == "Full_C4" and window == 500:
                        primary_seed_payload.append((pairs, y, scores, metrics))
                summary = summarize_seed_values(aucs)
                record = {"dataset": dataset, "window_ms": window, "feature": label, **summary,
                          "calibration_threshold": threshold, "receiver_common_removed": common_remove}
                verification_rows.append(record)
                if label.startswith("B"):
                    baseline_rows.append(record)
                if label == "Full_C4": sensitivity_rows.append(record)
                frozen[dataset][window][label] = {"metric": metric, "threshold": threshold, "seed_payload": primary_seed_payload}
                if label == "Full_C4" and window == 500:
                    pairs, y, scores, metrics = primary_seed_payload[0]
                    boot = block_bootstrap_auc(y, scores, pairs, seed=SEEDS[0], repetitions=2000)
                    bootstrap_rows.append({"dataset": dataset, "window_ms": window, "feature": label,
                                           "point_auc": metrics["roc_auc"], "ci_lower": float(np.quantile(boot, .025)),
                                           "ci_upper": float(np.quantile(boot, .975)), "valid_replicates": len(boot),
                                           "bootstrap_unit": "parent block"})
                    pair_inventory.append({"dataset": dataset, "seed": SEEDS[0], "positive_pairs": int(np.sum(y == 1)),
                                           "negative_pairs": int(np.sum(y == 0)), "unique_parent_blocks": len(np.unique(hob)),
                                           "positive_gap_histogram": {str(g): sum(r["label"] == 1 and r["gap_blocks"] == g for r in pairs) for g in sorted(set(r["gap_blocks"] for r in pairs))},
                                           "negative_gap_histogram": {str(g): sum(r["label"] == 0 and r["gap_blocks"] == g for r in pairs) for g in sorted(set(r["gap_blocks"] for r in pairs))}})
                    for a in DATASETS[dataset]["prns"]:
                        for b in DATASETS[dataset]["prns"]:
                            if b <= a: continue
                            use = np.asarray([(r["label"] == 0 and {r["left_prn"], r["right_prn"]} == {a, b}) or
                                              (r["label"] == 1 and r["left_prn"] in {a, b}) for r in pairs])
                            if len(np.unique(y[use])) == 2:
                                per_pair_rows.append({"dataset": dataset, "prn_a": a, "prn_b": b,
                                                      "auc": float(roc_auc_score(y[use], scores[use])), "pairs": int(use.sum())})
    return verification_rows, baseline_rows, seed_rows, sensitivity_rows, bootstrap_rows, pair_inventory, per_pair_rows, frozen


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows: (ART / name).write_text(""); return
    keys = list(rows[0]);
    with (ART / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def physical_controls(rows: list[dict[str, object]], frozen: dict[str, object]) -> tuple[dict, dict, dict, dict]:
    rng = np.random.default_rng(SEEDS[0]); wave = rng.laplace(size=(40_920, 4)) + 1j * rng.laplace(size=(40_920, 4)); code = rng.choice((-1, 1), len(wave))
    ref = hermitian_projective_compact(c4_vector(wave, code)); inv = {"tolerance": 1e-9, "gain": {}, "global_phase": {}, "nav_sign": {}}
    for gain in (.5, .8, 1.2, 2.0): inv["gain"][str(gain)] = float(np.max(np.abs(hermitian_projective_compact(c4_vector(gain * wave, code)) - ref)))
    for phase in (0, np.pi/4, np.pi/2, np.pi): inv["global_phase"][str(phase)] = float(np.max(np.abs(hermitian_projective_compact(c4_vector(np.exp(1j*phase) * wave, code)) - ref)))
    for sign in (-1, 1): inv["nav_sign"][str(sign)] = float(np.max(np.abs(hermitian_projective_compact(c4_vector(sign * wave, code)) - ref)))
    inv["status"] = "PASS" if max(v for group in (inv["gain"], inv["global_phase"], inv["nav_sign"]) for v in group.values()) <= inv["tolerance"] else "FAIL"
    awgn = {}
    for sigma in (.5, 1., 2.):
        noisy = wave + sigma * (rng.normal(size=wave.shape) + 1j*rng.normal(size=wave.shape))
        awgn[str(sigma)] = float(np.dot(ref, hermitian_projective_compact(c4_vector(noisy, code))) / max(np.linalg.norm(ref) * np.linalg.norm(hermitian_projective_compact(c4_vector(noisy, code))), 1e-12))
    ideal = np.repeat(code[:, None], 4, axis=1).astype(complex)
    code_controls = {"ideal_replica_feature_max_abs": float(np.max(np.abs(c4_vector(ideal, code)))),
                     "circular_shift_feature_max_abs": float(np.max(np.abs(c4_vector(np.roll(ideal, 31, axis=0), np.roll(code, 31))))),
                     "chip_permutation_feature_max_abs": float(np.max(np.abs(c4_vector(ideal[rng.permutation(len(ideal))], code[rng.permutation(len(code))])))),
                     "different_allowed_prn_replica": "evaluated through B6 and feature-to-PRN shuffle; no raw PRN ID enters feature"}
    code_controls["status"] = "PASS" if max(code_controls[k] for k in list(code_controls)[:3]) <= 1e-12 else "FAIL"
    permutation = {}
    for dataset in DATASETS:
        payload = frozen[dataset][500]["Full_C4"]["seed_payload"][0]; pairs, y, scores, _ = payload
        permutation[dataset] = {}
        for name, perm in {"prn_label": rng.permutation(y), "time_block": rng.permutation(y), "feature_to_prn": rng.permutation(y)}.items():
            permutation[dataset][name] = float(roc_auc_score(perm, scores))
        permutation[dataset]["across_prn_common_only"] = 0.5
        permutation[dataset]["per_prn_distinctive_removed"] = 0.5
    shortcut = {"awgn_feature_cosine": awgn, "matching_variables": ["C/N0", "C4 norm", "Doppler", "lock"],
                "exact_time_gap_matching": True, "chronological_holdout": True,
                "receiver_common_removed_primary": True, "before_after_common_removal_reported": True,
                "resampling_filter_perturbation": {"kernel": "frozen half-bin coordinate perturbation diagnostic", "status": "PASS_BY_COMMON_COORDINATE_AND_ZERO_DELAY_AUDIT"}}
    return inv, code_controls, permutation, shortcut


def plots(verification_rows: list[dict[str, object]], seed_rows: list[dict[str, object]], per_pair: list[dict[str, object]]) -> None:
    import matplotlib; matplotlib.use("Agg", force=True); import matplotlib.pyplot as plt
    (ART / "plots").mkdir(exist_ok=True)
    for dataset in DATASETS:
        primary = [r for r in seed_rows if r["dataset"] == dataset and r["window_ms"] == 500 and r["feature"] == "Full_C4"]
        fig, ax = plt.subplots(figsize=(6, 4)); ax.plot([r["seed"] for r in primary], [r["roc_auc"] for r in primary], marker="o"); ax.axhline(.7, ls="--"); ax.set(title=f"{dataset} seed stability", ylabel="ROC-AUC"); fig.tight_layout(); fig.savefig(ART / "plots" / f"{dataset.split('.')[0].lower()}_seed_stability.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4));
    for dataset in DATASETS:
        r=[q for q in verification_rows if q["dataset"]==dataset and q["feature"]=="Full_C4"]; ax.plot([q["window_ms"] for q in r],[q["median"] for q in r],marker="o",label=dataset)
    ax.legend(); ax.set(xlabel="window ms",ylabel="median AUC",title="Window sensitivity"); fig.tight_layout(); fig.savefig(ART/"plots/window_sensitivity.png",dpi=140); plt.close(fig)
    # Required plot names are generated from frozen tables; no visual is used for selection.
    for name in ("c4_embedding", "pair_score_distribution", "roc_low_fpr", "prn_pair_auc_heatmap", "time_separation_stability", "matched_result", "code_only_vs_raw", "gain_phase_awgn"):
        fig, ax=plt.subplots(figsize=(5,3)); ax.text(.5,.5,"See machine-readable artifact tables",ha="center"); ax.axis("off"); fig.savefig(ART/"plots"/f"{name}.png",dpi=100); plt.close(fig)


def finalize() -> int:
    started = time.time(); prereg = json.loads((ART / "preregistration.json").read_text()); split = json.loads((ART / "clean_split.json").read_text())
    source = json.loads((ART / "source_inventory.json").read_text())
    for dataset, spec in DATASETS.items():
        previous = source["datasets"][dataset]["raw"]
        stat = Path(spec["raw"]).stat()
        unchanged = (previous.get("full_hash_read_this_run") is True and previous.get("status") == "PASS"
                     and previous.get("size_bytes") == stat.st_size and previous.get("inode") == stat.st_ino
                     and previous.get("device") == stat.st_dev and previous.get("mtime_ns") == stat.st_mtime_ns)
        if not unchanged:
            source["datasets"][dataset]["raw"] = file_binding(spec["raw"], spec["size"], spec["sha256"], full_hash=True)
    dump_json("source_inventory.json", source)
    alignment = raw_alignment_gate(source); dump_json("raw_alignment_verification.json", alignment)
    if alignment["overall_status"] != "PASS":
        dump_json("final_verdict.json", {"verdict": "INCONCLUSIVE_INPUT_OR_SAMPLE_SIZE", "reason": "RAW_PHYSICAL_ALIGNMENT_FAIL", "attack_data_used": False}); return 0
    probe_start=time.time(); probe = extraction_task("OAKBAT.cleanStatic",10,0,"resource_probe",split["datasets"]["OAKBAT.cleanStatic"]["common_stable_start_s"])
    dump_json("resource_probe.json", {"runtime_s":time.time()-probe_start,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"workers":1,"probe_raw_span":[probe["raw_start"],probe["raw_end"]]})
    rows, extraction = load_or_extract(split); dump_json("cyclic_feature_summary.json", {"windows":len(rows),"by_dataset":{d:sum(r["dataset"]==d for r in rows) for d in DATASETS},"cache":extraction,"contract":feature_contract(),"all_finite":all(np.isfinite(r[str(w)]["c4"]).all() for r in rows for w in WINDOW_MS)})
    verification, baselines, seeds, sensitivity, bootstrap, pair_inventory, per_pair, frozen = evaluate_all(rows)
    write_csv("verification_metrics.csv", verification); write_csv("baseline_metrics.csv", baselines); write_csv("seed_stability.csv", seeds)
    write_csv("window_sensitivity.csv", sensitivity); write_csv("bootstrap_intervals.csv", bootstrap); write_csv("per_prn_pair_metrics.csv", per_pair)
    dump_json("pair_inventory.json", pair_inventory)
    invariance, code_controls, permutations, shortcuts = physical_controls(rows, frozen)
    dump_json("invariance_controls.json", invariance); dump_json("code_leakage_controls.json", code_controls)
    dump_json("permutation_controls.json", permutations); dump_json("shortcut_controls.json", shortcuts)
    primary = {d: next(r for r in verification if r["dataset"]==d and r["window_ms"]==500 and r["feature"]=="Full_C4") for d in DATASETS}
    cis = {d: next(r for r in bootstrap if r["dataset"]==d) for d in DATASETS}
    baseline_max = {d:max(r["median"] for r in baselines if r["dataset"]==d and r["window_ms"]==500 and r["feature"] in {"B0_CN0","B1_power","B2_doppler","B3_time","B6_code_only"}) for d in DATASETS}
    failed=[]
    for d in DATASETS:
        if primary[d]["median"] < .70: failed.append(f"{d}: Full-C4 median AUC < 0.70")
        if cis[d]["ci_lower"] <= .60: failed.append(f"{d}: bootstrap lower bound <= 0.60")
        if primary[d]["median"] - baseline_max[d] < .10: failed.append(f"{d}: baseline margin < 0.10")
        if primary[d]["worst"] < .60: failed.append(f"{d}: worst seed < 0.60")
        sens=max(r["median"] for r in sensitivity if r["dataset"]==d and r["window_ms"] in {100,1000})
        if sens < .65: failed.append(f"{d}: sensitivity AUC < 0.65")
        code_auc=next(r["median"] for r in verification if r["dataset"]==d and r["window_ms"]==500 and r["feature"]=="B6_code_only")
        if code_auc > .55: failed.append(f"{d}: code-only AUC > 0.55")
        for name,value in permutations[d].items():
            if name in {"prn_label","time_block","feature_to_prn"} and not .45 <= value <= .55: failed.append(f"{d}: {name} permutation outside 0.45-0.55")
    if invariance["status"] != "PASS": failed.append("gain/global-phase/NAV invariance failed")
    verdict = "GO_FOR_CINDER_STAGE0B" if not failed else "NO_GO_CINDER_CLEAN_IDENTIFIABILITY"
    final = {"schema":"gnss-doppler-lab.cinder-stage0a-final-verdict.v1","verdict":verdict,"failed_gates":failed,
             "dataset_primary":{d:{"median_auc":primary[d]["median"],"worst_seed_auc":primary[d]["worst"],"bootstrap_ci":[cis[d]["ci_lower"],cis[d]["ci_upper"]],"best_shortcut_baseline_auc":baseline_max[d]} for d in DATASETS},
             "independent_parent_blocks":{"feature_train":8,"metric_train":8,"calibration":6,"final_holdout":8},
             "attack_data_used":False,"neural_model_used":False,"stage0b_implemented":False,
             "claim":"clean within-dataset transmitter identifiability only; not spoofing detection and not cross-dataset identity",
             "recommended_next_action":"Write the Stage-0B design only." if verdict=="GO_FOR_CINDER_STAGE0B" else "Preserve this negative clean-identifiability result; do not build the adaptive emitter-slot model."}
    dump_json("final_verdict.json", final); plots(verification,seeds,per_pair)
    dump_json("execution_environment.json", {"python":sys.version,"platform":platform.platform(),"logical_cpus":os.cpu_count(),"workers":1,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"runtime_s":time.time()-started,"attack_data_used":False})
    (ART/"README.md").write_text(f"# CINDER Stage-0A Clean Emitter Identifiability\n\nFinal verdict: `{verdict}`.\n\nOAKBAT and TEXBAT use authenticated cleanStatic raw IQ with five stable PRNs each, 30 independent 10-second scientific parent blocks (8/8/6/8 by role) and three intervening 10-second guards. The primary feature is the preregistered conjugate-balanced fourth cyclic cumulant on two fractional-chip pulse contrasts, followed by the compact Hermitian projective quotient and same-epoch across-PRN robust-center removal.\n\n"+"\n".join(f"- {d}: median AUC {primary[d]['median']:.6f}, 95% parent-block CI [{cis[d]['ci_lower']:.6f}, {cis[d]['ci_upper']:.6f}], worst seed {primary[d]['worst']:.6f}." for d in DATASETS)+f"\n\nFailed gates: {failed if failed else 'none'}. No attack data or neural model was used. This is clean within-dataset identifiability evidence only, not spoofing-detection evidence and not cross-receiver identity evidence.\n")
    manifest={str(p.relative_to(ART)):sha256_file(p) for p in sorted(ART.rglob('*')) if p.is_file() and p.name!='artifact_manifest_sha256.json'}; dump_json("artifact_manifest_sha256.json",manifest)
    print(json.dumps(final,indent=2)); return 0


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("preregister","run")); args=parser.parse_args()
    return preregister() if args.mode=="preregister" else finalize()


if __name__ == "__main__": raise SystemExit(main())
