#!/usr/bin/env python3
"""Durable MIRAGE Stage-0A R1 alignment, clean, replay, control, and verdict runner."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.mirage_r1 import (  # noqa: E402
    SCALES, abs_spearman, epoch_features, full_score, paired_bootstrap,
    robust_reference, scale_surprise,
)
from gnss_doppler_lab.mirage_r1_executor import (  # noqa: E402
    generate_transient, load_mapping, reconstruct_records, role_minor_bundle,
    role_record_slice, run_receiver, trace_for_prn,
)
from gnss_doppler_lab.mosaic_raw_recorrelation import sha256_file  # noqa: E402
from gnss_doppler_lab.trace_native_1ms import read_records  # noqa: E402

ART = ROOT / "artifacts/mirage_stage0a_r1_full_execution"
EXTERNAL = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mirage-stage0a-r1-full-execution")
BRANCH = "research/mirage-stage0a-r1-full-execution"
PREREG_SHA = "bd27782c54c6f6df603f9a08b831013826d24046"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, text=True, check=True, stdout=subprocess.PIPE).stdout.strip()


def specs() -> dict[str, dict[str, object]]:
    inventory = json.loads((ART / "data_inventory.json").read_text())
    split = json.loads((ART / "clean_split_audit.json").read_text())
    result = {}
    for dataset, raw in inventory["raw_sources"].items():
        traces = inventory["trace_sources"][dataset]
        result[dataset] = {
            "fs": int(raw["sample_rate_hz"]), "raw": Path(raw["path"]), "raw_sha": raw["expected_sha256"],
            "trace_dir": Path(traces[0]["path"]).parent, "traces": {int(x["prn"]): Path(x["path"]) for x in traces},
            "trace_sha": {int(x["prn"]): x["sha256"] for x in traces}, "prns": [int(x["prn"]) for x in traces],
            "roles": split["datasets"][dataset]["roles"],
        }
    return result


def heartbeat(phase: str, detail: str) -> None:
    value = {"phase": phase, "detail": detail, "unix_time": time.time(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dump(EXTERNAL / "heartbeat.json", value)
    print(f"HEARTBEAT {phase}: {detail}", flush=True)


def require_frozen() -> str:
    head = command("git", "rev-parse", "HEAD")
    remote = command("git", "rev-parse", f"origin/{BRANCH}")
    binding = json.loads((ART / "execution_code_binding.json").read_text())
    if head != remote or not binding["status"] == "FROZEN":
        raise SystemExit(f"execution requires pushed frozen branch tip; local={head} remote={remote}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_SHA, head], cwd=ROOT).returncode:
        raise SystemExit("execution tip does not descend from preregistration")
    for name, expected in binding["code_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise SystemExit(f"frozen execution code mismatch: {name}")
    if command("git", "status", "--porcelain"):
        raise SystemExit("execution requires a clean worktree")
    return head


def update_phase(name: str, status: str) -> None:
    path = EXTERNAL / "phase_checkpoint.json"
    value = json.loads(path.read_text()) if path.exists() else {"phases": {}, "frozen_execution_sha": command("git", "rev-parse", "HEAD")}
    value["phases"][name] = {"status": status, "time": time.time()}
    dump(path, value)


def alignment() -> None:
    require_frozen(); EXTERNAL.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(ART / "extended_nav_mapping.csv.gz")
    rows = []
    source = specs()
    for dataset, spec in source.items():
        if sha256_file(spec["raw"]) != spec["raw_sha"]:
            raise SystemExit(f"raw SHA mismatch: {dataset}")
        for prn in spec["prns"]:
            trace = spec["traces"][prn]
            if sha256_file(trace) != spec["trace_sha"][prn]:
                raise SystemExit(f"TRACE SHA mismatch: {dataset} PRN {prn}")
            _, records = read_records(trace, mmap=True)
            candidates = []
            for role in ("train", "calibration", "holdout"):
                item = spec["roles"][role]
                take = role_record_slice(trace, item["raw_start_sample"], item["raw_end_sample_exclusive"])
                candidates.extend(take[np.linspace(0, len(take) - 1, 17, dtype=int)].tolist())
            from gnss_doppler_lab.mirage_r1 import native_alignment
            rows.extend({"dataset": dataset, **observed}
                        for observed in native_alignment(spec["raw"], trace, candidates[:50]))
            heartbeat("alignment", f"{dataset} PRN {prn}; {len(rows)}/500")
    by_dataset = {}
    for dataset in source:
        subset = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "selected": len(subset), "complex_cosine_median": float(np.median([r["complex_cosine"] for r in subset])),
            "magnitude_spearman_median": float(np.median([r["magnitude_spearman"] for r in subset])),
            "center_pass_fraction": float(np.mean([abs(r["center_error_chips"]) <= .125 for r in subset])),
            "gate_pass_fraction": float(np.mean([r["gate_pass"] for r in subset])),
        }
    passed = len(rows) >= 500 and all(
        v["selected"] >= 200 and v["complex_cosine_median"] >= .995 and
        v["magnitude_spearman_median"] >= .99 and v["center_pass_fraction"] >= .95
        for v in by_dataset.values())
    result = {"schema": "gnss-doppler-lab.mirage-r1-raw-recorrelation.v1", "rows": rows,
              "datasets": by_dataset, "selected_total": len(rows), "raw_and_trace_sha_match": True,
              "status": "PASS" if passed else "FAIL"}
    dump(EXTERNAL / "caf_reconstruction_validation.json", result)
    update_phase("alignment", result["status"])
    if not passed:
        raise SystemExit("INCONCLUSIVE_RAW_RECORRELATION_ALIGNMENT")


def _clean_task(args):
    dataset, prn, role_name, spec, mapping = args
    return dataset, prn, role_name, role_minor_bundle(
        Path(spec["raw"]), Path(spec["trace"]), spec["role"], mapping_rows=mapping, dataset=dataset)


def clean() -> None:
    require_frozen()
    gate = json.loads((EXTERNAL / "caf_reconstruction_validation.json").read_text())
    if gate["status"] != "PASS": raise SystemExit("alignment gate is not PASS")
    mapping = load_mapping(ART / "extended_nav_mapping.csv.gz")
    source = specs(); tasks = []
    for dataset, spec in source.items():
        for prn in spec["prns"]:
            subset = [r for r in mapping if r["dataset"] == dataset and int(r["prn"]) == prn]
            for role in ("train", "calibration", "holdout"):
                tasks.append((dataset, prn, role, {"raw": str(spec["raw"]), "trace": str(spec["traces"][prn]),
                                                   "role": spec["roles"][role]}, subset))
    bundles = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_clean_task, task) for task in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            dataset, prn, role, bundle = future.result()
            bundles[(dataset, prn, role)] = bundle
            heartbeat("clean_recorrelation", f"task {number}/{len(tasks)} {dataset} PRN {prn} {role}")
            dump(EXTERNAL / "clean" / f"{dataset.replace('.','_')}_prn{prn}_{role}.json", {
                "audit": bundle["audit"], "window_count": len(bundle["windows"]), "status": "PASS"})
    references = {}; score_rows = []; metrics = []; thresholds = {}
    reference_arrays = {}
    for dataset, spec in source.items():
        references[dataset] = {}
        for scale_index, scale in enumerate(SCALES):
            fields = [window["minors"][scale_index] for prn in spec["prns"]
                      for window in bundles[(dataset, prn, "train")]["windows"]]
            references[dataset][scale] = robust_reference(fields)
            reference_arrays[f"{dataset}|{scale}|location"] = references[dataset][scale]["location"]
            reference_arrays[f"{dataset}|{scale}|scale"] = references[dataset][scale]["scale"]
            reference_arrays[f"{dataset}|{scale}|train"] = references[dataset][scale]["train_statistics"]
        full_by_role = {}; scale_full_by_role = {}
        for role in ("train", "calibration", "holdout"):
            count = min(len(bundles[(dataset, p, role)]["windows"]) for p in spec["prns"])
            full_by_role[role] = []; scale_full_by_role[role] = [[], [], []]
            for ordinal in range(count):
                nodes = []; scale_nodes = [[], [], []]
                for prn in spec["prns"]:
                    window = bundles[(dataset, prn, role)]["windows"][ordinal]
                    scales = [scale_surprise(window["minors"][i], references[dataset][scale]) for i, scale in enumerate(SCALES)]
                    node = max(scales); nodes.append(node)
                    for scale_index, scale_value in enumerate(scales): scale_nodes[scale_index].append(scale_value)
                    score_rows.append({"dataset": dataset, "role": role, "epoch": ordinal, "segment": min(2, 3 * ordinal // max(count, 1)),
                                       "level": "PRN", "prn": prn, "score": node, "scale_20": scales[0],
                                       "scale_100": scales[1], "scale_500": scales[2], "energy": window["energy"],
                                       "svd": float(np.max(window["svd"])), "magnitude_minor": float(np.max(window["magnitude_minor"])),
                                       "raw_power": bundles[(dataset, prn, role)]["audit"]["mean_raw_power"],
                                       "cn0_db_hz": bundles[(dataset, prn, role)]["audit"]["mean_cn0_db_hz"]})
                value = full_score(nodes); full_by_role[role].append(value)
                for scale_index in range(3): scale_full_by_role[role][scale_index].append(full_score(scale_nodes[scale_index]))
                score_rows.append({"dataset": dataset, "role": role, "epoch": ordinal, "segment": min(2, 3 * ordinal // max(count, 1)),
                                   "level": "Full", "prn": "", "score": value, "scale_20": "", "scale_100": "",
                                   "scale_500": "", "energy": "", "svd": "", "magnitude_minor": "", "raw_power": "", "cn0_db_hz": ""})
        q99 = float(np.quantile(full_by_role["calibration"], .99, method="higher"))
        q995 = float(np.quantile(full_by_role["calibration"], .995, method="higher"))
        thresholds[dataset] = {"q99": q99, "q99_5_diagnostic": q995,
                               "q99_by_scale": [float(np.quantile(values, .99, method="higher")) for values in scale_full_by_role["calibration"]],
                               "calibration_count": len(full_by_role["calibration"])}
        hold = np.asarray(full_by_role["holdout"]); fpr = float(np.mean(hold > q99))
        segments = np.array_split(hold, 3); worst = max(float(np.mean(x > q99)) for x in segments)
        metrics.append({"dataset": dataset, "train_epochs": len(full_by_role["train"]),
                        "calibration_epochs": len(full_by_role["calibration"]), "holdout_epochs": len(hold),
                        "threshold_q99": q99, "holdout_fpr": fpr, "worst_segment_fpr": worst,
                        "status": "PASS" if fpr <= .015 and worst <= .05 else "FAIL"})
    np.savez_compressed(EXTERNAL / "clean_reference.npz", **reference_arrays)
    dump(EXTERNAL / "thresholds.json", thresholds)
    with gzip.open(EXTERNAL / "per_epoch_scores.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(score_rows)
    with (EXTERNAL / "clean_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(metrics)
    update_phase("clean", "PASS")
    heartbeat("clean", f"complete: {len(score_rows)} score rows")


def load_references(dataset: str) -> dict[float, dict[str, np.ndarray]]:
    archive = np.load(EXTERNAL / "clean_reference.npz")
    return {scale: {"location": archive[f"{dataset}|{scale}|location"],
                    "scale": archive[f"{dataset}|{scale}|scale"],
                    "train_statistics": archive[f"{dataset}|{scale}|train"]} for scale in SCALES}


def tap_scores(taps_by_prn: dict[int, np.ndarray], references, transform=None) -> dict[str, object]:
    sequences = {}; diagnostics = {}
    for prn, taps in taps_by_prn.items():
        node = []; scale_values = []; energy = []; svd = []; mag = []
        for i in range(0, len(taps) - 499, 500):
            value = taps[i:i + 500]
            if transform is not None: value = transform(prn, value, i // 500)
            feature = epoch_features(value, references)
            node.append(feature["node_score"]); scale_values.append([feature["scale_scores"][s] for s in SCALES])
            energy.append(feature["energy"]); svd.append(max(feature["svd"].values())); mag.append(max(feature["magnitude_minor"].values()))
        sequences[prn] = node; diagnostics[prn] = {"scales": scale_values, "energy": energy, "svd": svd, "magnitude": mag}
    epochs = min(len(x) for x in sequences.values())
    full = [full_score([sequences[p][i] for p in sorted(sequences)]) for i in range(epochs)]
    return {"node_sequences": {str(k): v for k, v in sequences.items()}, "full_sequence": full,
            "case_score": float(np.median(full)), "diagnostics": {str(k): v for k, v in diagnostics.items()}}


def authentic_bundle(dataset: str, spec: dict[str, object], case: dict[str, object], mapping, cache: Path):
    if cache.exists():
        z = np.load(cache); return {int(k[3:]): z[k] for k in z.files}
    fs = spec["fs"]; lo = case["anchor_start_sample"] + fs // 4; hi = case["anchor_start_sample"] + 7 * fs // 4
    output = {}
    for prn in spec["prns"]:
        trace = spec["traces"][prn]; indices = role_record_slice(trace, lo, hi)
        indices = indices[:len(indices) - len(indices) % 500]
        output[prn], _ = reconstruct_records(spec["raw"], trace, indices, mapping_rows=mapping, dataset=dataset)
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, **{f"prn{p}": x for p, x in output.items()})
    return output


def raw_awgn_bundle(dataset: str, spec: dict[str, object], case: dict[str, object], mapping,
                    cache: Path, power_ratio: float, seed_tag: int):
    if cache.exists():
        z = np.load(cache); return {int(k[3:]): z[k] for k in z.files}
    fs = spec["fs"]; lo = case["anchor_start_sample"] + fs // 4; hi = case["anchor_start_sample"] + 7 * fs // 4
    output = {}
    for prn in spec["prns"]:
        trace = spec["traces"][prn]; indices = role_record_slice(trace, lo, hi)
        indices = indices[:len(indices) - len(indices) % 500]
        def transform(iq, raw_start, epoch_begin, prn=prn):
            rng = np.random.default_rng(20260819 + seed_tag * 10000019 + prn * 1009 + raw_start % 1000003)
            centered = iq - np.mean(iq); sigma = np.sqrt(np.mean(np.abs(centered) ** 2) * power_ratio / 2)
            return iq + sigma * (rng.normal(size=len(iq)) + 1j * rng.normal(size=len(iq)))
        output[prn], _ = reconstruct_records(spec["raw"], trace, indices, mapping_rows=mapping,
            dataset=dataset, raw_transform=transform)
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, **{f"prn{p}": x for p, x in output.items()})
    return output


def execute_cases() -> None:
    require_frozen(); source = specs(); mapping = load_mapping(ART / "extended_nav_mapping.csv.gz")
    design = json.loads((ART / "injection_design.json").read_text())["cases"]
    inventory = json.loads((ART / "data_inventory.json").read_text()); receiver = Path(inventory["receiver"]["path"])
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for number, case in enumerate(design, 1):
        result_path = EXTERNAL / "cases" / case["case_id"] / "case_result.json"
        if result_path.exists():
            heartbeat("cases", f"resume skip {number}/84 {case['case_id']}"); continue
        spec = source[case["dataset"]]; root = result_path.parent; attempts = 0; error = None
        while attempts < 3 and not result_path.exists():
            attempts += 1
            try:
                temp = EXTERNAL / "temp" / case["case_id"]; shutil.rmtree(temp, ignore_errors=True)
                raw = temp / "raw" / "injected.bin"
                receipt = generate_transient(spec["raw"], raw, dataset=case["dataset"], fs=spec["fs"],
                    anchor=case["anchor_start_sample"], targets=case["target_prns"], rho_db=case["rho_db"],
                    delay_chips=case["delay_chips"], doppler_hz=case["doppler_hz"], phase_rad=case["phase_rad"],
                    trace_dir=spec["trace_dir"], mapping_rows=mapping)
                replay_dir = temp / "receiver"
                replay = run_receiver(receiver, spec["trace_dir"] / "receiver.conf", raw, replay_dir)
                if replay["status"] != "PASS": raise RuntimeError(f"receiver replay failed: {replay}")
                source_start = receipt["source_absolute_range"][0]; lo = int(30.25 * spec["fs"]); hi = int(31.75 * spec["fs"])
                taps = {}; replay_audits = {}
                for prn in spec["prns"]:
                    trace = trace_for_prn(replay_dir, prn); indices = role_record_slice(trace, lo, hi)
                    indices = indices[:len(indices) - len(indices) % 500]
                    taps[prn], replay_audits[prn] = reconstruct_records(raw, trace, indices, mapping_rows=mapping,
                        dataset=case["dataset"], absolute_sample_offset=source_start)
                references = load_references(case["dataset"]); primary = tap_scores(taps, references)
                cache = EXTERNAL / "controls" / "authentic_cache" / f"{case['dataset'].replace('.','_')}_{case['anchor_start_sample']}.npz"
                authentic_taps = authentic_bundle(case["dataset"], spec, case, mapping, cache)
                authentic = tap_scores(authentic_taps, references)
                ratio = 10 ** (case["rho_db"] / 20); collapsed_factor = 1 + ratio * np.exp(1j * case["phase_rad"])
                controls = {
                    "authentic_only": authentic,
                    "common_gain_scaling": tap_scores(authentic_taps, references, lambda p, x, i: 2 * x),
                    "global_phase_rotation": tap_scores(authentic_taps, references, lambda p, x, i: np.exp(1.234j) * x),
                    "matched_output_rms_gain": tap_scores(authentic_taps, references, lambda p, x, i: x * receipt["output_injection_region_rms"] / max(receipt["clean_injection_region_rms"], 1e-12)),
                    "collapsed_source": tap_scores(authentic_taps, references, lambda p, x, i: x * (collapsed_factor if p in case["target_prns"] else 1)),
                    "single_source_delay_drift": tap_scores(authentic_taps, references, lambda p, x, i: np.stack([np.interp(np.arange(9) - .5 * j / 499, np.arange(9), row.real) + 1j * np.interp(np.arange(9) - .5 * j / 499, np.arange(9), row.imag) for j, row in enumerate(x)])),
                    "single_source_doppler_drift": tap_scores(authentic_taps, references, lambda p, x, i: x * np.exp(1j * 2 * np.pi * 5 * (np.arange(500) / 1000) ** 2)[:, None]),
                }
                control_root = EXTERNAL / "controls" / "raw_awgn_cache"
                key = f"{case['dataset'].replace('.','_')}_{case['anchor_start_sample']}"
                awgn_taps = raw_awgn_bundle(case["dataset"], spec, case, mapping, control_root / f"{key}_awgn.npz", .10, 1)
                cn0_taps = raw_awgn_bundle(case["dataset"], spec, case, mapping, control_root / f"{key}_cn0_3db.npz", 1.0, 2)
                controls["empirical_raw_iq_awgn"] = tap_scores(awgn_taps, references)
                controls["cn0_degradation_3db"] = tap_scores(cn0_taps, references)
                one = {p: (taps[p] if p == case["target_prns"][0] else authentic_taps[p]) for p in spec["prns"]}
                controls["one_prn_secondary_path"] = tap_scores(one, references)
                sequences = primary["node_sequences"]; prns = sorted(int(p) for p in sequences)
                evaluation_epochs = min(len(v) for v in sequences.values())
                drop = [float(np.median([full_score([sequences[str(p)][i] for p in prns if p != gone]) for i in range(evaluation_epochs)])) for gone in prns]
                controls["prn_drop_add"] = {"drop_scores": drop, "add_restores_score": primary["case_score"], "case_score": float(np.median(drop))}
                result = {"case": case, "attempts": attempts, "injected_iq": receipt, "receiver": replay,
                          "primary": primary, "controls": controls,
                          "mean_replay_cn0_db_hz": float(np.mean([a["mean_cn0_db_hz"] for a in replay_audits.values()])),
                          "replay_tracking_audit": {str(p): a for p, a in replay_audits.items()}, "trace_receipts": [
                              {"name": p.name, "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                              for p in sorted(replay_dir.glob("trace_native_1ms_ch_*.bin"))], "status": "PASS"}
                dump(result_path, result); shutil.rmtree(temp)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"; dump(root / f"failure_attempt_{attempts}.json", {"error": error, "traceback": traceback.format_exc()})
        if not result_path.exists(): dump(result_path, {"case": case, "attempts": attempts, "error": error, "status": "FAIL"})
        heartbeat("cases", f"{number}/84 {case['case_id']} {json.loads(result_path.read_text())['status']}")
    update_phase("cases", "COMPLETE")


def write_csv(path: Path, rows: list[dict[str, object]], compressed=False) -> None:
    opener = gzip.open if compressed else Path.open
    kwargs = {"mode": "wt", "newline": ""} if compressed else {"mode": "w", "newline": ""}
    with opener(path, **kwargs) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def artifact_manifest() -> None:
    files = []
    for path in sorted(ART.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            files.append({"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    dump(ART / "artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.artifact-manifest-sha256.v1", "files": files})


def materialize_results(results, valid, verdict: str, gates) -> None:
    names = ["caf_reconstruction_validation.json", "thresholds.json", "clean_metrics.csv", "per_epoch_scores.csv.gz",
             "per_case_scores.csv.gz", "injection_metrics.csv", "control_metrics.csv", "scale_ablation.csv",
             "relation_destruction_metrics.json", "prn_dominance.json", "shortcut_audit.json", "bootstrap_intervals.csv",
             "invariance_validation.json", "multiscale_gate.json", "final_verdict.json"]
    for name in names: shutil.copy2(EXTERNAL / name, ART / name)
    with (ART / "case_execution_status.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "dataset", "mode", "status", "attempts", "reason"], lineterminator="\n")
        writer.writeheader()
        for result in results:
            case = result["case"]; writer.writerow({"case_id": case["case_id"], "dataset": case["dataset"],
                "mode": case["mode"], "status": result["status"], "attempts": result.get("attempts", 0), "reason": result.get("error", "")})
    checkpoint = json.loads((EXTERNAL / "phase_checkpoint.json").read_text())
    dump(ART / "runner_phase_evidence.json", {**checkpoint, "completed_cases": len(valid),
        "failed_cases": len(results) - len(valid), "receiver_replays_max_concurrent": 1, "caf_workers_max": 4,
        "attack_data_accessed": False, "temporary_iq_retained": False})
    clean = list(csv.DictReader((ART / "clean_metrics.csv").open()))
    injections = list(csv.DictReader((ART / "injection_metrics.csv").open()))
    (ART / "CURRENT_STATE.md").write_text(f"# MIRAGE Stage-0A R1 current state\n\nScientific execution complete: **{verdict}**. "
        f"Completed {len(valid)}/84 receiver-in-the-loop cases. No DS/OS attack data or neural model was used.\n")
    (ART / "README.md").write_text("# MIRAGE Stage-0A R1 full execution\n\n"
        f"Final verdict: **{verdict}**. The experiment used authenticated OAK/TEX long NAV/NCO support, actual raw-IQ "
        "nine-tap recorrelation, three complex CAF scales, clean-only calibration, 84 controlled raw-IQ injections, "
        "pinned receiver replay, all frozen controls/ablations, and relation destruction.\n\n"
        f"Clean metrics: `{json.dumps(clean, sort_keys=True)}`. Injection metrics: `{json.dumps(injections, sort_keys=True)}`. "
        f"Gate outcomes: `{json.dumps([{'gate': n, 'pass': ok} for n, ok in gates], sort_keys=True)}`.\n")
    plots = ART / "plots"; plots.mkdir(exist_ok=True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    case_rows = list(csv.DictReader(gzip.open(ART / "per_case_scores.csv.gz", "rt")))
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset in sorted({r["dataset"] for r in case_rows}):
        values = sorted(float(r["score"]) for r in case_rows if r["dataset"] == dataset)
        ax.plot(values, np.arange(1, len(values)+1)/len(values), label=dataset)
    ax.set(xlabel="MIRAGE Full score", ylabel="Empirical CDF", title="Controlled injection scores"); ax.legend(); fig.tight_layout(); fig.savefig(plots / "injection_score_ecdf.png", dpi=160); plt.close(fig)
    controls = list(csv.DictReader((ART / "control_metrics.csv").open()))
    labels = ["collapsed_source", "common_gain_scaling", "empirical_raw_iq_awgn"]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.boxplot([[float(r["injected_minus_control"]) for r in controls if r["control"] == label] for label in labels], labels=labels)
    ax.axhline(0, color="black", lw=1); ax.set(ylabel="Injected minus control score", title="Frozen control contrasts"); fig.tight_layout(); fig.savefig(plots / "control_effects.png", dpi=160); plt.close(fig)
    artifact_manifest()


def finalize() -> None:
    require_frozen(); source = specs(); thresholds = json.loads((EXTERNAL / "thresholds.json").read_text())
    results = [json.loads(p.read_text()) for p in sorted((EXTERNAL / "cases").glob("*/case_result.json"))]
    valid = [r for r in results if r["status"] == "PASS"]
    case_rows = []; control_rows = []; ablations = []; bootstrap_rows = []; injection_metrics = []
    for result in valid:
        case = result["case"]; threshold = thresholds[case["dataset"]]["q99"]; score = result["primary"]["case_score"]
        case_rows.append({"case_id": case["case_id"], "dataset": case["dataset"], "mode": case["mode"],
                          "strong": case["strong_resolvable"], "score": score, "threshold": threshold,
                          "detected": score > threshold, "rho_db": case["rho_db"], "delay_chips": case["delay_chips"],
                          "doppler_hz": case["doppler_hz"], "phase_rad": case["phase_rad"],
                          "rms": result["injected_iq"]["output_injection_region_rms"],
                          "cn0_db_hz": result["mean_replay_cn0_db_hz"],
                          "anchor_start_sample": case["anchor_start_sample"],
                          "clipping_ratio": result["injected_iq"]["clipping_ratio"]})
        for name, value in result["controls"].items():
            control_rows.append({"case_id": case["case_id"], "dataset": case["dataset"], "control": name,
                                 "score": value["case_score"], "injected_minus_control": score - value["case_score"]})
        diag = result["primary"]["diagnostics"]
        scale_scores = [float(np.median([v for p in diag.values() for v in np.asarray(p["scales"])[:, i]])) for i in range(3)]
        values = {"E0_total_energy": float(np.median([v for p in diag.values() for v in p["energy"]])),
                  "E1_magnitude_distortion": float(np.median([v for p in diag.values() for v in p["magnitude"]])),
                  "E2_complex_svd_second_energy": float(np.median([v for p in diag.values() for v in p["svd"]])),
                  "E3_20ms_complex_minor": scale_scores[0], "E4_100ms_complex_minor": scale_scores[1],
                  "E5_500ms_complex_minor": scale_scores[2],
                  "E6_magnitude_minor": float(np.median([v for p in diag.values() for v in p["magnitude"]])),
                  "E7_single_prn_node": float(max(np.median(v) for v in result["primary"]["node_sequences"].values())),
                  "Full": score}
        for name, value in values.items(): ablations.append({"case_id": case["case_id"], "dataset": case["dataset"], "ablation": name, "score": value})
    for dataset in source:
        rows = [r for r in case_rows if r["dataset"] == dataset]
        summary = {"dataset": dataset, "valid_cases": len(rows), "coverage": len(rows) / 42}
        for mode, label in (("single_prn", "single"), ("simultaneous_four_prn", "four")):
            strong = [r for r in rows if r["mode"] == mode and r["strong"]]
            summary[f"strong_{label}_count"] = len(strong); summary[f"strong_{label}_detection"] = float(np.mean([r["detected"] for r in strong])) if strong else 0
        injection_metrics.append(summary)
        for control in ("collapsed_source", "common_gain_scaling", "empirical_raw_iq_awgn"):
            diff = [r["injected_minus_control"] for r in control_rows if r["dataset"] == dataset and r["control"] == control]
            ci = paired_bootstrap(diff); bootstrap_rows.append({"dataset": dataset, "contrast": f"injected-{control}", **ci})
    relation = {"datasets": {}}
    for dataset in source:
        differences = []; permutation_errors = []
        for result in valid:
            if result["case"]["dataset"] != dataset or result["case"]["mode"] != "simultaneous_four_prn": continue
            sequences = {int(p): v for p, v in result["primary"]["node_sequences"].items()}; prns = sorted(sequences)
            evaluation_epochs = min(len(v) for v in sequences.values())
            original = [full_score([sequences[p][i] for p in prns]) for i in range(evaluation_epochs)]
            shifted = [full_score([sequences[p][(i + j) % evaluation_epochs] for j, p in enumerate(prns)]) for i in range(evaluation_epochs)]
            permuted = [full_score([sequences[p][i] for p in reversed(prns)]) for i in range(evaluation_epochs)]
            differences.append(float(np.median(original) - np.median(shifted)))
            permutation_errors.append(float(np.max(np.abs(np.asarray(original) - permuted))))
        relation["datasets"][dataset] = {"cases": len(differences), "paired_bootstrap": paired_bootstrap(differences),
                                           "per_prn_distributions_preserved": True, "minor_distributions_preserved": True,
                                           "caf_energy_preserved": True, "tracked_prn_count_preserved": True,
                                           "prn_order_permutation_max_error": max(permutation_errors, default=0)}
    clean_metrics = list(csv.DictReader((EXTERNAL / "clean_metrics.csv").open()))
    dominance = {"datasets": {}}; scale_gate = {}; invariance = {}
    for dataset in source:
        subset = [r for r in valid if r["case"]["dataset"] == dataset]
        four = [r for r in subset if r["case"]["mode"] == "simultaneous_four_prn"]
        fractions = []; prn_effect = {str(p): 0.0 for p in source[dataset]["prns"]}; anchor_effect = {}
        for result in four:
            primary = result["primary"]["node_sequences"]; authentic = result["controls"]["authentic_only"]["node_sequences"]
            effects = {p: max(float(np.median(primary[p]) - np.median(authentic[p])), 0) for p in primary}
            total = sum(effects.values()); fractions.append(max(effects.values(), default=0) / total if total else 0)
            for prn, effect in effects.items(): prn_effect[prn] += effect
            delta = max(result["primary"]["case_score"] - result["controls"]["authentic_only"]["case_score"], 0)
            anchor = str(result["case"]["anchor_start_sample"]); anchor_effect[anchor] = anchor_effect.get(anchor, 0) + delta
        total_prn = sum(prn_effect.values()); total_anchor = sum(anchor_effect.values())
        dominance["datasets"][dataset] = {
            "case_max_single_prn_positive_fraction": max(fractions, default=0),
            "aggregate_prn_positive_fraction": {p: v / total_prn if total_prn else 0 for p, v in prn_effect.items()},
            "max_anchor_positive_fraction": max(anchor_effect.values(), default=0) / total_anchor if total_anchor else 0,
            "status": "PASS" if max(fractions, default=0) <= .5 else "FAIL"}
        strong = [r for r in subset if r["case"]["strong_resolvable"]]
        full_rate = float(np.mean([r["primary"]["case_score"] > thresholds[dataset]["q99"] for r in strong])) if strong else 0
        scale_rates = []; scale_effects = []
        for index in range(3):
            injected = [float(np.median([v for d in r["primary"]["diagnostics"].values() for v in np.asarray(d["scales"])[:, index]])) for r in strong]
            authentic = [float(np.median([v for d in r["controls"]["authentic_only"]["diagnostics"].values() for v in np.asarray(d["scales"])[:, index]])) for r in strong]
            scale_rates.append(float(np.mean(np.asarray(injected) > thresholds[dataset]["q99_by_scale"][index])) if strong else 0)
            scale_effects.append(float(np.mean(np.asarray(injected) - authentic)) if strong else 0)
        scale_gate[dataset] = {"full_detection": full_rate, "scale_detection": scale_rates,
                               "full_minus_best_scale": full_rate - max(scale_rates), "scale_effects": scale_effects,
                               "positive_effect_scale_count": sum(x > 0 for x in scale_effects)}
        gain_error = max(abs(r["controls"]["common_gain_scaling"]["case_score"] - r["controls"]["authentic_only"]["case_score"]) for r in subset)
        phase_error = max(abs(r["controls"]["global_phase_rotation"]["case_score"] - r["controls"]["authentic_only"]["case_score"]) for r in subset)
        invariance[dataset] = {"gain_max_abs_score_error": gain_error, "phase_max_abs_score_error": phase_error,
                               "status": "PASS" if gain_error <= 1e-9 and phase_error <= 1e-9 else "FAIL"}
    gates = []
    gates.append(("extended_support", json.loads((ART / "extended_nav_validation.json").read_text())["status"] == "PASS"))
    gates.append(("alignment", json.loads((EXTERNAL / "caf_reconstruction_validation.json").read_text())["status"] == "PASS"))
    gates.append(("clean_fpr", all(float(r["holdout_fpr"]) <= .015 and float(r["worst_segment_fpr"]) <= .05 for r in clean_metrics)))
    gates.append(("invariance", all(v["status"] == "PASS" for v in invariance.values())))
    gates.append(("coverage", all(r["coverage"] >= .8 for r in injection_metrics)))
    gates.append(("strong_detection", all(r["strong_single_detection"] >= .75 and r["strong_four_detection"] >= .75 for r in injection_metrics)))
    gates.append(("paired_controls", all(r["lower_95"] > 0 for r in bootstrap_rows)))
    gates.append(("relation_destruction", all(v["paired_bootstrap"]["lower_95"] > 0 for v in relation["datasets"].values())))
    gates.append(("prn_dominance", all(v["case_max_single_prn_positive_fraction"] <= .5 for v in dominance["datasets"].values())))
    gates.append(("multiscale", all(v["full_minus_best_scale"] >= -.05 and v["positive_effect_scale_count"] >= 2 for v in scale_gate.values())))
    gates.append(("anchor_concentration", all(v["max_anchor_positive_fraction"] <= .5 for v in dominance["datasets"].values())))
    clipping_ok = all(r["clipping_ratio"] <= 1e-3 for r in case_rows); gates.append(("clipping", clipping_ok))
    shortcut = {dataset: {
        "abs_score_rms_spearman": abs_spearman([r["score"] for r in case_rows if r["dataset"] == dataset], [r["rms"] for r in case_rows if r["dataset"] == dataset]),
        "abs_score_cn0_spearman": abs_spearman([r["score"] for r in case_rows if r["dataset"] == dataset], [r["cn0_db_hz"] for r in case_rows if r["dataset"] == dataset])} for dataset in source}
    gates.append(("rms_cn0_shortcut", all(v["abs_score_rms_spearman"] < .3 and v["abs_score_cn0_spearman"] < .3 for v in shortcut.values())))
    dump(EXTERNAL / "prn_dominance.json", dominance); dump(EXTERNAL / "invariance_validation.json", invariance)
    dump(EXTERNAL / "multiscale_gate.json", scale_gate)
    verdict = "GO_FOR_FROZEN_STAGE0B_REAL_STATIC_EVALUATION" if all(ok for _, ok in gates) else "NO_GO_MIRAGE_PHYSICAL_HYPOTHESIS"
    write_csv(EXTERNAL / "per_case_scores.csv.gz", case_rows, compressed=True); write_csv(EXTERNAL / "control_metrics.csv", control_rows)
    write_csv(EXTERNAL / "scale_ablation.csv", ablations); write_csv(EXTERNAL / "bootstrap_intervals.csv", bootstrap_rows)
    write_csv(EXTERNAL / "injection_metrics.csv", injection_metrics)
    dump(EXTERNAL / "relation_destruction_metrics.json", relation); dump(EXTERNAL / "shortcut_audit.json", shortcut)
    dump(EXTERNAL / "final_verdict.json", {"verdict": verdict, "gates": [{"gate": n, "pass": ok} for n, ok in gates],
                                            "completed_cases": len(valid), "failed_cases": len(results) - len(valid),
                                            "attack_data_accessed": False, "neural_model_executed": False})
    update_phase("finalize", "PASS"); materialize_results(results, valid, verdict, gates); heartbeat("finalize", verdict)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["alignment", "clean", "cases", "finalize"])
    args = parser.parse_args(); {"alignment": alignment, "clean": clean, "cases": execute_cases, "finalize": finalize}[args.phase]()


if __name__ == "__main__": main()
