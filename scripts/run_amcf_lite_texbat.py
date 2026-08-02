#!/usr/bin/env python3
"""Run the post-exposure AMCF-Lite TEXBAT feasibility campaign.

Outputs are written through a sibling staging directory and renamed only after
all QA/artifacts succeed. Existing output paths are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from gnss_doppler_lab.amcf_lite import (
    FIXED_ORDERS, TAP_NAMES, MaskedSetModel, adaptive_query_order,
    aggregate_epoch_scores, assign_clean_role, binary_auc, calibrate_normal_thresholds,
    causal_decision_rows, first_sustained_alarm_delay, fit_prompt_gate, normalize_prompt, phase_masks,
    random_query_order, represent_values, score_query_path, tapwise_complex_qa,
)

CANONICAL = {
    "cleanStatic": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/exports/cleanStatic.npz"), "fcd1d378c28e79fe4a550b65fc1208cde3c8fb334db11406a07fed4d90fba237"),
    "DS1": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds1.npz"), "b24d947c83890dbfa1c801bfbcb72e1fd192dd66509e927eb5afb8118902b072"),
    "DS2": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds2.npz"), "dae0f245cbb107febd220c6de33b9a279a2bad356cb0ba772daf9418bc75d7c9"),
    "DS3": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds3.npz"), "38eb5842dfec306d99bf0c5d61df6cffcb6faa25ed63721cafa8e3c3776f9b3e"),
    "DS7": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/exports/ds7.npz"), "d0e6da4e27d51e3e96abf2ef7786501124072f28667671e4e40da756eb35f3c8"),
    "DS8": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/exports/ds8.npz"), "d1973fa150b7b4e7359df4827f36ce60289f206e9db11c1ac2bc1fd33a0df533"),
}
ONSETS = {"DS1": 100.0, "DS2": 100.0, "DS3": 100.0, "DS7": 110.0, "DS8": 110.0}
B0_DEFAULT = ROOT / "artifacts/cmte_a2_texbat_epochfix/per_epoch"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return _jsonable(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(_jsonable(document), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(""); return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows: writer.writerow({k: row.get(k, "") for k in fields})


def parse_scenarios(items: list[str] | None) -> dict[str, Path]:
    if not items: return {name: CANONICAL[name][0] for name in ("DS1", "DS2", "DS3", "DS7", "DS8")}
    result = {}
    for item in items:
        if "=" not in item: raise ValueError("--scenario requires NAME=/path")
        name, raw = item.split("=", 1); name = name.upper()
        if name not in ONSETS or name in result: raise ValueError(f"unsupported/duplicate scenario {name}")
        result[name] = Path(raw)
    return result


def load_npz(name: str, path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = path.resolve(strict=True)
    if name not in CANONICAL: raise ValueError(f"no pinned input for {name}")
    actual = sha256(path); expected = CANONICAL[name][1]
    if actual != expected: raise ValueError(f"{name} SHA-256 mismatch: {actual}")
    with np.load(path, allow_pickle=False) as source:
        required = {"complex_iq", "sample_count", "time_s", "prn", "channel", "segment_index"}
        missing = sorted(required - set(source.files))
        if missing: raise ValueError(f"{name} NPZ missing {missing}")
        data = {key: source[key].copy() for key in source.files}
    iq = data["complex_iq"]; n = len(iq)
    if iq.shape != (n, 9, 2) or not np.issubdtype(iq.dtype, np.floating):
        raise ValueError(f"{name} complex_iq must be floating [N,9,2]")
    for key in required - {"complex_iq"}:
        if data[key].shape != (n,): raise ValueError(f"{name} {key} alignment mismatch")
    if not np.isfinite(iq).all() or not np.isfinite(data["time_s"]).all():
        raise ValueError(f"{name} contains nonfinite I/Q or timestamps")
    qa = {"path": str(path), "sha256": actual, "npz_fields": sorted(data), "rows": n,
          "complex_iq_shape": list(iq.shape), "component_order": ["I", "Q"],
          "tap_order": list(TAP_NAMES), "alignment_passed": True,
          "time_range_s": [float(data["time_s"].min()), float(data["time_s"].max())],
          "prn_count": int(len(np.unique(data["prn"]))),
          "segment_count": int(len(np.unique(data["segment_index"]))),
          "channel_count": int(len(np.unique(data["channel"]))),
          "cn0_present": "cn0_db_hz" in data}
    return data, qa


def _distribution(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x): return {"q01": float("nan"), "median": float("nan"), "q99": float("nan")}
    q = np.quantile(x, [.01, .5, .99])
    return {"q01": float(q[0]), "median": float(q[1]), "q99": float(q[2])}


def prepare(name: str, data: dict[str, np.ndarray], gate, qa: dict[str, Any], smoke: bool):
    selected = causal_decision_rows(data["complex_iq"], time_s=data["time_s"], prn=data["prn"],
        channel=data["channel"], segment_index=data["segment_index"], sample_count=data["sample_count"],
        recording_id=name)
    normalized, valid = normalize_prompt(selected.complex_iq, gate)
    selected = selected.take(np.flatnonzero(valid)); normalized = normalized[valid]
    if smoke:
        if name == "cleanStatic":
            keep = []
            for role in ("train", "validation", "calibration", "clean_test"):
                idx = np.flatnonzero(np.array([assign_clean_role(t) == role for t in selected.decision_time_s]))
                keep.extend(idx[:600].tolist())
            keep = np.asarray(sorted(set(keep)), int)
        else:
            epochs = np.unique(selected.decision_time_s)
            chosen = set(epochs[epochs <= ONSETS[name] + 50][:320].tolist())
            keep = np.flatnonzero(np.array([t in chosen for t in selected.decision_time_s]))
        selected = selected.take(keep); normalized = normalized[keep]
    prompt = np.abs(data["complex_iq"][:, 4, 0] + 1j * data["complex_iq"][:, 4, 1])
    mag = np.abs(normalized)
    phase = np.angle(normalized)
    curvature = np.diff(mag, n=2, axis=1)
    unique_t, counts = np.unique(selected.decision_time_s, return_counts=True)
    qa.update({"selected_rows": len(selected), "decision_epochs": int(len(unique_t)),
               "causal_future_rows": int(np.sum(selected.time_s > selected.decision_time_s + 1e-9)),
               "duplicate_epoch_prn": int(len(selected) - len(set(zip(selected.decision_time_s, selected.prn)))),
               "prompt_magnitude": _distribution(prompt),
               "prompt_near_zero_count": int(np.sum(prompt < max(gate.min_prompt_magnitude, np.finfo(float).eps))),
               "prompt_rejected_selected_count": int(np.sum(~valid)),
               "tracked_N": _distribution(counts), "tap_magnitude": _distribution(mag),
               "tap_phase_rad": _distribution(phase), "tap_magnitude_curvature": _distribution(curvature),
               "tapwise_complex_distribution": tapwise_complex_qa(normalized)})
    # Runtime invariance QA on actual accepted rows.
    sample = selected.complex_iq[:min(32, len(selected))]
    if len(sample):
        base, vb = normalize_prompt(sample, gate)
        z = sample[..., 0] + 1j * sample[..., 1]; z *= np.exp(1j * .731)
        rotated = np.stack((z.real, z.imag), -1); rot, vr = normalize_prompt(rotated, gate)
        signed, vs = normalize_prompt(-sample, gate)
        qa["invariance"] = {"global_phase_max_abs_error": float(np.nanmax(np.abs(base-rot))),
                            "navigation_sign_max_abs_error": float(np.nanmax(np.abs(base-signed))),
                            "valid_masks_equal": bool(np.array_equal(vb, vr) and np.array_equal(vb, vs))}
    return selected, normalized, qa


def _score_rows(model: MaskedSetModel, values: np.ndarray, selected, name: str,
                order_kind: str, k: int = 9, random_seed: int | None = None) -> list[dict[str, Any]]:
    rows = []
    fixed = list(FIXED_ORDERS[k]) if order_kind == "fixed" else None
    random_order = random_query_order(k, seed=int(random_seed)) if order_kind == "random" else None
    for i, value in enumerate(values):
        if order_kind == "adaptive": order = adaptive_query_order(model, value, k)
        elif order_kind == "random": order = random_order
        else: order = fixed
        result = score_query_path(model, value, order)
        rows.append({"recording_id": str(selected.recording_id[i]), "decision_time_s": float(selected.decision_time_s[i]),
                     "prn": str(selected.prn[i]), "score": result["score"],
                     "query_order": result["query_order"]})
    epoch = aggregate_epoch_scores(rows)
    for row in epoch: row["model"] = name
    return epoch


def score_all(models, normalized, selected, random_seeds: list[int]) -> list[dict[str, Any]]:
    representations = {kind: represent_values(normalized, kind) for kind in ("complex", "magnitude", "phase")}
    specs = [
        ("magnitude all9", "magnitude", "fixed", 9, None),
        ("phase all9", "phase", "fixed", 9, None),
        ("complex K3", "complex", "fixed", 3, None),
        ("complex fixed K5", "complex", "fixed", 5, None),
        ("complex fixed K7", "complex", "fixed", 7, None),
        ("complex adaptive K5", "complex", "adaptive", 5, None),
        ("complex adaptive K7", "complex", "adaptive", 7, None),
        ("complex all9", "complex", "fixed", 9, None),
    ]
    for seed in random_seeds:
        specs.extend([(f"complex random K5 seed{seed}", "complex", "random", 5, seed),
                      (f"complex random K7 seed{seed}", "complex", "random", 7, seed)])
    output = []
    for label, rep, mode, k, seed in specs:
        output.extend(_score_rows(models[rep], representations[rep], selected, label, mode, k, seed))
    return output


def read_b0(path: Path, scenario: str) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({"recording_id": scenario, "decision_time_s": float(row["window_end_s"]),
                         "tracked_prn_count": int(float(row["tracked_prn_count"])),
                         "score": float(row["score_B0_Exact"]), "model": "B0-Exact"})
    return rows


def common_with_b0(amcf: list[dict[str, Any]], b0: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not b0: return amcf
    common = {r["decision_time_s"] for r in b0}
    common &= {r["decision_time_s"] for r in amcf}
    return [r for r in amcf if r["decision_time_s"] in common] + [r for r in b0 if r["decision_time_s"] in common]


def add_phase(rows: list[dict[str, Any]], scenario: str) -> None:
    times = np.asarray([r["decision_time_s"] for r in rows], float)
    if scenario == "cleanStatic":
        for row in rows: row["phase"] = assign_clean_role(row["decision_time_s"])
        return
    masks = phase_masks(times, ONSETS[scenario])
    for i, row in enumerate(rows):
        row.update({key: bool(mask[i]) for key, mask in masks.items()})
        row["phase"] = "persistent" if masks["persistent"][i] else "post" if masks["post"][i] else "stable_pre" if masks["stable_pre"][i] else "excluded"


def thresholds_from_clean(clean: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    models = sorted({r["model"] for r in clean})
    for model in models:
        rows = [r for r in clean if r["model"] == model and r.get("phase") == "calibration"]
        result[model] = calibrate_normal_thresholds([r["score"] for r in rows], ["calibration"]*len(rows), ["cleanStatic"]*len(rows))
    return result


def metrics(rows_by_scenario, thresholds):
    output = []
    for scenario, rows in rows_by_scenario.items():
        for model in sorted({r["model"] for r in rows}):
            subset = [r for r in rows if r["model"] == model]
            times = np.asarray([r["decision_time_s"] for r in subset]); scores = np.asarray([r["score"] for r in subset])
            for operating in ("q99", "q995"):
                threshold = thresholds[model][operating]; alarm = scores > threshold
                row = {"scenario": scenario, "model": model, "operating_point": operating,
                       "threshold": threshold, "epoch_count": len(subset), "alarm_comparison": "strict_greater"}
                if scenario == "cleanStatic":
                    test = np.array([r.get("phase") == "clean_test" for r in subset])
                    row.update({"clean_test_fpr": float(np.mean(alarm[test])) if test.any() else None,
                                "stable_pre_fpr": None, "roc_auc": None, "pr_auc": None,
                                "post_detection_rate": None, "persistent_detection_rate": None,
                                "persistent_alarm_ratio": None, "first_alarm_delay_s": None,
                                "first_sustained_alarm_delay_s": None})
                else:
                    masks = phase_masks(times, ONSETS[scenario]); hits = np.flatnonzero(alarm & masks["post"])
                    roc, pr = binary_auc(scores[masks["stable_pre"]], scores[masks["post"]])
                    sustained = first_sustained_alarm_delay(times, alarm, onset_s=ONSETS[scenario], run_length=3)
                    persistent_rate = float(np.mean(alarm[masks["persistent"]])) if masks["persistent"].any() else None
                    row.update({"clean_test_fpr": None, "roc_auc": roc, "pr_auc": pr,
                                "stable_pre_fpr": float(np.mean(alarm[masks["stable_pre"]])) if masks["stable_pre"].any() else None,
                                "post_detection_rate": float(np.mean(alarm[masks["post"]])) if masks["post"].any() else None,
                                "persistent_detection_rate": persistent_rate,
                                "persistent_alarm_ratio": persistent_rate,
                                "first_alarm_delay_s": float(times[hits[0]]-ONSETS[scenario]) if len(hits) else None,
                                "first_sustained_alarm_delay_s": sustained,
                                "pre_onset_alarm": bool(np.any(alarm & (times <= ONSETS[scenario])))})
                counts = np.asarray([r["tracked_prn_count"] for r in subset], int)
                row.update({"tracked_prn_count_median": float(np.median(counts)),
                            "tracked_prn_count_min": int(np.min(counts)),
                            "tracked_prn_count_max": int(np.max(counts))})
                output.append(row)
                for i, item in enumerate(subset): item[f"alarm_{operating}"] = bool(alarm[i])
    return output


def matched_fpr(clean: list[dict[str, Any]], thresholds, primary="complex adaptive K7") -> list[dict[str, Any]]:
    primary_rows = [r for r in clean if r["model"] == primary and r.get("phase") == "clean_test"]
    target = float(np.mean([r["score"] > thresholds[primary]["q99"] for r in primary_rows]))
    output = []
    for model in sorted(thresholds):
        values = np.asarray([r["score"] for r in clean if r["model"] == model and r.get("phase") == "clean_test"])
        candidates = np.unique(values)
        if not len(candidates): continue
        rates = np.asarray([np.mean(values > c) for c in candidates])
        index = int(np.lexsort((candidates, np.abs(rates-target)))[0])
        output.append({"model": model, "threshold": float(candidates[index]), "clean_test_fpr": float(rates[index]),
                       "target_primary_clean_test_fpr": target, "fit_source": "normal-only clean_test diagnostic",
                       "diagnostic_only": True, "attack_fit": False})
    return output


def tap_histogram(rows_by_scenario: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, int], int] = {}
    epochs: dict[tuple[str, str], int] = {}
    for scenario, rows in rows_by_scenario.items():
        for row in rows:
            model = str(row["model"])
            if model == "B0-Exact":
                continue
            epochs[(scenario, model)] = epochs.get((scenario, model), 0) + 1
            raw = row.get("selected_tap_histogram_json", "{}")
            item = json.loads(raw) if isinstance(raw, str) else dict(raw)
            for tap, count in item.items():
                key = (scenario, model, int(tap)); totals[key] = totals.get(key, 0) + int(count)
    output = []
    for (scenario, model, tap), count in sorted(totals.items()):
        output.append({"scenario": scenario, "model": model, "tap_index": tap, "tap_name": TAP_NAMES[tap],
                       "query_count": count, "epoch_count": epochs[(scenario, model)],
                       "queries_per_epoch": count / epochs[(scenario, model)]})
    return output


def make_plots(out: Path, rows_by_scenario, thresholds) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    shown = {"B0-Exact", "complex K3", "complex fixed K7", "complex adaptive K7", "complex all9", "magnitude all9", "phase all9"}
    for scenario, rows in rows_by_scenario.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in sorted(shown & {r["model"] for r in rows}):
            part = sorted((r for r in rows if r["model"] == model), key=lambda r:r["decision_time_s"])
            ax.plot([r["decision_time_s"] for r in part], [r["score"] for r in part], label=model, lw=.8)
        if scenario != "cleanStatic": ax.axvline(ONSETS[scenario], color="k", ls="--", lw=1, label="onset")
        ax.set(title=f"AMCF-Lite {scenario}", xlabel="decision time (s)", ylabel="epoch robust-median score")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(out/f"{scenario}.png", dpi=130); plt.close(fig)


def run(args) -> Path:
    out = args.out.resolve()
    if out.exists(): raise FileExistsError(f"non-overwrite output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(out.name + f".tmp-{os.getpid()}")
    if staging.exists(): raise FileExistsError(staging)
    staging.mkdir()
    try:
        scenarios = parse_scenarios(args.scenario)
        clean_data, clean_qa = load_npz("cleanStatic", args.clean)
        gate = fit_prompt_gate(clean_data["complex_iq"], clean_data["time_s"], quantile=args.prompt_quantile)
        selected, normalized, clean_qa = prepare("cleanStatic", clean_data, gate, clean_qa, args.smoke)
        roles = np.array([assign_clean_role(t) for t in selected.decision_time_s], object)
        train = roles == "train"; validation = roles == "validation"
        if train.sum() < 10 or validation.sum() < 2: raise ValueError("insufficient clean chronological train/validation rows")
        epochs = min(args.epochs, 3) if args.smoke else args.epochs
        models = {}
        for i, kind in enumerate(("complex", "magnitude", "phase")):
            values = represent_values(normalized, kind)
            model = MaskedSetModel(values.shape[-1], hidden=args.hidden, seed=args.seed+i, epochs=epochs)
            models[kind] = model.fit(values[train], values[validation])
        clean_rows = score_all(models, normalized[roles != None], selected.take(np.flatnonzero(roles != None)), args.random_seeds)
        b0_clean = read_b0(args.b0_dir/"cleanStatic_test.csv", "cleanStatic")
        clean_rows = common_with_b0(clean_rows, b0_clean); add_phase(clean_rows, "cleanStatic")
        thresholds = thresholds_from_clean(clean_rows)
        qa = {"schema": "gnss-doppler-lab.amcf-lite-qa.v1", "passed": True,
              "tap_order": list(TAP_NAMES), "component_order": ["I", "Q"],
              "prompt_gate": gate.to_dict(), "cleanStatic": clean_qa, "scenarios": {},
              "attack_fit_or_selection_or_calibration": False,
              "causal_grid": {"origin_s": 0.0, "stride_s": .5, "mapping": "first causal grid; latest row per epoch/PRN", "future_allowed": False}}
        rows_by_scenario = {"cleanStatic": clean_rows}
        for name, path in scenarios.items():
            data, item_qa = load_npz(name, path)
            item_selected, item_norm, item_qa = prepare(name, data, gate, item_qa, args.smoke)
            rows = score_all(models, item_norm, item_selected, args.random_seeds)
            rows = common_with_b0(rows, read_b0(args.b0_dir/f"{name}.csv", name)); add_phase(rows, name)
            rows_by_scenario[name] = rows; qa["scenarios"][name] = item_qa
        metric_rows = metrics(rows_by_scenario, thresholds)
        matched = matched_fpr(clean_rows, thresholds)
        matched_thresholds = {item["model"]: {"q99": item["threshold"], "q995": item["threshold"]} for item in matched}
        matched_metric_rows = [row for row in metrics(rows_by_scenario, matched_thresholds)
                               if row["scenario"] != "cleanStatic" and row["operating_point"] == "q99"]
        for row in matched_metric_rows:
            row["operating_point"] = "matched_clean_fpr_diagnostic"
            row["diagnostic_only"] = True
        write_json(staging/"qa.json", qa)
        write_json(staging/"thresholds.json", thresholds)
        write_json(staging/"model_audit.json", {k:v.fit_audit for k,v in models.items()})
        write_json(staging/"config.json", {"schema":"gnss-doppler-lab.amcf-lite-config.v1", "seed":args.seed,
            "epochs":epochs, "hidden":args.hidden, "smoke":args.smoke, "random_seeds":args.random_seeds,
            "roles":{"train":"[0,240)","validation":"[250,330)","calibration":"[340,410)","clean_test":"[420,inf)"},
            "onsets":ONSETS, "primary_threshold":"q99 normal-only calibration", "diagnostic_threshold":"q995",
            "status":"post-exposure developmental feasibility; not confirmatory"})
        write_csv(staging/"metrics.csv", metric_rows); write_csv(staging/"matched_fpr.csv", matched)
        write_csv(staging/"matched_metrics.csv", matched_metric_rows)
        write_csv(staging/"tap_selection_histogram.csv", tap_histogram(rows_by_scenario))
        for scenario, rows in rows_by_scenario.items(): write_csv(staging/"per_epoch"/f"{scenario}.csv", rows)
        make_plots(staging/"plots", rows_by_scenario, thresholds)
        readme = f"""# AMCF-Lite TEXBAT feasibility output

**Status:** developmental/post-exposure feasibility only; not confirmatory evidence.  No attack row was used for quality-gate fitting, model fitting/selection, or q99/q99.5 calibration.

- source commit: `{os.popen(f'git -C {ROOT} rev-parse HEAD').read().strip()}`
- clean fit roles: train `[0,240)`, validation `[250,330)`, calibration `[340,410)`, independent clean test `>=420`
- causal grid: first 0.5 s grid at/after each raw timestamp; latest deterministic row per epoch/PRN; no future values
- normalization: `C*conj(P)/|P|^2`, q{args.prompt_quantile:g} low-prompt gate fitted on clean train only
- seed taps: E/P/L (indices 3/4/5), scored leave-one-out; added taps scored before reveal
- primary PRN score: mean of top two queried Student-t(df=4) NLLs; epoch score: PRN-order-invariant median
- B0-Exact: reused from `artifacts/cmte_a2_texbat_epochfix/per_epoch` and q99/q99.5 recomputed on common clean calibration epochs
- matched-FPR: normal clean-test diagnostic only, explicitly non-independent
- smoke mode: `{args.smoke}`

Files: `qa.json`, `model_audit.json`, `thresholds.json`, `metrics.csv`, `matched_fpr.csv`, `per_epoch/`, `plots/`, and `hashes.json`.
"""
        (staging/"README.md").write_text(readme)
        inventory = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file(): inventory[str(path.relative_to(staging))] = sha256(path)
        write_json(staging/"hashes.json", {"algorithm":"sha256", "files":inventory})
        os.replace(staging, out)
        return out
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", type=Path, default=CANONICAL["cleanStatic"][0])
    p.add_argument("--scenario", action="append", help="repeat NAME=/canonical/path; default DS1/2/3/7/8")
    p.add_argument("--b0-dir", type=Path, default=B0_DEFAULT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=25, choices=range(1,51), metavar="1..50")
    p.add_argument("--hidden", type=int, default=32, choices=range(1,65), metavar="1..64")
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--random-seeds", type=lambda s:[int(x) for x in s.split(",")], default=[11,23,37])
    p.add_argument("--prompt-quantile", type=float, default=.005)
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = run(args); print(json.dumps({"out":str(result), "smoke":args.smoke}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
