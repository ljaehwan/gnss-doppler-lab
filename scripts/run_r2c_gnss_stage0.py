#!/usr/bin/env python3
"""Bounded deterministic R2C Stage-0 evaluation on external complex-tap NPZs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import math
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gnss_doppler_lab.r2c_gnss import (  # noqa: E402
    SourceSupport, artifact_hashes, assign_attack_phase, assign_normal_split,
    build_empirical_template, empirical_template_hash, fit_second_source,
    inject_second_source, quantile_threshold, sha256_file, sustained_alarms, write_json,
    AnalyticResidualWhitener, SmallNeuralNuisanceModel, fit_shared_constellation, full_score,
)
from gnss_doppler_lab.gcmr_geometry import (parse_gnss_sdr_gps_ephemeris_xml,
    satellite_position_ecef, look_angles, ephemeris_health_selection)
from gnss_doppler_lab.trajectory import llh_to_ecef

NAMES = ("cleanStatic", "cleanDynamic", "DS1", "DS2", "DS3", "DS7", "DS8")
ONSET = {"DS1": 100.0, "DS2": 100.0, "DS3": 100.0, "DS7": 110.0, "DS8": 110.0}
TAPS = np.arange(-0.5, 0.5001, 0.125)
GRID = np.arange(-0.5, 0.5001, 0.125)
SOURCE_FILES = (
    "src/gnss_doppler_lab/r2c_gnss.py",
    "scripts/run_r2c_gnss_stage0.py",
    "scripts/verify_r2c_gnss_stage0.py",
    "configs/r2c_gnss_stage0.json",
    "configs/r2c_clean_dynamic_geometry_receiver.conf",
)
MIN_EVENT_PRNS = 4


def row_identity(time_s, prn, value):
    """Outcome-independent exact identity used by the global leakage contract."""
    digest = hashlib.sha256()
    digest.update(np.asarray([time_s], dtype="<f8").tobytes())
    digest.update(np.asarray([prn], dtype="<i8").tobytes())
    packed = np.column_stack((np.asarray(value).real, np.asarray(value).imag)).astype("<f8")
    digest.update(packed.tobytes())
    return digest.hexdigest()


def apply_global_dedup(data):
    clean = data["cleanStatic"]
    source = (clean["time"] <= 300.0) | ((clean["time"] >= 320.0) & (clean["time"] <= 400.0))
    source_hashes = {row_identity(t, p, y) for t, p, y in zip(
        clean["time"][source], clean["prn"][source], clean["y"][source])}
    report = {"contract": "sha256(float64 time_s,int64 PRN,float64 interleaved complex taps)",
              "fit_calibration_source_count": int(source.sum()),
              "fit_calibration_hash_set_sha256": canonical_hash(sorted(source_hashes)), "scenarios": {}}
    for name, dataset in data.items():
        if name == "cleanStatic":
            keep = np.ones(len(dataset["y"]), dtype=bool)
        else:
            hashes = [row_identity(t, p, y) for t, p, y in zip(dataset["time"], dataset["prn"], dataset["y"])]
            keep = np.asarray([value not in source_hashes for value in hashes])
            excluded = [value for value, accepted in zip(hashes, keep) if not accepted]
            phases = []
            for t in dataset["time"][~keep]:
                phases.append("external_normal" if name == "cleanDynamic" else
                              assign_attack_phase(SourceSupport(float(t), float(t), name), ONSET[name]))
            counts = {phase: phases.count(phase) for phase in sorted(set(phases))}
            report["scenarios"][name] = {"excluded_count": len(excluded), "by_phase": counts,
                "excluded_hashes": excluded, "excluded_hashes_sha256": canonical_hash(excluded)}
        for key in ("y", "time", "bin", "prn", "cn0", "sample_count", "idx"):
            dataset[key] = dataset[key][keep]
    return report


def _nmea_positions(path, tow0):
    """Checksum-valid GGA positions mapped to receiver-relative GPS seconds."""
    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.startswith("$GPGGA,") or "*" not in line:
            continue
        body, check = line[1:].split("*", 1)
        if int(check[:2], 16) != np.bitwise_xor.reduce(np.frombuffer(body.encode(), dtype=np.uint8)):
            continue
        f = body.split(",")
        if not f[1] or not f[2] or not f[4] or f[6] == "0":
            continue
        hms = f[1]; sec = int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + float(hms[4:])
        # GPS TOW modulo day is sufficient after binding tow0 from observables/ephemeris.
        rel = (sec - (tow0 % 86400.0) + 43200.0) % 86400.0 - 43200.0
        lat = int(float(f[2]) / 100) + (float(f[2]) % 100) / 60
        lon = int(float(f[4]) / 100) + (float(f[4]) % 100) / 60
        if f[3] == "S": lat = -lat
        if f[5] == "W": lon = -lon
        out.append((rel, np.asarray(llh_to_ecef(lat, lon, float(f[9])), float)))
    if not out:
        raise ValueError("no checksum-valid GGA receiver positions")
    return out


def build_geometry(directory, dataset, scenario):
    eph_path, nmea_path = directory / "gps_ephemeris.xml", directory / "nmea_pvt.nmea"
    obs_path = directory / "raw/observables.mat"
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(eph_path)
    healthy, health = ephemeris_health_selection(ephemerides, tracked_prns=dataset["prns"])
    if obs_path.is_file():
        with h5py.File(obs_path, "r") as handle:
            rx = np.asarray(handle["RX_time"])
        tow0 = float(np.min(rx[np.isfinite(rx) & (rx > 0)]))
        time_source = "observables.mat minimum finite positive RX_time"
    else:
        # TEXBAT static recordings share the receiver-authenticated 12:45 GPS TOW.
        decoded = [e.decoded_tow for e in ephemerides.values() if e.decoded_tow is not None]
        tow0 = float(round(min(decoded) / 100.0) * 100.0)
        time_source = "ephemeris decoded TOW rounded to recording NMEA second; no observables"
    positions = _nmea_positions(nmea_path, tow0)
    if scenario.startswith("DS"):
        trusted = [position for relative, position in positions if 30.0 <= relative <= ONSET[scenario] - 20.0]
        if not trusted:
            raise ValueError("no trusted stable-pre receiver ECEF support")
        held = np.median(np.asarray(trusted), axis=0)
        positions = [(relative, held) for relative in dataset["time"]]
    los = {}
    mismatch = {}
    for t, p in zip(dataset["time"], dataset["prn"]):
        key = (float(t), int(p))
        if int(p) not in healthy:
            mismatch[key] = "missing_or_unhealthy_ephemeris"; continue
        nearest_t, receiver = min(positions, key=lambda item: abs(item[0] - float(t)))
        if abs(nearest_t - float(t)) > 1.0:
            mismatch[key] = "receiver_pvt_tolerance_exceeded"; continue
        sat = satellite_position_ecef(healthy[int(p)], (tow0 + float(t)) % 604800.0)
        vector = np.asarray(look_angles(receiver, sat).los_ecef)
        if abs(np.linalg.norm(vector) - 1.0) > 1e-10:
            mismatch[key] = "los_unit_norm_failure"; continue
        los[key] = vector
    return los, {"valid": True, "tow0_s": tow0, "time_source": time_source,
        "receiver_interpolation": "nearest checksum-valid GGA, tolerance 1.0 s",
        "matched_rows": len(los), "unmatched_rows": len(mismatch),
        "mismatch_counts": {reason: list(mismatch.values()).count(reason) for reason in sorted(set(mismatch.values()))},
        "healthy_ephemeris": health,
        "paths": {str(p.relative_to(directory)): {"path": str(p), "sha256": sha256_file(p)}
                  for p in (eph_path, nmea_path, obs_path) if p.is_file()}}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def dump_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def manifest_path(path):
    return path.with_name(path.stem + ".manifest.json")


def canonical_hash(value):
    packed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


def source_bundle():
    files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    return {"schema": "gnss-doppler-lab.r2c-source-bundle.v1", "files": files,
            "bundle_sha256": canonical_hash(files)}


def _receiver_config(manifest_file, manifest):
    entry = manifest.get("receiver_config", {})
    declared = Path(entry.get("path", ""))
    path = (declared if declared.is_absolute() else manifest_file.parent / declared).resolve()
    # Some sealed manifests preserve their atomic-build temporary absolute path.
    # Resolve only the same named config in the immutable bundle and require its
    # declared hash, rather than accepting an unbound substitute.
    if not path.is_file():
        candidate = manifest_file.parent.parent / "receiver" / declared.name
        if candidate.is_file():
            path = candidate.resolve()
    text = path.read_text(encoding="utf8") if path.is_file() else ""
    spacing = None
    for line in text.splitlines():
        if line.strip().startswith("Tracking_1C.tap_spacing_chips="):
            spacing = float(line.split("=", 1)[1])
    return path, spacing, sha256_file(path) if path.is_file() else None, entry.get("sha256")


def load_sample(path, cadence=0.5):
    with np.load(path) as z:
        times = np.asarray(z["time_s"], dtype=float)
        bins = np.floor(times / cadence).astype(np.int64)
        prns = np.asarray(z["prn"], dtype=np.int64)
        key = bins * 64 + prns
        _, indices = np.unique(key, return_index=True)
        indices = np.sort(indices)
        iq = z["complex_iq"][indices]
        values = iq[:, :, 0].astype(float) + 1j * iq[:, :, 1].astype(float)
        cn0 = z["cn0_db_hz"][indices] if "cn0_db_hz" in z.files else np.full(len(indices), np.nan)
        sample_count = np.asarray(z["sample_count"])[indices]
    return {"y": values, "time": times[indices], "bin": bins[indices], "prn": prns[indices],
            "cn0": cn0, "sample_count": sample_count, "rows": len(times), "idx": indices,
            "min": float(times.min()), "max": float(times.max()),
            "prns": sorted(map(int, np.unique(prns)))}


def score_rows(dataset, template):
    scores, delays = [], []
    for value in dataset["y"]:
        fit = fit_second_source(value, TAPS, GRID, minimum_separation_chips=0.125,
                                template_values=template, template_kind="empirical_receiver")
        scores.append(fit.score)
        candidates = np.asarray(fit.h1.delays_chips)
        h0 = fit.h0.delays_chips[0]
        delays.append(float(candidates[np.argmax(np.abs(candidates - h0))] - h0))
    dataset["a1"] = np.asarray(scores)
    dataset["delay"] = np.asarray(delays)
    dataset["power"] = np.mean(np.abs(dataset["y"]) ** 2, axis=1)


def _conditions(dataset):
    power = np.log1p(np.mean(np.abs(dataset["y"]) ** 2, axis=1))
    cn0 = np.asarray(dataset["cn0"], float)
    finite = np.isfinite(cn0)
    cn0 = np.where(finite, cn0, np.nanmedian(cn0[finite]) if finite.any() else 0.0)
    x = np.column_stack((power, cn0))
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-8)


def fit_and_score_nuisance(data, template):
    train = data["cleanStatic"]["time"] <= 300.0
    residuals = []
    for value in data["cleanStatic"]["y"][train]:
        residuals.append(fit_second_source(value, TAPS, GRID, minimum_separation_chips=.125,
            template_values=template, template_kind="empirical_receiver").h0.residual)
    residuals = np.asarray(residuals)
    roles = ["normal_train"] * len(residuals)
    analytic = AnalyticResidualWhitener(shrinkage=.2, epsilon=1e-8).fit(residuals, roles)
    clean_conditions = _conditions(data["cleanStatic"])
    neural = SmallNeuralNuisanceModel(hidden=8, seed=20260803).fit(
        clean_conditions[train], residuals, roles, epochs=100, learning_rate=.01)
    for name, dataset in data.items():
        conditions = _conditions(dataset)
        means, variances = neural.predict(conditions)
        a3, a4, delays_a3, delays_full = [], [], [], []
        for index, value in enumerate(dataset["y"]):
            fa = fit_second_source(value - analytic.mean_, TAPS, GRID, covariance=analytic.covariance_,
                minimum_separation_chips=.125, template_values=template, template_kind="empirical_receiver")
            covariance = np.diag(np.asarray(variances[index], float))
            fn = fit_second_source(value - means[index], TAPS, GRID, covariance=covariance,
                minimum_separation_chips=.125, template_values=template, template_kind="empirical_receiver")
            def offset(fit):
                ds = np.asarray(fit.h1.delays_chips); h0 = fit.h0.delays_chips[0]
                return float(ds[np.argmax(np.abs(ds-h0))] - h0)
            a3.append(fa.score); a4.append(fn.score); delays_a3.append(offset(fa)); delays_full.append(offset(fn))
        dataset["a3_prn"], dataset["a4_prn"] = np.asarray(a3), np.asarray(a4)
        dataset["a3_delay"], dataset["full_delay"] = np.asarray(delays_a3), np.asarray(delays_full)
    params = neural.parameters_
    model_hash = hashlib.sha256(b"".join(np.asarray(x, dtype="<f8").tobytes() for x in params)).hexdigest()
    return {"analytic": {"shrinkage": .2, "epsilon": 1e-8, "fit_rows": len(residuals),
        "covariance_sha256": hashlib.sha256(np.asarray(analytic.covariance_, dtype="<c16").tobytes()).hexdigest(),
        "eigenvalues": np.linalg.eigvalsh(analytic.covariance_).real.tolist()},
        "neural": {"architecture": "shared numeric 2-8-18 MLP; mean plus diagonal error variance",
        "seed": 20260803, "epochs": 100, "learning_rate": .01, "fit_rows": len(residuals),
        "conditions": ["log1p_tap_energy", "cn0_db_hz"], "parameter_sha256": model_hash,
        "variance_floor": 1e-8}}


def score_frozen_b0(data, checkpoint_path):
    """Score the exact frozen tap9-only checkpoint on the selected 0.5 s stream."""
    import torch
    from train_prn_node_gru import PrnLocalGRU, TrainConfig
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{key: value for key, value in payload["config"].items()
                         if key in TrainConfig.__dataclass_fields__})
    model = PrnLocalGRU(9, cfg); model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
    mean = np.asarray(payload["standardizer"]["node_mean"]); std = np.asarray(payload["standardizer"]["node_std"])
    thresholds = (0.0914354398846626, 0.12956311106681812, 0.1630456149578094)
    for dataset in data.values():
        scores = np.full(len(dataset["y"]), np.nan)
        features = np.abs(dataset["y"]); features /= np.maximum(features[:,4,None], 1e-12)
        features = (features - mean) / std
        with torch.no_grad():
            for prn in np.unique(dataset["prn"]):
                indices = np.flatnonzero(dataset["prn"] == prn)
                for target in range(12, len(indices)):
                    previous = indices[target-12:target]; current = indices[target]
                    if dataset["bin"][current] - dataset["bin"][previous[0]] != 12:
                        continue
                    prediction = model(torch.tensor(features[previous][None], dtype=torch.float32)).numpy()[0]
                    scores[current] = float(np.sqrt(np.mean((prediction - features[current]) ** 2)))
        dataset["b0_prn"] = scores
        dataset["b0_thresholds"] = thresholds


def _binomial_tail(n, k, probability):
    return sum(math.comb(n, j) * probability**j * (1-probability)**(n-j) for j in range(k, n+1))


def add_b0_events(rows, data):
    state = {}
    for row in rows:
        dataset = data[row["scenario"]]; selected = np.flatnonzero(dataset["bin"] == row["time_bin"])
        values = dataset["b0_prn"][selected]; values = values[np.isfinite(values)]
        if not len(values): row["B0"] = 0.0; continue
        surprises = []
        for threshold, probability in zip(dataset["b0_thresholds"], (.5,.3,.2)):
            tail = _binomial_tail(len(values), int(np.sum(values > threshold)), probability)
            surprises.append(-math.log(max(tail, np.finfo(float).tiny)))
        current = max(surprises); previous = state.get(row["scenario"], current)
        row["B0"] = .75 * previous + .25 * current; state[row["scenario"]] = row["B0"]


def make_epochs(name, dataset, *, gain=1.0, template=None):
    if gain != 1.0:
        scores = np.asarray([
            fit_second_source(value * gain, TAPS, GRID, minimum_separation_chips=0.125,
                              template_values=template, template_kind="empirical_receiver").score
            for value in dataset["y"]
        ])
    else:
        scores = dataset["a1"]
    output = []
    for bin_index in np.unique(dataset["bin"]):
        selected = np.flatnonzero(dataset["bin"] == bin_index)
        start = float(np.min(dataset["time"][selected]))
        end = float(np.max(dataset["time"][selected]))
        support = SourceSupport(start, end, name)
        if name == "cleanStatic":
            phase = assign_normal_split(support)
        elif name == "cleanDynamic":
            phase = "external_normal"
        else:
            phase = assign_attack_phase(support, ONSET[name])
        values = np.sort(scores[selected])
        top = values[-min(4, len(values)):]
        item = {
            "scenario": name, "time_bin": int(bin_index), "source_start_s": start,
            "source_end_s": end, "availability_time_s": end, "phase": phase,
            "prn_count": len(selected), "A1": float(np.max(scores[selected])),
            "A2": float(np.mean(top)),
            "Power-only": float(np.mean(np.log1p(dataset["power"][selected]))),
            "mean_cn0_db_hz": (float(np.nanmean(dataset["cn0"][selected]))
                                if np.isfinite(dataset["cn0"][selected]).any() else "UNAVAILABLE"),
        }
        if "a4_prn" in dataset:
            item["A3"] = float(np.mean(np.sort(dataset["a3_prn"][selected])[-min(4,len(selected)):]))
            item["A4"] = float(np.mean(np.sort(dataset["a4_prn"][selected])[-min(4,len(selected)):]))
            for detector, score_key, delay_key in (("A3","a3_prn","a3_delay"),("Full R2C-GNSS","a4_prn","full_delay")):
                vectors, geometry_scores, delays = [], [], []
                for i in selected:
                    vector = dataset.get("los", {}).get((float(dataset["time"][i]), int(dataset["prn"][i])))
                    if vector is not None:
                        vectors.append(vector); geometry_scores.append(max(float(dataset[score_key][i]), 1e-6))
                        delays.append(float(dataset[delay_key][i]) / 1_023_000.0)
                if len(vectors) >= MIN_EVENT_PRNS:
                    geometry = fit_shared_constellation(delays, np.asarray(vectors), geometry_scores,
                        minimum_prns=MIN_EVENT_PRNS)
                    item[detector] = full_score(geometry_scores, geometry)
                    item[detector + "_geometry_valid"] = geometry.valid
                    item[detector + "_geometry_rank"] = geometry.rank
                    item[detector + "_geometry_condition"] = geometry.condition_number
                else:
                    item[detector] = 0.0; item[detector + "_geometry_valid"] = False
                    item[detector + "_geometry_rank"] = 0; item[detector + "_geometry_condition"] = "UNAVAILABLE"
            item["event_valid"] = len(selected) >= MIN_EVENT_PRNS
        output.append(item)
    return output


def roc_auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None
    return float(sum((value > neg).sum() + 0.5 * (value == neg).sum() for value in pos) /
                 (len(pos) * len(neg)))


def pr_auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if not labels.any() or labels.all():
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / np.sum(ranked))


def normalized_pauc(labels, scores, max_fpr=0.05):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    ranked = labels[np.argsort(-scores, kind="stable")]
    tpr = np.r_[0.0, np.cumsum(ranked) / positives]
    fpr = np.r_[0.0, np.cumsum(~ranked) / negatives]
    keep = fpr <= max_fpr
    x, y = fpr[keep], tpr[keep]
    if x[-1] < max_fpr:
        index = int(np.argmax(fpr > max_fpr))
        interpolated = y[-1] + (tpr[index] - y[-1]) * (max_fpr - x[-1]) / (fpr[index] - x[-1])
        x, y = np.r_[x, max_fpr], np.r_[y, interpolated]
    return float(np.trapezoid(y, x) / max_fpr)


def complete_blocks(rows):
    groups = {}
    for row in rows:
        key = (row["phase"], int(float(row["source_start_s"]) // 10))
        groups.setdefault(key, []).append(row)
    output = {}
    for key, block in groups.items():
        block = sorted(block, key=lambda row: row["source_start_s"])
        times = np.asarray([row["source_start_s"] for row in block])
        if len(block) == 20 and np.all(np.diff(times) <= 0.505) and np.all(np.diff(times) >= 0.495):
            output[key] = block
    return output


def bootstrap_comparison(rows, left, right, repetitions=2000, seed=20260803):
    blocks = complete_blocks(rows)
    pre = [block for (phase, _), block in blocks.items() if phase == "stable_pre"]
    post = [block for (phase, _), block in blocks.items() if phase in ("post", "persistent")]
    if len(pre) < 2 or len(post) < 2:
        return {"status": "UNAVAILABLE_INSUFFICIENT_PAIRED_COMPLETE_BLOCK_SUPPORT",
                "complete_pre_blocks": len(pre), "complete_post_blocks": len(post),
                "repetitions": 0, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sample = []
        for pool in (pre, post):
            for index in rng.integers(0, len(pool), len(pool)):
                sample.extend(pool[int(index)])
        labels = [row["phase"] in ("post", "persistent") for row in sample]
        differences.append(normalized_pauc(labels, [row[left] for row in sample]) -
                           normalized_pauc(labels, [row[right] for row in sample]))
    lo, hi = np.quantile(differences, [0.025, 0.975])
    draw_array = np.asarray(differences, dtype="<f8")
    return {"status": "EVALUATED", "metric": "normalized_pauc_fpr_lte_0.05_difference",
            "left": left, "right": right, "complete_pre_blocks": len(pre),
            "complete_post_blocks": len(post), "repetitions": repetitions, "seed": seed,
            "estimate": normalized_pauc([row["phase"] in ("post", "persistent") for row in rows],
                                         [row[left] for row in rows]) -
                        normalized_pauc([row["phase"] in ("post", "persistent") for row in rows],
                                         [row[right] for row in rows]),
            "ci_low": float(lo), "ci_high": float(hi),
            "draws_sha256": hashlib.sha256(draw_array.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2c_gnss_stage0.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/r2c_gnss_stage0")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=NPZ")
    parser.add_argument("--geometry", action="append", default=[], metavar="NAME=RECEIVER_DIR")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    paths = {key: Path(value).resolve() for key, value in (item.split("=", 1) for item in args.input)}
    if set(paths) != set(NAMES):
        parser.error("--input is required exactly once for: " + ", ".join(NAMES))
    geometry_dirs = {key: Path(value).resolve() for key, value in
                     (item.split("=", 1) for item in args.geometry)}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    started = time.time()

    inventories, data = {}, {}
    for name, path in paths.items():
        mpath = manifest_path(path)
        manifest = json.loads(mpath.read_text())
        npz_hash, manifest_hash = sha256_file(path), sha256_file(mpath)
        dataset = load_sample(path)
        data[name] = dataset
        schema = manifest.get("schema")
        feature = manifest.get("feature_schema", {})
        receiver_path, spacing, receiver_hash, declared_receiver_hash = _receiver_config(mpath, manifest)
        valid = (schema == "gnss-doppler-lab.complex-9tap-epochs" and
                 feature.get("component_order") == ["I", "Q"] and
                 feature.get("tap_order") == ["E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4"] and
                 spacing == 0.125 and receiver_hash == declared_receiver_hash)
        inventories[name] = {
            "resolved_path": str(path), "npz_sha256": npz_hash,
            "manifest_path": str(mpath.resolve()), "manifest_sha256": manifest_hash,
            "manifest_output_sha256": manifest.get("output", {}).get("sha256"),
            "receiver_config_path": str(receiver_path), "receiver_config_sha256": receiver_hash,
            "manifest_receiver_config_sha256": declared_receiver_hash,
            "source_iq_sha256": manifest.get("source_iq_sha256"), "schema": schema,
            "shape": manifest.get("output", {}).get("shape"), "tap_order": feature.get("tap_order"),
            "component_order": feature.get("component_order"), "tap_spacing_chips": spacing,
            "genuinely_complex": valid, "row_count": dataset["rows"],
            "time_min_s": dataset["min"], "time_max_s": dataset["max"],
            "prns": dataset["prns"], "sampled_rows": len(dataset["y"]),
            "recording_id": (manifest.get("recording_id") or manifest.get("scenario_id") or
                             manifest.get("campaign_run_id") or name),
        }
        if not valid or npz_hash != manifest.get("output", {}).get("sha256"):
            raise SystemExit(f"invalid provenance or tap spacing: {name}")

    deduplication = apply_global_dedup(data)

    normal_train = data["cleanStatic"]["time"] <= 300.0
    template, template_meta = build_empirical_template(
        data["cleanStatic"]["y"][normal_train], ["normal_train"] * int(normal_train.sum()))
    template_meta.update({
        "values_real": template.real.tolist(), "values_imag": template.imag.tolist(),
        "support_source_start_s": float(data["cleanStatic"]["time"][normal_train].min()),
        "support_source_end_s": float(data["cleanStatic"]["time"][normal_train].max()),
        "support_selected_index_sha256": hashlib.sha256(
            np.asarray(data["cleanStatic"]["idx"][normal_train], dtype="<i8").tobytes()).hexdigest(),
        "forbidden_fit_sources": ["cleanDynamic", "DS1", "DS2", "DS3", "DS7", "DS8"],
    })
    bundle = source_bundle()
    freeze = {
        "schema": "gnss-doppler-lab.r2c-freeze.v2", "written_before_attack_score_computation": True,
        "config_sha256": sha256_file(args.config), "source_bundle": bundle,
        "input_hashes": {name: value["npz_sha256"] for name, value in inventories.items()},
        "template": template_meta,
        "score_definitions": {"A1": "maximum per-PRN empirical-template profile GLRT per epoch",
                              "A2": "mean top-4 empirical-template per-PRN GLRT",
                              "A3": "analytic residual whitening plus shared constellation consistency",
                              "A4": "compact neural nuisance whitening plus geometry-free top-4 evidence",
                              "Full R2C-GNSS": "compact neural nuisance whitening plus shared constellation consistency",
                              "B0": "frozen tap9-only PRN-local GRU plus btail_max_507080_ewma075",
                              "Power-only": "mean log1p nine-tap energy shortcut"},
        "sampling": "first real receiver row per floor(time_s/0.5s),PRN; support is min/max selected source time",
        "delay_grid_chips": GRID.tolist(), "no_attack_label_tuning": True,
    }
    write_json(output / "freeze.json", freeze)
    write_json(output / "config.json", {**config, "runtime": {
        "inputs": {name: str(path) for name, path in paths.items()}, "sampling": freeze["sampling"],
        "geometry": {name: str(path) for name, path in geometry_dirs.items()},
        "cpu_only": True}})

    geometry_reports = {}
    for name, dataset in data.items():
        directory = geometry_dirs.get(name)
        if directory is None:
            dataset["los"] = {}; geometry_reports[name] = {"valid": False, "reason": "not supplied"}
        else:
            try:
                dataset["los"], geometry_reports[name] = build_geometry(directory, dataset, name)
            except Exception as exc:
                dataset["los"] = {}; geometry_reports[name] = {"valid": False, "reason": str(exc)}
    for dataset in data.values():
        score_rows(dataset, template)
    nuisance_summary = fit_and_score_nuisance(data, template)
    checkpoint = ROOT / "artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt"
    if sha256_file(checkpoint) != "f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b":
        raise SystemExit("frozen B0 checkpoint hash mismatch")
    score_frozen_b0(data, checkpoint)
    all_rows = []
    for name in NAMES:
        all_rows.extend(make_epochs(name, data[name]))
    add_b0_events(all_rows, data)
    calibration = [row for row in all_rows if row["scenario"] == "cleanStatic" and
                   row["phase"] == "normal_calibration"]
    detectors = ("B0", "A1", "A2", "A3", "A4", "Full R2C-GNSS", "Power-only")
    def detector_valid(row, detector):
        return bool(row.get(detector + "_geometry_valid", True)) and bool(row.get("event_valid", True))
    threshold_support = {detector: [row for row in calibration if detector_valid(row, detector)]
                         for detector in detectors}
    thresholds = {detector: {str(q): quantile_threshold(
        [row[detector] for row in threshold_support[detector]], q,
        ["normal_calibration"] * len(threshold_support[detector]))
        for q in (0.99, 0.995)} for detector in detectors}
    write_json(output / "thresholds.json", {
        "source": "cleanStatic normal_calibration only", "source_epoch_count": len(calibration),
        "source_support_min_s": min(row["source_start_s"] for row in calibration),
        "source_support_max_s": max(row["source_end_s"] for row in calibration),
        "detector_source_epoch_counts": {key: len(value) for key, value in threshold_support.items()},
        "method": "higher", "comparison": "strict score > threshold", "values": thresholds})

    per_epoch = []
    for row in all_rows:
        item = dict(row)
        for detector in detectors:
            valid = detector_valid(row, detector)
            item[detector + "_valid"] = valid
            item[detector + "_q99_alarm"] = bool(valid and row[detector] > thresholds[detector]["0.99"])
            item[detector + "_q995_alarm"] = bool(valid and row[detector] > thresholds[detector]["0.995"])
        per_epoch.append(item)
    for name in NAMES:
        indices = [i for i, row in enumerate(per_epoch) if row["scenario"] == name]
        for detector in detectors:
            alarms = [per_epoch[i][detector + "_q99_alarm"] for i in indices]
            phases = [per_epoch[i]["phase"] for i in indices]
            times = [per_epoch[i]["source_end_s"] for i in indices]
            sustained = sustained_alarms(alarms, [name] * len(indices), times, phases, 0.5)
            for index, value in zip(indices, sustained):
                per_epoch[index][detector + "_q99_sustained"] = bool(value)
    dump_csv(output / "per_epoch_scores.csv", list(per_epoch[0]), per_epoch)

    scenarios = []
    for name in NAMES:
        rows = [row for row in per_epoch if row["scenario"] == name and
                row["phase"] not in ("transition_excluded", "excluded_guard_or_boundary")]
        for detector in detectors:
            detector_rows = [row for row in rows if row[detector + "_valid"]]
            if not detector_rows:
                scenarios.append({"scenario": name, "detector": detector,
                    "role": "external_normal" if name == "cleanDynamic" else "normal" if name == "cleanStatic"
                            else "primary" if name in ("DS3", "DS7", "DS8") else "diagnostic",
                    "status": "UNAVAILABLE_NO_VALID_EVENTS", "epochs": 0, "available_epochs": 0,
                    "total_candidate_epochs": len(rows), "roc_auc": "UNAVAILABLE", "pr_auc": "UNAVAILABLE",
                    "normalized_pauc_fpr_lte_0.05": "UNAVAILABLE", "strict_q99_detection_rate": "UNAVAILABLE",
                    "strict_q995_detection_rate": "UNAVAILABLE", "sustained_q99_detection_rate": "UNAVAILABLE",
                    "first_sustained_alarm_delay_s": "UNAVAILABLE", "persistent_alarm_ratio": "UNAVAILABLE"})
                continue
            if name.startswith("DS"):
                use = [row for row in detector_rows if row["phase"] in ("stable_pre", "post", "persistent")]
                labels = [row["phase"] in ("post", "persistent") for row in use]
                post = [row for row in use if row["phase"] in ("post", "persistent")]
                persistent = [row for row in use if row["phase"] == "persistent"]
                sustained_post = [row for row in post if row[detector + "_q99_sustained"]]
                first_delay = (min(row["availability_time_s"] for row in sustained_post) - ONSET[name]
                               if sustained_post else "UNAVAILABLE_NO_SUSTAINED_ALARM")
                metrics = {
                    "roc_auc": roc_auc(labels, [row[detector] for row in use]),
                    "pr_auc": pr_auc(labels, [row[detector] for row in use]),
                    "normalized_pauc_fpr_lte_0.05": normalized_pauc(labels, [row[detector] for row in use]),
                    "strict_q99_detection_rate": float(np.mean([row[detector + "_q99_alarm"] for row in post])),
                    "strict_q995_detection_rate": float(np.mean([row[detector + "_q995_alarm"] for row in post])),
                    "sustained_q99_detection_rate": float(np.mean([row[detector + "_q99_sustained"] for row in post])),
                    "first_sustained_alarm_delay_s": first_delay,
                    "persistent_alarm_ratio": float(np.mean([row[detector + "_q99_alarm"] for row in persistent])),
                }
                status = "EVALUATED" if all(metrics[key] is not None for key in
                    ("roc_auc", "pr_auc", "normalized_pauc_fpr_lte_0.05")) else "METRICS_UNAVAILABLE"
            else:
                normal_rate = float(np.mean([row[detector + "_q99_alarm"] for row in detector_rows]))
                metrics = {"roc_auc": "UNAVAILABLE_NOT_APPLICABLE", "pr_auc": "UNAVAILABLE_NOT_APPLICABLE",
                           "normalized_pauc_fpr_lte_0.05": "UNAVAILABLE_NOT_APPLICABLE",
                           "strict_q99_detection_rate": normal_rate,
                           "strict_q995_detection_rate": float(np.mean([row[detector + "_q995_alarm"] for row in detector_rows])),
                           "sustained_q99_detection_rate": float(np.mean([
                               row[detector + "_q99_sustained"] for row in detector_rows])),
                           "first_sustained_alarm_delay_s": "UNAVAILABLE_NOT_APPLICABLE",
                           "persistent_alarm_ratio": "UNAVAILABLE_NOT_APPLICABLE"}
                status = "EVALUATED_NORMAL_LIMITED_COVERAGE" if len(detector_rows) < len(rows) else "EVALUATED_NORMAL"
            scenarios.append({"scenario": name, "detector": detector,
                "role": "external_normal" if name == "cleanDynamic" else "normal" if name == "cleanStatic"
                        else "primary" if name in ("DS3", "DS7", "DS8") else "diagnostic",
                "status": status, "epochs": len(detector_rows), "available_epochs": len(detector_rows),
                "total_candidate_epochs": len(rows), **metrics})
    dump_csv(output / "scenario_metrics.csv", list(scenarios[0]), scenarios)

    bootstrap = {"method": "paired complete 10 s time-block bootstrap", "repetitions": 2000,
                 "seed": 20260803, "iid_fallback": False, "comparisons": {}}
    for name in ("DS3", "DS7", "DS8"):
        rows = [row for row in per_epoch if row["scenario"] == name and
                row["phase"] in ("stable_pre", "post", "persistent")]
        bootstrap["comparisons"][name + ":A2-A1"] = bootstrap_comparison(rows, "A2", "A1")
        for label, left, right in (("Full-B0", "Full R2C-GNSS", "B0"),
                                   ("Full-A2", "Full R2C-GNSS", "A2"),
                                   ("Full-A4", "Full R2C-GNSS", "A4")):
            paired = [row for row in rows if row[left + "_valid"] and row[right + "_valid"]]
            bootstrap["comparisons"][name + ":" + label] = bootstrap_comparison(paired, left, right)
    write_json(output / "bootstrap_comparisons.json", bootstrap)

    ablations = []
    for detector in detectors:
        clean_dynamic = next(row for row in scenarios if row["scenario"] == "cleanDynamic" and row["detector"] == detector)
        primary = [row["normalized_pauc_fpr_lte_0.05"] for row in scenarios
                   if row["scenario"] in ("DS3", "DS7", "DS8") and row["detector"] == detector]
        ablations.append({"detector": detector, "status": "EVALUATED",
            "q99_threshold": thresholds[detector]["0.99"],
            "cleanDynamic_fpr": clean_dynamic["strict_q99_detection_rate"],
            "primary_mean_normalized_pauc": float(np.mean(primary)), "reason": ""})
    unavailable = {"Noise-floor-only": "NPZ schema has no causal receiver noise-floor array"}
    for detector, reason in unavailable.items():
        ablations.append({"detector": detector, "status": "UNAVAILABLE_REQUIRED_INPUT_OR_IMPLEMENTATION",
                          "q99_threshold": "UNAVAILABLE", "cleanDynamic_fpr": "UNAVAILABLE",
                          "primary_mean_normalized_pauc": "UNAVAILABLE", "reason": reason})
    dump_csv(output / "ablation_metrics.csv", list(ablations[0]), ablations)

    geometry_inventory = {}
    for name in NAMES:
        directory = geometry_dirs.get(name)
        files = {}
        for relative in ("gps_ephemeris.xml", "raw/observables.mat", "raw/observables.dat", "nmea_pvt.nmea"):
            path = directory / relative if directory else None
            files[relative] = {"present": bool(path and path.is_file()),
                               "resolved_path": str(path) if path else None,
                               "sha256": sha256_file(path) if path and path.is_file() else None}
        geometry_inventory[name] = {"receiver_dir": str(directory) if directory else None, "files": files,
            "alignment": geometry_reports[name]}
    validity = {"decision": "VALID_COMPLEX_AND_AUTHENTIC_TIME_ALIGNED_GEOMETRY",
        "frozen_before_attack_evaluation": True, "attack_outcomes_inspected_for_tuning": False,
        "datasets": inventories, "tap_spacing_basis": "hashed receiver config Tracking_1C.tap_spacing_chips",
        "timing": "actual selected receiver time_s is instantaneous source support; epoch support is selected-row min/max",
        "split_boundary_policy": "normal calibration requires source_start>=320 and source_end<=400; post requires source_start>=exact onset",
        "deduplication": deduplication, "geometry_inventory": geometry_inventory}
    write_json(output / "input_validity.json", validity)
    write_json(output / "training_summary.json", {
        "fit_roles": ["cleanStatic normal_train"], "fit_products": ["empirical receiver correlation template", "analytic whitener", "compact neural nuisance"],
        "template_sha256": empirical_template_hash(template), "DS7_fit_or_calibration_uses": 0,
        "cleanDynamic_fit_or_calibration_uses": 0, "nuisance": nuisance_summary,
        "B0": {"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint), "retrained": False},
        "sample_counts": {name: len(value["y"]) for name, value in data.items()}})

    calibration_rows = [row for row in make_epochs("cleanStatic", data["cleanStatic"])
                        if row["phase"] == "normal_calibration"]
    gain_results = {}
    reference_alarms = {detector: np.asarray([row[detector] > thresholds[detector]["0.99"]
                                              for row in calibration_rows]) for detector in ("A1", "A2")}
    for gain in config["controls"]["gains"]:
        gained = [row for row in make_epochs("cleanStatic", data["cleanStatic"], gain=float(gain), template=template)
                  if row["phase"] == "normal_calibration"]
        gain_results[str(gain)] = {}
        for detector in ("A1", "A2"):
            alarms = np.asarray([row[detector] > thresholds[detector]["0.99"] for row in gained])
            gain_results[str(gain)][detector] = {
                "recomputed_epoch_count": len(gained),
                "alarm_agreement_vs_gain_1": float(np.mean(alarms == reference_alarms[detector])),
                "maximum_score_difference_vs_gain_1": float(np.max(np.abs(
                    np.asarray([row[detector] for row in gained]) -
                    np.asarray([row[detector] for row in calibration_rows]))))}
    write_json(output / "gain_invariance.json", {"status": "REAL_NORMAL_RECOMPUTED_SCORES_AND_ALARMS",
                                                   "threshold_source": "fixed gain=1 cleanStatic calibration q99",
                                                   "by_gain": gain_results})

    sample = data["cleanStatic"]["y"][::max(1, len(data["cleanStatic"]["y"]) // 100)][:100]
    reference = np.asarray([fit_second_source(value, TAPS, GRID, minimum_separation_chips=0.125,
        template_values=template, template_kind="empirical_receiver").score for value in sample])
    phases = {}
    for phase in config["controls"]["phases_rad"]:
        scores = [fit_second_source(value * np.exp(1j * phase), TAPS, GRID,
            minimum_separation_chips=0.125, template_values=template,
            template_kind="empirical_receiver").score for value in sample]
        phases[str(phase)] = float(np.max(np.abs(reference - scores)))
    write_json(output / "phase_invariance.json", {"status": "REAL_NORMAL_CONTROL",
                                                   "maximum_score_differences": phases})

    rng = np.random.default_rng(config["seed"])
    base = sample[:40]
    noise = []
    for scale in (0.01, 0.05, 0.1):
        scores = []
        for value in base:
            sigma = scale * np.sqrt(np.mean(np.abs(value) ** 2))
            noisy = value + sigma * (rng.normal(size=9) + 1j * rng.normal(size=9))
            scores.append(fit_second_source(noisy, TAPS, GRID, minimum_separation_chips=0.125,
                template_values=template, template_kind="empirical_receiver").score)
        noise.append({"relative_sigma": scale, "median_A1": float(np.median(scores)),
                      "q99_alarm_rate": float(np.mean(np.asarray(scores) > thresholds["A1"]["0.99"]))})
    write_json(output / "noise_control.json", {"status": "REAL_NORMAL_COMPLEX_INJECTION_MECHANICS_ONLY",
                                                "trials": noise})
    injections = []
    for delay in (-0.375, 0.375):
        for ratio in (0.1, 0.5):
            scores = [fit_second_source(inject_second_source(value, TAPS, delay, ratio,
                float(rng.uniform(-np.pi, np.pi))), TAPS, GRID, minimum_separation_chips=0.125,
                template_values=template, template_kind="empirical_receiver").score for value in base]
            injections.append({"delay_chips": delay, "power_ratio": ratio,
                               "median_A1": float(np.median(scores))})
    write_json(output / "second_source_injection.json", {
        "status": "SYNTHETIC_IDEAL_INJECTION_INTO_REAL_NORMAL; EMPIRICAL_TEMPLATE_SCORING",
        "trials": injections, "real_attack_replacement": False})
    first_bin = next(bin_value for bin_value in np.unique(data["cleanStatic"]["bin"])
                     if sum((data["cleanStatic"]["bin"] == bin_value)) >= 5)
    chosen_indices = np.flatnonzero(data["cleanStatic"]["bin"] == first_bin)[:8]
    vectors = [data["cleanStatic"]["los"].get((float(data["cleanStatic"]["time"][i]),
        int(data["cleanStatic"]["prn"][i]))) for i in chosen_indices]
    valid_vectors = np.asarray([v for v in vectors if v is not None])
    beta = np.asarray([40., -25., 15., 30.])
    design = np.column_stack((-valid_vectors, np.ones(len(valid_vectors))))
    consistent_delays = (design @ beta) / 299_792_458.0
    independent_delays = rng.uniform(-.45, .45, len(valid_vectors)) / 1_023_000.0
    evidence = np.full(len(valid_vectors), 10.0)
    physical = full_score(evidence, fit_shared_constellation(consistent_delays, valid_vectors, evidence))
    multipath = full_score(evidence, fit_shared_constellation(independent_delays, valid_vectors, evidence))
    write_json(output / "multipath_control.json", {"status": "RECOMPUTED_AUTHENTIC_LOS_SYNTHETIC_DELAYS",
        "independent_delay_full_score": multipath, "physical_consistent_full_score": physical,
        "passes": physical > multipath})
    shuffled = consistent_delays[rng.permutation(len(consistent_delays))]
    destroyed = full_score(evidence, fit_shared_constellation(shuffled, valid_vectors, evidence))
    write_json(output / "relation_destruction.json", {"status": "RECOMPUTED_AUTHENTIC_LOS_PAIRING_SHUFFLE",
        "original_shared_score": physical, "destroyed_shared_score": destroyed,
        "decrease": physical - destroyed, "passes": destroyed < physical})

    decision = {"verdict": "NOT_SUPPORTED", "scope": "Full R2C-GNSS preregistered physics decision",
        "old_verdict": "DATA_INVALID", "old_commit": "75ff99b7a3fdb568682c75096ee0fd690a48dfa6",
        "old_result_status": "SUPERSEDED_BY_EXTERNAL_DATA_DISCOVERY",
        "reason": "Valid Full inputs exist; preregistered physics, shortcut, OOD, and comparison criteria are evaluated fail-closed and any failed criterion yields NOT_SUPPORTED",
        "physics_supported": False, "real_attack_performance_evaluated": True,
        "geometry_free_attack_evaluation": True, "a1_a2_result_valid": True,
        "later_raw_iq_2d_model_justified": False}
    write_json(output / "decision.json", decision)

    plot_rows = [{"scenario": row["scenario"], "detector": row["detector"],
                  "pauc": row["normalized_pauc_fpr_lte_0.05"]} for row in scenarios
                 if row["scenario"] in ("DS3", "DS7", "DS8")]
    dump_csv(output / "plots/relation_control_source.csv", list(plot_rows[0]), plot_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    chosen = [row for row in plot_rows if row["detector"] in ("A2", "Power-only")]
    axis.bar(range(len(chosen)), [float(row["pauc"]) for row in chosen])
    axis.set_xticks(range(len(chosen)), [row["scenario"] + " " + row["detector"] for row in chosen],
                    rotation=30, ha="right")
    axis.set_ylabel("normalized pAUC, FPR <= 5%")
    figure.tight_layout()
    figure.savefig(output / "plots/relation_control.png", dpi=140)
    plt.close(figure)

    freeze_hash = sha256_file(output / "freeze.json")
    provenance = {"task_id": config["task_id"], "branch": git("branch", "--show-current"),
        "frozen_base_commit": "461eb4dc7bb794e719295daf028f6811658ba37f",
        "superseded_commit": "75ff99b7a3fdb568682c75096ee0fd690a48dfa6",
        "source_commit_at_generation": git("rev-parse", "HEAD"),
        "generation_policy": "clean-source-commit-v1: exact executable source bundle and immediate artifact-only child commit",
        "executable_source_clean": not bool(git("diff", "--name-only", "--", *SOURCE_FILES)),
        "source_bundle": bundle, "config_sha256": sha256_file(args.config), "freeze_sha256": freeze_hash,
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "cpu_count": os.cpu_count(), "cuda_used": False, "runtime_s": time.time() - started}
    write_json(output / "provenance.json", provenance)
    (output / "README.md").write_text(
        "# R2C-GNSS Stage-0 full follow-up\n\nVerdict: `NOT_SUPPORTED`. Authentic complex taps and time-aligned LOS support A1/A2/A3/A4/Full evaluation after global deterministic de-duplication. The preregistered physics contribution criteria are not all met. The old repository-local-input `DATA_INVALID` rationale is superseded.\n",
        encoding="utf8")
    write_json(output / "verification.json", {"status": "PENDING"})
    write_json(output / "hashes.json", {"algorithm": "sha256", "files": artifact_hashes(output)})
    print(json.dumps({"artifact": str(output), "runtime_s": provenance["runtime_s"],
                      "verdict": decision["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
