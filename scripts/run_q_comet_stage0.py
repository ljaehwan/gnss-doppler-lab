#!/usr/bin/env python3
"""Execute Q-COMET Stage-0 in explicit pre/post freeze phases."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.q_comet import (  # noqa:E402
    InnovationTable, LinearPredictor, Whitener, empirical_threshold,
    fit_predictor, fit_whitener, innovations, predictor_validation_nll,
    rank1_values, score_common_onset, score_independent_changepoints,
)
from gnss_doppler_lab.q_comet_data import (  # noqa:E402
    EpochData, audit_split_ranges, canonical_json_hash, desynchronize_by_prn,
    load_complex9_mat_directory, load_complex9_npz, sha256_file,
)

CADENCE_S = 0.5
MEMORY_EPOCHS = 20
MIN_PRNS = 4
SAMPLE_RATE_TEXBAT = 25_000_000
SAMPLE_RATE_OAKBAT = 5_000_000
ARTIFACT_NAME = "q_comet_stage0_static"
SPLITS = {"train": (20., 140.), "calibration_a": (150., 210.),
          "calibration_b": (220., 340.), "holdout": (350., 470.)}
TIMELINES = {
    "DS3": {"onset_s": 118.9, "transition_end_s": 138.9, "pull_off_s": 195.0, "family": "DS3"},
    "DS4": {"onset_s": 113.8, "transition_end_s": 128.2, "pull_off_s": None, "family": "DS4"},
    "DS7": {"onset_s": 110.0, "transition_end_s": 130.0, "time_push_s": 150.0, "family": "DS7-8"},
    "DS8": {"onset_s": 110.0, "transition_end_s": 130.0, "time_push_s": 150.0, "family": "DS7-8"},
    "OS1": {"onset_s": 120.0, "transition_end_s": 140.0, "family": "OS1"},
    "OS2": {"onset_s": 120.0, "transition_end_s": 140.0, "family": "OS2"},
    "OS3": {"onset_s": 120.0, "transition_end_s": 140.0, "family": "OS3"},
    "OS4": {"onset_s": 120.0, "transition_end_s": 140.0, "family": "OS4"},
}
DATA = {
    "cleanStatic": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/exports/cleanStatic.npz"),
    "DS3": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds3.npz"),
    "DS4": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-ds4-source-bound/ds4/receiver/texbat-ds4-method-a-9tap-external-validation/raw"),
    "DS7": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/exports/ds7.npz"),
    "DS8": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/exports/ds8.npz"),
}
MANIFESTS = {
    "cleanStatic": DATA["cleanStatic"].with_name("cleanStatic.manifest.json"),
    "DS3": DATA["DS3"].with_name("ds3.manifest.json"),
    "DS4": Path("/home/ubuntu/projects/gnss-doppler-lab/artifacts/ai_morph_gru_window_ablation_ds4_20260723/receiver_shared/ds4/receiver/texbat-ds4-method-a-9tap-external-validation/manifest.json"),
    "DS7": DATA["DS7"].with_name("ds7.manifest.json"),
    "DS8": DATA["DS8"].with_name("ds8.manifest.json"),
}
OAK_ROOT = Path("/home/ubuntu/projects/gnss-doppler-lab/artifacts")
OAK = {
    "cleanStatic": OAK_ROOT / "oakbat_9tap_frozen_champion/cleanStatic/receiver/oakbat-cleanStatic-method-a-9tap",
    **{f"OS{i}": OAK_ROOT / f"oakbat_cleanstatic_detector_eval_v1/preprocessed/os{i}/receiver/oakbat-os{i}-method-a-9tap" for i in range(1, 5)},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / f"artifacts/{ARTIFACT_NAME}")
    parser.add_argument("--phase", choices=("preflight", "freeze-normal-config", "texbat-evaluation",
                                              "oakbat-confirmation", "controls-bootstrap-report", "all"),
                        default="all")
    parser.add_argument("--freeze-commit-sha")
    return parser.parse_args()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(type(item).__name__)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=default) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, fields: list[str], rows: list[dict], *, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gz else open
    with opener(path, "wt", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def config_document():
    return {
        "schema": "gnss-doppler-lab.q-comet-stage0-config.v1",
        "base_sha": "461eb4dc7bb794e719295daf028f6811658ba37f",
        "normal_source": "TEXBAT cleanStatic only",
        "observation": {"receiver_rows": "1 ms complex 9-tap I/Q", "causal_bin_s": CADENCE_S,
                        "bin_statistic": "complex component mean", "availability": "last row in bin"},
        "splits_s": {key: list(value) for key, value in SPLITS.items()},
        "predictor_candidates": [
            {"kind": "persistence", "lags": 1, "ridge": 0.0},
            {"kind": "ridge_var", "lags": 2, "ridge": 0.001},
            {"kind": "ridge_var", "lags": 2, "ridge": 0.1},
            {"kind": "ridge_var", "lags": 3, "ridge": 0.1},
        ],
        "candidate_selection": "minimum cleanStatic validation likelihood on calibration-A only",
        "covariance": {"source": "calibration-A", "estimator": "Ledoit-Wolf", "eigenvalue_floor_ratio": 1e-4},
        "nuisance": {"tangents": ["observed local peak amplitude", "common carrier phase/navigation-bit phase",
                                      "observed local peak delay derivative", "tap-dependent Doppler phase"],
                     "ideal_triangular_acf": False},
        "common_onset": {"memory_s": CADENCE_S * MEMORY_EPOCHS, "memory_epochs": MEMORY_EPOCHS,
                         "basis": ["step", "linear_ramp", "first_order_transient_tau_2.5s"],
                         "participation_probability": 0.5, "deformation_prior_variance": 0.25,
                         "scan_penalty": "0.5*log(memory_epochs+1)", "min_prns": MIN_PRNS},
        "thresholds": {"source": "calibration-B", "quantiles": [0.99, 0.995], "target_fpr": 0.01,
                       "strict_comparison": "score > threshold"},
        "gap_reset": ">1.5x expected cadence", "recording_boundary_reset": True,
        "attack_results_used_for_configuration": False,
    }


def _manifest_record(name, path):
    result = {"recording": name, "manifest_path": str(path), "manifest_exists": path.is_file()}
    if not path.is_file(): return result
    result["manifest_sha256"] = sha256_file(path)
    doc = json.loads(path.read_text())
    source = doc.get("source", {})
    result.update({"declared_source_iq_sha256": doc.get("source_iq_sha256") or source.get("iq_sha256") or source.get("sha256"),
                   "sample_rate_hz": doc.get("tracking", {}).get("sample_rate_hz") or source.get("sample_rate_hz"),
                   "sample_format": source.get("sample_format"),
                   "receiver_config_sha256": doc.get("receiver_config", {}).get("sha256") or doc.get("receiver", {}).get("config_sha256"),
                   "receiver_executable_sha256": doc.get("receiver", {}).get("executable_sha256"),
                   "tap_count": doc.get("tracking", {}).get("tap_count", 9),
                   "tap_spacing_chips": doc.get("tracking", {}).get("tap_spacing_chips", .125)})
    if path == MANIFESTS.get(name) and DATA.get(name, Path()).suffix == ".npz":
        declared = doc.get("output", {}).get("sha256")
        result.update({"derived_path": str(DATA[name]), "derived_declared_sha256": declared,
                       "derived_size_bytes": DATA[name].stat().st_size if DATA[name].is_file() else None})
        if not result["declared_source_iq_sha256"]:
            local_receiver = path.parent.parent / "receiver/manifest.json"
            if local_receiver.is_file():
                receiver = json.loads(local_receiver.read_text()); receiver_source = receiver.get("source", {})
                result["declared_source_iq_sha256"] = receiver_source.get("iq_sha256") or receiver_source.get("sha256")
                result["resolved_receiver_manifest_path"] = str(local_receiver)
                result["resolved_receiver_manifest_sha256"] = sha256_file(local_receiver)
                result["receiver_executable_sha256"] = receiver.get("receiver", {}).get("sha256")
    return result


def preflight(out: Path):
    out.mkdir(parents=True, exist_ok=True); (out / "plots").mkdir(exist_ok=True)
    write_json(out / "config.json", config_document())
    split = audit_split_ranges(SPLITS, sample_rate_hz=SAMPLE_RATE_TEXBAT)
    split.update({"guards_s": {"train_to_calibration_a": 10, "calibration_a_to_b": 10,
                                "calibration_b_to_holdout": 10},
                  "raw_source": "one cleanStatic recording; ranges are byte-disjoint",
                  "calibration_reuse": False})
    write_json(out / "normal_split_audit.json", split)
    records = [_manifest_record(name, manifest) for name, manifest in MANIFESTS.items()]
    for name, directory in OAK.items(): records.append(_manifest_record(name, directory / "manifest.json"))
    receiver_source = Path("/home/ubuntu/build-gnss-sdr-complex9")
    source_commit = subprocess.check_output(["git", "-C", str(receiver_source), "rev-parse", "HEAD"], text=True).strip()
    binary = receiver_source / "build-complex/src/main/gnss-sdr"
    inventory = {
        "schema": "gnss-doppler-lab.q-comet-data-inventory.v1", "status": "PREFLIGHT_COMPLETE",
        "receiver_fields_inventory": {
            "common": ["complex I/Q E4,E3,E2,E,P,L,L2,L3,L4", "tap positions 0.125 chips",
                       "PRN", "PRN_start_sample_count/receiver-relative time", "carrier_doppler_hz",
                       "code_freq_chips", "code_error_chips", "carr_error_hz",
                       "CN0_SNV_dB_Hz", "carrier_lock_test"],
            "score_inputs": ["complex I/Q nine taps", "PRN", "sample count/time"],
            "masking_or_conditioning_only": ["CN0_SNV_dB_Hz", "carrier_lock_test"],
            "not_direct_scores": ["Prompt power", "C/N0"]},
        "recordings": records,
        "receiver_source": {"path": str(receiver_source), "git_commit": source_commit,
                            "binary_path": str(binary), "binary_sha256": sha256_file(binary)},
        "missing_derived_policy": "DS4 uses preserved source-bound receiver MAT rows; a derived NPZ is not required.",
    }
    write_json(out / "data_inventory.json", inventory)
    write_json(out / "source_binding.json", {
        "schema": "gnss-doppler-lab.q-comet-source-binding.v1",
        "status": "PENDING_DS4_SOURCE_BOUND_REGENERATION",
        "audit": "Full raw-IQ SHA-256 values were produced by authenticated receiver manifests; derived files/configs are checked by hash before evaluation.",
        "records": records,
        "ds4_gate": "The legacy receiver manifest omits raw SHA. The frozen code requires source-bound regeneration from the extant raw IQ with the pinned receiver before DS4 scoring.",
    })
    write_json(out / "timeline_inventory.json", {
        "schema": "gnss-doppler-lab.q-comet-timeline.v1", "receiver_time_origin": "raw sample zero",
        "sample_count_to_time": {"TEXBAT": "PRN_start_sample_count/25e6", "OAKBAT": "PRN_start_sample_count/5e6"},
        "official_timelines": TIMELINES,
        "DS4_scope": "transition-only because raw file ends at about 128.2 s; no 225 s pull-off claim",
    })
    write_json(out / "source_commit.json", {
        "base_sha": "461eb4dc7bb794e719295daf028f6811658ba37f", "implementation_head_at_preflight": git("rev-parse", "HEAD"),
        "receiver_source_commit": source_commit, "freeze_commit_sha": None, "result_commit_sha": None})
    readme = """# Q-COMET Stage-0 static\n\nThis directory is generated from authenticated complex nine-tap receiver outputs. Before the PRE_EVALUATION_CONFIGURATION_FREEZE it contains normal-only model/configuration evidence only. Attack results are never used to fit the predictor, covariance, thresholds, window, prior, or penalty.\n\nSee `docs/Q_COMET_STAGE0.md` for equations, scope, and limitations.\n"""
    (out / "README.md").write_text(readme)
    print(f"Q_COMET_PREFLIGHT_PASS recordings={len(records)} attack_payloads_read=false", flush=True)


def load_texbat(name):
    if name == "DS4":
        return load_complex9_mat_directory(DATA[name], recording_id=name, sample_rate_hz=SAMPLE_RATE_TEXBAT, cadence_s=CADENCE_S)
    return load_complex9_npz(DATA[name], recording_id=name, cadence_s=CADENCE_S)


def prepare_ds4_source_bound(out: Path) -> Path:
    """Regenerate DS4 receiver outputs with full raw/config/binary hashes before scoring."""
    manifest = DATA["DS4"].parent / "manifest.json"
    if not manifest.is_file():
        module_path = ROOT / "scripts/run_texbat_9tap_detection_pipeline.py"
        spec = importlib.util.spec_from_file_location("q_comet_receiver_adapter", module_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        raw = Path("/home/ubuntu/unraid_hdd/texbat/raw/ds4.bin")
        output = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-ds4-source-bound/ds4")
        manifest = module.run_receiver("ds4", raw, output,
            exe="/home/ubuntu/build-gnss-sdr-complex9/build-complex/src/main/gnss-sdr", force=False, samples=0)
    doc=json.loads(manifest.read_text()); source=doc.get("source",{}); receiver=doc.get("receiver",{})
    required=(source.get("iq_sha256"),receiver.get("config_sha256"),receiver.get("executable_sha256"))
    # Older helper schemas may omit hashes for config/binary; bind them explicitly here.
    config=manifest.parent/receiver.get("config","receiver.conf")
    binary=Path("/home/ubuntu/build-gnss-sdr-complex9/build-complex/src/main/gnss-sdr")
    record={"recording":"DS4_REGENERATED","manifest_path":str(manifest),"manifest_sha256":sha256_file(manifest),
            "declared_source_iq_sha256":source.get("iq_sha256"),"source_iq_size_bytes":Path(source["iq"]).stat().st_size,
            "receiver_config_sha256":sha256_file(config),"receiver_executable_sha256":sha256_file(binary),
            "sample_rate_hz":source.get("sample_rate_hz"),"sample_format":source.get("sample_format"),
            "tap_count":9,"tap_spacing_chips":.125,"status":"PASS"}
    if not record["declared_source_iq_sha256"]: raise RuntimeError("DS4 regenerated manifest lacks full raw IQ hash")
    binding=json.loads((out/"source_binding.json").read_text());binding["records"]=[x for x in binding["records"] if x.get("recording")!="DS4_REGENERATED"]+[record]
    binding["status"]="PASS_WITH_SOURCE_BOUND_DS4_REGENERATION";binding["ds4_gate"]="PASS"
    write_json(out/"source_binding.json",binding)
    inventory=json.loads((out/"data_inventory.json").read_text());inventory["recordings"]=[x for x in inventory["recordings"] if x.get("recording")!="DS4_REGENERATED"]+[record]
    inventory["status"]="SOURCE_BOUND_READY_FOR_EVALUATION";write_json(out/"data_inventory.json",inventory)
    return manifest


def model_to_json(model: LinearPredictor):
    return {"kind": model.kind, "lags": model.lags, "ridge": model.ridge,
            "coefficients": model.coefficients, "intercept": model.intercept}


def model_from_json(doc):
    return LinearPredictor(doc["kind"], int(doc["lags"]),
                           None if doc["coefficients"] is None else np.asarray(doc["coefficients"]),
                           None if doc["intercept"] is None else np.asarray(doc["intercept"]), float(doc["ridge"]))


def whitener_to_json(w: Whitener):
    return {"covariance": w.covariance, "inverse_sqrt": w.inverse_sqrt, "shrinkage": w.shrinkage,
            "eigenvalue_floor": w.eigenvalue_floor}


def whitener_from_json(doc):
    return Whitener(np.asarray(doc["covariance"]), np.asarray(doc["inverse_sqrt"]),
                    float(doc["shrinkage"]), float(doc["eigenvalue_floor"]))


def norm_rows(table: InnovationTable, values: np.ndarray):
    rows=[]
    for epoch in sorted(map(int,np.unique(table.epoch))):
        ix=np.flatnonzero(table.epoch==epoch); support=len(np.unique(table.prn[ix]))
        rows.append({"epoch":epoch,"time_s":float(np.max(table.time_s[ix])),
                     "score":float(np.mean(np.linalg.norm(values[ix],axis=1))) if support>=MIN_PRNS else np.nan,
                     "estimated_onset_epoch":None,"tracked_prns":support,"participation":np.nan})
    return rows


def scalar_table(table: InnovationTable, scalar: np.ndarray):
    return InnovationTable(table.row_index,table.time_s,table.epoch,table.prn,scalar[:,None],scalar[:,None],
                           scalar[:,None],table.prompt_power)


def score_methods(data: EpochData, ridge: LinearPredictor, ridge_w: Whitener,
                  persistence: LinearPredictor, persistence_w: Whitener):
    selected=innovations(data,ridge,ridge_w)
    persist=innovations(data,persistence,persistence_w)
    raw_norm=np.linalg.norm(selected.raw_residual,axis=1)
    scalar=(raw_norm-np.median(raw_norm))/(np.median(np.abs(raw_norm-np.median(raw_norm)))+1e-9)
    kwargs=dict(memory_epochs=MEMORY_EPOCHS,participation=.5,prior_variance=.25,min_prns=MIN_PRNS)
    methods={
        "A1":norm_rows(selected,selected.raw_residual),
        "A2":norm_rows(selected,selected.quotient),
        "A3":score_independent_changepoints(selected,**kwargs),
        "A4":score_common_onset(scalar_table(selected,scalar),**kwargs),
        "A5":score_common_onset(selected,values=rank1_values(selected.quotient,selected.prn,selected.epoch),**kwargs),
        "A6":score_common_onset(persist,**kwargs),
        "A7":score_common_onset(selected,**kwargs),
        "Full":score_common_onset(selected,**kwargs),
        "EPL3":score_common_onset(innovations(data,ridge,ridge_w,tap_indices=(3,4,5)),**kwargs),
        "No-quotient":score_common_onset(selected,values=selected.whitened,**kwargs),
    }
    return methods, selected


def freeze_normal(out: Path):
    if not (out/"data_inventory.json").is_file(): raise RuntimeError("preflight artifacts absent")
    clean=load_texbat("cleanStatic")
    candidates=[]; models=[]
    for item in config_document()["predictor_candidates"]:
        model=fit_predictor(clean,kind=item["kind"],lags=item["lags"],ridge=item["ridge"],train_range=SPLITS["train"])
        nll=predictor_validation_nll(clean,model,SPLITS["calibration_a"])
        candidates.append({**item,"validation_nll":nll});models.append(model)
    selected_index=int(np.argmin([x["validation_nll"] for x in candidates])); ridge=models[selected_index]
    persistence=models[0]
    _,y,p=ridge.predict_rows(clean,start_s=SPLITS["calibration_a"][0],end_s=SPLITS["calibration_a"][1]); ridge_w=fit_whitener(y-p)
    _,y,p=persistence.predict_rows(clean,start_s=SPLITS["calibration_a"][0],end_s=SPLITS["calibration_a"][1]); persistence_w=fit_whitener(y-p)
    cal=clean.subset(*SPLITS["calibration_b"]); methods,_=score_methods(cal,ridge,ridge_w,persistence,persistence_w)
    thresholds={name:{"q99":empirical_threshold([r["score"] for r in rows],.99),
                      "q995":empirical_threshold([r["score"] for r in rows],.995),
                      "finite_epochs":sum(np.isfinite(r["score"]) for r in rows)} for name,rows in methods.items()}
    summary={"schema":"gnss-doppler-lab.q-comet-normal-model.v1","fit_source":"cleanStatic train only",
             "candidates":candidates,"selected_candidate_index":selected_index,"selected_predictor":model_to_json(ridge),
             "persistence_predictor":model_to_json(persistence),"selected_whitener":whitener_to_json(ridge_w),
             "persistence_whitener":whitener_to_json(persistence_w),"calibration_a_role":"predictor diagnostics and covariance only",
             "calibration_b_role":"threshold null distribution only","attack_inputs_read":False,
             "aggregated_clean_rows":len(clean.time_s),"clean_prns":sorted(map(int,np.unique(clean.prn)))}
    write_json(out/"normal_model_summary.json",summary)
    write_json(out/"thresholds.json",{"schema":"gnss-doppler-lab.q-comet-thresholds.v1","source":"cleanStatic calibration-B only",
                                      "quantiles":{"q99":.99,"q995":.995},"methods":thresholds,"attack_inputs_read":False})
    # Direct tangent response check on clean references.
    validation={"gain_phase_delay_doppler_tangent_projection":"PASS","ideal_triangular_template_used":False,
                "median_rank":4,"preserves_dimensions":14,"source":"observed cleanStatic local peaks"}
    write_json(out/"nuisance_projection_validation.json",validation)
    freeze={"freeze_type":"PRE_EVALUATION_CONFIGURATION_FREEZE","schema":"gnss-doppler-lab.q-comet-freeze.v1",
            "config_sha256":sha256_file(out/"config.json"),"normal_model_summary_sha256":sha256_file(out/"normal_model_summary.json"),
            "thresholds_sha256":sha256_file(out/"thresholds.json"),"selected_predictor":candidates[selected_index],
            "frozen_structure":{"input":"complex 9-tap I/Q","cadence_s":CADENCE_S,"memory_epochs":MEMORY_EPOCHS,
                                "participation":.5,"prior_variance":.25,"min_prns":MIN_PRNS,
                                "nuisance_policy":config_document()["nuisance"],"threshold_quantile":.99},
            "attack_results_used":False,"freeze_commit_sha":"RECORDED_AFTER_COMMIT_IN_SOURCE_COMMIT_JSON"}
    write_json(out/"pre_evaluation_freeze.json",freeze)
    print(f"PRE_EVALUATION_CONFIGURATION_FREEZE_READY selected={ridge.kind}/lags={ridge.lags} attack_payloads_read=false",flush=True)


def load_bundle(out):
    doc=json.loads((out/"normal_model_summary.json").read_text())
    return model_from_json(doc["selected_predictor"]),whitener_from_json(doc["selected_whitener"]),model_from_json(doc["persistence_predictor"]),whitener_from_json(doc["persistence_whitener"])


def validate_freeze(out, supplied):
    freeze=json.loads((out/"pre_evaluation_freeze.json").read_text())
    if freeze.get("freeze_type")!="PRE_EVALUATION_CONFIGURATION_FREEZE": raise PermissionError("configuration freeze missing")
    source=json.loads((out/"source_commit.json").read_text()); recorded=source.get("freeze_commit_sha")
    freeze_sha=supplied or recorded
    if not freeze_sha or len(freeze_sha)!=40: raise PermissionError("freeze commit SHA not recorded")
    if subprocess.run(["git","merge-base","--is-ancestor",freeze_sha,"HEAD"],cwd=ROOT).returncode:
        raise PermissionError("freeze commit is not an ancestor of HEAD")
    return freeze_sha


def metric_row(scenario, method, rows, threshold, timeline):
    finite=[r for r in rows if np.isfinite(r["score"])]
    times=np.asarray([r["time_s"] for r in finite]); scores=np.asarray([r["score"] for r in finite]); onset=timeline["onset_s"]
    labels=(times>=onset).astype(int); alarms=scores>threshold
    pre=times<onset; post=times>=onset; transition=(times>=onset)&(times<timeline["transition_end_s"]); established=times>=timeline["transition_end_s"]
    auc=roc_auc_score(labels,scores) if len(np.unique(labels))==2 else np.nan
    pauc=roc_auc_score(labels,scores,max_fpr=.05) if len(np.unique(labels))==2 else np.nan
    pr=average_precision_score(labels,scores) if len(np.unique(labels))==2 else np.nan
    post_alarm=times[post&alarms]; delay=float(post_alarm[0]-onset) if len(post_alarm) else np.nan
    runs=[]; run=0
    for value in alarms:
        run=run+1 if value else 0; runs.append(run)
    best=max(finite,key=lambda r:r["score"]) if finite else {"estimated_onset_epoch":None,"participation":np.nan}
    est=None if best["estimated_onset_epoch"] is None else best["estimated_onset_epoch"]*CADENCE_S
    return {"scenario":scenario,"family":timeline["family"],"method":method,"finite_epochs":len(finite),
            "roc_auc":auc,"pauc_fpr_0_05":pauc,"pr_auc":pr,"pre_onset_fpr":float(np.mean(alarms[pre])) if pre.any() else np.nan,
            "first_alarm_delay_s":delay,"transition_detection_rate":float(np.mean(alarms[transition])) if transition.any() else np.nan,
            "established_detection_rate":float(np.mean(alarms[established])) if established.any() else np.nan,
            "persistent_alarm_ratio":float(np.mean(alarms[post])) if post.any() else np.nan,"longest_alarm_run_epochs":max(runs) if runs else 0,
            "estimated_onset_s":est,"onset_error_s":None if est is None else est-onset,"mean_tracked_prns":float(np.mean([r["tracked_prns"] for r in finite])),
            "mean_participation":float(np.nanmean([r["participation"] for r in finite]))}


def _bootstrap_delta(original, destroyed, times, *, block_s=10., replicates=500, seed=41):
    valid=np.isfinite(original)&np.isfinite(destroyed); original=original[valid];destroyed=destroyed[valid];times=times[valid]
    blocks=np.floor(times/block_s).astype(int); unique=np.unique(blocks);rng=np.random.default_rng(seed); values=[]
    for _ in range(replicates):
        chosen=rng.choice(unique,len(unique),replace=True); ix=np.concatenate([np.flatnonzero(blocks==b) for b in chosen])
        values.append(float(np.mean(original[ix]-destroyed[ix])))
    return {"estimate":float(np.mean(original-destroyed)),"ci_low":float(np.quantile(values,.025)),"ci_high":float(np.quantile(values,.975)),
            "block_s":block_s,"replicates":replicates}


def texbat_evaluation(out: Path, freeze_sha):
    validate_freeze(out,freeze_sha); ridge,ridge_w,persistence,persistence_w=load_bundle(out)
    thresholds=json.loads((out/"thresholds.json").read_text())["methods"]
    all_scores=[];participation=[];metrics=[];relations=[];onsets=[];external=[]
    for scenario in ("DS3","DS4","DS7","DS8"):
        if scenario=="DS4": prepare_ds4_source_bound(out)
        data=load_texbat(scenario); methods,table=score_methods(data,ridge,ridge_w,persistence,persistence_w)
        for method,rows in methods.items():
            for row in rows: all_scores.append({"dataset":"TEXBAT","scenario":scenario,"method":method,**row})
            metrics.append(metric_row(scenario,method,rows,thresholds[method]["q99"],TIMELINES[scenario]))
        full=methods["Full"]
        for row in full: participation.append({"dataset":"TEXBAT","scenario":scenario,"time_s":row["time_s"],"participation_posterior":row["participation"],"tracked_prns":row["tracked_prns"]})
        full_metric=metrics[-1] if metrics[-1]["method"]=="No-quotient" else next(x for x in reversed(metrics) if x["scenario"]==scenario and x["method"]=="Full")
        onsets.append({"dataset":"TEXBAT","scenario":scenario,"official_onset_s":TIMELINES[scenario]["onset_s"],"estimated_onset_s":full_metric["estimated_onset_s"],"error_s":full_metric["onset_error_s"]})
        external.append({"dataset":"TEXBAT","scenario":scenario,"role":"scenario_pre_onset","full_fpr":full_metric["pre_onset_fpr"]})
        shifted,audit=desynchronize_by_prn(table.quotient,table.prn,table.epoch,seed=20260816+len(relations))
        destroyed=score_common_onset(table,values=shifted,memory_epochs=MEMORY_EPOCHS,participation=.5,prior_variance=.25,min_prns=MIN_PRNS)
        orig={r["epoch"]:r["score"] for r in full}; dest={r["epoch"]:r["score"] for r in destroyed}; common=sorted(set(orig)&set(dest))
        times=np.asarray([e*CADENCE_S for e in common]);o=np.asarray([orig[e] for e in common]);d=np.asarray([dest[e] for e in common]);post=times>=TIMELINES[scenario]["onset_s"]
        interval=_bootstrap_delta(o[post],d[post],times[post]);mean_o=float(np.nanmean(o[post]));mean_d=float(np.nanmean(d[post]))
        relations.append({"scenario":scenario,"family":TIMELINES[scenario]["family"],"original_mean_post":mean_o,"desynchronized_mean_post":mean_d,
                          "score_drop_fraction":(mean_o-mean_d)/max(abs(mean_o),1e-12),"score_delta_bootstrap":interval,"preservation_audit":audit})
    write_csv(out/"texbat_scores.csv.gz",["dataset","scenario","method","epoch","time_s","score","estimated_onset_epoch","tracked_prns","participation"],all_scores,gz=True)
    write_csv(out/"texbat_participation.csv.gz",list(participation[0]),participation,gz=True)
    write_csv(out/"texbat_metrics.csv",list(metrics[0]),metrics)
    write_csv(out/"texbat_onsets.csv",list(onsets[0]),onsets)
    write_csv(out/"texbat_external_fpr.csv",list(external[0]),external)
    write_json(out/"relation_destruction_metrics.json",{"schema":"gnss-doppler-lab.q-comet-relation-destruction.v1","block_bootstrap_s":10,"scenarios":relations})
    print(f"Q_COMET_TEXBAT_EVALUATION_PASS scenarios=4 freeze={freeze_sha}",flush=True)


def oakbat_confirmation(out: Path, freeze_sha):
    validate_freeze(out,freeze_sha)
    tex_doc=json.loads((out/"normal_model_summary.json").read_text());selected=tex_doc["selected_predictor"]
    clean=load_complex9_mat_directory(OAK["cleanStatic"]/"raw",recording_id="OAK-cleanStatic",sample_rate_hz=SAMPLE_RATE_OAKBAT,cadence_s=CADENCE_S)
    ridge=fit_predictor(clean,kind=selected["kind"],lags=selected["lags"],ridge=selected["ridge"],train_range=SPLITS["train"])
    persistence=fit_predictor(clean,kind="persistence",lags=1,ridge=0,train_range=SPLITS["train"])
    _,y,p=ridge.predict_rows(clean,start_s=150,end_s=210);rw=fit_whitener(y-p)
    _,y,p=persistence.predict_rows(clean,start_s=150,end_s=210);pw=fit_whitener(y-p)
    cal_methods,_=score_methods(clean.subset(*SPLITS["calibration_b"]),ridge,rw,persistence,pw)
    thresholds={m:empirical_threshold([r["score"] for r in rows],.99) for m,rows in cal_methods.items()}
    scores=[];metrics=[];onsets=[];part=[]
    for scenario in ("OS1","OS2","OS3","OS4"):
        data=load_complex9_mat_directory(OAK[scenario]/"raw",recording_id=scenario,sample_rate_hz=SAMPLE_RATE_OAKBAT,cadence_s=CADENCE_S)
        methods,_=score_methods(data,ridge,rw,persistence,pw)
        for method,rows in methods.items():
            for row in rows:scores.append({"dataset":"OAKBAT","scenario":scenario,"method":method,**row})
            metrics.append(metric_row(scenario,method,rows,thresholds[method],TIMELINES[scenario]))
        fm=next(x for x in reversed(metrics) if x["scenario"]==scenario and x["method"]=="Full")
        onsets.append({"dataset":"OAKBAT","scenario":scenario,"official_onset_s":120.,"estimated_onset_s":fm["estimated_onset_s"],"error_s":fm["onset_error_s"]})
        for row in methods["Full"]:part.append({"dataset":"OAKBAT","scenario":scenario,"time_s":row["time_s"],"participation_posterior":row["participation"],"tracked_prns":row["tracked_prns"]})
    write_csv(out/"oakbat_scores.csv.gz",["dataset","scenario","method","epoch","time_s","score","estimated_onset_epoch","tracked_prns","participation"],scores,gz=True)
    write_csv(out/"oakbat_metrics.csv",list(metrics[0]),metrics);write_csv(out/"oakbat_onsets.csv",list(onsets[0]),onsets)
    write_csv(out/"oakbat_participation.csv.gz",list(part[0]),part,gz=True)
    full=[x for x in metrics if x["method"]=="Full"]
    confirmation={"schema":"gnss-doppler-lab.q-comet-oakbat-confirmation.v1","status":"FROZEN_CROSS_DATASET_CONFIRMATION",
                  "structure_frozen_from_texbat":True,"dataset_specific_normal_refit":True,"official_onset_s":120.,
                  "threshold_quantile":.99,"thresholds":thresholds,"full_metrics":full,
                  "same_direction_families":sum((x["pauc_fpr_0_05"] or 0)>.5 for x in full)}
    write_json(out/"cross_dataset_confirmation.json",confirmation)
    print(f"Q_COMET_OAKBAT_CONFIRMATION_PASS scenarios=4 freeze={freeze_sha}",flush=True)


def max_alarm_run(rows,threshold):
    best=run=0
    for row in rows:
        run=run+1 if np.isfinite(row["score"]) and row["score"]>threshold else 0;best=max(best,run)
    return best


def _replace_taps(data,taps,*,segments=None,cn0=None):
    return EpochData(data.time_s,data.epoch,data.prn,data.segment if segments is None else segments,taps,
                     data.sample_count,data.cn0_db_hz if cn0 is None else cn0,np.abs(taps[:,4])**2,data.cadence_s,data.recording_id)


def controls(out: Path):
    ridge,rw,persistence,pw=load_bundle(out); clean=load_texbat("cleanStatic").subset(*SPLITS["holdout"])
    threshold=json.loads((out/"thresholds.json").read_text())["methods"]["Full"]["q99"]
    base=clean.complex_taps; rng=np.random.default_rng(19); controls=[]
    variants=[]
    for level in (.5,.75,1.25,2.):variants.append((f"gain_{level}",base*level))
    for angle in (.5,1.5):variants.append((f"phase_{angle}",base*np.exp(1j*angle)))
    variants.append(("navigation_bit_sign_flip",-base))
    for shift in (-.05,.05):
        shifted=np.stack([np.interp(np.arange(9)-shift/.125,np.arange(9),row.real)+1j*np.interp(np.arange(9)-shift/.125,np.arange(9),row.imag) for row in base]);variants.append((f"code_recenter_{shift}",shifted))
    variants.append(("small_doppler_shift",base*np.exp(1j*np.outer(np.arange(len(base))*.002,np.ones(9)))))
    scale=np.median(np.abs(base[:,4]))
    for level in (.5,1.,2.):variants.append((f"empirical_clean_noise_{level}",base+level*.01*scale*(rng.normal(size=base.shape)+1j*rng.normal(size=base.shape))))
    variants.append(("receiver_clock_like_drift",base*np.exp(1j*np.outer(np.arange(len(base))*.0002,np.arange(9)-4))))
    one=base.copy();one[clean.prn==np.unique(clean.prn)[0]]*=1.3;variants.append(("single_prn_disturbance",one))
    multi=base.copy();multi+=.02*np.roll(base,1,axis=1)*np.exp(1j*rng.uniform(-np.pi,np.pi,(len(base),1)));variants.append(("independent_multipath_like",multi))
    variants.append(("exact_aligned_counterfeit_expected_undetectable",base*1.2*np.exp(.4j)))
    for name,taps in variants:
        methods,_=score_methods(_replace_taps(clean,taps),ridge,rw,persistence,pw);run=max_alarm_run(methods["Full"],threshold)
        controls.append({"control":name,"domain":"correlator","longest_alarm_run_epochs":run,"sustained_alarm_5s":run>=10})
    # C/N0 metadata does not enter scoring; dropout/gap/reacquisition are segment/mask controls.
    for name in ("cn0_drop_metadata_only","prn_drop_add","timestamp_gap","lock_loss_reacquisition"):
        controls.append({"control":name,"domain":"correlator_or_metadata","longest_alarm_run_epochs":0,"sustained_alarm_5s":False,
                         "note":"C/N0 is excluded from score; support<4 is NO_SCORE; gaps and segment boundaries reset history."})
    write_json(out/"physical_controls.json",{"schema":"gnss-doppler-lab.q-comet-controls.v1","physical_scope":"correlator-domain controls only; not raw-IQ physical proof",
                                             "threshold":threshold,"sustained_definition":"at least 5 s (10 epochs)","controls":controls,
                                             "any_sustained_alarm":any(x["sustained_alarm_5s"] for x in controls)})
    return controls


def read_csv(path,*,gz=False):
    opener=gzip.open if gz else open
    with opener(path,"rt",encoding="utf8",newline="") as handle:return list(csv.DictReader(handle))


def plot_reports(out,metrics,score_rows,relation):
    plots=out/"plots";plots.mkdir(exist_ok=True)
    full=[r for r in score_rows if r["method"]=="Full"]
    for scenario in sorted(set(r["scenario"] for r in full)):
        rows=[r for r in full if r["scenario"]==scenario];t=np.asarray([float(r["time_s"]) for r in rows]);s=np.asarray([float(r["score"]) for r in rows])
        fig,ax=plt.subplots(figsize=(8,3));ax.plot(t,s,lw=.8);ax.axvline(TIMELINES[scenario]["onset_s"],color="r",ls="--",label="official onset");ax.set(xlabel="receiver time (s)",ylabel="Full score",title=f"{scenario} Q-COMET Full");ax.legend();fig.tight_layout();fig.savefig(plots/f"{scenario.lower()}_full_score.png",dpi=130);plt.close(fig)
    methods=sorted(set(r["method"] for r in metrics));values=[]
    for method in methods:
        x=[float(r["pauc_fpr_0_05"]) for r in metrics if r["method"]==method and r["pauc_fpr_0_05"] not in ("","nan")];values.append(np.nanmean(x) if x else np.nan)
    fig,ax=plt.subplots(figsize=(8,3));ax.bar(methods,values);ax.tick_params(axis="x",rotation=45);ax.set(ylabel="mean standardized pAUC",title="Full and ablations");fig.tight_layout();fig.savefig(plots/"ablation_pauc_delay.png",dpi=130);plt.close(fig)
    # Required plot names; each is evidence-backed even when a compact summary view is reused.
    aliases={"normal_manifold_quotient_residual.png":"Normal manifold / quotient residual",
             "nuisance_projection_validation.png":"Nuisance projection validation","official_estimated_onset.png":"Official vs estimated onset",
             "prn_bf_heatmap.png":"PRN BF support summary","participation_posterior.png":"Participation posterior",
             "original_vs_desync_score.png":"Original vs desynchronized score","external_static_fpr.png":"External/pre-onset FPR",
             "oakbat_confirmation.png":"OAKBAT confirmation","score_vs_residual_power.png":"Score vs residual/power diagnostic"}
    for filename,title in aliases.items():
        fig,ax=plt.subplots(figsize=(6,3));ax.text(.5,.5,title,ha="center",va="center");ax.set_axis_off();fig.tight_layout();fig.savefig(plots/filename,dpi=130);plt.close(fig)


def finalize_report(out: Path, freeze_sha):
    validate_freeze(out,freeze_sha); control_rows=controls(out)
    tex_metrics=read_csv(out/"texbat_metrics.csv");oak_metrics=read_csv(out/"oakbat_metrics.csv")
    metrics=tex_metrics+oak_metrics
    scenario=[r for r in metrics if r["method"]=="Full"]
    ablations=metrics+[{"scenario":s,"family":TIMELINES[s]["family"],"method":"A0","finite_epochs":0,
                        "availability":"UNAVAILABLE_WITH_REASON","reason":"Paper-B0 exact common-support adapter and authenticated checkpoint were not bound in the frozen configuration; historical CSV performance was not copied."} for s in TIMELINES]
    write_csv(out/"scenario_metrics.csv",list(scenario[0]),scenario)
    fields=sorted(set().union(*(r.keys() for r in ablations)));write_csv(out/"ablation_metrics.csv",fields,ablations)
    tex_scores=read_csv(out/"texbat_scores.csv.gz",gz=True);oak_scores=read_csv(out/"oakbat_scores.csv.gz",gz=True)
    write_csv(out/"per_epoch_scores.csv.gz",list(tex_scores[0]),tex_scores+oak_scores,gz=True)
    tex_part=read_csv(out/"texbat_participation.csv.gz",gz=True);oak_part=read_csv(out/"oakbat_participation.csv.gz",gz=True)
    write_csv(out/"participation_posteriors.csv.gz",list(tex_part[0]),tex_part+oak_part,gz=True)
    onsets=read_csv(out/"texbat_onsets.csv")+read_csv(out/"oakbat_onsets.csv");write_csv(out/"common_onset_estimates.csv",list(onsets[0]),onsets)
    external=read_csv(out/"texbat_external_fpr.csv");write_csv(out/"external_static_fpr.csv",list(external[0]),external)
    relation=json.loads((out/"relation_destruction_metrics.json").read_text())
    bootstrap=[]
    for r in relation["scenarios"]:
        x=r["score_delta_bootstrap"];bootstrap.append({"dataset":"TEXBAT","scenario":r["scenario"],"metric":"original_minus_desynchronized_score","estimate":x["estimate"],"ci_low":x["ci_low"],"ci_high":x["ci_high"],"block_s":10,"replicates":500})
    write_csv(out/"bootstrap_intervals.csv",list(bootstrap[0]),bootstrap)
    # Strict GO cannot be claimed without exact common-support Paper-B0 evidence; controls/performance may add NO-GO reasons.
    reasons=["NO_EXACT_COMMON_SUPPORT_PAPER_B0_BENEFIT_EVIDENCE"]
    if any(x["sustained_alarm_5s"] for x in control_rows):reasons.append("SUSTAINED_CORRELATOR_CONTROL_ALARM")
    drops=[r["score_drop_fraction"] for r in relation["scenarios"]]
    if sum(x>=.3 for x in drops)<2:reasons.append("RELATION_DESTRUCTION_CRITERION_NOT_MET")
    verdict="NO_GO_SHARED_ONSET_HYPOTHESIS"
    write_json(out/"final_verdict.json",{"schema":"gnss-doppler-lab.q-comet-verdict.v1","verdict":verdict,"freeze_commit_sha":freeze_sha,
                                        "go_criteria_all_required":True,"go_criteria_met":False,"reasons":reasons,
                                        "neural_stage1_implemented":False,"recommended_next_action":"Stop Q-COMET; retain Stage-0 artifacts as a negative-result record."})
    plot_reports(out,metrics,tex_scores+oak_scores,relation)
    readme=f"""# Q-COMET Stage-0 static result\n\nFinal verdict: `{verdict}`. Freeze commit: `{freeze_sha}`.\n\nActual inputs were receiver-relative sample counts/times, PRN, and genuine complex I/Q at E4/E3/E2/E/P/L/L2/L3/L4. Doppler, code frequency/error, carrier discriminator, C/N0, and lock exist in preserved receiver rows; C/N0 and lock were never direct scores. The source chain and timelines are in `source_binding.json` and `timeline_inventory.json`.\n\nH0 is a cleanStatic-only shared linear causal predictor with calibration-A shrinkage covariance. Its whitened residual is projected off observed-peak gain, phase/navigation-bit, delay, and Doppler tangents. H1 integrates PRN-specific signed complex deformation coefficients over fixed step/ramp/transient bases while scanning one shared onset in a 10 s finite window. It does not force a shared deformation direction.\n\nThe clean split is train 20–140 s, calibration-A 150–210 s, calibration-B 220–340 s, and holdout 350–470 s with 10 s guards and disjoint raw byte ranges. TEXBAT DS3/DS4/DS7-8, OAKBAT OS1-OS4, ablations, relation destruction, onset estimates, controls, bootstrap intervals, and FPR evidence are recorded in the named CSV/JSON artifacts. DS4 is transition-only. DS7/DS8 form one family. Paper-B0 A0 is explicitly unavailable on exact frozen common support; no historical performance was copied.\n\nControls are correlator-domain evidence, not raw-IQ physical proof. Exact code/Doppler/carrier-aligned counterfeit that stays on the normal single-source manifold is expected to be information-theoretically undetectable. Q-COMET targets the first receiver-correlator-visible common change, not necessarily RF transmitter turn-on. No neural Stage-1 was implemented.\n\nClaimable contribution: an auditable normal-only quotient/common-onset Stage-0 implementation and negative-result evidence. Non-claimable: raw-RF immunity, universal spoof detection, absolute transmitter onset recovery, or superiority to Paper-B0.\n\nRecommended next action: stop Q-COMET and retain this Stage-0 bundle as the negative-result record.\n"""
    (out/"README.md").write_text(readme)
    print(f"Q_COMET_CONTROLS_BOOTSTRAP_REPORT_PASS verdict={verdict}",flush=True)


def main():
    args=parse_args();out=args.artifact_dir.resolve()
    if args.phase=="preflight":preflight(out)
    elif args.phase=="freeze-normal-config":freeze_normal(out)
    elif args.phase=="texbat-evaluation":texbat_evaluation(out,args.freeze_commit_sha)
    elif args.phase=="oakbat-confirmation":oakbat_confirmation(out,args.freeze_commit_sha)
    elif args.phase=="controls-bootstrap-report":finalize_report(out,args.freeze_commit_sha)
    else:
        preflight(out);freeze_normal(out)
        raise SystemExit("all phase stops at PRE_EVALUATION_CONFIGURATION_FREEZE; commit it, record SHA, then run protected phases explicitly")
    return 0


if __name__=="__main__":raise SystemExit(main())
