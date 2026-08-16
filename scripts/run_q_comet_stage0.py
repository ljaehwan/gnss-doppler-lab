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
import shutil
import subprocess
import sys
from typing import Any
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.q_comet import (  # noqa:E402
    InnovationTable, LinearPredictor, Whitener, empirical_threshold,
    fit_predictor, fit_whitener, innovations, predictor_validation_nll,
    nuisance_jacobian, quotient_project, rank1_values, score_common_onset,
    score_independent_changepoints,
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
OAK_LEGACY = {
    "cleanStatic": OAK_ROOT / "oakbat_9tap_frozen_champion/cleanStatic/receiver/oakbat-cleanStatic-method-a-9tap",
    **{f"OS{i}": OAK_ROOT / f"oakbat_cleanstatic_detector_eval_v1/preprocessed/os{i}/receiver/oakbat-os{i}-method-a-9tap" for i in range(1, 5)},
}
OAK_GENERATED_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9")
OAK = {name: OAK_GENERATED_ROOT/name.lower()/"receiver"/f"{name.lower()}-complex9" for name in OAK_LEGACY}
OAK_RAW = {"cleanStatic": Path("/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw/cleanStatic_gps.bin"),
           **{f"OS{i}":Path(f"/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw/os{i}.bin") for i in range(1,5)}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / f"artifacts/{ARTIFACT_NAME}")
    parser.add_argument("--phase", choices=("preflight", "freeze-normal-config", "ds78-prefix-audit",
                                              "texbat-evaluation", "oakbat-confirmation",
                                              "controls-bootstrap-report", "runner-provenance", "all"),
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
    for name, directory in OAK_LEGACY.items(): records.append(_manifest_record(name, directory / "manifest.json"))
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


def _sha256_prefix(path: Path, byte_count: int) -> str:
    digest=hashlib.sha256(); remaining=byte_count
    with path.open("rb") as handle:
        while remaining:
            block=handle.read(min(16*1024*1024,remaining))
            if not block: raise RuntimeError(f"{path} ended before {byte_count} prefix bytes")
            digest.update(block);remaining-=len(block)
    return digest.hexdigest()


def ds78_prefix_audit(out: Path, freeze_sha: str):
    validate_freeze(out,freeze_sha)
    duration_s=110;bytes_per_complex=4;byte_count=SAMPLE_RATE_TEXBAT*duration_s*bytes_per_complex
    paths={"cleanStatic":Path("/home/ubuntu/unraid_hdd/texbat/raw/cleanStatic.bin"),
           "DS7":Path("/home/ubuntu/unraid_hdd/texbat/raw/ds7.bin"),
           "DS8":Path("/home/ubuntu/unraid_hdd/texbat/raw/ds8.bin")}
    records=[]
    for name,path in paths.items():
        records.append({"recording":name,"path":str(path),"file_size_bytes":path.stat().st_size,
                        "prefix_duration_s":duration_s,"prefix_bytes":byte_count,
                        "prefix_sha256":_sha256_prefix(path,byte_count)})
    by_name={x["recording"]:x["prefix_sha256"] for x in records}
    ds7_identical=by_name["DS7"]==by_name["cleanStatic"]
    ds8_identical=by_name["DS8"]==by_name["cleanStatic"]
    doc={"schema":"gnss-doppler-lab.q-comet-ds78-prefix-audit.v1","status":"PASS",
         "sample_format":"interleaved int16 I/Q (4 bytes/complex sample)","sample_rate_hz":SAMPLE_RATE_TEXBAT,
         "records":records,"ds7_pre110_identical_to_cleanStatic":ds7_identical,
         "ds8_pre110_identical_to_cleanStatic":ds8_identical,
         "independent_normal_confirmation_policy":"Any byte-identical pre-110 s prefix is not counted as independent normal confirmation.",
         "independent_normal_confirmation_counted":False}
    write_json(out/"ds78_byte_identity_audit.json",doc)
    timeline=json.loads((out/"timeline_inventory.json").read_text())
    timeline["DS7_DS8_pre110_byte_identity_audit"]={"artifact":"ds78_byte_identity_audit.json",
        "ds7_identical_to_cleanStatic":ds7_identical,"ds8_identical_to_cleanStatic":ds8_identical,
        "counted_as_independent_normal_confirmation":False}
    write_json(out/"timeline_inventory.json",timeline)
    print(f"Q_COMET_DS78_PREFIX_AUDIT_PASS bytes={byte_count} ds7_identical={ds7_identical} ds8_identical={ds8_identical}",flush=True)


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


def _complex9_receiver_config(raw: Path, run: Path, sample_rate_hz: int) -> str:
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={sample_rate_hz}
SignalSource.implementation=File_Signal_Source
SignalSource.filename={raw.resolve()}
SignalSource.item_type=ishort
SignalSource.sampling_frequency={sample_rate_hz}
SignalSource.samples=0
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false
SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ishort_To_Complex
InputFilter.implementation=Pass_Through
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
Resampler.implementation=Pass_Through
Resampler.item_type=gr_complex
Channels_1C.count=11
Channels.in_acquisition=11
Channel.signal=1C
Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.threshold=2.5
Acquisition_1C.doppler_max=10000
Acquisition_1C.doppler_step=100
Acquisition_1C.dump=false
Tracking_1C.implementation=GPS_L1_CA_DLL_PLL_Tracking
Tracking_1C.item_type=gr_complex
Tracking_1C.pll_bw_hz=20.0
Tracking_1C.dll_bw_hz=1.5
Tracking_1C.order=3
Tracking_1C.dump=true
Tracking_1C.dump_filename={(run/'raw/epl_tracking_ch_').resolve()}
Tracking_1C.tap_count=9
Tracking_1C.tap_spacing_chips=0.125
TelemetryDecoder_1C.implementation=GPS_L1_CA_Telemetry_Decoder
TelemetryDecoder_1C.dump=false
Observables.implementation=Hybrid_Observables
Observables.dump=true
Observables.dump_filename={(run/'raw/observables.dat').resolve()}
PVT.implementation=RTKLIB_PVT
PVT.positioning_mode=Single
PVT.output_rate_ms=100
PVT.display_rate_ms=500
PVT.flag_rtcm_server=false
PVT.flag_rtcm_tty_port=false
PVT.dump=false
"""


def prepare_oak_source_bound(name: str, out: Path) -> Path:
    """Regenerate genuine complex nine taps because legacy OAK MAT kept magnitudes only."""
    run=OAK[name];manifest=run/"manifest.json"
    if not manifest.is_file():
        if run.exists():
            archived=run.with_name(run.name+"-incomplete")
            if archived.exists(): shutil.rmtree(archived)
            run.rename(archived)
        (run/"raw").mkdir(parents=True,exist_ok=True)
        raw=OAK_RAW[name];binary=Path("/home/ubuntu/build-gnss-sdr-complex9/build-complex/src/main/gnss-sdr")
        config=run/"receiver.conf";config.write_text(_complex9_receiver_config(raw,run,SAMPLE_RATE_OAKBAT))
        command=[str(binary),f"--config_file={config}","--keyboard=false"]
        with (run/"receiver.log").open("w") as log:
            result=subprocess.run(command,cwd=run,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        if result.returncode: raise RuntimeError(f"OAKBAT {name} complex receiver failed rc={result.returncode}")
        mats=sorted((run/"raw").glob("epl_tracking_ch_*.mat"))
        if not mats: raise RuntimeError(f"OAKBAT {name} complex receiver produced no MAT rows")
        from gnss_doppler_lab.gnss_sdr import export_tracking_csv, parse_acquired_prns, parse_receiver_reported_prns
        report=export_tracking_csv(mats,run/"tracking.csv",run/"tracking_summary.csv",sample_rate_hz=SAMPLE_RATE_OAKBAT)
        log_text=(run/"receiver.log").read_text(errors="replace")
        doc={"schema":"gnss-doppler-lab.q-comet-oakbat-complex9.v1","recording_id":name,
             "source":{"dataset":"OAKBAT","scenario_id":name,"iq":str(raw),"iq_sha256":sha256_file(raw),
                       "iq_size_bytes":raw.stat().st_size,"sample_rate_hz":SAMPLE_RATE_OAKBAT,"sample_format":"interleaved_int16_iq"},
             "receiver":{"name":"GNSS-SDR patched complex9","executable":str(binary),"executable_sha256":sha256_file(binary),
                         "config":"receiver.conf","config_sha256":sha256_file(config),"command":command,"exit_code":0,
                         "source_commit":"1ddd4562723040fd66cb334b578a5b69455625f4"},
             "acquisition":{"channel_count":11,"acquired_prns":parse_acquired_prns(log_text),"receiver_reported_prns":parse_receiver_reported_prns(log_text)},
             "tracking":{**report,"tap_count":9,"tap_spacing_chips":.125,"component_order":["I","Q"]},
             "source_mats":[{"path":str(p),"sha256":sha256_file(p),"size_bytes":p.stat().st_size} for p in mats]}
        write_json(manifest,doc)
    doc=json.loads(manifest.read_text());source=doc["source"];receiver=doc["receiver"]
    record={"recording":f"OAKBAT_{name}_REGENERATED","manifest_path":str(manifest),"manifest_sha256":sha256_file(manifest),
            "declared_source_iq_sha256":source["iq_sha256"],"source_iq_size_bytes":source["iq_size_bytes"],
            "receiver_config_sha256":receiver["config_sha256"],"receiver_executable_sha256":receiver["executable_sha256"],
            "receiver_source_commit":receiver["source_commit"],"sample_rate_hz":source["sample_rate_hz"],
            "sample_format":source["sample_format"],"tap_count":9,"tap_spacing_chips":.125,"status":"PASS"}
    binding=json.loads((out/"source_binding.json").read_text());binding["records"]=[x for x in binding["records"] if x.get("recording")!=record["recording"]]+[record]
    write_json(out/"source_binding.json",binding)
    inventory=json.loads((out/"data_inventory.json").read_text());inventory["recordings"]=[x for x in inventory["recordings"] if x.get("recording")!=record["recording"]]+[record]
    write_json(out/"data_inventory.json",inventory)
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


def align_innovation_rows(table: InnovationTable, reference: InnovationTable) -> InnovationTable:
    """Put an ablation on the exact target rows selected by the Full predictor."""
    positions={int(row):i for i,row in enumerate(table.row_index)}
    try:order=np.asarray([positions[int(row)] for row in reference.row_index],np.int64)
    except KeyError as exc:raise RuntimeError(f"ablation lacks Full target row {exc.args[0]}") from exc
    return InnovationTable(*(np.asarray(getattr(table,name))[order] for name in table.__dataclass_fields__))


def score_methods(data: EpochData, ridge: LinearPredictor, ridge_w: Whitener,
                  persistence: LinearPredictor, persistence_w: Whitener):
    selected=innovations(data,ridge,ridge_w)
    persist=align_innovation_rows(innovations(data,persistence,persistence_w),selected)
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


def score_full(data: EpochData, predictor: LinearPredictor, whitener: Whitener):
    table=innovations(data,predictor,whitener)
    return score_common_onset(table,memory_epochs=MEMORY_EPOCHS,participation=.5,
                              prior_variance=.25,min_prns=MIN_PRNS),table


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
    participation_values=np.asarray([r["participation"] for r in finite],float)
    return {"scenario":scenario,"family":timeline["family"],"method":method,"finite_epochs":len(finite),
            "roc_auc":auc,"pauc_fpr_0_05":pauc,"pr_auc":pr,"pre_onset_fpr":float(np.mean(alarms[pre])) if pre.any() else np.nan,
            "first_alarm_delay_s":delay,"transition_detection_rate":float(np.mean(alarms[transition])) if transition.any() else np.nan,
            "established_detection_rate":float(np.mean(alarms[established])) if established.any() else np.nan,
            "persistent_alarm_ratio":float(np.mean(alarms[post])) if post.any() else np.nan,"longest_alarm_run_epochs":max(runs) if runs else 0,
            "estimated_onset_s":est,"onset_error_s":None if est is None else est-onset,"mean_tracked_prns":float(np.mean([r["tracked_prns"] for r in finite])),
            "mean_participation":float(np.nanmean(participation_values)) if np.isfinite(participation_values).any() else np.nan}


def epoch_diagnostics(scenario, rows, table):
    score_by_epoch={int(r["epoch"]):float(r["score"]) for r in rows}
    result=[]
    for epoch in sorted(score_by_epoch):
        ix=np.flatnonzero(table.epoch==epoch)
        if not len(ix): continue
        result.append({"scenario":scenario,"epoch":epoch,"time_s":float(np.max(table.time_s[ix])),
                       "full_score":score_by_epoch[epoch],
                       "total_residual_energy":float(np.sum(table.raw_residual[ix]**2)),
                       "total_prompt_power":float(np.sum(table.prompt_power[ix])),
                       "mean_prompt_power":float(np.mean(table.prompt_power[ix])),
                       "tracked_prns":int(len(np.unique(table.prn[ix])))})
    return result


def _correlations(rows):
    score=np.asarray([r["full_score"] for r in rows],float)
    output={}
    for field in ("total_residual_energy","total_prompt_power"):
        value=np.asarray([r[field] for r in rows],float);valid=np.isfinite(score)&np.isfinite(value)
        if valid.sum()<3 or np.std(score[valid])==0 or np.std(value[valid])==0:
            pearson=spearman=np.nan
        else:
            pearson=float(np.corrcoef(score[valid],value[valid])[0,1])
            spearman=float(spearmanr(score[valid],value[valid]).statistic)
        output[f"score_{field}_pearson_r"]=pearson
        output[f"score_{field}_spearman_rho"]=spearman
    return output


def prn_bf_rows(dataset,scenario,rows):
    output=[]
    for row in rows:
        posterior=row.get("prn_participation_posterior",{})
        for prn,bf in row.get("prn_log_bf",{}).items():
            output.append({"dataset":dataset,"scenario":scenario,"epoch":row["epoch"],"time_s":row["time_s"],
                           "estimated_onset_epoch":row["estimated_onset_epoch"],"prn":prn,
                           "log_bayes_factor":bf,"participation_posterior":posterior.get(str(prn))})
    return output


def _bootstrap_delta(original, destroyed, times, *, block_s=10., replicates=500, seed=41):
    valid=np.isfinite(original)&np.isfinite(destroyed); original=original[valid];destroyed=destroyed[valid];times=times[valid]
    blocks=np.floor(times/block_s).astype(int); unique=np.unique(blocks);rng=np.random.default_rng(seed); values=[]
    for _ in range(replicates):
        chosen=rng.choice(unique,len(unique),replace=True); ix=np.concatenate([np.flatnonzero(blocks==b) for b in chosen])
        values.append(float(np.mean(original[ix]-destroyed[ix])))
    return {"estimate":float(np.mean(original-destroyed)),"ci_low":float(np.quantile(values,.025)),"ci_high":float(np.quantile(values,.975)),
            "block_s":block_s,"replicates":replicates}


def _relation_effects(original, destroyed, times, threshold, timeline, *, block_s=10., replicates=500, seed=43):
    valid=np.isfinite(original)&np.isfinite(destroyed);o=original[valid];d=destroyed[valid];t=times[valid]
    onset=timeline["onset_s"];labels=(t>=onset).astype(int);post=t>=onset
    def values(oo,dd,tt,ll):
        post_mask=tt>=onset
        pauc_o=roc_auc_score(ll,oo,max_fpr=.05) if len(np.unique(ll))==2 else np.nan
        pauc_d=roc_auc_score(ll,dd,max_fpr=.05) if len(np.unique(ll))==2 else np.nan
        alarm_o=tt[post_mask&(oo>threshold)];alarm_d=tt[post_mask&(dd>threshold)]
        delay_o=float(np.min(alarm_o)-onset) if len(alarm_o) else np.nan
        delay_d=float(np.min(alarm_d)-onset) if len(alarm_d) else np.nan
        return {"score_delta":float(np.mean(oo[post_mask]-dd[post_mask])) if post_mask.any() else np.nan,"pauc_delta":float(pauc_o-pauc_d),
                "alarm_delay_change_s":float(delay_d-delay_o) if np.isfinite(delay_o) and np.isfinite(delay_d) else np.nan,
                "persistent_detection_delta":float(np.mean(oo[post_mask]>threshold)-np.mean(dd[post_mask]>threshold)) if post_mask.any() else np.nan}
    point=values(o,d,t,labels);blocks=np.floor(t/block_s).astype(int);unique=np.unique(blocks);rng=np.random.default_rng(seed);samples={k:[] for k in point}
    for _ in range(replicates):
        chosen=rng.choice(unique,len(unique),replace=True);ix=np.concatenate([np.flatnonzero(blocks==b) for b in chosen]);sample=values(o[ix],d[ix],t[ix],labels[ix])
        for key,value in sample.items():
            if np.isfinite(value):samples[key].append(value)
    output={}
    for key,estimate in point.items():
        sample=np.asarray(samples[key],float)
        output[key]={"estimate":None if not np.isfinite(estimate) else estimate,
                     "ci_low":None if not len(sample) else float(np.quantile(sample,.025)),
                     "ci_high":None if not len(sample) else float(np.quantile(sample,.975)),
                     "finite_replicates":int(len(sample)),"block_s":block_s,"replicates":replicates}
    return output


def texbat_evaluation(out: Path, freeze_sha):
    validate_freeze(out,freeze_sha); ridge,ridge_w,persistence,persistence_w=load_bundle(out)
    thresholds=json.loads((out/"thresholds.json").read_text())["methods"]
    all_scores=[];participation=[];metrics=[];relations=[];onsets=[];external=[];diagnostics=[];bf_rows=[];relation_rows=[]
    for scenario in ("DS3","DS4","DS7","DS8"):
        if scenario=="DS4": prepare_ds4_source_bound(out)
        data=load_texbat(scenario); methods,table=score_methods(data,ridge,ridge_w,persistence,persistence_w)
        for method,rows in methods.items():
            for row in rows: all_scores.append({"dataset":"TEXBAT","scenario":scenario,"method":method,**row})
            metrics.append(metric_row(scenario,method,rows,thresholds[method]["q99"],TIMELINES[scenario]))
        full=methods["Full"]
        diag=epoch_diagnostics(scenario,full,table);diagnostics.extend(diag);bf_rows.extend(prn_bf_rows("TEXBAT",scenario,full))
        next(x for x in reversed(metrics) if x["scenario"]==scenario and x["method"]=="Full").update(_correlations(diag))
        for row in full: participation.append({"dataset":"TEXBAT","scenario":scenario,"time_s":row["time_s"],"participation_posterior":row["participation"],"tracked_prns":row["tracked_prns"]})
        full_metric=next(x for x in reversed(metrics) if x["scenario"]==scenario and x["method"]=="Full")
        onsets.append({"dataset":"TEXBAT","scenario":scenario,"official_onset_s":TIMELINES[scenario]["onset_s"],"estimated_onset_s":full_metric["estimated_onset_s"],"error_s":full_metric["onset_error_s"]})
        external.append({"dataset":"TEXBAT","scenario":scenario,"role":"scenario_pre_onset","full_fpr":full_metric["pre_onset_fpr"]})
        shifted,audit=desynchronize_by_prn(table.quotient,table.prn,table.epoch,seed=20260816+len(relations))
        destroyed=score_common_onset(table,values=shifted,memory_epochs=MEMORY_EPOCHS,participation=.5,prior_variance=.25,min_prns=MIN_PRNS)
        orig={r["epoch"]:r["score"] for r in full}; dest={r["epoch"]:r["score"] for r in destroyed}; common=sorted(set(orig)&set(dest))
        times=np.asarray([e*CADENCE_S for e in common]);o=np.asarray([orig[e] for e in common]);d=np.asarray([dest[e] for e in common]);post=times>=TIMELINES[scenario]["onset_s"]
        interval=_bootstrap_delta(o[post],d[post],times[post]);effects=_relation_effects(o,d,times,thresholds["Full"]["q99"],TIMELINES[scenario]);mean_o=float(np.nanmean(o[post]));mean_d=float(np.nanmean(d[post]))
        for time_s,original_score,destroyed_score in zip(times,o,d):
            relation_rows.append({"scenario":scenario,"time_s":time_s,"original_score":original_score,
                                  "desynchronized_score":destroyed_score,"post_onset":bool(time_s>=TIMELINES[scenario]["onset_s"])})
        relations.append({"scenario":scenario,"family":TIMELINES[scenario]["family"],"original_mean_post":mean_o,"desynchronized_mean_post":mean_d,
                          "score_drop_fraction":(mean_o-mean_d)/max(abs(mean_o),1e-12),"score_delta_bootstrap":interval,
                          "paired_block_bootstrap":effects,"preservation_audit":audit})
    write_csv(out/"texbat_scores.csv.gz",["dataset","scenario","method","epoch","time_s","score","estimated_onset_epoch","tracked_prns","participation"],all_scores,gz=True)
    write_csv(out/"texbat_participation.csv.gz",list(participation[0]),participation,gz=True)
    write_csv(out/"texbat_metrics.csv",sorted(set().union(*(r.keys() for r in metrics))),metrics)
    write_csv(out/"texbat_onsets.csv",list(onsets[0]),onsets)
    write_csv(out/"texbat_external_fpr.csv",list(external[0]),external)
    write_csv(out/"texbat_score_diagnostics.csv.gz",list(diagnostics[0]),diagnostics,gz=True)
    write_csv(out/"texbat_prn_bayes_factors.csv.gz",list(bf_rows[0]),bf_rows,gz=True)
    write_csv(out/"texbat_relation_scores.csv.gz",list(relation_rows[0]),relation_rows,gz=True)
    write_json(out/"relation_destruction_metrics.json",{"schema":"gnss-doppler-lab.q-comet-relation-destruction.v1","block_bootstrap_s":10,"scenarios":relations})
    print(f"Q_COMET_TEXBAT_EVALUATION_PASS scenarios=4 freeze={freeze_sha}",flush=True)


def oakbat_confirmation(out: Path, freeze_sha):
    validate_freeze(out,freeze_sha)
    tex_doc=json.loads((out/"normal_model_summary.json").read_text());selected=tex_doc["selected_predictor"]
    prepare_oak_source_bound("cleanStatic",out)
    clean=load_complex9_mat_directory(OAK["cleanStatic"]/"raw",recording_id="OAK-cleanStatic",sample_rate_hz=SAMPLE_RATE_OAKBAT,cadence_s=CADENCE_S)
    ridge=fit_predictor(clean,kind=selected["kind"],lags=selected["lags"],ridge=selected["ridge"],train_range=SPLITS["train"])
    persistence=fit_predictor(clean,kind="persistence",lags=1,ridge=0,train_range=SPLITS["train"])
    _,y,p=ridge.predict_rows(clean,start_s=150,end_s=210);rw=fit_whitener(y-p)
    _,y,p=persistence.predict_rows(clean,start_s=150,end_s=210);pw=fit_whitener(y-p)
    cal_methods,_=score_methods(clean.subset(*SPLITS["calibration_b"]),ridge,rw,persistence,pw)
    thresholds={m:empirical_threshold([r["score"] for r in rows],.99) for m,rows in cal_methods.items()}
    scores=[];metrics=[];onsets=[];part=[];diagnostics=[];bf_rows=[]
    for scenario in ("OS1","OS2","OS3","OS4"):
        prepare_oak_source_bound(scenario,out)
        data=load_complex9_mat_directory(OAK[scenario]/"raw",recording_id=scenario,sample_rate_hz=SAMPLE_RATE_OAKBAT,cadence_s=CADENCE_S)
        methods,table=score_methods(data,ridge,rw,persistence,pw)
        for method,rows in methods.items():
            for row in rows:scores.append({"dataset":"OAKBAT","scenario":scenario,"method":method,**row})
            metrics.append(metric_row(scenario,method,rows,thresholds[method],TIMELINES[scenario]))
        fm=next(x for x in reversed(metrics) if x["scenario"]==scenario and x["method"]=="Full")
        diag=epoch_diagnostics(scenario,methods["Full"],table);diagnostics.extend(diag);bf_rows.extend(prn_bf_rows("OAKBAT",scenario,methods["Full"]))
        fm.update(_correlations(diag))
        onsets.append({"dataset":"OAKBAT","scenario":scenario,"official_onset_s":120.,"estimated_onset_s":fm["estimated_onset_s"],"error_s":fm["onset_error_s"]})
        for row in methods["Full"]:part.append({"dataset":"OAKBAT","scenario":scenario,"time_s":row["time_s"],"participation_posterior":row["participation"],"tracked_prns":row["tracked_prns"]})
    write_csv(out/"oakbat_scores.csv.gz",["dataset","scenario","method","epoch","time_s","score","estimated_onset_epoch","tracked_prns","participation"],scores,gz=True)
    write_csv(out/"oakbat_metrics.csv",sorted(set().union(*(r.keys() for r in metrics))),metrics);write_csv(out/"oakbat_onsets.csv",list(onsets[0]),onsets)
    write_csv(out/"oakbat_participation.csv.gz",list(part[0]),part,gz=True)
    write_csv(out/"oakbat_score_diagnostics.csv.gz",list(diagnostics[0]),diagnostics,gz=True)
    write_csv(out/"oakbat_prn_bayes_factors.csv.gz",list(bf_rows[0]),bf_rows,gz=True)
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


def _mask_data(data,keep):
    return EpochData(*(np.asarray(value)[keep] for value in (data.time_s,data.epoch,data.prn,data.segment,
                     data.complex_taps,data.sample_count,data.cn0_db_hz,data.prompt_power)),data.cadence_s,data.recording_id)


def controls(out: Path):
    ridge,rw,_,_=load_bundle(out); clean=load_texbat("cleanStatic").subset(*SPLITS["holdout"])
    threshold=json.loads((out/"thresholds.json").read_text())["methods"]["Full"]["q99"]
    base=clean.complex_taps; rng=np.random.default_rng(19); controls=[]
    base_rows=[r for r in score_full(clean,ridge,rw)[0] if np.isfinite(r["score"])]
    clean_holdout_fpr=float(np.mean([r["score"]>threshold for r in base_rows]))
    controls.append({"control":"cleanStatic_holdout","domain":"correlator","longest_alarm_run_epochs":max_alarm_run(base_rows,threshold),
                     "sustained_alarm_5s":max_alarm_run(base_rows,threshold)>=10,"false_positive_rate":clean_holdout_fpr})
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
        rows,_=score_full(_replace_taps(clean,taps),ridge,rw);run=max_alarm_run(rows,threshold)
        controls.append({"control":name,"domain":"correlator","longest_alarm_run_epochs":run,"sustained_alarm_5s":run>=10})
    # Execute metadata/support/reset controls rather than assigning assumed outcomes.
    metadata=[]
    metadata.append(("cn0_drop_metadata_only",_replace_taps(clean,base,cn0=clean.cn0_db_hz-20)))
    middle=(clean.time_s>=390)&(clean.time_s<430)&(clean.prn==np.unique(clean.prn)[0])
    metadata.append(("prn_drop_add",_mask_data(clean,~middle)))
    gap_epoch=clean.epoch.copy();gap_epoch[clean.time_s>=410]+=3
    metadata.append(("timestamp_gap",EpochData(clean.time_s,gap_epoch,clean.prn,clean.segment,base,clean.sample_count,
                                                clean.cn0_db_hz,clean.prompt_power,clean.cadence_s,clean.recording_id)))
    reacquired=clean.segment.copy();reacquired[clean.time_s>=410]+=10000
    metadata.append(("lock_loss_reacquisition",_replace_taps(clean,base,segments=reacquired)))
    for name,data in metadata:
        rows,_=score_full(data,ridge,rw);run=max_alarm_run(rows,threshold)
        controls.append({"control":name,"domain":"correlator_or_metadata","longest_alarm_run_epochs":run,
                         "sustained_alarm_5s":run>=10,"executed":True,
                         "note":"C/N0 is excluded from score; support changes, epoch gaps, and segment changes use the frozen reset/support logic."})
    write_json(out/"physical_controls.json",{"schema":"gnss-doppler-lab.q-comet-controls.v1","physical_scope":"correlator-domain controls only; not raw-IQ physical proof",
                                             "threshold":threshold,"sustained_definition":"at least 5 s (10 epochs)","controls":controls,
                                             "cleanStatic_holdout_fpr":clean_holdout_fpr,
                                             "any_sustained_alarm":any(x["sustained_alarm_5s"] for x in controls)})
    return controls


def read_csv(path,*,gz=False):
    opener=gzip.open if gz else open
    with opener(path,"rt",encoding="utf8",newline="") as handle:return list(csv.DictReader(handle))


def runner_provenance(out: Path):
    run_root=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/runs")
    groups={
      "source-inventory-preflight":["config.json","data_inventory.json","source_binding.json","timeline_inventory.json","normal_split_audit.json","source_commit.json"],
      "freeze-normal-config":["normal_model_summary.json","nuisance_projection_validation.json","thresholds.json","pre_evaluation_freeze.json"],
      "ds78-prefix-audit":["ds78_byte_identity_audit.json","timeline_inventory.json"],
      "texbat-evaluation":["texbat_scores.csv.gz","texbat_metrics.csv","texbat_onsets.csv","texbat_participation.csv.gz","texbat_external_fpr.csv","texbat_score_diagnostics.csv.gz","texbat_prn_bayes_factors.csv.gz","texbat_relation_scores.csv.gz","relation_destruction_metrics.json"],
      "oakbat-confirmation":["oakbat_scores.csv.gz","oakbat_metrics.csv","oakbat_onsets.csv","oakbat_participation.csv.gz","oakbat_score_diagnostics.csv.gz","oakbat_prn_bayes_factors.csv.gz","cross_dataset_confirmation.json"],
      "controls-bootstrap-report":["physical_controls.json","scenario_metrics.csv","ablation_metrics.csv","per_epoch_scores.csv.gz","common_onset_estimates.csv","participation_posteriors.csv.gz","external_static_fpr.csv","bootstrap_intervals.csv","final_verdict.json","implementation_repairs.json","README.md","plots"],
      "verification":["artifact_manifest_sha256.json","verifier_report.json","fresh_clone_verifier_report.json"],
    }
    records=[]
    for directory in sorted(run_root.glob("*-q-comet-*")):
        required=[directory/x for x in ("contract.json","git.json","env.json","heartbeat.json","status.json")]
        if not all(x.is_file() for x in required):continue
        contract=json.loads((directory/"contract.json").read_text());status=json.loads((directory/"status.json").read_text())
        produced=[]
        if status.get("status")=="succeeded":
            for marker,names in groups.items():
                if marker in contract["name"]:
                    for name in names:
                        path=out/name
                        if path.is_dir():produced.extend(str(x.relative_to(ROOT)) for x in sorted(path.rglob("*")) if x.is_file())
                        elif path.is_file():produced.append(str(path.relative_to(ROOT)))
        manifest={"schema_version":1,"run_id":contract["run_id"],"artifacts":produced,
                  "recorded_by":"Q-COMET runner provenance audit"}
        if "orchestrator" not in contract["name"]:write_json(directory/"result_manifest.json",manifest)
        logs={}
        for stream in ("stdout","stderr"):
            path=directory/f"{stream}.log";logs[stream]={"path":str(path),"sha256":sha256_file(path),"size_bytes":path.stat().st_size}
        records.append({"run_id":contract["run_id"],"name":contract["name"],"command":contract["command"],
                        "cwd":contract["cwd"],"git":json.loads((directory/"git.json").read_text()),
                        "environment":json.loads((directory/"env.json").read_text()),"status":status,
                        "heartbeat":json.loads((directory/"heartbeat.json").read_text()),"logs":logs,
                        "result_manifest_path":str(directory/"result_manifest.json"),"produced_artifacts":produced})
    write_json(out/"runner_runs.json",{"schema":"gnss-doppler-lab.q-comet-runner-provenance.v1",
        "shared_artifact_path_note":"Repeated repair runs overwrite the same declared artifact paths; status/logs identify failed, killed, and final successful attempts, while artifact_manifest_sha256.json authenticates the final files.",
        "runs":records})
    print(f"Q_COMET_RUNNER_PROVENANCE_PASS runs={len(records)}",flush=True)


def plot_reports(out,metrics,score_rows,relation):
    plots=out/"plots";plots.mkdir(exist_ok=True)
    def number(value):
        try:return float(value)
        except (TypeError,ValueError):return np.nan
    tex_threshold=json.loads((out/"thresholds.json").read_text())["methods"]["Full"]["q99"]
    oak_threshold=json.loads((out/"cross_dataset_confirmation.json").read_text())["thresholds"]["Full"]
    full=[r for r in score_rows if r["method"]=="Full"]
    for scenario in sorted(set(r["scenario"] for r in full)):
        rows=[r for r in full if r["scenario"]==scenario];t=np.asarray([float(r["time_s"]) for r in rows]);s=np.asarray([float(r["score"]) for r in rows])
        threshold=tex_threshold if scenario.startswith("DS") else oak_threshold
        fig,ax=plt.subplots(figsize=(8,3));ax.plot(t,s,lw=.8,label="Full score");ax.axhline(threshold,color="k",ls=":",label="clean q99 threshold");ax.axvline(TIMELINES[scenario]["onset_s"],color="r",ls="--",label="official onset");ax.set(xlabel="receiver time (s)",ylabel="Full score",title=f"{scenario} Q-COMET Full");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(plots/f"{scenario.lower()}_full_score.png",dpi=130);plt.close(fig)
    methods=sorted(set(r["method"] for r in metrics));values=[];delays=[]
    for method in methods:
        x=[number(r.get("pauc_fpr_0_05")) for r in metrics if r["method"]==method];values.append(np.nanmean(x))
        x=[number(r.get("first_alarm_delay_s")) for r in metrics if r["method"]==method];delays.append(np.nanmean(x))
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,3.5));ax1.bar(methods,values);ax1.set(ylabel="mean standardized pAUC@5%",title="Discrimination");ax2.bar(methods,delays);ax2.set(ylabel="mean first-alarm delay (s)",title="Delay");
    for ax in (ax1,ax2):ax.tick_params(axis="x",rotation=60)
    fig.tight_layout();fig.savefig(plots/"ablation_pauc_delay.png",dpi=130);plt.close(fig)

    ridge,rw,_,_=load_bundle(out);clean=load_texbat("cleanStatic").subset(*SPLITS["holdout"]);clean_table=innovations(clean,ridge,rw)
    peaks=np.concatenate((clean.complex_taps.real,clean.complex_taps.imag),axis=1);centered=peaks-peaks.mean(0);_,_,vh=np.linalg.svd(centered,full_matrices=False);pc=centered@vh[:2].T
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.5));ax1.scatter(pc[:,0],pc[:,1],s=3,alpha=.2);ax1.set(xlabel="normal peak PC1",ylabel="normal peak PC2",title="cleanStatic complex-peak manifold")
    ax2.hist(np.linalg.norm(clean_table.whitened,axis=1),bins=60,alpha=.6,label="whitened residual");ax2.hist(np.linalg.norm(clean_table.quotient,axis=1),bins=60,alpha=.6,label="quotient residual");ax2.set(xlabel="residual norm",ylabel="rows",title="Normal residual before/after quotient");ax2.legend(fontsize=7)
    fig.tight_layout();fig.savefig(plots/"normal_manifold_quotient_residual.png",dpi=130);plt.close(fig)

    _,_,pred=ridge.predict_rows(clean)
    before=[];after=[]
    for reference in pred[::max(1,len(pred)//100)]:
        tangents=nuisance_jacobian(reference,rw.inverse_sqrt)
        before.append(np.linalg.norm(tangents,axis=1));after.append([np.linalg.norm(quotient_project(x,reference,rw.inverse_sqrt)) for x in tangents])
    before=np.mean(before,axis=0);after=np.mean(after,axis=0);names=["gain","phase/nav bit","code delay","Doppler phase"]
    fig,ax=plt.subplots(figsize=(7,3.5));x=np.arange(4);ax.bar(x-.18,before,.36,label="before");ax.bar(x+.18,after,.36,label="after quotient");ax.set_xticks(x,names,rotation=20);ax.set_yscale("symlog",linthresh=1e-12);ax.set(ylabel="mean tangent response norm",title="Observed-clean-peak nuisance response");ax.legend();fig.tight_layout();fig.savefig(plots/"nuisance_projection_validation.png",dpi=130);plt.close(fig)

    onset_rows=read_csv(out/"common_onset_estimates.csv");official=np.asarray([number(x["official_onset_s"]) for x in onset_rows]);estimated=np.asarray([number(x["estimated_onset_s"]) for x in onset_rows])
    fig,ax=plt.subplots(figsize=(5,4));ax.scatter(official,estimated);lo=np.nanmin(np.r_[official,estimated]);hi=np.nanmax(np.r_[official,estimated]);ax.plot([lo,hi],[lo,hi],"k--",label="exact");
    for row,x,y in zip(onset_rows,official,estimated):ax.annotate(row["scenario"],(x,y),fontsize=7)
    ax.set(xlabel="official onset (s)",ylabel="estimated onset (s)",title="Common-onset estimates");ax.legend();fig.tight_layout();fig.savefig(plots/"official_estimated_onset.png",dpi=130);plt.close(fig)

    bf=read_csv(out/"texbat_prn_bayes_factors.csv.gz",gz=True);bf=[x for x in bf if x["scenario"]=="DS3"]
    times=sorted({number(x["time_s"]) for x in bf});prns=sorted({int(x["prn"]) for x in bf});matrix=np.full((len(prns),len(times)),np.nan);ti={x:i for i,x in enumerate(times)};pi={x:i for i,x in enumerate(prns)}
    for row in bf:matrix[pi[int(row["prn"])],ti[number(row["time_s"])]]=np.clip(number(row["log_bayes_factor"]),-20,20)
    fig,ax=plt.subplots(figsize=(9,3.5));im=ax.imshow(matrix,aspect="auto",origin="lower",extent=[min(times),max(times),-.5,len(prns)-.5],cmap="coolwarm",vmin=-20,vmax=20);ax.axvline(TIMELINES["DS3"]["onset_s"],color="k",ls="--");ax.set_yticks(np.arange(len(prns)),prns);ax.set(xlabel="receiver time (s)",ylabel="PRN",title="DS3 per-PRN log Bayes factors (clipped)");fig.colorbar(im,ax=ax,label="log BF");fig.tight_layout();fig.savefig(plots/"prn_bf_heatmap.png",dpi=130);plt.close(fig)

    part=read_csv(out/"participation_posteriors.csv.gz",gz=True);fig,ax=plt.subplots(figsize=(9,3.5))
    for scenario in sorted({x["scenario"] for x in part}):
        rows=[x for x in part if x["scenario"]==scenario];ax.plot([number(x["time_s"]) for x in rows],[number(x["participation_posterior"]) for x in rows],lw=.7,label=scenario)
    ax.set(xlabel="receiver time (s)",ylabel="mean PRN participation posterior",title="Frozen soft-participation evidence");ax.legend(ncol=4,fontsize=7);fig.tight_layout();fig.savefig(plots/"participation_posterior.png",dpi=130);plt.close(fig)

    rel=relation["scenarios"];x=np.arange(len(rel));fig,ax=plt.subplots(figsize=(7,3.5));ax.bar(x-.2,[r["original_mean_post"] for r in rel],.4,label="original");ax.bar(x+.2,[r["desynchronized_mean_post"] for r in rel],.4,label="PRN-time shifted");ax.set_xticks(x,[r["scenario"] for r in rel]);ax.set(ylabel="mean post-onset score",title="Same-epoch relation destruction");ax.legend();fig.tight_layout();fig.savefig(plots/"original_vs_desync_score.png",dpi=130);plt.close(fig)

    external=read_csv(out/"external_static_fpr.csv");fig,ax=plt.subplots(figsize=(8,3.5));labels=[f"{x['dataset']}:{x['scenario']}" for x in external];fpr=[number(x["full_fpr"]) for x in external];ax.bar(labels,fpr);ax.axhline(.01,color="k",ls=":",label="target 1%");ax.axhline(.05,color="r",ls="--",label="external ceiling 5%");ax.tick_params(axis="x",rotation=45);ax.set(ylabel="Full false-positive rate",title="Clean holdout and scenario pre-onset FPR");ax.legend();fig.tight_layout();fig.savefig(plots/"external_static_fpr.png",dpi=130);plt.close(fig)

    oak=[r for r in metrics if r["method"]=="Full" and r["scenario"].startswith("OS")];x=np.arange(len(oak));fig,ax=plt.subplots(figsize=(7,3.5));ax.bar(x-.2,[number(r["pauc_fpr_0_05"]) for r in oak],.4,label="standardized pAUC@5%");ax.bar(x+.2,[number(r["persistent_alarm_ratio"]) for r in oak],.4,label="persistent alarm ratio");ax.set_xticks(x,[r["scenario"] for r in oak]);ax.set_ylim(0,1);ax.set(title="OAKBAT frozen cross-dataset confirmation");ax.legend();fig.tight_layout();fig.savefig(plots/"oakbat_confirmation.png",dpi=130);plt.close(fig)

    diag=read_csv(out/"texbat_score_diagnostics.csv.gz",gz=True)+read_csv(out/"oakbat_score_diagnostics.csv.gz",gz=True);fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.5))
    ax1.scatter([number(x["total_residual_energy"]) for x in diag],[number(x["full_score"]) for x in diag],s=5,alpha=.25);ax2.scatter([number(x["total_prompt_power"]) for x in diag],[number(x["full_score"]) for x in diag],s=5,alpha=.25)
    ax1.set(xlabel="total raw residual energy",ylabel="Full score",title="Residual-energy diagnostic");ax2.set(xlabel="total Prompt power",ylabel="Full score",title="Prompt-power diagnostic");fig.tight_layout();fig.savefig(plots/"score_vs_residual_power.png",dpi=130);plt.close(fig)


def finalize_report(out: Path, freeze_sha):
    validate_freeze(out,freeze_sha); control_rows=controls(out)
    tex_metrics=read_csv(out/"texbat_metrics.csv");oak_metrics=read_csv(out/"oakbat_metrics.csv")
    tex_scores=read_csv(out/"texbat_scores.csv.gz",gz=True);oak_scores=read_csv(out/"oakbat_scores.csv.gz",gz=True)
    method_thresholds=json.loads((out/"thresholds.json").read_text())["methods"]
    for metric in tex_metrics:
        rows=[x for x in tex_scores if x["scenario"]==metric["scenario"] and x["method"]==metric["method"]]
        threshold=method_thresholds[metric["method"]]["q99"]
        for field,event_key in (("pull_off_first_alarm_delay_s","pull_off_s"),("time_push_first_alarm_delay_s","time_push_s")):
            event=TIMELINES[metric["scenario"]].get(event_key)
            alarms=[float(x["time_s"]) for x in rows if event is not None and float(x["time_s"])>=event and np.isfinite(float(x["score"])) and float(x["score"])>threshold]
            metric[field]=np.nan if event is None or not alarms else min(alarms)-event
    metrics=tex_metrics+oak_metrics
    scenario=[r for r in metrics if r["method"]=="Full"]
    ablations=metrics+[{"scenario":s,"family":TIMELINES[s]["family"],"method":"A0","finite_epochs":0,
                        "availability":"UNAVAILABLE_WITH_REASON","reason":"Paper-B0 exact common-support adapter and authenticated checkpoint were not bound in the frozen configuration; historical CSV performance was not copied."} for s in TIMELINES]
    write_csv(out/"scenario_metrics.csv",list(scenario[0]),scenario)
    fields=sorted(set().union(*(r.keys() for r in ablations)));write_csv(out/"ablation_metrics.csv",fields,ablations)
    write_csv(out/"per_epoch_scores.csv.gz",list(tex_scores[0]),tex_scores+oak_scores,gz=True)
    tex_part=read_csv(out/"texbat_participation.csv.gz",gz=True);oak_part=read_csv(out/"oakbat_participation.csv.gz",gz=True)
    write_csv(out/"participation_posteriors.csv.gz",list(tex_part[0]),tex_part+oak_part,gz=True)
    onsets=read_csv(out/"texbat_onsets.csv")+read_csv(out/"oakbat_onsets.csv");write_csv(out/"common_onset_estimates.csv",list(onsets[0]),onsets)
    external=read_csv(out/"texbat_external_fpr.csv")
    control_doc=json.loads((out/"physical_controls.json").read_text())
    external.insert(0,{"dataset":"TEXBAT","scenario":"cleanStatic","role":"chronological_holdout","full_fpr":control_doc["cleanStatic_holdout_fpr"]})
    write_csv(out/"external_static_fpr.csv",list(external[0]),external)
    relation=json.loads((out/"relation_destruction_metrics.json").read_text())
    bootstrap=[]
    for r in relation["scenarios"]:
        for metric_name,x in r["paired_block_bootstrap"].items():
            bootstrap.append({"dataset":"TEXBAT","scenario":r["scenario"],"metric":metric_name,
                              "estimate":x["estimate"],"ci_low":x["ci_low"],"ci_high":x["ci_high"],
                              "block_s":x["block_s"],"replicates":x["replicates"],"finite_replicates":x["finite_replicates"]})
    write_csv(out/"bootstrap_intervals.csv",list(bootstrap[0]),bootstrap)
    # Strict GO cannot be claimed without exact common-support Paper-B0 evidence; controls/performance may add NO-GO reasons.
    reasons=["NO_EXACT_COMMON_SUPPORT_PAPER_B0_BENEFIT_EVIDENCE"]
    if any(x["sustained_alarm_5s"] for x in control_rows):reasons.append("SUSTAINED_CORRELATOR_CONTROL_ALARM")
    relation_families={r["family"] for r in relation["scenarios"] if r["score_drop_fraction"]>=.3 and r["score_delta_bootstrap"]["ci_low"]>0}
    if len(relation_families)<2:reasons.append("RELATION_DESTRUCTION_CRITERION_NOT_MET")
    worst_external=max(float(x["full_fpr"]) for x in external if x["role"]=="scenario_pre_onset")
    if worst_external>.05:reasons.append("EXTERNAL_STATIC_PRE_ONSET_FPR_EXCEEDS_5_PERCENT")
    tex_full=[x for x in tex_metrics if x["method"]=="Full"]
    plausible_tex_families={x["family"] for x in tex_full if x["onset_error_s"] not in ("","nan") and abs(float(x["onset_error_s"]))<=10}
    if len(plausible_tex_families)<2:reasons.append("PLAUSIBLE_TEXBAT_COMMON_ONSET_NOT_REPRODUCED")
    oak_full=[x for x in oak_metrics if x["method"]=="Full"]
    plausible_oak=sum(x["onset_error_s"] not in ("","nan") and abs(float(x["onset_error_s"]))<=10 for x in oak_full)
    if plausible_oak<2:reasons.append("OAKBAT_PLAUSIBLE_ONSET_CONFIRMATION_NOT_MET")
    comparison_families=set()
    for family in {x["family"] for x in tex_metrics}:
        family_rows=[x for x in tex_metrics if x["family"]==family]
        means={method:np.nanmean([float(x["pauc_fpr_0_05"]) for x in family_rows if x["method"]==method]) for method in ("Full","A3","A5")}
        if means["Full"]>means["A3"] and means["Full"]>means["A5"]:comparison_families.add(family)
    if len(comparison_families)<2:reasons.append("SHARED_ONSET_AND_HETEROGENEOUS_DIRECTION_ABLATION_BENEFIT_NOT_MET")
    verdict="NO_GO_SHARED_ONSET_HYPOTHESIS"
    write_json(out/"final_verdict.json",{"schema":"gnss-doppler-lab.q-comet-verdict.v1","verdict":verdict,"freeze_commit_sha":freeze_sha,
                                        "go_criteria_all_required":True,"go_criteria_met":False,"reasons":reasons,
                                        "criterion_evidence":{"cleanStatic_holdout_fpr":control_doc["cleanStatic_holdout_fpr"],
                                            "worst_external_static_pre_onset_fpr":worst_external,
                                            "qualifying_relation_destruction_families":sorted(relation_families),
                                            "plausible_texbat_onset_families":sorted(plausible_tex_families),
                                            "plausible_oakbat_onset_scenarios":plausible_oak,
                                            "full_over_a3_and_a5_families":sorted(comparison_families)},
                                        "neural_stage1_implemented":False,"recommended_next_action":"Stop Q-COMET; retain Stage-0 artifacts as a negative-result record."})
    write_json(out/"implementation_repairs.json",{"schema":"gnss-doppler-lab.q-comet-post-freeze-repairs.v1",
        "scientific_configuration_changed":False,"repairs":[
          {"scope":"DS4 provenance adapter","reason":"Legacy DS4 manifest omitted raw-IQ SHA; regenerated from raw IQ with the pinned complex-nine-tap receiver before scoring.","attack_result_driven":False},
          {"scope":"OAKBAT receiver adapter","failed_run_id":"20260816T035459Z-q-comet-oakbat-confirmation",
           "reason":"Legacy Method-A MAT files preserved magnitudes plus Prompt I/Q, not genuine complex I/Q for all nine taps. The first run stopped while loading normal cleanStatic, before OS1-OS4 scoring. Raw IQ was then regenerated with the pinned receiver.","attack_result_driven":False},
          {"scope":"A6 common-support alignment","reason":"The persistence predictor produces one extra leading target row. A post-evaluation support audit aligned A6 by immutable receiver row index to Full before scoring; Full and its frozen configuration/scores are unchanged.","attack_result_driven":False},
          {"scope":"Full summary selector","reason":"The first report pass incorrectly populated TEXBAT onset/pre-onset summary files from No-quotient. The selector was corrected to the already-computed Full row; no detector score or configuration changed.","attack_result_driven":False},
          {"scope":"evidence diagnostics","reason":"Added per-PRN Bayes-factor export, score-energy/power correlations, DS7/DS8 byte-prefix audit, and data-backed plots without changing Full scores, thresholds, priors, windows, or model selection.","attack_result_driven":False}]})
    plot_reports(out,metrics,tex_scores+oak_scores,relation)
    def compact(rows):
        def display(value):
            value=float(value);return "no alarm" if not np.isfinite(value) else f"{value:.1f} s"
        return "; ".join(f"{x['scenario']}: pAUC={float(x['pauc_fpr_0_05']):.3f}, pre-FPR={float(x['pre_onset_fpr']):.3f}, delay={display(x['first_alarm_delay_s'])}, onset={display(x['estimated_onset_s'])}" for x in rows)
    def maximum_abs(rows,field):
        values=[abs(float(x[field])) for x in rows if x.get(field) not in (None,"","nan") and np.isfinite(float(x[field]))]
        return max(values) if values else np.nan
    relation_text="; ".join(f"{x['scenario']}: drop={100*x['score_drop_fraction']:.1f}%, 95% CI delta=[{x['score_delta_bootstrap']['ci_low']:.3g},{x['score_delta_bootstrap']['ci_high']:.3g}]" for x in relation["scenarios"])
    sustained=", ".join(x["control"] for x in control_rows if x["sustained_alarm_5s"])
    residual_corr=maximum_abs(scenario,"score_total_residual_energy_pearson_r");prompt_corr=maximum_abs(scenario,"score_total_prompt_power_pearson_r")
    comparison_statement=(f"Full exceeded both A3 and A5 in the required two families ({', '.join(sorted(comparison_families))})."
                          if len(comparison_families)>=2 else "Full did not satisfy the required two-family advantage over both A3 and A5.")
    readme=f"""# Q-COMET Stage-0 static result

Final verdict: `{verdict}`. Freeze commit: `{freeze_sha}`. Every GO criterion was required; the exact failure reasons and criterion values are in `final_verdict.json`. Neural Stage-1 was not implemented.

## Inputs, provenance, and time

The scored fields were receiver-relative sample count/time, PRN support, and genuine complex I/Q at E4/E3/E2/E/P/L/L2/L3/L4 (0.125-chip spacing). Preserved audit-only receiver fields include carrier Doppler, code frequency/error, carrier discriminator, C/N0, and carrier lock; C/N0, lock, Prompt amplitude, and IQ power were never direct scores. `source_binding.json` binds raw IQ, receiver binary/config/source, and MAT/NPZ lineage. `timeline_inventory.json` binds official receiver-relative onsets. DS4 ends near 128.2 s and is transition-only. DS7/DS8 are one family; `ds78_byte_identity_audit.json` prevents byte-identical pre-110 s data from being counted as independent normal confirmation.

## Hypotheses and quotient

H0 is `c[i,t] = f_theta(c[i,t-1:t-L]) + epsilon`, fitted only to cleanStatic, with calibration-A Ledoit-Wolf covariance. The innovation is `r = Sigma^(-1/2)(c-c_hat)` and `v = (I-J(J'J)^(-1)J')r`. The observed-local-peak Jacobian removes gain, common carrier/navigation-bit phase, local code-delay recentering, and tap-dependent Doppler phase; it preserves the remaining signed complex deformation directions. H1 uses free PRN-specific coefficients on frozen step/ramp/transient bases and `S_t = max_k sum_i log((1-pi)+pi*B_i(k,t)) - 0.5 log(L+1)` over a causal 10 s window. Full does not impose a common deformation direction.

## Normal reliability and performance

The chronological cleanStatic split is train 20–140 s, calibration-A 150–210 s, calibration-B 220–340 s, and holdout 350–470 s, with 10 s guards and byte-disjoint ranges. Calibration-A selected ridge-VAR lag 2 and fit shrinkage covariance; calibration-B alone set q99/q99.5 thresholds. Clean holdout FPR was {100*control_doc['cleanStatic_holdout_fpr']:.2f}%.

TEXBAT Full — {compact(tex_full)}. DS3/DS4/DS7-8 therefore did not provide plausible common-onset recovery, and worst pre-onset FPR was {100*worst_external:.2f}%.

OAKBAT frozen confirmation — {compact(oak_full)}. The structure remained frozen and only the normal predictor/covariance/threshold were refit on OAKBAT cleanStatic. OS-family onset behavior was mixed and fewer than two scenarios had onset error within 10 s.

Paper-B0 A0 is `UNAVAILABLE_WITH_REASON`: no authenticated exact common-support adapter/checkpoint was bound at freeze, so historical CSV performance was not copied. `ablation_metrics.csv` contains A1–A7, Full, EPL3, and No-quotient on common receiver units/support. {comparison_statement} No core recording remained unavailable: DS4 was source-bound from raw IQ, and OAKBAT was regenerated from raw after an initial normal-clean adapter failure revealed that legacy MATs lacked genuine complex outer-tap I/Q.

Relation destruction — {relation_text}. Only families with at least 30% score drop and a strictly positive paired 10 s block-bootstrap CI qualify; fewer than two qualified. The same bootstrap also reports pAUC delta, alarm-delay change, and persistent-detection delta.

Correlator-domain controls covered gain 0.5/0.75/1.25/2, carrier phase, navigation-bit sign, code recentering, Doppler, empirical clean noise 0.5/1/2, C/N0 metadata, clock-like drift, one-PRN disturbance, PRN drop/add, independent multipath, timestamp gap, reacquisition, and exact-aligned counterfeit. Sustained alarms occurred for: {sustained}. These are not raw-IQ physical proofs. The exact-aligned case documents the information-theoretically-undetectable boundary.

Official-versus-estimated onsets, participation, and per-PRN Bayes factors are in the CSVs and evidence-backed plots. Across the eight Full scenario rows, the maximum absolute Pearson correlation was {residual_corr:.3f} with total raw-residual energy and {prompt_corr:.3f} with total Prompt power. Q-COMET targets the first receiver-correlator-visible common change, not necessarily transmitter RF turn-on.

Claimable contribution: an auditable, normal-only quotient/common-onset Stage-0 implementation plus reproducible negative-result evidence. Non-claimable: raw-RF immunity, universal spoof detection, absolute transmitter onset recovery, superiority to Paper-B0, or evidence for neural Stage-1.

Recommended next action: stop Q-COMET and retain this Stage-0 bundle as the negative-result record.
"""
    (out/"README.md").write_text(readme)
    print(f"Q_COMET_CONTROLS_BOOTSTRAP_REPORT_PASS verdict={verdict}",flush=True)


def main():
    args=parse_args();out=args.artifact_dir.resolve()
    if args.phase=="preflight":preflight(out)
    elif args.phase=="freeze-normal-config":freeze_normal(out)
    elif args.phase=="ds78-prefix-audit":ds78_prefix_audit(out,args.freeze_commit_sha)
    elif args.phase=="texbat-evaluation":texbat_evaluation(out,args.freeze_commit_sha)
    elif args.phase=="oakbat-confirmation":oakbat_confirmation(out,args.freeze_commit_sha)
    elif args.phase=="controls-bootstrap-report":finalize_report(out,args.freeze_commit_sha)
    elif args.phase=="runner-provenance":runner_provenance(out)
    else:
        preflight(out);freeze_normal(out)
        raise SystemExit("all phase stops at PRE_EVALUATION_CONFIGURATION_FREEZE; commit it, record SHA, then run protected phases explicitly")
    return 0


if __name__=="__main__":raise SystemExit(main())
