#!/usr/bin/env python3
"""Frozen, leakage-safe GCMR external inference on TEXBAT DS1--DS4."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, NamedTuple
import numpy as np
import torch
import gnss_doppler_lab.gcmr_experiment as experiment
from gnss_doppler_lab.gcmr_experiment import (
    cache_events, load_event_cache, parse_preonset_nmea_position,
    preflight_oakbat_geometry, save_score_csv, score_events, source_hashes,
)
from gnss_doppler_lab.gcmr_geometry import (
    ephemeris_health_selection, parse_gnss_sdr_gps_ephemeris_xml,
    validate_ephemeris_time_alignment,
)
from gnss_doppler_lab.gcmr_relations import (
    build_gcmr_pair_relation_events, load_gnss_sdr_tracking_rows,
)

TEXBAT_CONTRACT_VERSION = "gcmr-texbat-external-v1"
ADAPTER_REPOSITORY_PATH = "scripts/run_gcmr_texbat_external.py"
FROZEN_THRESHOLD = 2.8224338538365963
CHECKPOINT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-oakbat-poc-v5-seed23/model.pt")
DEFAULT_OUTPUT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-ds1-ds4-external-v1")
DEFAULT_CACHE = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-event-cache-v1")
DS123_PARENT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver")
DS4_ROOT = Path("/home/ubuntu/projects/gnss-doppler-lab/artifacts/ai_morph_gru_window_ablation_ds4_20260723/receiver_shared/ds4/receiver/texbat-ds4-method-a-9tap-external-validation")
SCENARIOS = {
    "DS1": {"root": DS123_PARENT/"ds1-complex9", "sample_rate_hz": 25e6, "tow0_s": 477900.0, "preflight": "observables_rx_time"},
    "DS2": {"root": DS123_PARENT/"ds2-complex9", "sample_rate_hz": 25e6, "tow0_s": 477900.02, "preflight": "observables_rx_time"},
    "DS3": {"root": DS123_PARENT/"ds3-complex9", "sample_rate_hz": 25e6, "tow0_s": 477900.0, "preflight": "observables_rx_time"},
    "DS4": {"root": DS4_ROOT, "sample_rate_hz": 25e6, "tow0_s": 477900.0, "preflight": "alternate_common_replay_nmea_tow0_no_observables"},
}
EVENT_CONTRACT = {
    "version": TEXBAT_CONTRACT_VERSION, "window_s": 1.0, "stride_s": 0.5,
    "resample_bin_s": 0.02, "min_common_samples": 20, "min_prns": 4,
    "window_interval": "[start,end)", "score_available_at": "window_end_s",
    "healthy_ephemerides_only": True, "max_toe_age_s": 7200.0,
}
ONSET_CONTRACT = {
    "primary_nominal_onset_s": 100.0, "stable_pre": "[30,90)",
    "transition": "[90,110)", "stable_post": "[110,+inf)",
    "post120_sensitivity": "[120,+inf) secondary only",
    "ds4_auxiliary_script_conflict": {"auxiliary_onset_s": 110.0, "primary_onset_s": 100.0,
       "resolution": "primary frozen TEXBAT contract retained; post>=120 reported as sensitivity only"},
}

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def external_evaluation_implementation_manifest(frozen_core=None, *, adapter_path: Path | None=None) -> dict:
    """Describe the frozen-core/external-adapter boundary using runtime file bytes.

    The digest is deliberately computed from the adapter file itself rather than
    compared with a value embedded in that file, avoiding a circular self-hash.
    """
    path=Path(adapter_path or __file__).resolve()
    core=(frozen_core if frozen_core is not None else {
        "provenance_source":"loaded_checkpoint",
        "validation":"checkpoint-embedded frozen implementation manifest is revalidated before evaluation",
    })
    return {
        "boundary":"frozen core is checkpoint-embedded and hash-validated; external adapter is evaluation-only and identified by its runtime SHA256",
        "frozen_core":core,
        "external_adapter":{"path":ADAPTER_REPOSITORY_PATH,"sha256":_sha(path)},
        "adapter_hash_contract":"SHA256 is computed directly from adapter bytes at runtime; this output manifest is not an embedded expected hash",
    }


def _validate_adapter_cache_provenance(metadata: dict) -> None:
    """Reject cache operations unless metadata binds them to these adapter bytes."""
    try:
        adapter=metadata["external_evaluation_implementation"]["external_adapter"]
        saved_path=adapter["path"]
        saved_hash=adapter["sha256"]
    except (KeyError,TypeError):
        raise ValueError("adapter provenance missing from event cache metadata") from None
    current=_sha(Path(__file__).resolve())
    if saved_path != ADAPTER_REPOSITORY_PATH or saved_hash != current:
        raise ValueError("adapter provenance is stale or does not match the running external adapter")


def frozen_implementation_manifest(saved: dict, repo_root: Path | None=None) -> dict:
    """Revalidate exactly the saved manifest; external adapter files stay outside it."""
    root=(repo_root or Path(__file__).resolve().parents[1]).resolve()
    records=[]; aggregate=hashlib.sha256()
    for expected in saved.get("files",[]):
        rel=expected.get("path",""); path=root/rel
        if not rel or not path.is_file(): raise ValueError(f"frozen implementation file missing: {rel}")
        digest=_sha(path)
        if digest != expected.get("sha256"): raise ValueError(f"frozen implementation file changed: {rel}")
        records.append({"path":rel,"sha256":digest})
        aggregate.update(rel.encode()); aggregate.update(b"\0"); aggregate.update(bytes.fromhex(digest))
    actual={"files":records,"aggregate_sha256":aggregate.hexdigest()}
    if actual != saved: raise ValueError("frozen implementation aggregate manifest mismatch")
    return actual

@contextlib.contextmanager
def _external_manifest_boundary(manifest):
    """Keep newly added external-only files out of the already frozen implementation set."""
    original=experiment.implementation_manifest
    experiment.implementation_manifest=lambda anchor=None: manifest
    try: yield
    finally: experiment.implementation_manifest=original

class FrozenCheckpoint(NamedTuple):
    """Immutable external provenance wrapper around the frozen core loader result."""
    core: object
    checkpoint_sha256: str

    @property
    def model(self):
        return self.core.model

    @property
    def calibrator(self):
        return self.core.calibrator

    @property
    def threshold(self):
        return self.core.threshold

    @property
    def provenance(self):
        return self.core.provenance


def load_frozen_checkpoint(path=CHECKPOINT, *, device="cpu"):
    """Call the validated loader and bind the result to exact checkpoint bytes."""
    path=Path(path)
    try: checkpoint_sha256=_sha(path)
    except OSError as exc: raise ValueError(f"invalid frozen checkpoint: {exc}") from exc
    try: payload=torch.load(path,map_location="cpu",weights_only=True)
    except (OSError,RuntimeError,ValueError) as exc: raise ValueError(f"invalid frozen checkpoint: {exc}") from exc
    saved=payload.get("provenance",{}).get("implementation") if isinstance(payload,dict) else None
    if not isinstance(saved,dict): raise ValueError("checkpoint has no frozen implementation manifest")
    actual=frozen_implementation_manifest(saved)
    with _external_manifest_boundary(actual):
        loaded=experiment.load_checkpoint(path,device=device)
    if loaded.threshold != FROZEN_THRESHOLD:
        raise ValueError(f"frozen threshold mismatch: {loaded.threshold!r} != {FROZEN_THRESHOLD!r}")
    try: checkpoint_sha256_after=_sha(path)
    except OSError as exc: raise ValueError(f"invalid frozen checkpoint after load: {exc}") from exc
    if checkpoint_sha256_after != checkpoint_sha256:
        raise ValueError("frozen checkpoint bytes changed while loading")
    return FrozenCheckpoint(core=loaded,checkpoint_sha256=checkpoint_sha256)

def _valid_rmc_records(path: Path, tow0_s: float):
    records=[]
    for line in path.read_text(errors="replace").splitlines():
        fields=experiment._valid_sentence(line)
        if not fields or fields[0][-3:]!="RMC" or len(fields)<=9 or fields[2]!="A": continue
        try:
            week,tow=experiment._gps_week_and_tow(datetime.strptime(fields[9],"%d%m%y").date(),experiment._hms(fields[1]))
        except (ValueError,IndexError): continue
        rel=(tow-float(tow0_s)+302400)%604800-302400
        records.append((week,float(tow),float(rel)))
    return records

def preflight_ds4_alternate(root, ephemerides, *, configured_tow0_s, tracked_prns,
                            min_prns=4, max_toe_age_s=7200.0,
                            expected_first_rmc_relative_s=18.18, rmc_tolerance_s=0.25):
    """Fail-closed DS4 proof for the explicitly observables-free common replay path."""
    root=Path(root); tow0=float(configured_tow0_s)
    if not math.isfinite(tow0): raise ValueError("user-supplied fixed tow0 must be finite")
    observables=root/"raw/observables.mat"
    if observables.exists(): raise ValueError("DS4 alternate preflight requires observables.mat to be absent")
    nmea=root/"nmea_pvt.nmea"
    records=_valid_rmc_records(nmea,tow0)
    if not records: raise ValueError("no checksum-valid active RMC records")
    first_rel=records[0][2]
    if abs(first_rel-expected_first_rmc_relative_s)>rmc_tolerance_s:
        raise ValueError(f"first valid RMC is inconsistent with common replay start: {first_rel}")
    pre=[x for x in records if 0<=x[2]<ONSET_CONTRACT["primary_nominal_onset_s"]]
    if not pre or len({x[0] for x in pre})!=1: raise ValueError("pre-onset RMC GPS weeks are missing or inconsistent")
    alignment=validate_ephemeris_time_alignment(ephemerides,full_gps_week=pre[0][0],recording_start_tow_s=tow0,max_toe_age_s=max_toe_age_s)
    _,health=ephemeris_health_selection(ephemerides,tracked_prns=tracked_prns,min_prns=min_prns)
    position=parse_preonset_nmea_position(nmea,gps_tow_at_time_zero_s=tow0,onset_s=100.0,position_window_s=(20,90))
    return {
        "classification":"alternate_common_replay_nmea_tow0_no_observables",
        "configured_fixed_tow0_s":tow0, "observables_present":False,
        "first_valid_rmc_relative_s":first_rel, "expected_first_valid_rmc_relative_s":expected_first_rmc_relative_s,
        "rmc_tolerance_s":rmc_tolerance_s, "full_gps_week":pre[0][0],
        "ephemeris_alignment":alignment, "ephemeris_health":health,
        "receiver_position_contract":position,
        "proof":"checksum-valid NMEA/RMC + user-fixed common-replay tow0; observables are unavailable and not claimed",
    }

def scenario_sources(root: Path, include_observables: bool):
    raw=root/"raw"
    paths=[root/"nmea_pvt.nmea",root/"gps_ephemeris.xml",*sorted(raw.glob("epl_tracking_ch_*.mat"))]
    if include_observables: paths.insert(2,raw/"observables.mat")
    missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise ValueError(f"missing scenario sources: {missing}")
    return paths

def cache_metadata(name, spec, preflight):
    return {"scenario":name,"evaluation":"external_frozen_inference_only","texbat_contract":EVENT_CONTRACT,
            "tow0_s":spec["tow0_s"],"sample_rate_hz":spec["sample_rate_hz"],"geometry_preflight":preflight,
            "external_evaluation_implementation":external_evaluation_implementation_manifest()}

def event_cache_roundtrip(path, events, *, source_paths, metadata, force=False):
    _validate_adapter_cache_provenance(metadata)
    path=Path(path)
    if path.exists() and not force:
        return load_event_cache(path,source_paths=source_paths,expected_metadata=metadata)[0]
    cache_events(path,events,source_paths=source_paths,metadata=metadata)
    return load_event_cache(path,source_paths=source_paths,expected_metadata=metadata)[0]

def load_scenario(name, cache_dir, *, force_cache=False):
    spec=SCENARIOS[name];root=Path(spec["root"])
    eph=parse_gnss_sdr_gps_ephemeris_xml(root/"gps_ephemeris.xml")
    rows=load_gnss_sdr_tracking_rows(root/"raw",sample_rate_hz=spec["sample_rate_hz"])
    tracked={r.prn for r in rows}
    if name=="DS4":
        preflight=preflight_ds4_alternate(root,eph,configured_tow0_s=spec["tow0_s"],tracked_prns=tracked)
        position=preflight["receiver_position_contract"]
        sources=scenario_sources(root,False)
    else:
        preflight=preflight_oakbat_geometry(root/"raw/observables.mat",root/"nmea_pvt.nmea",eph,
          configured_tow0_s=spec["tow0_s"],max_toe_age_s=EVENT_CONTRACT["max_toe_age_s"],
          tow_tolerance_s=.05,onset_s=100.,tracked_prns=tracked,min_prns=4)
        position=parse_preonset_nmea_position(root/"nmea_pvt.nmea",gps_tow_at_time_zero_s=spec["tow0_s"],onset_s=100.,position_window_s=(20,90))
        sources=scenario_sources(root,True)
    metadata=cache_metadata(name,spec,preflight)
    target=Path(cache_dir)/f"{name.lower()}.relations.npz"
    if target.exists() and not force_cache:
        events=event_cache_roundtrip(target,[],source_paths=sources,metadata=metadata)
    else:
        events=build_gcmr_pair_relation_events(rows,ephemerides=eph,receiver_ecef=position["ecef"],
          gps_tow_at_time_zero_s=spec["tow0_s"],window_s=1.,stride_s=.5,resample_bin_s=.02,
          min_common_samples=20,min_prns=4)
        events=event_cache_roundtrip(target,events,source_paths=sources,metadata=metadata,force=True)
    return events,metadata,sources,position

def region_masks(availability_s):
    t=np.asarray(availability_s,float)
    return {"acquisition":t<30.,"stable_pre":(t>=30.)&(t<90.),"transition":(t>=90.)&(t<110.),
            "stable_post":t>=110.,"post120_sensitivity":t>=120.}

def _metrics(scores, threshold, mask):
    score=np.asarray(scores["combined_score"])[mask]; time=np.asarray(scores["availability_s"])[mask]
    alarm=score>threshold
    return {"event_count":int(len(score)),"alarm_count":int(alarm.sum()),
      "alarm_rate":float(alarm.mean()) if len(score) else None,
      "score_median":float(np.median(score)) if len(score) else None,
      "score_q99":float(np.quantile(score,.99)) if len(score) else None,
      "first_alarm_score_end_s":float(time[np.flatnonzero(alarm)[0]]) if alarm.any() else None}

def summarize_scenario(name, scored, threshold, metadata, sources):
    t=np.asarray(scored["availability_s"],float); masks=region_masks(t)
    result={k:_metrics(scored,threshold,m) for k,m in masks.items()}
    post_onset=(np.asarray(scored["combined_score"])>threshold)&(t>=100.)
    first=float(t[np.flatnonzero(post_onset)[0]]) if post_onset.any() else None
    result.update({"actual_available_event_count":int(len(t)),
      "actual_available_event_range_s":[float(t.min()),float(t.max())] if len(t) else None,
      "score_availability":"window_end_s", "first_alarm_score_end_s":first,
      "first_alarm_delay_from_primary_onset_s":None if first is None else first-100.,
      "source_sha256":source_hashes(sources),"geometry_preflight":metadata["geometry_preflight"]})
    if name=="DS4": result["onset_conflict"]=ONSET_CONTRACT["ds4_auxiliary_script_conflict"]
    return result

def build_summary(checkpoint_path, loaded, scenarios_requested, results):
    checkpoint_path=Path(checkpoint_path).resolve()
    checkpoint_sha256=_sha(checkpoint_path)
    loaded_checkpoint_sha256=loaded.checkpoint_sha256
    if loaded_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("checkpoint provenance mismatch: checkpoint bytes changed after loading")
    provenance=getattr(loaded,"provenance",None)
    if not isinstance(provenance,dict) or not isinstance(provenance.get("implementation"),dict):
        raise ValueError("loaded checkpoint provenance is missing its frozen implementation manifest")
    implementation=external_evaluation_implementation_manifest(provenance["implementation"])
    return {"evaluation_classification":"external_evaluation_frozen_inference_only",
      "leakage_contract":{"training":False,"scaler_fitting":False,"calibrator_fitting":False,"threshold_fitting":False,"model_selection":False,"texbat_data_used_for_inference_only":True},
      "checkpoint":str(checkpoint_path),"checkpoint_sha256":checkpoint_sha256,
      "loaded_checkpoint_provenance":provenance,"adapter_sha256":implementation["external_adapter"]["sha256"],
      "external_evaluation_implementation":implementation,
      "frozen_threshold":FROZEN_THRESHOLD,"checkpoint_threshold":loaded.threshold,
      "threshold_equality_verified":loaded.threshold==FROZEN_THRESHOLD,"texbat_contract":EVENT_CONTRACT,
      "onset_contract":ONSET_CONTRACT,"scenarios_requested":list(scenarios_requested),"results":results}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    p.add_argument("--checkpoint",type=Path,default=CHECKPOINT);p.add_argument("--force-cache",action="store_true")
    p.add_argument("--scenarios",nargs="+",choices=sorted(SCENARIOS),default=list(SCENARIOS));p.add_argument("--device",default="cpu")
    a=p.parse_args(argv);a.output_dir.mkdir(parents=True,exist_ok=True);a.cache_dir.mkdir(parents=True,exist_ok=True)
    frozen=load_frozen_checkpoint(a.checkpoint,device=a.device)
    results={}
    for name in a.scenarios:
        events,metadata,sources,_=load_scenario(name,a.cache_dir,force_cache=a.force_cache)
        scored=score_events(frozen.model,events,frozen.calibrator,device=a.device)
        save_score_csv(a.output_dir/f"{name.lower()}_scores.csv",scored,FROZEN_THRESHOLD)
        results[name]=summarize_scenario(name,scored,FROZEN_THRESHOLD,metadata,sources)
    summary=build_summary(a.checkpoint,frozen,a.scenarios,results)
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output_dir":str(a.output_dir),"results":results},indent=2,default=str));return 0
if __name__=="__main__": raise SystemExit(main())
