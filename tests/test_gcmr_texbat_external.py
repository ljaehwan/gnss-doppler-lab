import ast
import importlib.util
import copy
import hashlib
import json
from dataclasses import dataclass, FrozenInstanceError
from datetime import date
from pathlib import Path
import numpy as np
import pytest
from gnss_doppler_lab.gcmr_geometry import GpsEphemeris
from gnss_doppler_lab.gcmr_relations import GcmrPairRelationEvent

SCRIPT=Path(__file__).parents[1]/"scripts/run_gcmr_texbat_external.py"
spec=importlib.util.spec_from_file_location("gcmr_texbat_external",SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def ck(body):
    value=0
    for c in body:value^=ord(c)
    return f"${body}*{value:02X}"

def eph(prn,week):
    return GpsEphemeris(prn,.2,4e-9,.01,5153.7,.7,.94,-.3,-8e-9,1e-10,1e-6,2e-6,200.,-80.,3e-8,-2e-8,475200.,week%1024,SV_health=0)

def ds4_fixture(tmp_path,bad_checksum=False,first="124500.18"):
    root=tmp_path; (root/"raw").mkdir(parents=True)
    lines=[ck(f"GPRMC,{first},A,3017.25,N,09744.14,W,0,0,300432,,,A")]
    for seconds in ("124502.00","124530.00","124610.00"):
        lines += [ck(f"GPRMC,{seconds},A,3017.25,N,09744.14,W,0,0,300432,,,A"),ck(f"GPGGA,{seconds},3017.25,N,09744.14,W,1,08,1.0,174.0,M,0.0,M,,")]
    if bad_checksum: lines=[lines[0][:-2]+"00"]
    (root/"nmea_pvt.nmea").write_text("\n".join(lines)+"\n")
    week,_=module.experiment._gps_week_and_tow(date(2032,4,30),(12,45,0.18))
    ep={p:eph(p,week) for p in range(1,5)}
    return root,ep

def event(start=0.):
    return GcmrPairRelationEvent(start,start+1,np.array([[1,2]]),np.ones((1,10)),np.ones((1,10),bool),np.ones((1,8)))

def test_import_and_contract_are_frozen():
    assert module.TEXBAT_CONTRACT_VERSION=="gcmr-texbat-external-v1"
    assert module.EVENT_CONTRACT["window_interval"]=="[start,end)"
    assert module.EVENT_CONTRACT["score_available_at"]=="window_end_s"
    assert module.FROZEN_THRESHOLD==2.8224338538365963

def test_adapter_has_no_fit_or_training_api_calls():
    tree=ast.parse(SCRIPT.read_text()); called=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            called.append(node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else "")
    assert not ({"fit","fit_scaler","train_clean_model","calibration_threshold","save_checkpoint"}&set(called))

def test_frozen_loader_gate_uses_validated_loader_and_fixed_threshold(monkeypatch,tmp_path):
    @dataclass(frozen=True)
    class Loaded:
        threshold: float = module.FROZEN_THRESHOLD
        model: object = "frozen-model"
        calibrator: object = "frozen-calibrator"
        provenance: object = None

    core_loaded=Loaded()
    with pytest.raises(FrozenInstanceError):
        core_loaded.checkpoint_sha256="forbidden"
    payload={"provenance":{"implementation":{"files":[],"aggregate_sha256":__import__("hashlib").sha256().hexdigest()}}}
    monkeypatch.setattr(module.torch,"load",lambda *a,**k:payload)
    seen=[];monkeypatch.setattr(module.experiment,"load_checkpoint",lambda *a,**k:seen.append((a,k)) or core_loaded)
    checkpoint=tmp_path/"model.pt";checkpoint.write_bytes(b"checkpoint fixture")
    got=module.load_frozen_checkpoint(checkpoint)
    assert got.threshold==module.FROZEN_THRESHOLD and seen
    assert got.checkpoint_sha256==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert got.core is core_loaded
    assert not hasattr(core_loaded,"checkpoint_sha256")
    assert got.model=="frozen-model" and got.calibrator=="frozen-calibrator"
    with pytest.raises(AttributeError):
        got.checkpoint_sha256="cannot mutate wrapper"
    class Bad: threshold=1.
    monkeypatch.setattr(module.experiment,"load_checkpoint",lambda *a,**k:Bad())
    with pytest.raises(ValueError,match="threshold"):module.load_frozen_checkpoint(checkpoint)

@pytest.mark.skipif(not module.CHECKPOINT.is_file(),reason="real frozen checkpoint is not available on this host")
def test_real_frozen_checkpoint_load_smoke():
    loaded=module.load_frozen_checkpoint(module.CHECKPOINT)
    assert isinstance(loaded,module.FrozenCheckpoint)
    assert loaded.core is not loaded
    assert loaded.threshold==module.FROZEN_THRESHOLD
    assert loaded.checkpoint_sha256==module._sha(module.CHECKPOINT)
    assert isinstance(loaded.provenance,dict)
    assert isinstance(loaded.provenance.get("implementation"),dict)
    summary=module.build_summary(module.CHECKPOINT,loaded,["DS4"],{})
    assert summary["checkpoint_sha256"]==loaded.checkpoint_sha256
    assert summary["loaded_checkpoint_provenance"]==loaded.provenance
    assert summary["threshold_equality_verified"] is True


def test_ds4_alternate_preflight_passes_without_observables(tmp_path):
    root,ep=ds4_fixture(tmp_path)
    report=module.preflight_ds4_alternate(root,ep,configured_tow0_s=477900.,tracked_prns=ep)
    assert report["classification"]=="alternate_common_replay_nmea_tow0_no_observables"
    assert report["observables_present"] is False
    assert report["first_valid_rmc_relative_s"]==pytest.approx(18.18)
    assert report["receiver_position_contract"]["timing"]["position_window_s"]==[20.,90.]

def test_ds4_alternate_preflight_fails_closed(tmp_path):
    root,ep=ds4_fixture(tmp_path,bad_checksum=True)
    with pytest.raises(ValueError,match="checksum-valid"):module.preflight_ds4_alternate(root,ep,configured_tow0_s=477900.,tracked_prns=ep)
    root,ep=ds4_fixture(tmp_path/"other",first="124510.18")
    with pytest.raises(ValueError,match="common replay"):module.preflight_ds4_alternate(root,ep,configured_tow0_s=477900.,tracked_prns=ep)
    (root/"raw/observables.mat").write_bytes(b"not allowed")
    with pytest.raises(ValueError,match="absent"):module.preflight_ds4_alternate(root,ep,configured_tow0_s=477900.,tracked_prns=ep)

def test_onset_region_boundaries_and_no_primary_contract_change():
    t=np.array([29.999,30.,89.999,90.,109.999,110.,119.999,120.])
    m=module.region_masks(t)
    assert np.flatnonzero(m["acquisition"]).tolist()==[0]
    assert np.flatnonzero(m["stable_pre"]).tolist()==[1,2]
    assert np.flatnonzero(m["transition"]).tolist()==[3,4]
    assert np.flatnonzero(m["stable_post"]).tolist()==[5,6,7]
    assert np.flatnonzero(m["post120_sensitivity"]).tolist()==[7]
    assert module.ONSET_CONTRACT["primary_nominal_onset_s"]==100.

def test_cache_roundtrip_and_stale_source_rejection(tmp_path):
    src=tmp_path/"source";src.write_bytes(b"v1");p=tmp_path/"cache.npz"
    metadata={"texbat_contract":module.EVENT_CONTRACT,"scenario":"DSX",
              "external_evaluation_implementation":module.external_evaluation_implementation_manifest()}
    got=module.event_cache_roundtrip(p,[event()],source_paths=[src],metadata=metadata,force=True)
    assert len(got)==1 and got[0].window_end_s==1.
    with np.load(p,allow_pickle=False) as z:meta=json.loads(str(z["metadata_json"]))
    assert meta["texbat_contract"]["version"]==module.TEXBAT_CONTRACT_VERSION
    src.write_bytes(b"v2")
    with pytest.raises(ValueError,match="stale"):module.event_cache_roundtrip(p,[],source_paths=[src],metadata=metadata)

def test_score_availability_fixed_threshold_and_external_summary_shape():
    scored={"availability_s":np.array([29.,89.,90.,100.,110.,120.]),"combined_score":np.array([0.,3.,0.,3.,0.,3.])}
    meta={"geometry_preflight":{"classification":"fixture"}};src=[]
    out=module.summarize_scenario("DS4",scored,module.FROZEN_THRESHOLD,meta,src)
    assert out["score_availability"]=="window_end_s"
    assert out["first_alarm_score_end_s"]==100. and out["first_alarm_delay_from_primary_onset_s"]==0.
    assert out["actual_available_event_range_s"]==[29.,120.]
    assert out["post120_sensitivity"]["alarm_count"]==1
    assert out["onset_conflict"]["auxiliary_onset_s"]==110.


def test_adapter_manifest_hashes_runtime_bytes_without_embedded_expected(tmp_path):
    frozen_core={"files":[],"aggregate_sha256":hashlib.sha256().hexdigest()}
    manifest=module.external_evaluation_implementation_manifest(frozen_core)
    runtime_digest=hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert manifest["external_adapter"]["sha256"]==runtime_digest
    assert runtime_digest not in SCRIPT.read_text()
    assert manifest["frozen_core"]==frozen_core
    assert "runtime" in manifest["adapter_hash_contract"]
    copied=tmp_path/"adapter.py";copied.write_bytes(SCRIPT.read_bytes()+b"\n# changed copy\n")
    changed=module.external_evaluation_implementation_manifest(frozen_core,adapter_path=copied)
    assert changed["external_adapter"]["sha256"]==hashlib.sha256(copied.read_bytes()).hexdigest()
    assert changed["external_adapter"]["sha256"]!=manifest["external_adapter"]["sha256"]


def test_cache_reuse_rejects_changed_or_missing_adapter_provenance(tmp_path):
    src=tmp_path/"source";src.write_bytes(b"v1");p=tmp_path/"cache.npz"
    metadata={"scenario":"DSX","external_evaluation_implementation":module.external_evaluation_implementation_manifest()}
    module.event_cache_roundtrip(p,[event()],source_paths=[src],metadata=metadata,force=True)
    changed=copy.deepcopy(metadata)
    changed["external_evaluation_implementation"]["external_adapter"]["sha256"]="0"*64
    with pytest.raises(ValueError,match="adapter provenance"):
        module.event_cache_roundtrip(p,[],source_paths=[src],metadata=changed)
    missing=copy.deepcopy(metadata);missing.pop("external_evaluation_implementation")
    with pytest.raises(ValueError,match="adapter provenance"):
        module.event_cache_roundtrip(p,[],source_paths=[src],metadata=missing)


def test_summary_explicitly_marks_external_evaluation_and_no_fitting(tmp_path):
    frozen_core={"files":[],"aggregate_sha256":hashlib.sha256().hexdigest()}
    loaded_provenance={"implementation":frozen_core,"training_run":"oakbat-fixture"}
    class Loaded:
        threshold=module.FROZEN_THRESHOLD
        provenance=loaded_provenance
    checkpoint=tmp_path/"model.pt";checkpoint.write_bytes(b"checkpoint fixture bytes")
    loaded=module.FrozenCheckpoint(Loaded(),hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    summary=module.build_summary(checkpoint,loaded,["DS4"],{})
    assert summary["evaluation_classification"]=="external_evaluation_frozen_inference_only"
    assert summary["threshold_equality_verified"] is True
    assert summary["checkpoint_sha256"]==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert summary["loaded_checkpoint_provenance"]==loaded_provenance
    implementation=summary["external_evaluation_implementation"]
    assert implementation["frozen_core"]==frozen_core
    assert implementation["external_adapter"]["sha256"]==hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert summary["adapter_sha256"]==implementation["external_adapter"]["sha256"]
    leakage=summary["leakage_contract"]
    assert leakage["texbat_data_used_for_inference_only"] is True
    assert all(leakage[k] is False for k in ("training","scaler_fitting","calibrator_fitting","threshold_fitting","model_selection"))
