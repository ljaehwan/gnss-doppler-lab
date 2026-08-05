import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r2c_gnss_stage0_fix"
sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_fix import ScoreResult,run_full_controls


def verifier():
    name = "correction6_verifier"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/verify_r2c_gnss_stage0_fix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_unavailable_fail_closed_campaign_has_no_semantic_errors():
    errors = verifier().verify(ARTIFACT, require_committed=False, recompute=False, allow_pending=True)
    assert errors == []


def test_dynamic_metric_roster_rejects_required_deletion_and_unsupported_fabrication():
    module = verifier()
    scores = read_csv(ARTIFACT / "per_epoch_scores.csv")
    config = json.loads((ARTIFACT / "config.json").read_text())
    expected = module.expected_scenario_metric_keys(scores, config)
    actual = read_csv(ARTIFACT / "scenario_metrics.csv")
    assert {(r["scenario"], r["detector"], r["threshold"]) for r in actual} == expected
    required = next(key for key in expected if key[0] == "cleanStatic")
    deleted = [r for r in actual if (r["scenario"], r["detector"], r["threshold"]) != required]
    assert "scenario_metrics row roster/cardinality mismatch" in module.validate_metric_roster(deleted, expected)
    unsupported = ("DS1", "A4", "q99")
    assert unsupported not in expected
    fabricated = actual + [{"scenario": unsupported[0], "detector": unsupported[1], "threshold": unsupported[2]}]
    assert "scenario_metrics row roster/cardinality mismatch" in module.validate_metric_roster(fabricated, expected)


CONTROL_GATE_MAP = {
    "gain_invariance.json": ("gain_invariance",),
    "phase_invariance.json": ("phase_invariance",),
    "noise_control.json": ("noise_gain_alarms",),
    "multipath_control.json": ("shortcut_controls",),
    "second_source_injection.json": ("complex_second_source",),
    "relation_destruction.json": ("relation_destruction", "geometry_removal"),
}


@pytest.mark.parametrize("filename,gates", CONTROL_GATE_MAP.items())
def test_not_evaluated_control_requires_empty_rows_and_matching_unavailable_gates(filename, gates):
    module = verifier()
    control = {"status": "NOT_EVALUATED", "threshold": float("inf"), "rows": []}
    decision = {gate: {"status": "UNAVAILABLE_CONTROL_SUPPORT", "rows": 0} for gate in gates}
    assert module.validate_control(filename, control, decision, synthetic=False) == []
    bad_rows = {**control, "rows": [{"kind": "fabricated"}]}
    assert module.validate_control(filename, bad_rows, decision, synthetic=False)
    bad_gate = {gate: dict(value) for gate, value in decision.items()}
    bad_gate[gates[0]]["rows"] = 1
    assert module.validate_control(filename, control, bad_gate, synthetic=False)


def realistic_control(*,seed=20260803):
    permutation=[2,3,1];rng=np.random.default_rng(seed)
    for _ in range(6):rng.normal(size=30*9)
    rng.uniform(-math.pi,math.pi,len(permutation))
    specs=[]
    specs += [("gain",{"gain":x}) for x in (.5,.75,1.,1.5,2.)]
    specs += [("slow_agc",{"minimum":.75,"maximum":1.25})]
    specs += [("global_phase",{"radians":x}) for x in (0.,math.pi/4,math.pi/2,math.pi)]
    specs += [("awgn",{"relative_sigma":.05}),("cn0_degradation",{"db":6.}),
              ("matched_power_noise",{"relative_power":1.}),("quantization",{"bits":8}),
              ("non_shared_multipath",{"per_prn":True})]
    for delay in (-.35,.35):
        for ratio in (.1,.25,.5,1.):specs.append(("second_source_injection",{"delay_chips":delay,"power_ratio":ratio,"phase_rad":float(rng.uniform(-math.pi,math.pi))}))
    specs.append(("relation_destruction",{"permutation":permutation}))
    rows=[]
    for index,(kind,parameters) in enumerate(specs):
        post=-1. if kind=="non_shared_multipath" else .5 if kind=="relation_destruction" else 2.+index/100
        rows.append({"kind":kind,"parameters":parameters,"seed":seed,"threshold":0.,"support_count":30,
          "baseline_status":"AVAILABLE","perturbed_status":"AVAILABLE","baseline_reason":None,"perturbed_reason":None,
          "pre_score":1.,"post_score":post,"effect_size":post-1.,"pre_alarm":True,"post_alarm":post>0})
    return {"schema":"gnss-doppler-lab.r2c-full-controls.v2","seed":seed,"threshold":0.,"support_count":30,
      "rows":rows,"computed_rows":24,"status":"PASS","baseline_status":"AVAILABLE","baseline_score":1.,
      "baseline_reason":None,"criteria":{"invariance_alarm_agreement":1.}}


def producer_control(*,unavailable=False,threshold=0.):
    observations={p:np.full((2,9),p,dtype=complex) for p in (1,2,3)}
    los={1:np.array([1.,0.,0.]),2:np.array([0.,1.,0.]),3:np.array([0.,0.,1.])}
    calls=0
    def scorer(value,pairing,**kwargs):
        nonlocal calls
        calls+=1
        if unavailable and calls==2:return ScoreResult("UNAVAILABLE_INVALID_SCORE",None,"nonfinite")
        return ScoreResult("AVAILABLE",1.+calls/100)
    return run_full_controls(scorer,observations,los,threshold,seed=20260803)


def expected_gates(control,minimum=.95):
    rows=control["rows"]
    def selected(kinds):return [r for r in rows if r["kind"] in kinds]
    def gate(kinds,value):return {"status":"PASS" if value else "FAIL","rows":len(selected(kinds))}
    gain=selected({"gain","slow_agc"});phase=selected({"global_phase"});noise=selected({"awgn","cn0_degradation","matched_power_noise","quantization"})
    relation=selected({"relation_destruction"});second=selected({"second_source_injection"});shortcut=selected({"non_shared_multipath"})
    agreement=lambda values:sum(r["pre_alarm"]==r["post_alarm"] for r in values)/len(values)>=minimum
    relation_gate=gate({"relation_destruction"},all(r["post_score"]<r["pre_score"] for r in relation))
    return {"gain_invariance":gate({"gain","slow_agc"},agreement(gain)),"phase_invariance":gate({"global_phase"},agreement(phase)),
      "noise_gain_alarms":gate({"awgn","cn0_degradation","matched_power_noise","quantization"},agreement(noise)),
      "relation_destruction":relation_gate,"geometry_removal":dict(relation_gate),
      "complex_second_source":gate({"second_source_injection"},sum(r["post_alarm"] for r in second if r["parameters"]["power_ratio"]>=.5)/4>=.5),
      "shortcut_controls":gate({"non_shared_multipath"},not any(r["post_alarm"] for r in shortcut))}


@pytest.mark.parametrize("filename", CONTROL_GATE_MAP)
def test_actual_full_control_payload_is_required_in_every_file(filename):
    module=verifier();control=realistic_control();gates=expected_gates(control)
    assert module.validate_control(filename,control,gates,synthetic=False,config={"controls":{"alarm_agreement_minimum":.95}})==[]


def test_payload_generated_by_frozen_producer_has_exact_rng_stream():
    module=verifier();control=producer_control();gates=expected_gates(control)
    assert module.validate_control_payload(control,{"seed":20260803},control_prns={1,2,3})==[]
    assert module.validate_control("second_source_injection.json",control,gates,config={"seed":20260803,"controls":{"alarm_agreement_minimum":.95}},thresholds={"detectors":{"Full":{"status":"AVAILABLE","q99":0.}}})==[]
    changed=json.loads(json.dumps(control));next(r for r in changed["rows"] if r["kind"]=="second_source_injection")["parameters"]["phase_rad"]+=.01
    assert module.validate_control_payload(changed,{"seed":20260803},control_prns={1,2,3})
    changed=json.loads(json.dumps(control));next(r for r in changed["rows"] if r["kind"]=="relation_destruction")["parameters"]["permutation"]=[3,2,1]
    assert module.validate_control_payload(changed,{"seed":20260803},control_prns={1,2,3})


def test_producer_unavailable_row_retains_only_permitted_baseline_score():
    module=verifier();control=producer_control(unavailable=True);row=control["rows"][0]
    assert row["pre_score"] is not None and row["post_score"] is row["effect_size"] is row["pre_alarm"] is row["post_alarm"] is None
    assert module.validate_control_payload(control,{"seed":20260803},control_prns={1,2,3})==[]
    for field,value in (("post_score",2.),("effect_size",1.),("pre_alarm",True),("perturbed_reason",None)):
        changed=json.loads(json.dumps(control));changed["rows"][0][field]=value
        assert module.validate_control_payload(changed,{"seed":20260803},control_prns={1,2,3})


def test_control_threshold_is_bound_to_verified_full_q99():
    module=verifier();config={"seed":20260803,"controls":{"alarm_agreement_minimum":.95}}
    for control_threshold,verified_threshold in ((0.,1.),(1.,0.)):
        control=producer_control(threshold=control_threshold);gates=expected_gates(control)
        assert module.validate_control("gain_invariance.json",control,gates,config=config,thresholds={"detectors":{"Full":{"status":"AVAILABLE","q99":verified_threshold}}})
    unavailable={"status":"NOT_EVALUATED","threshold":float("inf"),"rows":[]};gate={"gain_invariance":{"status":"UNAVAILABLE_CONTROL_SUPPORT","rows":0}}
    full_unavailable={"detectors":{"Full":{"status":"UNAVAILABLE_CALIBRATION_SUPPORT","q99":None}}}
    assert module.validate_control("gain_invariance.json",unavailable,gate,config=config,thresholds=full_unavailable)==[]
    control=producer_control();assert module.validate_control("gain_invariance.json",control,expected_gates(control),config=config,thresholds=full_unavailable)


def test_json_boolean_is_not_a_control_prn():
    module=verifier();control=producer_control();relation=next(r for r in control["rows"] if r["kind"]=="relation_destruction")
    relation["parameters"]["permutation"]=[2,True,1]
    assert module.validate_control_payload(control,{"seed":20260803},control_prns={1,2,3})


@pytest.mark.parametrize("location", ["gain","phase","relative_power","criteria","seed","support_count","computed_rows","row_seed","row_support"])
def test_control_contract_rejects_boolean_numeric_substitution(location):
    module=verifier();control=producer_control()
    if location=="gain":next(r for r in control["rows"] if r["kind"]=="gain" and r["parameters"]["gain"]==1.)["parameters"]["gain"]=True
    elif location=="phase":next(r for r in control["rows"] if r["kind"]=="global_phase")["parameters"]["radians"]=False
    elif location=="relative_power":next(r for r in control["rows"] if r["kind"]=="matched_power_noise")["parameters"]["relative_power"]=True
    elif location=="criteria":control["criteria"]["invariance_alarm_agreement"]=True
    elif location=="seed":control["seed"]=True
    elif location=="support_count":control["support_count"]=True
    elif location=="computed_rows":control["computed_rows"]=True
    elif location=="row_seed":control["rows"][0]["seed"]=True
    else:control["rows"][0]["support_count"]=True
    assert module.validate_control_payload(control,{"seed":20260803},control_prns={1,2,3})


def test_full_threshold_record_exact_available_and_unavailable_shapes():
    module=verifier();names=("q99","q99.5","target_fpr_1pct")
    available={"status":"AVAILABLE","q99":1.,"q99.5":2.,"target_fpr_1pct":1.}
    unavailable={"status":"UNAVAILABLE_CALIBRATION_SUPPORT","q99":None,"q99.5":None,"target_fpr_1pct":None}
    assert module.validate_full_threshold_aliases({**available,"detectors":{"Full":available}})==[]
    assert module.validate_full_threshold_aliases({**{name:float("inf") for name in names},"detectors":{"Full":unavailable}})==[]
    bad=[]
    for missing in names:bad.append({**{name:float("inf") for name in names},"detectors":{"Full":{k:v for k,v in unavailable.items() if k!=missing}}})
    bad += [{**{name:float("inf") for name in names},"detectors":{"Full":{**unavailable,"fabricated":1}}},
      {**available,"detectors":{"Full":{**available,"status":"MISSING"}}},
      {**available,"detectors":{"Full":{**available,"q99":True}}},
      {**available,"detectors":{"Full":{**available,"q99":"1"}}}]
    assert all(module.validate_full_threshold_aliases(item) for item in bad)


def test_complete_threshold_contract_rejects_non_full_and_top_level_mutations():
    module=verifier();scores=read_csv(ARTIFACT/"per_epoch_scores.csv");thresholds=json.loads((ARTIFACT/"thresholds.json").read_text())
    assert module.validate_threshold_contract(thresholds,scores)==[]
    mutations=[]
    for key,value in (("schema","wrong"),("method","linear"),("comparison","greater_equal")):mutations.append({**thresholds,key:value})
    for detector in ("A1","A3","Power-only"):
        changed=json.loads(json.dumps(thresholds));changed["detectors"][detector]["status"]="AVAILABLE" if changed["detectors"][detector]["status"]!="AVAILABLE" else "UNAVAILABLE_CALIBRATION_SUPPORT";mutations.append(changed)
        changed=json.loads(json.dumps(thresholds));changed["detectors"][detector]["fabricated"]=1;mutations.append(changed)
    changed=json.loads(json.dumps(thresholds));changed["detectors"]["A1"]["q99"]+=1.;mutations.append(changed)
    assert all(module.validate_threshold_contract(item,scores) for item in mutations)


def test_top_level_verify_rejects_non_full_threshold_mutation(tmp_path):
    module=verifier();target=tmp_path/"artifact";shutil.copytree(ARTIFACT,target)
    thresholds=json.loads((target/"thresholds.json").read_text());thresholds["detectors"]["A1"]["q99"]+=1.;(target/"thresholds.json").write_text(json.dumps(thresholds))
    hashes=json.loads((target/"hashes.json").read_text());hashes["files"]={str(p.relative_to(target)):module.sha(p) for p in target.rglob("*") if p.is_file() and p.name!="hashes.json"};(target/"hashes.json").write_text(json.dumps(hashes))
    errors=module.verify(target,require_committed=False,recompute=False,allow_pending=True)
    assert "threshold detector contract mismatch: A1" in errors


def test_control_candidate_context_is_derived_not_self_authenticated():
    module=verifier();scores=[{"scenario":"cleanStatic","time_bin":"8","detector":"Full","status":"AVAILABLE","score":"1","epoch_count":"6","prn_count":"3"}]
    geometry={"scenarios":{"cleanStatic":{"los_by_bin":{"8":{"1":[1,0,0],"2":[0,1,0],"3":[0,0,1]}}}}}
    assert module.derive_control_candidate_context(scores,geometry)==({1,2,3},6)
    control=producer_control();gates=expected_gates(control);config={"seed":20260803,"controls":{"alarm_agreement_minimum":.95}}
    assert module.validate_control("gain_invariance.json",control,gates,config=config,control_prns={1,2,3},expected_support_count=6)==[]
    for prns,support in (({7,8,9},6),({1,2,3},7)):
        assert module.validate_control("gain_invariance.json",control,gates,config=config,control_prns=prns,expected_support_count=support)


def test_frozen_producer_infinite_threshold_and_all_unavailable_paths():
    module=verifier();config={"seed":20260803,"controls":{"alarm_agreement_minimum":.95}}
    infinite=producer_control(threshold=float("inf"));gates=expected_gates(infinite)
    unavailable_thresholds={"detectors":{"Full":{"status":"UNAVAILABLE_CALIBRATION_SUPPORT","q99":None,"q99.5":None,"target_fpr_1pct":None}}}
    assert module.validate_control("gain_invariance.json",infinite,gates,config=config,thresholds=unavailable_thresholds,control_prns={1,2,3},expected_support_count=6)==[]
    available_thresholds={"detectors":{"Full":{"status":"AVAILABLE","q99":0.,"q99.5":1.,"target_fpr_1pct":0.}}}
    assert module.validate_control("gain_invariance.json",infinite,gates,config=config,thresholds=available_thresholds,control_prns={1,2,3},expected_support_count=6)
    observations={p:np.full((2,9),p,dtype=complex) for p in (1,2,3)};los={p:np.eye(3)[p-1] for p in observations}
    scorer=lambda *a,**k:ScoreResult("UNAVAILABLE_INVALID_SCORE",None,"nonfinite")
    all_unavailable=run_full_controls(scorer,observations,los,float("inf"));unavailable_gates={}
    for filename,gate_names in CONTROL_GATE_MAP.items():
        count=len([r for r in all_unavailable["rows"] if r["kind"] in module.CONTROL_FILES[filename]])
        unavailable_gates.update({name:{"status":"UNAVAILABLE_CONTROL_SUPPORT","rows":count} for name in gate_names})
    assert module.validate_control("gain_invariance.json",all_unavailable,unavailable_gates,config=config,thresholds=unavailable_thresholds,control_prns={1,2,3},expected_support_count=6)==[]


@pytest.mark.parametrize("kind", ["gain","slow_agc","global_phase","awgn","cn0_degradation","matched_power_noise","quantization","non_shared_multipath","second_source_injection","relation_destruction"])
def test_control_exact_support_rejects_selective_row_deletion(kind):
    module=verifier();control=realistic_control();control["rows"].remove(next(r for r in control["rows"] if r["kind"]==kind))
    assert module.validate_control_payload(control)


@pytest.mark.parametrize("mutation", ["duplicate","extra","parameter","status","nonfinite"])
def test_control_payload_rejects_malformed_evidence(mutation):
    module=verifier();control=realistic_control()
    if mutation=="duplicate":control["rows"].append(dict(control["rows"][0]))
    elif mutation=="extra":control["rows"].append({**control["rows"][0],"kind":"unknown"})
    elif mutation=="parameter":control["rows"][0]["parameters"]={"gain":.6}
    elif mutation=="status":control["rows"][0]["perturbed_status"]="MYSTERY"
    else:control["rows"][0]["post_score"]=float("nan")
    assert module.validate_control_payload(control)


@pytest.mark.parametrize("gate", ["gain_invariance","phase_invariance","noise_gain_alarms","complex_second_source","shortcut_controls","relation_destruction","geometry_removal"])
def test_control_gate_recomputation_rejects_pass_fail_counterfactuals(gate):
    module=verifier();control=realistic_control();gates=expected_gates(control);gates[gate]["status"]="FAIL"
    filename=next(name for name,names in CONTROL_GATE_MAP.items() if gate in names)
    assert module.validate_control(filename,control,gates,synthetic=False,config={"controls":{"alarm_agreement_minimum":.95}})
    # Make the underlying predicate fail, then lie in the opposite direction.
    control=realistic_control()
    if gate in {"gain_invariance","phase_invariance","noise_gain_alarms"}:
        kinds=module.CONTROL_FILES[filename];row=next(r for r in control["rows"] if r["kind"] in kinds)
        row["post_alarm"]=False;row["post_score"]=-1.;row["effect_size"]=-2.
        if gate in {"gain_invariance","phase_invariance"}:
            control["status"]="FAIL";control["criteria"]={"invariance_alarm_agreement":.9}
    elif gate in {"relation_destruction","geometry_removal"}:
        row=next(r for r in control["rows"] if r["kind"]=="relation_destruction");row["post_score"]=2.;row["effect_size"]=1.
    elif gate=="complex_second_source":
        for r in control["rows"]:
            if r["kind"]=="second_source_injection" and r["parameters"]["power_ratio"]>=.5:r["post_alarm"]=False;r["post_score"]=-1.;r["effect_size"]=-2.
    else:
        row=next(r for r in control["rows"] if r["kind"]=="non_shared_multipath");row["post_alarm"]=True;row["post_score"]=2.;row["effect_size"]=1.
    gates=expected_gates(control);gates[gate]["status"]="PASS"
    assert module.validate_control(filename,control,gates,synthetic=False,config={"controls":{"alarm_agreement_minimum":.95}})


def test_full_alias_available_and_unavailable_contracts():
    module = verifier()
    available = {"status": "AVAILABLE", "q99": 1.0, "q99.5": 2.0, "target_fpr_1pct": 1.0}
    assert module.validate_full_threshold_aliases({**available, "detectors": {"Full": available}}) == []
    unavailable = {"status": "UNAVAILABLE_CALIBRATION_SUPPORT", "q99": None, "q99.5": None, "target_fpr_1pct": None}
    aliases = {"q99": float("inf"), "q99.5": float("inf"), "target_fpr_1pct": float("inf"), "detectors": {"Full": unavailable}}
    assert module.validate_full_threshold_aliases(aliases) == []
    for bad in (
        {**aliases, "q99": 1.0},
        {**aliases, "q99": None},
        {**aliases, "detectors": {"Full": {**unavailable, "q99": 1.0}}},
        {**aliases, "detectors": {"Full": {**available, "status": "UNAVAILABLE_CALIBRATION_SUPPORT"}}},
    ):
        assert module.validate_full_threshold_aliases(bad)


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def commit(repo, message):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", message], cwd=repo, check=True, stdout=subprocess.PIPE)
    return git(repo, "rev-parse", "HEAD")


def graph_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "artifacts/r2c_gnss_stage0_fix").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "tests").mkdir()
    (repo / "source.txt").write_text("source\n")
    source = commit(repo, "source")
    return repo, source


def artifact_commit(repo):
    (repo / "artifacts/r2c_gnss_stage0_fix/result.json").write_text("{}\n")
    return commit(repo, "artifact")


def test_commit_graph_source_to_artifact_and_verifier_only_pass(tmp_path):
    module = verifier()
    repo, source = graph_repo(tmp_path)
    artifact = artifact_commit(repo)
    assert module.verify_artifact_commit(repo, repo / "artifacts/r2c_gnss_stage0_fix", source) == []
    (repo / "scripts/verify_r2c_gnss_stage0_fix.py").write_text("# maintenance\n")
    commit(repo, "verifier")
    assert module.verify_artifact_commit(repo, repo / "artifacts/r2c_gnss_stage0_fix", source) == []
    assert artifact != git(repo, "rev-parse", "HEAD")


def test_commit_graph_rejects_post_artifact_producer_or_config_change(tmp_path):
    module = verifier()
    repo, source = graph_repo(tmp_path)
    artifact_commit(repo)
    (repo / "configs").mkdir()
    (repo / "configs/config.json").write_text("{}\n")
    commit(repo, "config")
    assert module.verify_artifact_commit(repo, repo / "artifacts/r2c_gnss_stage0_fix", source)


def test_commit_graph_rejects_forbidden_change_even_when_later_restored(tmp_path):
    module=verifier();repo,source=graph_repo(tmp_path);artifact_commit(repo)
    (repo/"source.txt").write_text("laundered\n");commit(repo,"forbidden")
    (repo/"source.txt").write_text("source\n");commit(repo,"restore")
    assert module.verify_artifact_commit(repo,repo/"artifacts/r2c_gnss_stage0_fix",source)


def test_restore_chain_rejected_with_git_replacement_ref(tmp_path):
    module=verifier();repo,source=graph_repo(tmp_path);artifact=artifact_commit(repo)
    (repo/"source.txt").write_text("laundered\n");commit(repo,"forbidden")
    (repo/"source.txt").write_text("source\n");commit(repo,"restore")
    subprocess.run(["git","replace","HEAD",artifact],cwd=repo,check=True)
    assert module.verify_artifact_commit(repo,repo/"artifacts/r2c_gnss_stage0_fix",source)


def test_git_grafts_redirecting_config_and_environment_are_rejected(tmp_path,monkeypatch):
    module=verifier();repo,source=graph_repo(tmp_path);artifact_commit(repo)
    grafts=repo/".git/info/grafts";grafts.write_text(f"{git(repo,'rev-parse','HEAD')} {source}\n")
    assert module.verify_artifact_commit(repo,repo/"artifacts/r2c_gnss_stage0_fix",source)
    grafts.unlink();included=repo/"redirect.conf";included.write_text("[core]\nfilemode = false\n")
    subprocess.run(["git","config","--local","include.path",str(included)],cwd=repo,check=True)
    assert module.verify_artifact_commit(repo,repo/"artifacts/r2c_gnss_stage0_fix",source)
    subprocess.run(["git","config","--local","--unset","include.path"],cwd=repo,check=True)
    monkeypatch.setenv("GIT_DIR",str(repo/".git"))
    assert module.verify_artifact_commit(repo,repo/"artifacts/r2c_gnss_stage0_fix",source)


@pytest.mark.parametrize("attack", ["replace","graft","environment","include","includeif"])
def test_top_level_verify_rejects_git_resolution_before_artifact_or_recompute(tmp_path,monkeypatch,attack):
    module=verifier();repo,source=graph_repo(tmp_path);artifact=artifact_commit(repo)
    if attack=="replace":subprocess.run(["git","replace","HEAD",source],cwd=repo,check=True)
    elif attack=="graft":(repo/".git/info/grafts").write_text(f"{artifact} {source}\n")
    elif attack=="environment":monkeypatch.setenv("GIT_WORK_TREE",str(tmp_path))
    else:
        included=repo/"redirect.conf";included.write_text("[core]\nfilemode=false\n")
        key="include.path" if attack=="include" else "includeif.onbranch:main.path"
        subprocess.run(["git","config","--local",key,str(included)],cwd=repo,check=True)
    monkeypatch.setattr(module,"external_recompute",lambda *a,**k:pytest.fail("recompute reached"))
    errors=module.verify(tmp_path/"missing-artifact",repo=repo,require_committed=False,recompute=True,allow_pending=True)
    assert any("Git" in error for error in errors)


def test_finalize_main_rejects_redirect_before_recompute_or_write(tmp_path,monkeypatch,capsys):
    module=verifier();repo,_=graph_repo(tmp_path);artifact=repo/"artifacts/r2c_gnss_stage0_fix"
    monkeypatch.setattr(module,"ROOT",repo);monkeypatch.setenv("GIT_DIR",str(repo/".git"))
    monkeypatch.setattr(module,"external_recompute",lambda *a,**k:pytest.fail("finalize recompute reached"))
    monkeypatch.setattr(sys,"argv",["verify",str(artifact),"--uncommitted-test","--finalize"])
    with pytest.raises(SystemExit) as caught:module.main()
    assert caught.value.code==1 and "Git redirect environment is forbidden" in capsys.readouterr().out


def finalize_repo(tmp_path):
    repo,source=graph_repo(tmp_path);artifact=repo/"artifacts/r2c_gnss_stage0_fix"
    (repo/".gitignore").write_text("artifacts/*\n");commit(repo,"ignore artifact");source=git(repo,"rev-parse","HEAD")
    subprocess.run(["git","config","user.name","Test"],cwd=repo,check=True);subprocess.run(["git","config","user.email","test@example.com"],cwd=repo,check=True)
    (artifact/"verification.json").write_text('{"status":"PENDING_EXTERNAL_VERIFIER"}\n');(artifact/"hashes.json").write_text("{}\n");(artifact/"result.txt").write_text("result\n")
    return repo,artifact,source,git(repo,"branch","--show-current")


def test_finalize_transaction_force_stages_ignored_artifact_only(tmp_path):
    module=verifier();repo,artifact,source,branch=finalize_repo(tmp_path)
    result=module.finalize_transaction(artifact,repo,source,branch,lambda:([],"a"*64))
    assert result["errors"]==[] and git(repo,"diff-tree","--no-commit-id","--name-only","-r","HEAD").splitlines()==[
      "artifacts/r2c_gnss_stage0_fix/hashes.json","artifacts/r2c_gnss_stage0_fix/result.txt","artifacts/r2c_gnss_stage0_fix/verification.json"]
    assert git(repo,"status","--porcelain")==""


def test_finalize_rejects_prestaged_and_unsafe_config_hook_filter(tmp_path):
    module=verifier();repo,artifact,source,branch=finalize_repo(tmp_path);(repo/"source.txt").write_text("staged\n");subprocess.run(["git","add","source.txt"],cwd=repo,check=True)
    assert module.finalize_transaction(artifact,repo,source,branch,lambda:([],"a"*64))["errors"]
    subprocess.run(["git","reset","--","source.txt"],cwd=repo,check=True);(repo/"source.txt").write_text("source\n")
    hooks=tmp_path/"hooks";hooks.mkdir();marker=tmp_path/"hook-ran";(hooks/"commit-msg").write_text(f"#!/bin/sh\ntouch {marker}\n");(hooks/"commit-msg").chmod(0o755)
    subprocess.run(["git","config","--local","core.hooksPath",str(hooks)],cwd=repo,check=True)
    assert module.finalize_transaction(artifact,repo,source,branch,lambda:([],"a"*64))["errors"] and not marker.exists()
    subprocess.run(["git","config","--local","--unset","core.hooksPath"],cwd=repo,check=True);subprocess.run(["git","config","--local","filter.bad.clean","evil"],cwd=repo,check=True)
    assert module.finalize_transaction(artifact,repo,source,branch,lambda:([],"a"*64))["errors"]


def test_finalize_rejects_global_include_hook_and_system_redirect(tmp_path,monkeypatch):
    module=verifier();repo,artifact,source,branch=finalize_repo(tmp_path);home=tmp_path/"home";home.mkdir();hooks=tmp_path/"global-hooks";hooks.mkdir()
    included=tmp_path/"included.conf";included.write_text(f"[core]\n\thooksPath = {hooks}\n")
    (home/".gitconfig").write_text(f"[include]\n\tpath = {included}\n")
    monkeypatch.setenv("HOME",str(home))
    assert module.finalize_transaction(artifact,repo,source,branch,lambda:([],"a"*64))["errors"]
    monkeypatch.setenv("GIT_CONFIG_SYSTEM",str(included))
    assert module.finalize_transaction(artifact,repo,source,branch,lambda:([],"a"*64))["errors"]


@pytest.mark.parametrize("failure",["add","commit","postcheck"])
def test_finalize_transaction_rolls_back_files_index_and_head(tmp_path,monkeypatch,failure):
    module=verifier();repo,artifact,source,branch=finalize_repo(tmp_path);before={name:(artifact/name).read_bytes() for name in ("verification.json","hashes.json")};real=module._git_mutate
    if failure in {"add","commit"}:
        def injected(repo,*args,**kwargs):
            if args and args[0]==failure:raise subprocess.CalledProcessError(1,["git",*args])
            return real(repo,*args,**kwargs)
        monkeypatch.setattr(module,"_git_mutate",injected)
    else:monkeypatch.setattr(module,"_tree_blobs",lambda *a,**k:{})
    result=module.finalize_transaction(artifact,repo,source,branch,lambda:([],"a"*64))
    assert result["errors"] and git(repo,"rev-parse","HEAD")==source and git(repo,"diff","--cached","--name-only")==""
    assert {name:(artifact/name).read_bytes() for name in before}==before


def test_repository_layout_rejects_decoy_gitfile_and_commondir(tmp_path):
    module=verifier();repo,_,_,_=finalize_repo(tmp_path)
    assert module.validate_repository_layout(repo)==[]
    decoy=tmp_path/"decoy";decoy.mkdir();(decoy/".git").write_text(f"gitdir: {repo/'.git'}\n")
    assert module.validate_repository_layout(decoy)


def test_commit_graph_checks_every_merge_parent_edge(tmp_path):
    module=verifier();repo,source=graph_repo(tmp_path);artifact_commit(repo)
    branch=git(repo,"branch","--show-current")
    subprocess.run(["git","checkout","-q","-b","side"],cwd=repo,check=True)
    (repo/"source.txt").write_text("forbidden side\n");commit(repo,"side forbidden")
    subprocess.run(["git","checkout","-q",branch],cwd=repo,check=True)
    (repo/"scripts/verify_r2c_gnss_stage0_fix.py").write_text("main\n");commit(repo,"main allowed")
    # The merge tree keeps main's clean source, so the net artifact..HEAD diff is allowlisted;
    # the edge from the forbidden side parent changes source.txt and must still fail.
    subprocess.run(["git","-c","user.name=Test","-c","user.email=test@example.com","merge","--no-ff","-s","ours","side","-m","merge"],cwd=repo,check=True,stdout=subprocess.PIPE)
    assert module.verify_artifact_commit(repo,repo/"artifacts/r2c_gnss_stage0_fix",source)


def test_commit_graph_allows_multi_commit_verifier_maintenance_chain(tmp_path):
    module=verifier();repo,source=graph_repo(tmp_path);artifact_commit(repo)
    (repo/"scripts/verify_r2c_gnss_stage0_fix.py").write_text("one\n");commit(repo,"verifier one")
    (repo/"tests/test_r2c_stage0_correction6.py").write_text("two\n");commit(repo,"verifier two")
    assert module.verify_artifact_commit(repo,repo/"artifacts/r2c_gnss_stage0_fix",source)==[]


def test_commit_graph_rejects_wrong_parent_and_non_artifact_diff(tmp_path):
    module = verifier()
    repo, source = graph_repo(tmp_path)
    (repo / "source.txt").write_text("intermediate\n")
    commit(repo, "intermediate")
    artifact_commit(repo)
    assert module.verify_artifact_commit(repo, repo / "artifacts/r2c_gnss_stage0_fix", source)
    repo2, source2 = graph_repo(tmp_path / "second")
    (repo2 / "source.txt").write_text("changed with artifact\n")
    artifact_commit(repo2)
    assert module.verify_artifact_commit(repo2, repo2 / "artifacts/r2c_gnss_stage0_fix", source2)


def test_commit_graph_rejects_artifact_tree_change_after_artifact_commit(tmp_path):
    module = verifier()
    repo, source = graph_repo(tmp_path)
    artifact_commit(repo)
    (repo / "artifacts/r2c_gnss_stage0_fix/result.json").write_text('{"changed": true}\n')
    commit(repo, "changed artifact")
    assert module.verify_artifact_commit(repo, repo / "artifacts/r2c_gnss_stage0_fix", source)
