#!/usr/bin/env python3
"""Independent semantic verifier for R2C Stage-0-fix artifacts."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,subprocess,sys,tempfile
from pathlib import Path,PurePosixPath
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_artifact import (FIX_SOURCE_FILES,PRESERVED_TREE,TOP_LEVEL_DIRECTORIES,
 TOP_LEVEL_FILES,expected_hash_keys)
REQUIRED_DETECTORS={"A1","A2","A3","A4","Full","Neural-with-energy","Power-only"}
ALL_DETECTORS={"B0-native","A1","A2","A3","A4","Full","Neural-with-energy","Power-only","Noise-floor-only"}
SCENARIO_ORDER=("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8");SCENARIOS=set(SCENARIO_ORDER)
CONTROL_FILES={"gain_invariance.json":{"gain","slow_agc"},"phase_invariance.json":{"global_phase"},
 "noise_control.json":{"awgn","cn0_degradation","matched_power_noise","quantization"},
 "multipath_control.json":{"non_shared_multipath"},"second_source_injection.json":{"second_source_injection"},
 "relation_destruction.json":{"relation_destruction"}}
CONTROL_GATES={"gain_invariance.json":("gain_invariance",),"phase_invariance.json":("phase_invariance",),
 "noise_control.json":("noise_gain_alarms",),"multipath_control.json":("shortcut_controls",),
 "second_source_injection.json":("complex_second_source",),
 "relation_destruction.json":("relation_destruction","geometry_removal")}
VERIFIER_MAINTENANCE_FILES={"scripts/verify_r2c_gnss_stage0_fix.py","tests/test_r2c_stage0_correction6.py"}
LIKELIHOOD_DETECTORS={"A1","A3","A4","Full","Neural-with-energy"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def git_blob_sha1(path):
    data=path.read_bytes();return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def load(path):return json.loads(path.read_text())
GIT_REDIRECT_ENV={"GIT_DIR","GIT_WORK_TREE","GIT_INDEX_FILE","GIT_OBJECT_DIRECTORY","GIT_ALTERNATE_OBJECT_DIRECTORIES",
 "GIT_COMMON_DIR","GIT_REPLACE_REF_BASE","GIT_CONFIG","GIT_CONFIG_GLOBAL","GIT_CONFIG_SYSTEM","GIT_CONFIG_COUNT"}
def safe_git_env():
    env={key:value for key,value in os.environ.items() if not key.startswith("GIT_")};env.update({"GIT_NO_REPLACE_OBJECTS":"1","GIT_CONFIG_GLOBAL":os.devnull,"GIT_CONFIG_SYSTEM":os.devnull});return env
def git_inspection_env():
    env={key:value for key,value in os.environ.items() if not key.startswith("GIT_")};env["GIT_NO_REPLACE_OBJECTS"]="1";return env
def git(repo,*args):return subprocess.check_output(["git",*args],cwd=repo,text=True,env=safe_git_env()).strip()
def verify_file_record(record,label,errors):
    try:path=Path(record["path"])
    except (KeyError,TypeError):errors.append(f"{label} path/hash missing");return
    if not path.is_file() or sha(path)!=record.get("sha256"):errors.append(f"{label} path/hash mismatch")
def manifest_iq_hash(doc):return (doc.get("source",{}).get("iq_sha256") or doc.get("source",{}).get("sha256") or doc.get("authenticated_inputs",{}).get("iq_after_receiver",{}).get("sha256"))

def expected_scenario_metric_keys(score_rows,config):
    """Reproduce the producer's predeclared metric-support rules from score evidence."""
    expected=set();onsets=config.get("attacks",{}).get("onset_s",{})
    for scenario in SCENARIOS:
        onset=onsets.get(scenario)
        for detector in ALL_DETECTORS:
            support=[r for r in score_rows if r.get("scenario")==scenario and r.get("detector")==detector and r.get("score") not in (None,"")]
            required=not support or onset is None
            if support and onset is not None:
                pre=any(float(r["availability_time_s"])<onset-20 for r in support)
                attack=any(float(r["availability_time_s"])>=onset+40 for r in support)
                required=pre and attack
            if required:
                expected.update((scenario,detector,name) for name in ("q99","q99.5"))
    return expected

def validate_metric_roster(metric_rows,expected):
    keys=[(r.get("scenario"),r.get("detector"),r.get("threshold")) for r in metric_rows]
    return [] if set(keys)==expected and len(keys)==len(set(keys)) else ["scenario_metrics row roster/cardinality mismatch"]

def validate_full_threshold_aliases(thresholds):
    errors=[];full=thresholds.get("detectors",{}).get("Full",{});names=("q99","q99.5","target_fpr_1pct");fields={"status",*names}
    if not isinstance(full,dict) or set(full)!=fields:return ["Full threshold record shape mismatch"]
    if full.get("status")=="AVAILABLE":
        if any(type(full.get(name)) is not float or not np.isfinite(full[name]) for name in names):errors.append("available Full threshold value mismatch")
        if any(not strict_equal(thresholds.get(name),full.get(name)) for name in names):errors.append("top-level Full threshold mismatch")
    elif full.get("status")=="UNAVAILABLE_CALIBRATION_SUPPORT":
        if any(full.get(name) is not None for name in names):errors.append("unavailable Full threshold retained calibration values")
        if any(type(thresholds.get(name)) is not float or not np.isposinf(thresholds.get(name)) for name in names):errors.append("top-level Full threshold mismatch")
    else:errors.append("Full threshold availability status mismatch")
    return errors

def validate_threshold_contract(thresholds,score_rows):
    errors=[];top_fields={"schema","q99","q99.5","target_fpr_1pct","detectors","method","comparison"}
    if not isinstance(thresholds,dict) or set(thresholds)!=top_fields:errors.append("threshold top-level contract mismatch")
    if thresholds.get("schema")!="gnss-doppler-lab.r2c-thresholds.v2" or thresholds.get("method")!="higher" or thresholds.get("comparison")!="strict_greater":errors.append("threshold method/schema contract mismatch")
    detectors=thresholds.get("detectors",{})
    if not isinstance(detectors,dict) or set(detectors)!=ALL_DETECTORS:errors.append("threshold detector roster mismatch");return errors
    fields={"status","q99","q99.5","target_fpr_1pct"}
    for detector in ALL_DETECTORS:
        values=[float(row["score"]) for row in score_rows if row.get("scenario")=="cleanStatic" and row.get("detector")==detector and row.get("score") not in (None,"") and 320<=float(row["availability_time_s"])<=400]
        if values:
            q99=float(np.quantile(np.asarray(values),.99,method="higher"));q995=float(np.quantile(np.asarray(values),.995,method="higher"))
            expected={"status":"AVAILABLE","q99":q99,"q99.5":q995,"target_fpr_1pct":q99}
        else:expected={"status":"UNAVAILABLE_CALIBRATION_SUPPORT","q99":None,"q99.5":None,"target_fpr_1pct":None}
        item=detectors.get(detector)
        if not isinstance(item,dict) or set(item)!=fields or not strict_equal(item,expected):errors.append(f"threshold detector contract mismatch: {detector}")
    errors.extend(validate_full_threshold_aliases(thresholds));return errors

def derive_control_candidate_context(score_rows,geometry):
    candidates=[];los_bins=geometry.get("scenarios",{}).get("cleanStatic",{}).get("los_by_bin",{})
    for row in score_rows:
        if row.get("scenario")!="cleanStatic" or row.get("detector")!="Full" or row.get("status")!="AVAILABLE" or row.get("score") in (None,""):continue
        if str(row.get("geometry_valid","")).lower() not in {"true","1"} and "geometry_valid" in row:continue
        los=los_bins.get(str(int(row["time_bin"])),{});prns=set()
        try:prns={int(value) for value in los}
        except (TypeError,ValueError):continue
        try:support=int(row["epoch_count"]);count=int(row["prn_count"])
        except (TypeError,ValueError,KeyError):continue
        if prns and len(prns)==count and support>0:candidates.append((prns,support))
    return candidates[-1] if candidates else (None,None)

CONTROL_STATUSES={"AVAILABLE","UNAVAILABLE_INVALID_SHARED_FIT","UNAVAILABLE_INVALID_SCORE"}
CONTROL_ROW_FIELDS={"kind","parameters","seed","threshold","support_count","baseline_status","perturbed_status",
 "baseline_reason","perturbed_reason","pre_score","post_score","effect_size","pre_alarm","post_alarm"}
CONTROL_DOCUMENT_FIELDS={"schema","seed","threshold","support_count","rows","computed_rows","status",
 "baseline_status","baseline_score","baseline_reason","criteria"}

def _finite_number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value)
def strict_equal(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return set(actual)==set(expected) and all(strict_equal(actual[key],value) for key,value in expected.items())
    if isinstance(expected,(list,tuple)):return len(actual)==len(expected) and all(strict_equal(left,right) for left,right in zip(actual,expected))
    return actual==expected

def expected_control_parameters(seed,permutation,support_count=0,tap_count=9):
    expected=[("gain",{"gain":x}) for x in (.5,.75,1.,1.5,2.)]
    expected += [("slow_agc",{"minimum":.75,"maximum":1.25})]
    expected += [("global_phase",{"radians":x}) for x in (0.,math.pi/4,math.pi/2,math.pi)]
    expected += [("awgn",{"relative_sigma":.05}),("cn0_degradation",{"db":6.}),
      ("matched_power_noise",{"relative_power":1.}),("quantization",{"bits":8}),
      ("non_shared_multipath",{"per_prn":True})]
    rng=np.random.default_rng(seed)
    for _ in range(6):rng.normal(size=support_count*tap_count)
    rng.uniform(-math.pi,math.pi,len(permutation))
    for delay in (-.35,.35):
        for ratio in (.1,.25,.5,1.):expected.append(("second_source_injection",{"delay_chips":delay,"power_ratio":ratio,"phase_rad":float(rng.uniform(-math.pi,math.pi))}))
    expected.append(("relation_destruction",{"permutation":permutation}))
    return expected

def validate_control_payload(control,config=None,control_prns=None,expected_support_count=None):
    errors=[];rows=control.get("rows")
    if set(control)!=CONTROL_DOCUMENT_FIELDS or control.get("schema")!="gnss-doppler-lab.r2c-full-controls.v2" or not isinstance(rows,list):return ["control document schema mismatch"]
    seed=control.get("seed");expected_seed=(config or {}).get("seed",20260803)
    if type(seed) is not int or type(expected_seed) is not int or seed!=expected_seed:errors.append("control seed mismatch")
    relation=[r for r in rows if r.get("kind")=="relation_destruction"]
    permutation=relation[0].get("parameters",{}).get("permutation") if len(relation)==1 else None
    if not isinstance(permutation,list) or not permutation or any(type(x) is not int for x in permutation) or len(set(permutation))!=len(permutation) or permutation!=sorted(permutation)[1:]+sorted(permutation)[:1]:
        errors.append("control relation permutation mismatch");permutation=[]
    if control_prns is not None and (any(type(x) is not int for x in control_prns) or set(permutation)!=set(control_prns)):errors.append("control PRN roster mismatch")
    tap_count=len((config or {}).get("tap_offsets_chips",range(9)))
    expected=expected_control_parameters(seed,permutation,control.get("support_count",0),tap_count) if type(seed) is int else []
    actual=[(r.get("kind"),r.get("parameters")) for r in rows]
    if not strict_equal(actual,expected):errors.append("control row parameter roster/multiplicity mismatch")
    if type(control.get("computed_rows")) is not int or control.get("computed_rows")!=24 or len(rows)!=24:errors.append("control row count mismatch")
    threshold=control.get("threshold");support=control.get("support_count");baseline_status=control.get("baseline_status")
    if type(threshold) is not float or np.isnan(threshold) or np.isneginf(threshold) or type(support) is not int or support<=0:errors.append("control support/threshold mismatch")
    if expected_support_count is not None and (type(expected_support_count) is not int or support!=expected_support_count):errors.append("control candidate support mismatch")
    if baseline_status not in CONTROL_STATUSES:errors.append("unknown control status")
    baseline_available=baseline_status=="AVAILABLE";baseline_score=control.get("baseline_score")
    if baseline_available!=(type(baseline_score) is float and np.isfinite(baseline_score)):errors.append("control baseline evidence mismatch")
    if (baseline_available and control.get("baseline_reason") is not None) or (not baseline_available and not isinstance(control.get("baseline_reason"),str)):errors.append("control baseline reason mismatch")
    agreements=[]
    for row in rows:
        if set(row)!=CONTROL_ROW_FIELDS:errors.append("control row schema mismatch");continue
        if type(row.get("seed")) is not int or type(row.get("threshold")) is not float or type(row.get("support_count")) is not int or row.get("seed")!=seed or row.get("threshold")!=threshold or row.get("support_count")!=support:errors.append("control row shared evidence mismatch")
        before=row.get("baseline_status");after=row.get("perturbed_status")
        if before not in CONTROL_STATUSES or after not in CONTROL_STATUSES:errors.append("unknown control status");continue
        if before!=baseline_status or row.get("baseline_reason")!=control.get("baseline_reason"):errors.append("control baseline consistency mismatch")
        if (after=="AVAILABLE" and row.get("perturbed_reason") is not None) or (after!="AVAILABLE" and not isinstance(row.get("perturbed_reason"),str)):errors.append("control perturbed reason mismatch")
        baseline_row_available=before=="AVAILABLE";perturbed_available=after=="AVAILABLE";both=baseline_row_available and perturbed_available
        numeric=(row.get("pre_score"),row.get("post_score"),row.get("effect_size"))
        valid_float=lambda value:type(value) is float and np.isfinite(value)
        if baseline_row_available!=valid_float(row.get("pre_score")) or perturbed_available!=valid_float(row.get("post_score")):errors.append("control available-score evidence mismatch")
        if baseline_row_available and row.get("pre_score")!=baseline_score:errors.append("control baseline score mismatch")
        if both:
            if not valid_float(row.get("effect_size")) or row["effect_size"]!=row["post_score"]-row["pre_score"]:errors.append("control score/effect mismatch")
            if type(row.get("pre_alarm")) is not bool or type(row.get("post_alarm")) is not bool or row["pre_alarm"]!=(row["pre_score"]>threshold) or row["post_alarm"]!=(row["post_score"]>threshold):errors.append("control alarm mismatch")
        elif row.get("effect_size") is not None or row.get("pre_alarm") is not None or row.get("post_alarm") is not None:errors.append("unavailable control retained derived evidence")
        if row.get("kind") in {"gain","slow_agc","global_phase"}:agreements.append(both and row.get("pre_alarm")==row.get("post_alarm"))
    agreement=sum(agreements)/len(agreements) if agreements else None
    expected_status="PASS" if agreements and all(agreements) else "FAIL"
    if control.get("status")!=expected_status or not strict_equal(control.get("criteria"),{"invariance_alarm_agreement":agreement}):errors.append("control summary mismatch")
    return errors

def _expected_control_gate(gate_name,rows,minimum):
    kinds=next(CONTROL_FILES[name] for name,names in CONTROL_GATES.items() if gate_name in names)
    selected=[r for r in rows if r.get("kind") in kinds]
    available=[r for r in selected if r.get("baseline_status")==r.get("perturbed_status")=="AVAILABLE"]
    if not selected or len(available)!=len(selected):return {"status":"UNAVAILABLE_CONTROL_SUPPORT","rows":len(selected)}
    if gate_name in {"gain_invariance","phase_invariance","noise_gain_alarms"}:passed=np.mean([r["pre_alarm"]==r["post_alarm"] for r in available])>=minimum
    elif gate_name in {"relation_destruction","geometry_removal"}:passed=all(r["post_score"]<r["pre_score"] for r in available)
    elif gate_name=="complex_second_source":passed=np.mean([r["post_alarm"] for r in available if r["parameters"]["power_ratio"]>=.5])>=.5
    else:passed=not any(r["post_alarm"] for r in available)
    return {"status":"PASS" if passed else "FAIL","rows":len(selected)}

def validate_control(filename,control,gates,synthetic=False,config=None,thresholds=None,control_prns=None,expected_support_count=None):
    errors=[];rows=control.get("rows",[]);status=control.get("status")
    if thresholds is not None and not synthetic:
        full=thresholds.get("detectors",{}).get("Full",{});full_status=full.get("status");q99=full.get("q99")
        if full_status=="AVAILABLE":
            if not _finite_number(q99) or status=="NOT_EVALUATED" or control.get("threshold")!=q99:errors.append("control threshold binding mismatch")
        elif full_status=="UNAVAILABLE_CALIBRATION_SUPPORT":
            if q99 is not None or type(control.get("threshold")) is not float or not np.isposinf(control.get("threshold")):errors.append("unavailable control threshold binding mismatch")
        else:errors.append("control threshold availability mismatch")
    if status=="NOT_EVALUATED":
        unavailable_threshold=control.get("threshold")
        if set(control)!={"status","threshold","rows"} or rows or not isinstance(unavailable_threshold,(int,float)) or not np.isposinf(unavailable_threshold):errors.append(f"NOT_EVALUATED control evidence mismatch: {filename}")
        for gate_name in CONTROL_GATES[filename]:
            gate=gates.get(gate_name,{})
            if gate!={"status":"UNAVAILABLE_CONTROL_SUPPORT","rows":0}:errors.append(f"unavailable control gate mismatch: {gate_name}")
        return errors
    if synthetic:
        if not CONTROL_FILES[filename]<={r.get("kind") for r in rows}:errors.append(f"missing perturbation: {filename}")
    else:errors.extend(validate_control_payload(control,config,control_prns,expected_support_count))
    if not synthetic:
        minimum=(config or {}).get("controls",{}).get("alarm_agreement_minimum")
        if not _finite_number(minimum) or not 0<=minimum<=1:errors.append("control alarm agreement threshold invalid");minimum=np.nan
        for gate_name in CONTROL_GATES[filename]:
            gate=gates.get(gate_name,{})
            if gate!=_expected_control_gate(gate_name,rows,minimum):errors.append(f"control gate recomputation mismatch: {gate_name}")
    finite_posts=[r["post_score"] for r in rows if r.get("perturbed_status")=="AVAILABLE" and type(r.get("post_score")) is float and np.isfinite(r["post_score"])]
    if not synthetic and finite_posts and len(set(finite_posts))==1:errors.append("constant control scores")
    return errors

def _tree_blobs(repo,commit,path):
    prefix=f"{path}/";lines=git(repo,"ls-tree","-r",commit,path).splitlines()
    return {line.split("\t",1)[1].split(prefix,1)[1]:line.split()[2] for line in lines if line}

def git_resolution_security_errors(repo):
    errors=[]
    dangerous=[key for key in os.environ if key in GIT_REDIRECT_ENV or key.startswith("GIT_CONFIG_KEY_") or key.startswith("GIT_CONFIG_VALUE_")]
    if dangerous:errors.append("Git redirect environment is forbidden")
    try:
        if git(repo,"for-each-ref","--format=%(refname)","refs/replace").splitlines():errors.append("Git replacement refs are forbidden")
        paths={git(repo,"rev-parse","--git-dir"),git(repo,"rev-parse","--git-common-dir")}
        for value in paths:
            directory=Path(value);directory=(Path(repo)/directory).resolve() if not directory.is_absolute() else directory.resolve()
            for relative in ("info/grafts","objects/info/alternates"):
                candidate=directory/relative
                if candidate.is_file() and candidate.read_text().strip():errors.append(f"Git {relative} mechanism is forbidden")
        config=subprocess.run(["git","config","--no-includes","--show-origin","--get-regexp",".*"],cwd=repo,text=True,capture_output=True,env=git_inspection_env())
        for line in config.stdout.splitlines():
            parts=line.split(None,2);key=parts[1] if len(parts)>1 else "";value=parts[2] if len(parts)>2 else "";normalized=key.lower()
            unsafe=normalized.startswith(("include.","includeif.","filter.")) or normalized in {"core.worktree","core.gitdir","extensions.worktreeconfig","core.hookspath","core.attributesfile","core.autocrlf","core.eol","core.safecrlf"}
            if unsafe or (normalized=="core.bare" and value.lower()=="true"):errors.append(f"Git unsafe config is forbidden: {key}")
    except (subprocess.CalledProcessError,OSError):errors.append("Git resolution security inspection failed")
    return errors

TRUSTED_COMMON_GIT=(ROOT.parent/"gnss-doppler-lab"/".git").resolve()
def validate_repository_layout(repo):
    errors=[];repo=Path(repo).resolve()
    try:
        if Path(git(repo,"rev-parse","--show-toplevel")).resolve()!=repo:errors.append("repository top-level layout mismatch")
        gitdir_value=Path(git(repo,"rev-parse","--git-dir"));common_value=Path(git(repo,"rev-parse","--git-common-dir"))
        gitdir=(repo/gitdir_value).resolve() if not gitdir_value.is_absolute() else gitdir_value.resolve();common=(repo/common_value).resolve() if not common_value.is_absolute() else common_value.resolve()
        marker=repo/".git"
        if marker.is_dir():
            if gitdir!=marker.resolve() or common!=marker.resolve():errors.append("standalone repository layout mismatch")
        elif marker.is_file():
            declared=marker.read_text().strip()
            if not declared.startswith("gitdir: ") or Path(declared[8:]).resolve()!=gitdir:errors.append("linked worktree gitfile mismatch")
            if common!=TRUSTED_COMMON_GIT or gitdir.parent!=common/"worktrees":errors.append("untrusted linked worktree common repository")
            commondir=gitdir/"commondir";reciprocal=gitdir/"gitdir"
            if not commondir.is_file() or (gitdir/commondir.read_text().strip()).resolve()!=common:errors.append("linked worktree commondir mismatch")
            if not reciprocal.is_file() or Path(reciprocal.read_text().strip()).resolve()!=marker.resolve():errors.append("linked worktree reciprocal gitfile mismatch")
        else:errors.append("repository .git layout missing")
    except (subprocess.CalledProcessError,OSError,ValueError):errors.append("repository layout inspection failed")
    return errors

def _git_mutate(repo,*args,check=True):
    command=["git","-c",f"core.hooksPath={os.devnull}","-c","core.autocrlf=false","-c","core.safecrlf=false","-c",f"core.attributesFile={os.devnull}",*args]
    return subprocess.run(command,cwd=repo,env=safe_git_env(),text=True,capture_output=True,check=check)

def finalize_transaction(artifact,repo,source,expected_branch,recompute_fn):
    artifact=Path(artifact).resolve();repo=Path(repo).resolve();relative="artifacts/r2c_gnss_stage0_fix";errors=[]
    errors.extend(git_resolution_security_errors(repo));errors.extend(validate_repository_layout(repo))
    try:
        start_head=git(repo,"rev-parse","HEAD")
        if start_head!=source or git(repo,"branch","--show-current")!=expected_branch:errors.append("finalize branch/HEAD mismatch")
        if git(repo,"diff","--cached","--name-only"):errors.append("finalize requires empty index")
        if git(repo,"diff","--name-only"):errors.append("finalize requires clean tracked worktree")
        if git(repo,"ls-files","--others","--exclude-standard"):errors.append("unexpected untracked files before finalize")
    except subprocess.CalledProcessError:errors.append("finalize precondition Git failure");start_head=None
    if errors:return {"errors":sorted(set(errors)),"head":start_head}
    originals={name:(artifact/name).read_bytes() for name in ("verification.json","hashes.json")}
    committed=False
    try:
        regenerated,evidence=recompute_fn()
        if regenerated:raise RuntimeError("; ".join(regenerated))
        (artifact/"verification.json").write_text(json.dumps({"schema":"gnss-doppler-lab.r2c-production-verification.v1","status":"PASS","deterministic_external_recomputation":True,"recomputation_evidence_sha256":evidence},indent=2,sort_keys=True)+"\n")
        files={str(path.relative_to(artifact)):sha(path) for path in artifact.rglob("*") if path.is_file() and path.name!="hashes.json"}
        (artifact/"hashes.json").write_text(json.dumps({"algorithm":"sha256","policy":"all files recursively except hashes.json","files":files},indent=2,sort_keys=True)+"\n")
        _git_mutate(repo,"add","-f","--",relative)
        staged=git(repo,"diff","--cached","--name-only").splitlines();disk={f"{relative}/{path.relative_to(artifact)}":git_blob_sha1(path) for path in artifact.rglob("*") if path.is_file()}
        entries={};
        for line in git(repo,"ls-files","--stage","--",relative).splitlines():
            meta,path=line.split("\t",1);mode,blob,_=meta.split();entries[path]=(mode,blob)
        if set(staged)!=set(disk) or set(entries)!=set(disk) or any(not path.startswith(relative+"/") or entries[path]!=("100644",blob) for path,blob in disk.items()):raise RuntimeError("staged artifact index contract mismatch")
        _git_mutate(repo,"commit","-m","Add externally verified R2C Stage-0 fix campaign artifact");committed=True;new_head=git(repo,"rev-parse","HEAD")
        if git(repo,"show","-s","--format=%P",new_head).split()!=[source]:raise RuntimeError("finalize commit parent mismatch")
        changed=git(repo,"diff-tree","--no-commit-id","--name-only","-r",new_head).splitlines()
        if set(changed)!=set(disk) or any(not path.startswith(relative+"/") for path in changed):raise RuntimeError("finalize commit is not exact artifact-only")
        if _tree_blobs(repo,new_head,relative)!={path.split(relative+"/",1)[1]:blob for path,blob in disk.items()}:raise RuntimeError("finalize committed bytes mismatch")
        verification=load(artifact/"verification.json")
        if verification.get("status")!="PASS" or verification.get("recomputation_evidence_sha256")!=evidence or git(repo,"status","--porcelain"):raise RuntimeError("finalize post-check mismatch")
        return {"errors":[],"head":new_head,"evidence":evidence}
    except Exception as exc:
        if committed:
            current=git(repo,"rev-parse","HEAD");_git_mutate(repo,"update-ref","HEAD",start_head,current)
        _git_mutate(repo,"reset","-q","HEAD","--",relative,check=False)
        for name,data in originals.items():(artifact/name).write_bytes(data)
        return {"errors":[f"finalize transaction failed: {exc}"],"head":start_head}

def verify_artifact_commit(repo,artifact,source):
    errors=git_resolution_security_errors(repo);path="artifacts/r2c_gnss_stage0_fix"
    try:
        artifact_commit=git(repo,"rev-list","-1","HEAD","--",path)
        if not artifact_commit: return ["artifact commit not found"]
        if git(repo,"show","-s","--format=%P",artifact_commit).split()!=[source]:errors.append("artifact commit parent is not frozen source commit")
        changed=git(repo,"diff-tree","--no-commit-id","--name-only","-r",artifact_commit).splitlines()
        if any(not name.startswith(path+"/") for name in changed):errors.append("artifact commit diff is not artifact-only")
        committed=_tree_blobs(repo,artifact_commit,path);head=_tree_blobs(repo,"HEAD",path)
        if committed!=head:errors.append("artifact tree changed after artifact commit")
        disk={str(p.relative_to(artifact)):git_blob_sha1(p) for p in artifact.rglob("*") if p.is_file()}
        if committed!=disk:errors.append("on-disk artifact differs from committed tree")
        descendants=git(repo,"rev-list",f"{artifact_commit}..HEAD").splitlines()
        for commit in descendants:
            parents=git(repo,"show","-s","--format=%P",commit).split()
            if not parents:errors.append("post-artifact root commit is forbidden");continue
            for parent in parents:
                edge=git(repo,"diff-tree","--no-commit-id","--name-only","-r",parent,commit).splitlines()
                if any(name not in VERIFIER_MAINTENANCE_FILES for name in edge):errors.append(f"post-artifact commit is not verifier-only: {commit}")
    except subprocess.CalledProcessError:errors.append("git artifact commit verification failed")
    return errors

def external_recompute(artifact,provenance,source,repo):
    """Regenerate using the frozen source and external provenance, never artifact evidence."""
    with tempfile.TemporaryDirectory(prefix="r2c-verify-") as temporary:
        base=Path(temporary);worktree=base/"source";output=base/"regenerated"
        subprocess.run(["git","worktree","add","--detach",str(worktree),source],cwd=repo,env=safe_git_env(),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            external=provenance["external_inputs"]
            command=[sys.executable,str(worktree/"scripts/run_r2c_gnss_stage0_fix.py"),"--config",str(worktree/"configs/r2c_gnss_stage0_fix.json"),"--output",str(output),"--source-commit",source,"--verification-recompute"]
            by_scenario={item["scenario"]:item for item in external}
            for name in SCENARIO_ORDER:
                item=by_scenario[name];command += ["--input",f'{name}={item["selected"]["path"]}']
            command += ["--geometry",external[0]["geometry"]["path"],"--b0-validation",provenance["b0_validation"]["path"]]
            env={**safe_git_env(),"R2C_VERIFIER_RECOMPUTE":"1","R2C_ATTEMPT_ID":"verifier-recompute",
                 "R2C_ATTEMPT_DIR":str(base/"verifier-runtime"),"PYTHONHASHSEED":str(load(worktree/"configs/r2c_gnss_stage0_fix.json")["seed"])}
            subprocess.run(command,cwd=worktree,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            excluded={"hashes.json","verification.json"};mismatches=[]
            expected={str(p.relative_to(artifact)) for p in artifact.rglob("*") if p.is_file() and p.name not in excluded}
            actual={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name not in excluded}
            if expected!=actual:mismatches.append("recomputed file roster mismatch")
            for name in sorted(expected&actual):
                if (artifact/name).read_bytes()!=(output/name).read_bytes():mismatches.append(f"external recomputation mismatch: {name}")
            evidence=hashlib.sha256("\n".join(f"{name}:{sha(output/name)}" for name in sorted(actual)).encode()).hexdigest()
            return mismatches,evidence
        finally:subprocess.run(["git","worktree","remove","--force",str(worktree)],cwd=repo,env=safe_git_env(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def independent_decision(gates):
    disq=("clean_dynamic_fpr","gain_invariance","noise_gain_alarms","relation_destruction","geometry_removal","complex_second_source","shortcut_controls")
    required=("complex_provenance","time_los_alignment","geometry_coverage","clean_dynamic_fpr","gain_invariance","phase_invariance","noise_gain_alarms","relation_destruction","full_improvement","full_a2_two_scenarios","shortcut_controls")
    bad=[n for n in disq if gates.get(n,{}).get("status")=="FAIL"]
    coverage={"time_los_alignment","geometry_coverage"}
    missing=[n for n in required if gates.get(n,{}).get("status") not in {"PASS","FAIL"} or (n in coverage and gates.get(n,{}).get("status")!="PASS")]
    failed=[n for n in required if n not in coverage and gates.get(n,{}).get("status")=="FAIL"]
    core="R2C_CORE_NOT_SUPPORTED" if bad else "R2C_CORE_INCONCLUSIVE" if missing else "R2C_CORE_NOT_SUPPORTED" if failed else "R2C_CORE_SUPPORTED"
    paper=all(gates.get(n,{}).get("status")=="PASS" for n in ("b0_authentic_common_support","full_b0_comparison","empirical_wide_template","paper_gates"))
    return core,paper,"PHYSICS_SUPPORTED" if core=="R2C_CORE_SUPPORTED" and paper else core

def verify(artifact:Path,*,repo=ROOT,require_committed=True,allow_synthetic_test_artifact=False,recompute=True,allow_pending=False):
    errors=git_resolution_security_errors(repo)
    if errors:return sorted(set(errors))
    artifact=artifact.resolve()
    if not artifact.is_dir():return ["artifact missing"]
    files={p.name for p in artifact.iterdir() if p.is_file()};dirs={p.name for p in artifact.iterdir() if p.is_dir()}
    if files!=TOP_LEVEL_FILES:errors.append("exact top-level file set mismatch")
    if dirs!=TOP_LEVEL_DIRECTORIES:errors.append("exact top-level directory set mismatch")
    if any(p.is_dir() for p in artifact.rglob("*") if p.parent!=artifact):errors.append("extra nested campaign attempt")
    if not TOP_LEVEL_FILES<=files:return errors
    hashes=load(artifact/"hashes.json")
    if hashes.get("policy")!="all files recursively except hashes.json" or hashes.get("algorithm")!="sha256":errors.append("hash policy mismatch")
    manifest=hashes.get("files")
    if not isinstance(manifest,dict) or set(manifest)!=expected_hash_keys(artifact):errors.append("hash keys are not exact/nonempty")
    else:
        for key,value in manifest.items():
            pure=PurePosixPath(key)
            if pure.is_absolute() or ".." in pure.parts:errors.append("unsafe hash path")
            else:
                target=(artifact/key).resolve()
                if artifact not in target.parents or not target.is_file() or sha(target)!=value:errors.append(f"hash mismatch: {key}")
    provenance=load(artifact/"provenance.json");source=provenance.get("source_commit")
    if not isinstance(source,str) or len(source)!=40:errors.append("frozen source commit missing")
    bundle=provenance.get("source_bundle",{}).get("files",{})
    synthetic=bool(provenance.get("synthetic_test_mode"))
    production=(repo/"artifacts/r2c_gnss_stage0_fix").resolve()
    if synthetic and (artifact==production or production in artifact.parents):errors.append("synthetic artifact forbidden at production path")
    if synthetic and not allow_synthetic_test_artifact:errors.append("synthetic artifact requires explicit allow flag")
    if synthetic and (artifact==repo.resolve() or repo.resolve() in artifact.parents):errors.append("synthetic artifact must be outside repository")
    if not require_committed and source!=git(repo,"rev-parse","HEAD"):errors.append("uncommitted verification source commit must equal HEAD")
    if set(bundle)!=set(FIX_SOURCE_FILES):errors.append("source bundle file set mismatch")
    else:
        for name,value in bundle.items():
            if synthetic:blob=(repo/name).read_bytes() if (repo/name).is_file() else b""
            else:
                try:blob=subprocess.check_output(["git","show",f"{source}:{name}"],cwd=repo,env=safe_git_env())
                except subprocess.CalledProcessError:errors.append(f"source bundle blob missing: {name}");continue
            if hashlib.sha256(blob).hexdigest()!=value:errors.append(f"source bundle hash mismatch: {name}")
        expected_bundle_hash=hashlib.sha256(json.dumps(bundle,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if provenance.get("source_bundle",{}).get("bundle_sha256")!=expected_bundle_hash:errors.append("source bundle aggregate hash mismatch")
    if provenance.get("preserved_artifact_tree")!=PRESERVED_TREE:errors.append("preserved tree source constant mismatch")
    try:
        if git(repo,"rev-parse","HEAD:artifacts/r2c_gnss_stage0")!=PRESERVED_TREE:errors.append("old artifact tree changed")
        if source and subprocess.run(["git","merge-base","--is-ancestor",source,"HEAD"],cwd=repo,env=safe_git_env()).returncode:errors.append("source commit not ancestor")
        if require_committed:
            if git(repo,"status","--porcelain=v1"):errors.append("repository/worktree dirty")
            errors.extend(verify_artifact_commit(repo,artifact,source))
    except subprocess.CalledProcessError:errors.append("git ancestry verification failed")
    external=provenance.get("external_inputs",[])
    if synthetic and external:errors.append("synthetic artifact must not carry production external inputs")
    if not synthetic:
        if len(external)!=7 or {x.get("scenario") for x in external}!=SCENARIOS:errors.append("external input roster mismatch")
        for item in external:
            for role in ("selected","geometry"):
                value=item.get(role,{});path=Path(value.get("path",""))
                if not path.is_file() or sha(path)!=value.get("sha256"):errors.append("external input/checkpoint hash mismatch")
            value=item.get("b0",{})
            if "path" in value and (not Path(value["path"]).is_file() or sha(Path(value["path"]))!=value.get("sha256")):errors.append("B0 lineage hash mismatch")
        canonical=provenance.get("canonical_config",{});config_path=repo/"configs/r2c_gnss_stage0_fix.json"
        if canonical.get("path")!="configs/r2c_gnss_stage0_fix.json" or canonical.get("sha256")!=sha(config_path):errors.append("canonical config provenance mismatch")
        if (artifact/"config.json").read_bytes()!=config_path.read_bytes():errors.append("artifact config differs from canonical bytes")
        geometry=load(artifact/"time_geometry_validation.json")
        if geometry.get("schema")!="gnss-doppler-lab.r2c-strict-time-geometry.v2" or geometry.get("attack_scores_computed") is not False or set(geometry.get("scenarios",{}))!=SCENARIOS:errors.append("geometry wrapper schema/roster mismatch")
        external_by_name={x.get("scenario"):x for x in external}
        for name,item in geometry.get("scenarios",{}).items():
            if item.get("scenario")!=name:errors.append("geometry internal scenario mismatch");continue
            selected=item.get("lineage",{}).get("selected",{});expected=external_by_name.get(name,{}).get("selected",{})
            if selected.get("path")!=expected.get("path") or selected.get("sha256")!=expected.get("sha256"):errors.append("geometry selected lineage mismatch")
            lineage=item.get("lineage",{});receiver_iq=lineage.get("receiver_source_iq_sha256");export_iq=lineage.get("export_source_iq_sha256")
            valid=lambda x:isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)
            binding=lineage.get("source_iq_binding_status")
            if binding=="HASH_BOUND" and (not valid(receiver_iq) or not valid(export_iq) or receiver_iq!=export_iq):errors.append("geometry source-IQ binding invalid")
            elif binding!="HASH_BOUND" and binding!="LINEAGE_GAP":errors.append("geometry source-IQ lineage status invalid")
            for role in ("rinex","observables","ephemeris","nmea","selected"):
                verify_file_record(lineage.get(role),f"geometry {name} {role}",errors)
            if binding=="HASH_BOUND":verify_file_record(lineage.get("receiver_manifest"),f"geometry {name} receiver_manifest",errors)
            verify_file_record({"path":lineage.get("selected",{}).get("manifest_path"),"sha256":lineage.get("selected",{}).get("manifest_sha256")},f"geometry {name} selected manifest",errors)
            try:
                export_manifest=Path(lineage["selected"]["manifest_path"]);export_doc=load(export_manifest);receiver_manifest=Path(lineage["receiver_manifest"]["path"]);receiver_valid=receiver_manifest.is_file() and sha(receiver_manifest)==lineage["receiver_manifest"].get("sha256");receiver_doc=load(receiver_manifest) if receiver_valid else {}
                output=export_doc.get("output",{});declared=Path(output.get("path",""));declared=(export_manifest.parent/declared).resolve() if not declared.is_absolute() else declared.resolve()
                receiver_ref=export_doc.get("receiver_manifest",{});receiver_declared=Path(receiver_ref.get("path",""));receiver_declared=(export_manifest.parent/receiver_declared).resolve() if not receiver_declared.is_absolute() else receiver_declared.resolve()
                identity=export_doc.get("recording_id") or export_doc.get("scenario");manifest_export=export_doc.get("source_iq_sha256");manifest_receiver=manifest_iq_hash(receiver_doc)
                values=(receiver_iq,export_iq,lineage.get("selected_source_iq_sha256"),lineage["selected"].get("source_iq_sha256"),manifest_export,manifest_receiver)
                export_identity_ok=declared==Path(selected.get("path","")).resolve() and (identity is None or identity.lower()==name.lower())
                if output.get("sha256")!=selected.get("sha256") or int(output.get("row_count",-1))!=int(lineage["selected"].get("rows",-2)):errors.append("selected manifest export content mismatch")
                receiver_reference_ok=receiver_declared==receiver_manifest.resolve() and receiver_ref.get("sha256")==lineage["receiver_manifest"].get("sha256") and receiver_valid
                if binding=="HASH_BOUND" and (not receiver_reference_ok or not export_identity_ok or lineage["selected"].get("export_identity_status")!="PASS" or not all(valid(x) for x in values) or len(set(values))!=1):errors.append("manifest-derived source-IQ binding mismatch")
            except (KeyError,TypeError,ValueError,OSError,json.JSONDecodeError):errors.append("selected/receiver manifest parse failure")
            causal=item.get("event_time_causal_ephemeris_availability",{})
            if causal.get("causal_decode_history_verified_by")!="UNIMPLEMENTED_STAGE0" or causal.get("status")!="OFFLINE_ORACLE_ONLY" or causal.get("decoded_history_authenticated") is not False:errors.append("Stage-0 cannot authenticate causal ephemeris PASS")
        b0=load(artifact/"b0_interface_validation.json")
        if b0.get("schema")!="gnss-doppler-lab.r2c-b0-validation.v2" or set(b0.get("scenarios",{}))!=SCENARIOS:errors.append("B0 validation schema/roster mismatch")
        canonical_b0=set(load(config_path)["b0"]["saved_score_sha256"])
        for name,item in b0.get("scenarios",{}).items():
            if name not in canonical_b0 and (item.get("status")!="UNAVAILABLE_AUTHENTIC_INTERFACE" or item.get("event_rows")):errors.append("noncanonical B0 scenario exposed scores")
            if item.get("status")=="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY":errors.append("runner cannot assert authenticated node replay")
        if b0.get("paper_comparison_eligible"):errors.append("saved-only B0 replay cannot be paper eligible")
        if b0.get("aggregate_status") not in {"RECONSTRUCTABLE_WITH_LINEAGE_GAPS","UNAVAILABLE_AUTHENTIC_INTERFACE"}:errors.append("B0 aggregate status overclaims authentication")
        b0_provenance=provenance.get("b0_validation",{});b0_path=Path(b0_provenance.get("path",""))
        if not b0_path.is_file() or sha(b0_path)!=b0_provenance.get("sha256"):errors.append("B0 validation wrapper hash mismatch")
    with (artifact/"per_epoch_scores.csv").open() as handle:
        reader=csv.DictReader(handle);rows=list(reader);score_header=reader.fieldnames
    expected_score_header=(["scenario","availability_time_s","detector","status","score","ll0","ll1","n","k0","k1"] if synthetic else ["scenario","time_bin","availability_time_s","detector","status","reason","score","ll0","ll1","n","k0","k1","epoch_count","prn_count","geometry_valid"])
    if score_header!=expected_score_header:errors.append("per_epoch_scores exact header mismatch")
    key_fields=("scenario","detector") if synthetic else ("scenario","time_bin","detector")
    keys=[tuple(r.get(k) for k in key_fields) for r in rows]
    if len(keys)!=len(set(keys)):errors.append("duplicate per-event detector row")
    detectors={r.get("detector") for r in rows}
    if not REQUIRED_DETECTORS<=detectors:errors.append("required detector paths missing")
    by_detector={d:[] for d in REQUIRED_DETECTORS}
    for row in rows:
        d=row.get("detector")
        available=row.get("status")=="AVAILABLE" or row.get("status")=="AVAILABLE_AUTHENTIC_NODE_TO_SCORE_REPLAY" or row.get("status")=="AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP"
        try:score=float(row["score"])
        except (ValueError,TypeError):
            if available:errors.append("available detector has empty/nonfinite score")
            if any(row.get(x) not in (None,"") for x in ("ll0","ll1","n","k0","k1")):errors.append("unavailable detector retained likelihood evidence")
            continue
        if not np.isfinite(score):errors.append("nonfinite detector score")
        if not available:errors.append("invalid/unavailable fit exposed finite score")
        if d in by_detector:by_detector[d].append(score)
        fields=tuple(row.get(k,"") for k in ("ll0","ll1","n","k0","k1"))
        if available and d in LIKELIHOOD_DETECTORS and not all(x!="" for x in fields):errors.append("available likelihood detector missing BIC evidence")
        if all(x!="" for x in fields):
            expected=2*(float(row["ll1"])-float(row["ll0"]))-(float(row["k1"])-float(row["k0"]))*np.log(float(row["n"]))
            if not np.isclose(score,expected,rtol=1e-11,atol=1e-11):errors.append("BIC identity mismatch")
        if "availability_time_s" not in row or not np.isfinite(float(row["availability_time_s"])):errors.append("wrong/missing availability")
    for a,b in (("A1","A2"),("A3","Full"),("A4","Full"),("Full","Neural-with-energy")):
        if by_detector[a] and len(by_detector[a])==len(by_detector[b]) and np.array_equal(by_detector[a],by_detector[b]):errors.append(f"detector alias: {a}={b}")
    thresholds=load(artifact/"thresholds.json")
    if not {"q99","q99.5","target_fpr_1pct"}<=set(thresholds):errors.append("threshold artifact incomplete")
    if not synthetic:
        if {r.get("scenario") for r in rows}!=SCENARIOS:errors.append("scenario row roster mismatch")
        for scenario in SCENARIOS:
            if {r.get("detector") for r in rows if r.get("scenario")==scenario}!=ALL_DETECTORS:errors.append(f"detector row roster mismatch: {scenario}")
        event_keys={(r.get("scenario"),r.get("time_bin")) for r in rows}
        for event_key in event_keys:
            if {r.get("detector") for r in rows if (r.get("scenario"),r.get("time_bin"))==event_key}!=ALL_DETECTORS:errors.append(f"per-event detector roster mismatch: {event_key}")
        if not allow_pending and load(artifact/"verification.json").get("status") in {None,"PENDING_EXTERNAL_VERIFIER"}:errors.append("pending production verification")
    if not synthetic:
        per_detector=thresholds.get("detectors",{});errors.extend(validate_threshold_contract(thresholds,rows))
        metric_fields=["scenario","detector","threshold","status","threshold_value","normal_fpr","auroc","pr_auc","normalized_pauc_fpr_lte_0.05","attack_detection_rate","sustained_detection_rate","first_sustained_delay_s","persistent_alarm_ratio","causal_time_field"]
        with (artifact/"scenario_metrics.csv").open() as handle:metric_reader=csv.DictReader(handle);metric_rows=list(metric_reader)
        if metric_reader.fieldnames!=metric_fields:errors.append("scenario_metrics exact header mismatch")
        expected_metric_keys=expected_scenario_metric_keys(rows,load(artifact/"config.json"))
        errors.extend(validate_metric_roster(metric_rows,expected_metric_keys))
        score_groups={(s,d):[r for r in rows if r.get("scenario")==s and r.get("detector")==d and r.get("score") not in (None,"")] for s in SCENARIOS for d in ALL_DETECTORS}
        for metric in metric_rows:
            scenario=metric.get("scenario");detector=metric.get("detector");threshold_name=metric.get("threshold")
            if scenario not in {"cleanStatic","cleanDynamic"} or detector not in ALL_DETECTORS or threshold_name not in {"q99","q99.5"}:continue
            threshold=per_detector.get(detector,{}).get(threshold_name);support=score_groups[(scenario,detector)]
            support=[r for r in support if float(r["availability_time_s"])>=420] if scenario=="cleanStatic" else support
            expected_fpr=None if threshold is None or not support else float(np.mean([float(r["score"])>threshold for r in support]))
            actual=None if metric.get("normal_fpr") in (None,"") else float(metric["normal_fpr"])
            if actual!=expected_fpr:errors.append("normal FPR recomputation mismatch")
        with (artifact/"ablation_metrics.csv").open() as handle:ablation_reader=csv.DictReader(handle);ablation_rows=list(ablation_reader)
        if ablation_reader.fieldnames!=["detector","status","available_scenarios"] or {r.get("detector") for r in ablation_rows}!=ALL_DETECTORS or len(ablation_rows)!=len(ALL_DETECTORS):errors.append("ablation schema/roster mismatch")
    decision=load(artifact/"decision.json")
    control_documents={filename:load(artifact/filename) for filename in CONTROL_FILES}
    if any(document!=next(iter(control_documents.values())) for document in control_documents.values()):errors.append("control payload files are not identical")
    control_prns,control_support=(None,None) if synthetic else derive_control_candidate_context(rows,geometry)
    if not synthetic and any(document.get("rows") for document in control_documents.values()) and (control_prns is None or control_support is None):errors.append("authenticated control candidate context unavailable")
    for filename in CONTROL_FILES:
        errors.extend(validate_control(filename,control_documents[filename],decision.get("gates",{}),synthetic,load(artifact/"config.json"),thresholds,control_prns,control_support))
    bootstrap=load(artifact/"bootstrap_comparisons.json")
    comparison_hashes=[x.get("draw_index_sha256") for x in bootstrap.get("comparisons",[]) if x.get("draw_index_sha256")]
    if comparison_hashes:
        expected_draw_hash=hashlib.sha256(json.dumps(comparison_hashes,sort_keys=True).encode()).hexdigest()
        if bootstrap.get("draw_index_sha256")!=expected_draw_hash:errors.append("bootstrap draw hash mismatch")
    if not synthetic and (bootstrap.get("valid_draw_count")!=2000 or not bootstrap.get("draw_index_sha256") or not bootstrap.get("comparisons")):errors.append("bootstrap evidence incomplete")
    if not synthetic:
        expected_comparisons={(s,l,r) for s in ("DS3","DS7","DS8") for l,r in (("A2","A1"),("Full","A2"),("Full","A4"),("Full","B0-native"),("Full","Power-only"))}|{("controls","Full","relation-destruction")}
        actual_comparisons={(x.get("scenario"),x.get("left"),x.get("right")) for x in bootstrap.get("comparisons",[])}
        if actual_comparisons!=expected_comparisons or len(actual_comparisons)!=len(bootstrap.get("comparisons",[])):errors.append("bootstrap comparison roster mismatch")
        for comparison in bootstrap.get("comparisons",[]):
            if comparison.get("status")=="AVAILABLE":
                ids=comparison.get("eligible_block_ids",[])
                if hashlib.sha256("\n".join(ids).encode()).hexdigest()!=comparison.get("eligible_block_ids_sha256"):errors.append("bootstrap eligible block hash mismatch")
                if comparison.get("valid_draw_count")!=2000 or comparison.get("seed")!=load(artifact/"config.json").get("bootstrap",{}).get("seed"):errors.append("bootstrap draw evidence mismatch")
            elif not comparison.get("reason") and comparison.get("right")!="relation-destruction":errors.append("unavailable bootstrap missing reason")
    core,paper,verdict=independent_decision(decision.get("gates",{}))
    if (decision.get("core_physics_verdict"),decision.get("paper_comparison_ready"),decision.get("verdict"))!=(core,paper,verdict):errors.append("fabricated decision gates")
    if load(artifact/"config.json").get("template",{}).get("analytic_approximation") and decision.get("paper_comparison_ready"):errors.append("analytic template cannot be paper ready")
    for png in (artifact/"plots").glob("*.png"):
        if not png.with_suffix(".csv").is_file():errors.append("plot source data missing")
    if not synthetic and recompute and not errors:
        regenerated,evidence=external_recompute(artifact,provenance,source,repo);errors.extend(regenerated)
        if not regenerated and not allow_pending and load(artifact/"verification.json").get("recomputation_evidence_sha256")!=evidence:errors.append("external recomputation evidence hash mismatch")
    return sorted(set(errors))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("artifact",type=Path);ap.add_argument("--uncommitted-test",action="store_true");ap.add_argument("--allow-synthetic-test-artifact",action="store_true");ap.add_argument("--finalize",action="store_true");args=ap.parse_args()
    if args.finalize and not args.uncommitted_test:ap.error("--finalize requires --uncommitted-test")
    if args.finalize and args.artifact.resolve()!=(ROOT/"artifacts/r2c_gnss_stage0_fix").resolve():ap.error("--finalize is only for the canonical production artifact")
    errors=verify(args.artifact,require_committed=not args.uncommitted_test,allow_synthetic_test_artifact=args.allow_synthetic_test_artifact,allow_pending=args.finalize,recompute=not args.finalize)
    if args.finalize and not errors:
        provenance=load(args.artifact/"provenance.json");source=provenance["source_commit"]
        result=finalize_transaction(args.artifact,ROOT,source,"research/r2c-gnss-stage0-fix",lambda:external_recompute(args.artifact.resolve(),provenance,source,ROOT));errors.extend(result["errors"])
    print(json.dumps({"status":"PASS" if not errors else "FAIL","errors":errors},indent=2));raise SystemExit(bool(errors))
if __name__=="__main__":main()
