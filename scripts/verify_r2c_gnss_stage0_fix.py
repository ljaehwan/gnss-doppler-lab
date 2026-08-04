#!/usr/bin/env python3
"""Independent semantic verifier for R2C Stage-0-fix artifacts."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path,PurePosixPath
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_artifact import (FIX_SOURCE_FILES,PRESERVED_TREE,TOP_LEVEL_DIRECTORIES,
 TOP_LEVEL_FILES,expected_hash_keys)
REQUIRED_DETECTORS={"A1","A2","A3","A4","Full","Neural-with-energy","Power-only"}
ALL_DETECTORS={"B0-native","A1","A2","A3","A4","Full","Neural-with-energy","Power-only","Noise-floor-only"}
SCENARIOS={"cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8"}
CONTROL_FILES={"gain_invariance.json":{"gain","slow_agc"},"phase_invariance.json":{"global_phase"},
 "noise_control.json":{"awgn","cn0_degradation","matched_power_noise","quantization"},
 "multipath_control.json":{"non_shared_multipath"},"second_source_injection.json":{"second_source_injection"},
 "relation_destruction.json":{"relation_destruction"}}
LIKELIHOOD_DETECTORS={"A1","A3","A4","Full","Neural-with-energy"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def git_blob_sha1(path):
    data=path.read_bytes();return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def load(path):return json.loads(path.read_text())
def git(repo,*args):return subprocess.check_output(["git",*args],cwd=repo,text=True).strip()
def verify_file_record(record,label,errors):
    try:path=Path(record["path"])
    except (KeyError,TypeError):errors.append(f"{label} path/hash missing");return
    if not path.is_file() or sha(path)!=record.get("sha256"):errors.append(f"{label} path/hash mismatch")
def manifest_iq_hash(doc):return (doc.get("source",{}).get("iq_sha256") or doc.get("source",{}).get("sha256") or doc.get("authenticated_inputs",{}).get("iq_after_receiver",{}).get("sha256"))

def external_recompute(artifact,provenance,source,repo):
    """Regenerate using the frozen source and external provenance, never artifact evidence."""
    with tempfile.TemporaryDirectory(prefix="r2c-verify-") as temporary:
        base=Path(temporary);worktree=base/"source";output=base/"regenerated"
        subprocess.run(["git","worktree","add","--detach",str(worktree),source],cwd=repo,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            external=provenance["external_inputs"]
            command=[sys.executable,str(worktree/"scripts/run_r2c_gnss_stage0_fix.py"),"--config",str(worktree/"configs/r2c_gnss_stage0_fix.json"),"--output",str(output),"--source-commit",source,"--verification-recompute"]
            for item in sorted(external,key=lambda x:x["scenario"]):command += ["--input",f'{item["scenario"]}={item["selected"]["path"]}']
            command += ["--geometry",external[0]["geometry"]["path"],"--b0-validation",provenance["b0_validation"]["path"]]
            env={**os.environ,"R2C_VERIFIER_RECOMPUTE":"1","PYTHONHASHSEED":str(load(worktree/"configs/r2c_gnss_stage0_fix.json")["seed"])}
            subprocess.run(command,cwd=worktree,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            excluded={"hashes.json","verification.json"};mismatches=[]
            expected={str(p.relative_to(artifact)) for p in artifact.rglob("*") if p.is_file() and p.name not in excluded}
            actual={str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name not in excluded}
            if expected!=actual:mismatches.append("recomputed file roster mismatch")
            for name in sorted(expected&actual):
                if (artifact/name).read_bytes()!=(output/name).read_bytes():mismatches.append(f"external recomputation mismatch: {name}")
            evidence=hashlib.sha256("\n".join(f"{name}:{sha(output/name)}" for name in sorted(actual)).encode()).hexdigest()
            return mismatches,evidence
        finally:subprocess.run(["git","worktree","remove","--force",str(worktree)],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
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
    errors=[];artifact=artifact.resolve()
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
                try:blob=subprocess.check_output(["git","show",f"{source}:{name}"],cwd=repo)
                except subprocess.CalledProcessError:errors.append(f"source bundle blob missing: {name}");continue
            if hashlib.sha256(blob).hexdigest()!=value:errors.append(f"source bundle hash mismatch: {name}")
        expected_bundle_hash=hashlib.sha256(json.dumps(bundle,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if provenance.get("source_bundle",{}).get("bundle_sha256")!=expected_bundle_hash:errors.append("source bundle aggregate hash mismatch")
    if provenance.get("preserved_artifact_tree")!=PRESERVED_TREE:errors.append("preserved tree source constant mismatch")
    try:
        if git(repo,"rev-parse","HEAD:artifacts/r2c_gnss_stage0")!=PRESERVED_TREE:errors.append("old artifact tree changed")
        if source and subprocess.run(["git","merge-base","--is-ancestor",source,"HEAD"],cwd=repo).returncode:errors.append("source commit not ancestor")
        if require_committed:
            if git(repo,"status","--porcelain=v1"):errors.append("repository/worktree dirty")
            parents=git(repo,"show","-s","--format=%P","HEAD").split()
            if parents!=[source]:errors.append("artifact HEAD parent is not frozen source commit")
            changed=git(repo,"diff-tree","--no-commit-id","--name-only","-r","HEAD").splitlines()
            if any(not p.startswith("artifacts/r2c_gnss_stage0_fix/") for p in changed):errors.append("artifact commit diff is not artifact-only")
            tree_lines=git(repo,"ls-tree","-r","HEAD","artifacts/r2c_gnss_stage0_fix").splitlines();committed={line.split("\t",1)[1].split("artifacts/r2c_gnss_stage0_fix/",1)[1]:line.split()[2] for line in tree_lines}
            disk={str(p.relative_to(artifact)):git_blob_sha1(p) for p in artifact.rglob("*") if p.is_file()}
            if committed!=disk:errors.append("on-disk artifact differs from committed tree")
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
        per_detector=thresholds.get("detectors",{})
        if set(per_detector)!=ALL_DETECTORS:errors.append("threshold detector roster mismatch")
        for detector in ALL_DETECTORS:
            values=[float(r["score"]) for r in rows if r.get("scenario")=="cleanStatic" and r.get("detector")==detector and r.get("score") not in (None,"") and 320<=float(r["availability_time_s"])<=400]
            if not values: continue
            expected=float(np.quantile(np.asarray(values),.99,method="higher"));expected995=float(np.quantile(np.asarray(values),.995,method="higher"))
            item=per_detector.get(detector,{})
            if item.get("q99")!=expected or item.get("target_fpr_1pct")!=expected or item.get("q99.5")!=expected995:
                errors.append(f"threshold recomputation mismatch: {detector}")
        full=per_detector.get("Full",{})
        if full and any(thresholds.get(k)!=full.get(k) for k in ("q99","q99.5","target_fpr_1pct")):errors.append("top-level Full threshold mismatch")
        metric_fields=["scenario","detector","threshold","status","threshold_value","normal_fpr","auroc","pr_auc","normalized_pauc_fpr_lte_0.05","attack_detection_rate","sustained_detection_rate","first_sustained_delay_s","persistent_alarm_ratio","causal_time_field"]
        with (artifact/"scenario_metrics.csv").open() as handle:metric_reader=csv.DictReader(handle);metric_rows=list(metric_reader)
        if metric_reader.fieldnames!=metric_fields:errors.append("scenario_metrics exact header mismatch")
        metric_keys=[(r.get("scenario"),r.get("detector"),r.get("threshold")) for r in metric_rows]
        expected_metric_keys={(s,d,t) for s in SCENARIOS for d in ALL_DETECTORS for t in ("q99","q99.5")}
        if set(metric_keys)!=expected_metric_keys or len(metric_keys)!=len(set(metric_keys)):errors.append("scenario_metrics row roster/cardinality mismatch")
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
    for filename,kinds in CONTROL_FILES.items():
        control=load(artifact/filename);control_rows=control.get("rows",[])
        present={r.get("kind") for r in control_rows}
        if not kinds<=present:errors.append(f"missing perturbation: {filename}")
        for row in control_rows:
            if not all(k in row for k in ("pre_score","post_score","pre_alarm","post_alarm","effect_size","parameters","seed","support_count","baseline_status","perturbed_status","threshold")):errors.append("empty control evidence");continue
            threshold=float(control.get("threshold",np.nan))
            if float(row["threshold"])!=threshold:errors.append("control threshold mismatch")
            both=row["baseline_status"]=="AVAILABLE" and row["perturbed_status"]=="AVAILABLE"
            if both and (row["pre_alarm"]!=(float(row["pre_score"])>threshold) or row["post_alarm"]!=(float(row["post_score"])>threshold)):errors.append("constant/fabricated control alarm")
            if not both and any(row.get(x) is not None for x in ("pre_alarm","post_alarm")):errors.append("unavailable control produced alarm")
        if not synthetic and control_rows and len({r["post_score"] for r in control_rows})==1:errors.append("constant control scores")
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
    decision=load(artifact/"decision.json");core,paper,verdict=independent_decision(decision.get("gates",{}))
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
        provenance=load(args.artifact/"provenance.json");regen,evidence=external_recompute(args.artifact.resolve(),provenance,provenance["source_commit"],ROOT);errors.extend(regen)
        if not errors:
            (args.artifact/"verification.json").write_text(json.dumps({"schema":"gnss-doppler-lab.r2c-production-verification.v1","status":"PASS","deterministic_external_recomputation":True,"recomputation_evidence_sha256":evidence},indent=2,sort_keys=True)+"\n")
            files={str(p.relative_to(args.artifact)):sha(p) for p in args.artifact.rglob("*") if p.is_file() and p.name!="hashes.json"}
            (args.artifact/"hashes.json").write_text(json.dumps({"algorithm":"sha256","policy":"all files recursively except hashes.json","files":files},indent=2,sort_keys=True)+"\n")
            subprocess.run(["git","add","--","artifacts/r2c_gnss_stage0_fix"],cwd=ROOT,check=True)
            subprocess.run(["git","commit","-m","Add externally verified R2C Stage-0 fix campaign artifact"],cwd=ROOT,check=True)
    print(json.dumps({"status":"PASS" if not errors else "FAIL","errors":errors},indent=2));raise SystemExit(bool(errors))
if __name__=="__main__":main()
