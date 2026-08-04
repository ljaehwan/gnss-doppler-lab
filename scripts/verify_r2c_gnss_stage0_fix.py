#!/usr/bin/env python3
"""Independent semantic verifier for R2C Stage-0-fix artifacts."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path,PurePosixPath
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_artifact import (FIX_SOURCE_FILES,PRESERVED_TREE,TOP_LEVEL_DIRECTORIES,
 TOP_LEVEL_FILES,expected_hash_keys)
REQUIRED_DETECTORS={"A1","A2","A3","A4","Full","Neural-with-energy","Power-only"}
CONTROL_FILES={"gain_invariance.json":{"gain","slow_agc"},"phase_invariance.json":{"global_phase"},
 "noise_control.json":{"awgn","cn0_degradation","matched_power_noise","quantization"},
 "multipath_control.json":{"non_shared_multipath"},"second_source_injection.json":{"second_source_injection"},
 "relation_destruction.json":{"relation_destruction"}}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def git(repo,*args):return subprocess.check_output(["git",*args],cwd=repo,text=True).strip()
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

def verify(artifact:Path,*,repo=ROOT,require_committed=True):
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
    if set(bundle)!=set(FIX_SOURCE_FILES):errors.append("source bundle file set mismatch")
    else:
        for name,value in bundle.items():
            if not (repo/name).is_file() or sha(repo/name)!=value:errors.append(f"source bundle hash mismatch: {name}")
    if provenance.get("preserved_artifact_tree")!=PRESERVED_TREE:errors.append("preserved tree source constant mismatch")
    try:
        if git(repo,"rev-parse","HEAD:artifacts/r2c_gnss_stage0")!=PRESERVED_TREE:errors.append("old artifact tree changed")
        if source and subprocess.run(["git","merge-base","--is-ancestor",source,"HEAD"],cwd=repo).returncode:errors.append("source commit not ancestor")
        if require_committed:
            parents=git(repo,"show","-s","--format=%P","HEAD").split()
            if parents!=[source]:errors.append("artifact HEAD parent is not frozen source commit")
            changed=git(repo,"diff-tree","--no-commit-id","--name-only","-r","HEAD").splitlines()
            if any(not p.startswith("artifacts/r2c_gnss_stage0_fix/") for p in changed):errors.append("artifact commit diff is not artifact-only")
    except subprocess.CalledProcessError:errors.append("git ancestry verification failed")
    for item in provenance.get("external_inputs",[]):
        path=Path(item.get("path",""))
        if not path.is_file() or sha(path)!=item.get("sha256"):errors.append("external input/checkpoint hash mismatch")
    rows=list(csv.DictReader((artifact/"per_epoch_scores.csv").open()))
    detectors={r.get("detector") for r in rows}
    if not REQUIRED_DETECTORS<=detectors:errors.append("required detector paths missing")
    by_detector={d:[] for d in REQUIRED_DETECTORS}
    for row in rows:
        d=row.get("detector")
        try:score=float(row["score"])
        except (ValueError,TypeError):continue
        if d in by_detector:by_detector[d].append(score)
        fields=(row.get(k,"") for k in ("ll0","ll1","n","k0","k1"))
        if all(x!="" for x in fields):
            expected=2*(float(row["ll1"])-float(row["ll0"]))-(float(row["k1"])-float(row["k0"]))*np.log(float(row["n"]))
            if not np.isclose(score,expected,rtol=1e-11,atol=1e-11):errors.append("BIC identity mismatch")
        if "availability_time_s" not in row or not np.isfinite(float(row["availability_time_s"])):errors.append("wrong/missing availability")
    for a,b in (("A1","A2"),("A3","Full"),("A4","Full"),("Full","Neural-with-energy")):
        if by_detector[a] and len(by_detector[a])==len(by_detector[b]) and np.array_equal(by_detector[a],by_detector[b]):errors.append(f"detector alias: {a}={b}")
    thresholds=load(artifact/"thresholds.json")
    if not {"q99","q99.5","target_fpr_1pct"}<=set(thresholds):errors.append("threshold artifact incomplete")
    synthetic=bool(provenance.get("synthetic_test_mode"))
    if not synthetic:
        per_detector=thresholds.get("detectors",{})
        for detector in REQUIRED_DETECTORS:
            values=[float(r["score"]) for r in rows if r.get("scenario")=="cleanStatic" and r.get("detector")==detector and r.get("score") not in (None,"") and 320<=float(r["availability_time_s"])<=400]
            if not values: continue
            expected=float(np.quantile(np.asarray(values),.99,method="higher"));expected995=float(np.quantile(np.asarray(values),.995,method="higher"))
            item=per_detector.get(detector,{})
            if item.get("q99")!=expected or item.get("target_fpr_1pct")!=expected or item.get("q99.5")!=expected995:
                errors.append(f"threshold recomputation mismatch: {detector}")
        full=per_detector.get("Full",{})
        if full and any(thresholds.get(k)!=full.get(k) for k in ("q99","q99.5","target_fpr_1pct")):errors.append("top-level Full threshold mismatch")
    for filename,kinds in CONTROL_FILES.items():
        control=load(artifact/filename);control_rows=control.get("rows",[])
        present={r.get("kind") for r in control_rows}
        if not kinds<=present:errors.append(f"missing perturbation: {filename}")
        for row in control_rows:
            if not all(k in row for k in ("pre_score","post_score","pre_alarm","post_alarm","effect_size","parameters","seed","support_count")):errors.append("empty control evidence");continue
            threshold=float(control.get("threshold",np.nan))
            if row["pre_alarm"]!=(float(row["pre_score"])>threshold) or row["post_alarm"]!=(float(row["post_score"])>threshold):errors.append("constant/fabricated control alarm")
        if not synthetic and control_rows and len({r["post_score"] for r in control_rows})==1:errors.append("constant control scores")
    bootstrap=load(artifact/"bootstrap_comparisons.json")
    comparison_hashes=[x.get("draw_index_sha256") for x in bootstrap.get("comparisons",[]) if x.get("draw_index_sha256")]
    if comparison_hashes:
        expected_draw_hash=hashlib.sha256(json.dumps(comparison_hashes,sort_keys=True).encode()).hexdigest()
        if bootstrap.get("draw_index_sha256")!=expected_draw_hash:errors.append("bootstrap draw hash mismatch")
    if not synthetic and (bootstrap.get("valid_draw_count")!=2000 or not bootstrap.get("draw_index_sha256") or not bootstrap.get("comparisons")):errors.append("bootstrap evidence incomplete")
    decision=load(artifact/"decision.json");core,paper,verdict=independent_decision(decision.get("gates",{}))
    if (decision.get("core_physics_verdict"),decision.get("paper_comparison_ready"),decision.get("verdict"))!=(core,paper,verdict):errors.append("fabricated decision gates")
    if load(artifact/"config.json").get("template",{}).get("analytic_approximation") and decision.get("paper_comparison_ready"):errors.append("analytic template cannot be paper ready")
    for png in (artifact/"plots").glob("*.png"):
        if not png.with_suffix(".csv").is_file():errors.append("plot source data missing")
    return sorted(set(errors))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("artifact",type=Path);ap.add_argument("--uncommitted-test",action="store_true");args=ap.parse_args()
    errors=verify(args.artifact,require_committed=not args.uncommitted_test);print(json.dumps({"status":"PASS" if not errors else "FAIL","errors":errors},indent=2));raise SystemExit(bool(errors))
if __name__=="__main__":main()
