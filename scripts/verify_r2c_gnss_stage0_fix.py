#!/usr/bin/env python3
"""Independent semantic verifier for the eventual Stage-0 fix campaign."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.r2c_stage0_fix import calibration_thresholds, derive_two_layer_decision

PRESERVED_TREE="53f7cdab9ac324c08b94a5d6f38f6a32d3ec16b7"
FILES={"README.md","config.json","provenance.json","input_validity.json","b0_interface_validation.json",
 "time_geometry_validation.json","training_summary.json","thresholds.json","scenario_metrics.csv","ablation_metrics.csv",
 "per_epoch_scores.csv","bootstrap_comparisons.json","gain_invariance.json","phase_invariance.json","noise_control.json",
 "multipath_control.json","second_source_injection.json","relation_destruction.json","decision.json","verification.json","hashes.json"}

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())

def verify(root: Path, *, repo=ROOT):
    errors=[]; root=root.resolve()
    if root.name!="r2c_gnss_stage0_fix": errors.append("campaign root name is not exact")
    actual={p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else set()
    if actual!=FILES: errors.append(f"exact top-level file set mismatch missing={sorted(FILES-actual)} extra={sorted(actual-FILES)}")
    nested=[p for p in root.rglob("*") if p.is_dir() and p.relative_to(root).parts[0]!="plots"] if root.exists() else []
    if nested: errors.append("extra nested campaign attempt")
    if not FILES<=actual: return errors
    manifest=load(root/"hashes.json").get("files",{})
    for relative, expected in manifest.items():
        target=(root/relative).resolve()
        if root not in target.parents or not target.is_file(): errors.append(f"hash path containment/missing: {relative}")
        elif digest(target)!=expected: errors.append(f"hash mismatch: {relative}")
    provenance=load(root/"provenance.json")
    if provenance.get("preserved_artifact_tree")!=PRESERVED_TREE: errors.append("preserved tree is not source constant")
    try:
        tree=subprocess.check_output(["git","rev-parse","HEAD:artifacts/r2c_gnss_stage0"],cwd=repo,text=True).strip()
        if tree!=PRESERVED_TREE: errors.append("old artifact tree changed")
    except subprocess.CalledProcessError: errors.append("cannot verify old artifact tree")
    for item in provenance.get("external_inputs",[]):
        path=Path(item.get("path",""))
        if not path.is_file() or digest(path)!=item.get("sha256"): errors.append(f"external input hash mismatch: {path}")
    rows=list(csv.DictReader((root/"per_epoch_scores.csv").open()))
    detectors=set()
    for row in rows:
        detector=row.get("detector"); detectors.add(detector)
        if all(row.get(k) not in (None,"") for k in ("ll0","ll1","n","k0","k1","score")):
            expected=2*(float(row["ll1"])-float(row["ll0"]))-(float(row["k1"])-float(row["k0"]))*np.log(float(row["n"]))
            if not np.isclose(expected,float(row["score"]),rtol=1e-10,atol=1e-10): errors.append("CSV likelihood/BIC identity mismatch")
    required={"B0-native","A1","A2","A3","A4","Full","Neural-with-energy","Power-only"}
    if not required<=detectors: errors.append("missing detector path")
    if rows:
        columns={d:np.asarray([float(r["score"]) for r in rows if r.get("detector")==d and r.get("score") not in ("",None)]) for d in required}
        for a,b in (("A1","A2"),("A3","Full"),("A4","Full")):
            if columns[a].shape==columns[b].shape and columns[a].size and np.array_equal(columns[a],columns[b]): errors.append(f"alias detectors: {a}={b}")
    threshold=load(root/"thresholds.json")
    if "q99.5" not in json.dumps(threshold): errors.append("missing q99.5")
    controls=[load(root/name) for name in ("gain_invariance.json","phase_invariance.json","noise_control.json",
      "multipath_control.json","second_source_injection.json","relation_destruction.json")]
    if any(not c.get("rows") or not all("pre_score" in r and "post_score" in r for r in c.get("rows",[])) for c in controls): errors.append("empty/stub control evidence")
    bootstrap=load(root/"bootstrap_comparisons.json")
    if bootstrap.get("repetitions")!=2000 or not bootstrap.get("comparisons"): errors.append("empty/bootstrap contract mismatch")
    decision=load(root/"decision.json"); recomputed=derive_two_layer_decision(decision.get("gates",{}))
    if any(decision.get(k)!=recomputed[k] for k in ("core_physics_verdict","paper_comparison_ready","verdict")): errors.append("decision mismatch")
    for plot in (root/"plots").glob("*.png"):
        if not plot.with_suffix(".csv").is_file(): errors.append(f"plot lacks source data: {plot.name}")
    source_commit=provenance.get("source_commit")
    if source_commit:
        if subprocess.run(["git","merge-base","--is-ancestor",source_commit,"HEAD"],cwd=repo).returncode: errors.append("source commit is not ancestor")
        parents=subprocess.check_output(["git","show","-s","--format=%P","HEAD"],cwd=repo,text=True).split()
        if len(parents)>1: errors.append("artifact commit is not single-parent source child")
    return sorted(set(errors))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("artifact",type=Path); args=ap.parse_args()
    errors=verify(args.artifact)
    print(json.dumps({"status":"PASS" if not errors else "FAIL","errors":errors},indent=2))
    raise SystemExit(bool(errors))
if __name__=="__main__": main()
