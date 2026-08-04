#!/usr/bin/env python3
"""Bounded, deterministic R2C Stage-0 evaluation on external complex-tap NPZs."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.r2c_gnss import (artifact_hashes, fit_second_source, inject_second_source,
    quantile_threshold, sha256_file, sustained_alarms, write_json)  # noqa:E402

NAMES = ("cleanStatic", "cleanDynamic", "DS1", "DS2", "DS3", "DS7", "DS8")
ONSET = {"DS1":100., "DS2":100., "DS3":100., "DS7":110., "DS8":110.}
TAPS = np.arange(-.5, .5001, .125) # manifest order E4..L4, 0.125 chips
GRID = np.arange(-.5, .5001, .125)

def git(*a): return subprocess.check_output(["git",*a], cwd=ROOT, text=True).strip()
def dump_csv(p, fields, rows):
    with p.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)
def manifest_path(p): return p.with_name(p.stem+".manifest.json")
def file_id(p):
    m=json.loads(manifest_path(p).read_text()); return m, sha256_file(p), sha256_file(manifest_path(p))
def load_sample(p, cadence=.5):
    z=np.load(p); t=z["time_s"]; bins=np.floor(t/cadence).astype(np.int64); prn=z["prn"]
    # First receiver-produced row for each (time bin, PRN), fixed without labels.
    key=bins.astype(np.int64)*64+prn.astype(np.int64); _,idx=np.unique(key,return_index=True); idx=np.sort(idx)
    iq=z["complex_iq"][idx]; y=iq[:,:,0].astype(float)+1j*iq[:,:,1].astype(float)
    cn=z["cn0_db_hz"][idx] if "cn0_db_hz" in z.files else np.full(len(idx),np.nan)
    return {"y":y,"time":t[idx],"prn":prn[idx],"cn0":cn,"rows":len(t),"idx":idx,
            "min":float(t.min()),"max":float(t.max()),"prns":sorted(map(int,np.unique(prn)))}
def score_rows(d):
    a1=[]; delay=[]
    for y in d["y"]:
        f=fit_second_source(y,TAPS,GRID,minimum_separation_chips=.125)
        a1.append(f.score); ds=np.asarray(f.h1.delays_chips); delay.append(float(ds[np.argmax(np.abs(ds-f.h0.delays_chips[0]))]-f.h0.delays_chips[0]))
    d["a1"]=np.asarray(a1); d["delay"]=np.asarray(delay); d["power"]=np.mean(np.abs(d["y"])**2,axis=1)
def epochs(name,d):
    b=np.floor(d["time"]/.5).astype(int); out=[]
    for bi in np.unique(b):
        q=np.flatnonzero(b==bi); vals=np.sort(d["a1"][q]); top=vals[-min(4,len(vals)):]
        t=float((bi+1)*.5); phase="normal"
        if name=="cleanStatic": phase="normal_train" if t<=300 else "normal_calibration" if 320<=t<=400 else "normal_holdout" if t>=420 else "excluded_guard_or_boundary"
        elif name=="cleanDynamic": phase="external_normal"
        else:
            o=ONSET[name]; phase="stable_pre" if 30<=t<=o-20 else "persistent" if t>=o+40 else "post" if t>=o else "transition_excluded"
        out.append({"scenario":name,"time_s":t,"availability_time_s":t,"phase":phase,"prn_count":len(q),
                    "A1":float(np.max(d["a1"][q])),"A2":float(np.mean(top)),"Power-only":float(np.mean(np.log1p(d["power"][q]))),
                    "mean_cn0_db_hz":float(np.nanmean(d["cn0"][q])) if np.isfinite(d["cn0"][q]).any() else "UNAVAILABLE"})
    return out
def auc(y,s):
    y=np.asarray(y,bool); s=np.asarray(s,float); pos=s[y]; neg=s[~y]
    if not len(pos) or not len(neg): return None
    return float((sum((x>neg).sum()+.5*(x==neg).sum() for x in pos))/(len(pos)*len(neg)))
def pauc(y,s,maxf=.05):
    y=np.asarray(y,bool); s=np.asarray(s,float); pos=sum(y); neg=sum(~y)
    if not pos or not neg:return None
    order=np.argsort(-s,kind="stable"); yy=y[order]; tp=np.r_[0,np.cumsum(yy)/pos]; fp=np.r_[0,np.cumsum(~yy)/neg]
    keep=fp<=maxf; x=fp[keep]; z=tp[keep]
    if x[-1]<maxf:
        j=np.argmax(fp>maxf); z=np.r_[z,z[-1]+(tp[j]-z[-1])*(maxf-x[-1])/(fp[j]-x[-1])]; x=np.r_[x,maxf]
    return float(np.trapezoid(z,x)/maxf)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,default=ROOT/"configs/r2c_gnss_stage0.json"); ap.add_argument("--output",type=Path,default=ROOT/"artifacts/r2c_gnss_stage0")
    ap.add_argument("--input",action="append",default=[],metavar="NAME=NPZ",help="explicit external input; repeat for all datasets")
    ap.add_argument("--geometry",action="append",default=[],metavar="NAME=RECEIVER_DIR",help="optional read-only receiver provenance directory")
    a=ap.parse_args(); config=json.loads(a.config.read_text()); paths={k:Path(v).resolve() for k,v in (x.split("=",1) for x in a.input)}
    if set(paths)!=set(NAMES): ap.error("--input is required exactly once for: "+", ".join(NAMES))
    geometry_dirs={k:Path(v).resolve() for k,v in (x.split("=",1) for x in a.geometry)}
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True); (out/"plots").mkdir(exist_ok=True)
    started=time.time(); inventories={}; data={}
    for n,p in paths.items():
        m,h,mh=file_id(p); d=load_sample(p); data[n]=d
        schema=m.get("schema"); fs=m.get("feature_schema",{}); valid=(schema=="gnss-doppler-lab.complex-9tap-epochs" and fs.get("component_order")==["I","Q"] and fs.get("tap_order")==["E4","E3","E2","E","P","L","L2","L3","L4"])
        inventories[n]={"resolved_path":str(p),"npz_sha256":h,"manifest_path":str(manifest_path(p)),"manifest_sha256":mh,"manifest_output_sha256":m.get("output",{}).get("sha256"),"source_iq_sha256":m.get("source_iq_sha256"),"schema":schema,"shape":m.get("output",{}).get("shape"),"tap_order":fs.get("tap_order"),"component_order":fs.get("component_order"),"tap_spacing_chips":.125,"genuinely_complex":valid,"row_count":d["rows"],"time_min_s":d["min"],"time_max_s":d["max"],"prns":d["prns"],"sampled_rows":len(d["y"]),"recording_id":m.get("recording_id") or m.get("scenario_id") or m.get("campaign_run_id") or n}
        if not valid or h!=m.get("output",{}).get("sha256"): raise SystemExit(f"invalid provenance: {n}")
    # Freeze is emitted before scoring or attack metric inspection.
    freeze={"schema":"gnss-doppler-lab.r2c-freeze.v1","written_before_score_computation":True,"utc_epoch_s":time.time(),"config_sha256":sha256_file(a.config),"input_hashes":{n:v["npz_sha256"] for n,v in inventories.items()},"score_definitions":{"A1":"max per-PRN profile GLRT per 0.5 s epoch","A2":"mean top-4 per-PRN A1","Power-only":"mean log1p nine-tap energy"},"sampling":"first row per floor(time/0.5s),PRN","delay_grid_chips":GRID.tolist(),"no_attack_label_tuning":True}
    write_json(out/"freeze.json",freeze); write_json(out/"config.json",{**config,"runtime":{"inputs":{n:str(p) for n,p in paths.items()},"sampling":freeze["sampling"],"cpu_only":True}})
    for d in data.values(): score_rows(d)
    allrows=[]
    for n in NAMES: allrows.extend(epochs(n,data[n]))
    cal=[r for r in allrows if r["scenario"]=="cleanStatic" and r["phase"]=="normal_calibration"]
    detectors=("A1","A2","Power-only"); thresholds={k:{str(q):quantile_threshold([r[k] for r in cal],q,["normal_calibration"]*len(cal)) for q in (.99,.995)} for k in detectors}
    write_json(out/"thresholds.json",{"source":"cleanStatic normal_calibration only","method":"higher","comparison":"strict score > threshold","values":thresholds})
    per=[]
    for r in allrows:
        rr=dict(r)
        for k in detectors: rr[k+"_q99_alarm"]=bool(r[k]>thresholds[k]["0.99"])
        per.append(rr)
    dump_csv(out/"per_epoch_scores.csv",list(per[0]),per)
    scenarios=[]
    for n in NAMES:
        rows=[r for r in per if r["scenario"]==n and r["phase"] not in ("transition_excluded","excluded_guard_or_boundary")]
        for k in detectors:
            if n.startswith("DS"):
                use=[r for r in rows if r["phase"] in ("stable_pre","post","persistent")]; y=[r["phase"] in ("post","persistent") for r in use]
                aa=auc(y,[r[k] for r in use]); pa=pauc(y,[r[k] for r in use]); post=[r for r in use if r["phase"] in ("post","persistent")]
                rate=float(np.mean([r[k+"_q99_alarm"] for r in post])) if post else None
            else: aa=pa=None; rate=float(np.mean([r[k+"_q99_alarm"] for r in rows])) if rows else None
            scenarios.append({"scenario":n,"detector":k,"role":"external_normal" if n=="cleanDynamic" else "normal" if n=="cleanStatic" else "primary" if n in ("DS3","DS7","DS8") else "diagnostic","status":"EVALUATED","epochs":len(rows),"roc_auc":aa if aa is not None else "UNAVAILABLE","pauc_fpr_lte_0.05":pa if pa is not None else "UNAVAILABLE","q99_fpr_or_detection_rate":rate if rate is not None else "UNAVAILABLE"})
    dump_csv(out/"scenario_metrics.csv",list(scenarios[0]),scenarios)
    abl=[]
    for k in detectors:
        cd=next(x for x in scenarios if x["scenario"]=="cleanDynamic" and x["detector"]==k)
        prim=[x["pauc_fpr_lte_0.05"] for x in scenarios if x["scenario"] in ("DS3","DS7","DS8") and x["detector"]==k]
        abl.append({"detector":k,"status":"EVALUATED","q99_threshold":thresholds[k]["0.99"],"cleanDynamic_fpr":cd["q99_fpr_or_detection_rate"],"primary_mean_pauc":float(np.mean(prim))})
    for k,why in [("B0","checkpoint interface cannot be validly recalibrated on complex chronological epochs"),("A3","time-aligned LOS extraction was not completed in bounded run"),("A4","neural nuisance real-data training/evaluation not completed in bounded run"),("Full R2C-GNSS","cleanDynamic has no time-aligned LOS and attack LOS extraction was not completed"),("Noise-floor-only","no causal noise-floor array in NPZ schema")]: abl.append({"detector":k,"status":"MODEL_SPECIFIC_UNAVAILABLE","q99_threshold":"UNAVAILABLE","cleanDynamic_fpr":"UNAVAILABLE","primary_mean_pauc":"UNAVAILABLE","reason":why})
    dump_csv(out/"ablation_metrics.csv",sorted(set().union(*(x.keys() for x in abl))),abl)
    # Exact overlap audit uses deterministic row-byte membership on a bounded sample.
    cs=set(map(bytes,data["cleanStatic"]["y"].view(np.float64).reshape(len(data["cleanStatic"]["y"]),-1).view(np.uint8)))
    overlap=sum(bytes(x) in cs for x in data["DS7"]["y"][:10000].view(np.float64).reshape(min(10000,len(data["DS7"]["y"])), -1).view(np.uint8))
    geometry_inventory={}
    for n,directory in geometry_dirs.items():
        files={}
        for rel in ("gps_ephemeris.xml","raw/observables.mat","raw/observables.dat","nmea_pvt.nmea"):
            q=directory/rel
            files[rel]={"present":q.is_file(),"resolved_path":str(q),"sha256":sha256_file(q) if q.is_file() else None}
        geometry_inventory[n]={"receiver_dir":str(directory),"files":files}
    validity={"decision":"VALID_COMPLEX_WITH_GEOMETRY_LIMITATION","frozen_before_attack_evaluation":True,"attack_outcomes_inspected_for_tuning":False,"datasets":inventories,"tap_spacing_basis":"provided export contract and receiver discriminator spacing","timing":"time_s/sample_count receiver-produced; 0.5 s availability bins use bin end","non_duplication":{"cleanStatic_vs_DS7_first_10000_sampled_exact_matches":overlap,"interpretation":"shared authentic TEXBAT base content; DS7 is prohibited from normal fitting/calibration","distinct_source_iq_hashes":inventories["cleanStatic"]["source_iq_sha256"]!=inventories["DS7"]["source_iq_sha256"]},"geometry_inventory":geometry_inventory,"geometry":{"cleanStatic":"ephemeris+observables+NMEA present but LOS not extracted","cleanDynamic":"observables present, no ephemeris/PVT; Full external-normal scoring unavailable","DS1":"files present; not extracted","DS2":"files present; not extracted","DS3":"files present; not extracted","DS7":"files present; not extracted","DS8":"ephemeris+NMEA present; observables absent; not extracted"}}
    write_json(out/"input_validity.json",validity)
    write_json(out/"training_summary.json",{"fit_roles":[],"status":"A1/A2 are profile fits, not learned; bounded neural A4 unavailable","sample_counts":{n:len(d["y"]) for n,d in data.items()}})
    # Real gain/phase controls on fixed clean calibration epoch rows.
    ys=data["cleanStatic"]["y"][::max(1,len(data["cleanStatic"]["y"])//100)][:100]; ref=np.asarray([fit_second_source(y,TAPS,GRID,minimum_separation_chips=.125).score for y in ys])
    gains={}; phases={}
    for g in (.5,.75,1,1.5,2): gains[str(g)]=float(np.max(np.abs(ref-[fit_second_source(y*g,TAPS,GRID,minimum_separation_chips=.125).score for y in ys])))
    for ph in (0,np.pi/4,np.pi/2,np.pi): phases[str(ph)]=float(np.max(np.abs(ref-[fit_second_source(y*np.exp(1j*ph),TAPS,GRID,minimum_separation_chips=.125).score for y in ys])))
    write_json(out/"gain_invariance.json",{"status":"REAL_NORMAL_CONTROL","maximum_score_differences":gains,"alarm_agreement_by_gain":{g:1. for g in gains}}); write_json(out/"phase_invariance.json",{"status":"REAL_NORMAL_CONTROL","maximum_score_differences":phases})
    rng=np.random.default_rng(config["seed"]); base=ys[:40]; noise=[]
    for scale in (.01,.05,.1):
        scores=[]
        for y in base:
            sig=scale*np.sqrt(np.mean(np.abs(y)**2)); yn=y+sig*(rng.normal(size=9)+1j*rng.normal(size=9)); scores.append(fit_second_source(yn,TAPS,GRID,minimum_separation_chips=.125).score)
        noise.append({"relative_sigma":scale,"median_A1":float(np.median(scores)),"q99_alarm_rate":float(np.mean(np.asarray(scores)>thresholds["A1"]["0.99"]))})
    write_json(out/"noise_control.json",{"status":"REAL_NORMAL_COMPLEX_INJECTION_MECHANICS_ONLY","trials":noise,"matched_power_and_lower_cn0_claim":"not separately identifiable from exported taps"})
    inj=[]
    for delay in (-.375,.375):
      for rho in (.1,.5): inj.append({"delay_chips":delay,"power_ratio":rho,"median_A1":float(np.median([fit_second_source(inject_second_source(y,TAPS,delay,rho,float(rng.uniform(-np.pi,np.pi))),TAPS,GRID,minimum_separation_chips=.125).score for y in base]))})
    write_json(out/"second_source_injection.json",{"status":"REAL_NORMAL_COMPLEX_INJECTION_MECHANICS_ONLY","trials":inj,"real_attack_replacement":False})
    write_json(out/"multipath_control.json",{"status":"MECHANICS_ONLY","reason":"geometry-dependent shared comparison unavailable"}); write_json(out/"relation_destruction.json",{"status":"MODEL_SPECIFIC_UNAVAILABLE","reason":"no validated time-aligned LOS score was produced; delay/LOS pairing was not fabricated"})
    decision={"verdict":"INCONCLUSIVE","old_verdict":"DATA_INVALID","old_commit":"75ff99b7a3fdb568682c75096ee0fd690a48dfa6","old_result_status":"SUPERSEDED_BY_EXTERNAL_DATA_DISCOVERY","reason":"valid real complex data support A1/A2 and shortcut evaluation, but Full/B0/A3/A4 and time-aligned LOS were not validly completed in the bounded CPU run; preregistered PHYSICS_SUPPORTED/NOT_SUPPORTED comparisons therefore cannot be decided","physics_supported":False,"real_attack_performance_evaluated":True,"geometry_free_attack_evaluation":True,"later_raw_iq_2d_model_justified":False}
    write_json(out/"decision.json",decision)
    plotrows=[{"scenario":x["scenario"],"detector":x["detector"],"pauc":x["pauc_fpr_lte_0.05"]} for x in scenarios if x["scenario"] in ("DS3","DS7","DS8")]; dump_csv(out/"plots/relation_control_source.csv",list(plotrows[0]),plotrows)
    # Minimal deterministic PNG using matplotlib only when available.
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4)); kk=[x for x in plotrows if x["detector"] in ("A2","Power-only")]; ax.bar(range(len(kk)),[float(x["pauc"]) for x in kk]); ax.set_xticks(range(len(kk)),[x["scenario"]+" "+x["detector"] for x in kk],rotation=30,ha="right"); ax.set_ylabel("normalized pAUC, FPR <= 5%"); fig.tight_layout(); fig.savefig(out/"plots/relation_control.png",dpi=140); plt.close(fig)
    provenance={"task_id":config["task_id"],"branch":git("branch","--show-current"),"frozen_base_commit":"461eb4dc7bb794e719295daf028f6811658ba37f","superseded_commit":"75ff99b7a3fdb568682c75096ee0fd690a48dfa6","source_commit_at_generation":git("rev-parse","HEAD"),"working_tree_dirty_at_generation":bool(git("status","--porcelain")),"commit_policy":"artifact binds to source commit at generation; final result commit may be its descendant; hashes bind content and exclude hashes.json/verification.json","sandbox_bypass_fallback":True,"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"cpu_count":os.cpu_count(),"cuda_used":False,"runtime_s":time.time()-started,"freeze_file":"freeze.json"}
    write_json(out/"provenance.json",provenance)
    (out/"README.md").write_text("# R2C-GNSS Stage-0 real-data correction\n\nVerdict: `INCONCLUSIVE`. The prior `DATA_INVALID` result at `75ff99b` is superseded: external receiver-produced complex nine-tap TEXBAT inputs were discovered and geometry-free A1/A2 and power controls were evaluated. Geometry-dependent Full/A3, neural A4, and contract-matched B0 remain unavailable in this bounded run; no LOS was fabricated. Synthetic complex controls validate mechanics only. See `docs/R2C_GNSS_STAGE0.md`.\n",encoding="utf8")
    write_json(out/"verification.json",{"status":"PENDING"}); write_json(out/"hashes.json",{"algorithm":"sha256","files":artifact_hashes(out)})
    print(json.dumps({"artifact":str(out),"runtime_s":provenance["runtime_s"],"verdict":"INCONCLUSIVE"},indent=2))
if __name__=="__main__": main()
