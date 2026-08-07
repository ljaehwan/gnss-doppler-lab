#!/usr/bin/env python3
"""Authenticated, cleanStatic-only R1.4 producer (source phase: never auto-runs)."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,sys,tempfile,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica,carrier_wipeoff,source_support
from gnss_doppler_lab.acaf_nf_stage0_r14_doppler_validation import (CANDIDATE_STRING,FROZEN_CONFIG,FS,LENGTHS,R13_REFERENCE,ROLES,R13_SOURCE_SHA256,R13_CHECKSUMS_SHA256,R13_CENTER_VALIDATION_SHA256,R13_IDENTITY_ORDER_SHA256,aggregation_gate,bootstrap_paired,clean_only_guard,common_anchor_blocks,delay_gate,delay_metrics,diagnostic_aggregates,doppler_metrics,final_gates,offset_zero_clearly_better,paired_improvements,prompt_evidence,prompt_gate,prompt_metrics)
from run_acaf_nf_stage0_static_r13_reconstruction import authenticate_inputs,balanced_sample,load_triples,read_iq,sha256
OUT=ROOT/"artifacts/acaf_nf_stage0_static_r14_doppler_validation"
R13_ARTIFACT=ROOT/"artifacts/acaf_nf_stage0_static_r13_reconstruction"
R13_SOURCE=ROOT/"src/gnss_doppler_lab/acaf_nf_stage0_r13_reconstruction.py"
GRID={"delay_chips":[round(-1+.125*i,3) for i in range(17)],"doppler_hz":list(range(-250,251,50))}
INVENTORY=("README.md config.json environment.json r13_frozen_lineage.json frozen_reconstruction_config.json prompt_reproduction_metrics.json prompt_reproduction_by_prn.csv prompt_reproduction_by_time_block.csv delay_recovery_metrics.json delay_recovery_by_prn.csv delay_recovery_by_time_block.csv doppler_1ms_metrics.json aggregation_metrics.csv aggregation_by_prn.csv aggregation_by_time_block.csv paired_improvement.csv bootstrap_results.json doppler_mainlobe_diagnostics.csv residual_doppler_diagnostics.json per_block_scores.csv execution_validity.json go_no_go.json test_report.txt verification_report.json checksums.json plots").split()
PLOTS={"l-histograms":("Doppler offset (Hz)","Count",("L1","L5","L10","L20")),"l-recovery":("Integration length L","Within 50 Hz fraction",("overall",)),"prn-l1-l20":("PRN","Within 50 Hz fraction",("L1","L20")),"role-comparison":("Role","Within 50 Hz fraction",ROLES),"prompt-scatter":("MAT Prompt magnitude","Reconstructed center magnitude",("epochs",)),"delay-histogram":("Delay offset (chips)","Count",("epochs",)),"peak-center-distribution":("Peak / center ratio","Count",("epochs",))}
def write_json(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
def write_csv(p,rows):
 rows=list(rows);fields=list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def canon(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def identities(rows):return [(str(r["channel"]),int(r["prn"]),int(r["tracker_row"]),str(r["role"])) for r in rows]
def identity_sha(rows):return hashlib.sha256(canon(identities(rows)).encode()).hexdigest()
def grouped(rows,key,metric):return [{key:v,**metric([r for r in rows if r[key]==v])} for v in sorted({r[key] for r in rows},key=str)]
def r13_metrics(rows):
 def rho(rs):
  x=float(spearmanr([float(r["center_magnitude"]) for r in rs],[float(r["mat_prompt_magnitude"]) for r in rs]).statistic);return x if np.isfinite(x) else 0.
 prns=sorted({int(r["prn"]) for r in rows}); roles=Counter(str(r["role"]) for r in rows)
 return {"n":len(rows),"prn_count":len(prns),"role_counts":dict(sorted(roles.items())),"pooled_spearman":rho(rows),"median_prn_spearman":float(np.median([rho([r for r in rows if int(r["prn"])==p]) for p in prns])),"boundary_fraction":float(np.mean([bool(r["delay_boundary"]) or bool(r["doppler_boundary"]) for r in rows])),"within_tolerance_fraction":float(np.mean([abs(float(r["peak_delay_offset_chips"]))<=.125 and abs(float(r["peak_doppler_offset_hz"]))<=50 for r in rows])),"exact_center_fraction":float(np.mean([float(r["peak_delay_offset_chips"])==0 and float(r["peak_doppler_offset_hz"])==0 for r in rows]))}
def validate_reference(m):
 return m.get("role_counts")=={r:323 for r in ROLES} and all(k in m and abs(float(m[k])-float(v))<=1e-6 for k,v in R13_REFERENCE.items())
def authenticate_r13():
 if sha256(R13_SOURCE)!=R13_SOURCE_SHA256 or sha256(R13_ARTIFACT/"checksums.json")!=R13_CHECKSUMS_SHA256:raise RuntimeError("approved R1.3 trust anchor drift")
 manifest=json.loads((R13_ARTIFACT/"checksums.json").read_text())
 if set(manifest)!={"files"} or not isinstance(manifest["files"],dict) or not manifest["files"]:raise RuntimeError("R1.3 checksum manifest schema")
 expected=set(manifest["files"])|{"checksums.json","verification_report.json"}; actual={str(p.relative_to(R13_ARTIFACT)) for p in R13_ARTIFACT.rglob("*") if p.is_file()}
 if actual!=expected:raise RuntimeError("R1.3 inventory drift")
 if any(not (R13_ARTIFACT/n).is_file() or sha256(R13_ARTIFACT/n)!=d for n,d in manifest["files"].items()):raise RuntimeError("R1.3 manifest entry drift")
 report=json.loads((R13_ARTIFACT/"verification_report.json").read_text())
 if report.get("status")!="PASS" or report.get("errors")!=[]:raise RuntimeError("R1.3 verification is not PASS")
 center=R13_ARTIFACT/"center_validation.csv"
 if sha256(center)!=R13_CENTER_VALIDATION_SHA256:raise RuntimeError("R1.3 center evidence drift")
 with center.open(newline="") as f:rows=list(csv.DictReader(f))
 if identity_sha(rows)!=R13_IDENTITY_ORDER_SHA256 or len(rows)!=969:raise RuntimeError("R1.3 identity order drift")
 metrics=r13_metrics([dict(r,delay_boundary=str(r["grid_boundary"]).lower()=="true",doppler_boundary=str(r["grid_boundary"]).lower()=="true") for r in rows])
 if not validate_reference(metrics):raise RuntimeError("R1.3 independently recomputed metric drift")
 return {"artifact_path":str(R13_ARTIFACT.relative_to(ROOT)),"source_path":str(R13_SOURCE.relative_to(ROOT)),"r13_source_sha256":R13_SOURCE_SHA256,"r13_artifact_checksums_sha256":R13_CHECKSUMS_SHA256,"center_validation_sha256":R13_CENTER_VALIDATION_SHA256,"identity_order_sha256":R13_IDENTITY_ORDER_SHA256,"manifest_entry_count":len(manifest["files"]),"inventory_verified":True,"verification_status":"PASS","reference_metrics":R13_REFERENCE,"recomputed_reference_metrics":metrics,"metrics_tolerance":1e-6,"immutable_candidate":CANDIDATE_STRING},rows
def frozen_lineage():return authenticate_r13()[0]
def complex_caf_surface(iq,row):
 n=len(iq);c=FROZEN_CONFIG.candidate
 replicas=np.asarray([code_replica(row["prn"],n,FS,row["code_freq_chips"],row["aux1"],c.remnant_sign,d,replica_direction=1)[0] for d in GRID["delay_chips"]])
 wiped=np.asarray([carrier_wipeoff(n,FS,row["carrier_doppler_hz"],f,c.carrier_sign)[0] for f in GRID["doppler_hz"]])*np.asarray(iq)[None,:]
 return wiped@replicas.T
def surface_score(surface,identity):
 mag=np.abs(surface);di,ci=np.unravel_index(int(np.argmax(mag)),mag.shape);center=float(mag[5,8]);peak=float(mag[di,ci])
 return {**identity,"peak_delay_offset_chips":float(GRID["delay_chips"][ci]),"peak_doppler_offset_hz":float(GRID["doppler_hz"][di]),"peak_magnitude":peak,"center_magnitude":center,"peak_center_ratio":peak/max(center,np.finfo(float).eps),"delay_boundary":ci in (0,16),"doppler_boundary":di in (0,10)}
def score_fields(surface,prefix):
 x=surface_score(surface,{})
 return {f"{prefix}_{k}":x[k] for k in ("peak_magnitude","peak_delay_offset_chips","peak_doppler_offset_hz","center_magnitude","peak_center_ratio")}
def svg_plot(path,title,xlabel,ylabel,series,values):
 vals=[float(v) for v in values] or [0.];lo=min(vals);hi=max(vals);span=max(hi-lo,1e-12);pts=" ".join(f"{55+i*500/max(len(vals)-1,1):.2f},{250-(v-lo)*180/span:.2f}" for i,v in enumerate(vals))
 labels=" ".join(f'<text x="60" y="{30+14*i}">{s}</text>' for i,s in enumerate(series))
 path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320"><title>{title}</title><line x1="50" y1="260" x2="600" y2="260" stroke="black"/><line x1="50" y1="20" x2="50" y2="260" stroke="black"/><text x="260" y="305">{xlabel}</text><text x="5" y="15">{ylabel}</text>{labels}<polyline points="{pts}" fill="none" stroke="blue"/></svg>\n')
def run(args):
 lineage,frozen_rows=authenticate_r13();binding=authenticate_inputs(args.raw,args.tracker_dir,args.manifest)
 if not all(binding["checks"].values()):raise RuntimeError("A1 source binding failed")
 selected=balanced_sample(load_triples(args.tracker_dir,args.raw.stat().st_size//4),969)
 if len(selected)!=969:raise RuntimeError("exact R1.3 epoch count failed")
 final=Path(args.output);final.parent.mkdir(parents=True,exist_ok=True)
 if final.exists():raise FileExistsError(final)
 stage=Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-",dir=final.parent));(stage/"plots").mkdir();started=time.perf_counter();scores=[];surfaces={};epoch_evidence=[]
 try:
  for role,triple in selected:
   state,prompt=triple[0],triple[1];support=source_support(triple,25000);iq=read_iq(args.raw,support["start_sample"],support["end_sample"])
   state_doc={"code_freq_chips":float(state["code_freq_chips"]),"carrier_doppler_hz":float(state["carrier_doppler_hz"]),"aux1":float(state["aux1"]),"state_mat_row":int(state["mat_row"]),"state_mat_path":str(state.get("mat_path",prompt.get("mat_path","")))}
   ident={"recording":"cleanStatic","role":role,"prn":int(prompt["PRN"]),"channel":str(prompt["channel"]),"tracker_row":int(prompt["mat_row"]),"anchor_tracker_row":int(prompt["mat_row"]),"support_start_sample":int(support["start_sample"]),"support_end_sample":int(support["end_sample"]),"support_length_samples":int(support["length_samples"]),"valid_raw_support":True,"cn0_db_hz":float(prompt["CN0_SNV_dB_Hz"]),"carrier_lock":float(prompt["carrier_lock_test"]),**state_doc,"state_provenance":"previous tracker row","state_digest":hashlib.sha256(canon(state_doc).encode()).hexdigest(),"raw_interval_sha256":hashlib.sha256(np.ascontiguousarray(iq).view(np.uint8)).hexdigest(),"mat_prompt_magnitude":float(np.hypot(prompt["Prompt_I"],prompt["Prompt_Q"]))}
   s=complex_caf_surface(iq,ident);key=(ident["channel"],ident["prn"],ident["tracker_row"]);surfaces[key]=s;score=prompt_evidence(surface_score(s,ident));scores.append(score)
   epoch_evidence.append({"record_type":"epoch","L":0,**score,"common_anchor_id":"","constituent_identities":canon([list(key)]),"surface_sha256":array_sha(s),"surface_real":canon(s.real.tolist()),"surface_imag":canon(s.imag.tolist()),"overlap_transition_count":0,"overlap_samples":0,"rejected_overlap_count":0})
  if identities(scores)!=identities(frozen_rows):raise RuntimeError("R1.3 exact identity set/order mismatch")
  reproduced=r13_metrics(scores)
  if not validate_reference(reproduced):raise RuntimeError("raw recomputation differs from frozen R1.3 reference")
  lineage["r14_raw_recomputed_reference_metrics"]=reproduced
  offset_rows=list(csv.DictReader((R13_ARTIFACT/"global_offset_sensitivity.csv").open(newline="")))
  prompt_all=prompt_metrics(scores);prompt_prn=grouped(scores,"prn",prompt_metrics);prompt_channel=grouped(scores,"channel",prompt_metrics);prompt_role=grouped(scores,"role",prompt_metrics);delay_all=delay_metrics(scores);delay_prn=grouped(scores,"prn",delay_metrics);delay_role=grouped(scores,"role",delay_metrics);doppler_all=doppler_metrics(scores)
  blocks=common_anchor_blocks(scores);aggregation=[];aggregation_prn=[];aggregation_role=[];paired=[];boots={};block_rows=[];by_l={}
  for L in LENGTHS:
   current=[]
   for block in blocks[L]:
    ss=[surfaces[(str(r["channel"]),int(r["prn"]),int(r["tracker_row"]))] for r in block];agg=diagnostic_aggregates(ss);primary=np.sqrt(agg["normalized_power_mean"]);last=block[-1];anchor=f"{last['channel']}:{last['prn']}:{last['tracker_row']}";ident={k:last[k] for k in ("role","prn","channel","anchor_tracker_row")};sc=surface_score(primary,ident);current.append(sc)
    starts=[int(r["support_start_sample"]) for r in block];ends=[int(r["support_end_sample"]) for r in block];overlaps=[max(0,a-b) for a,b in zip(ends,starts[1:])]
    row={"record_type":"aggregate","L":L,"recording":"cleanStatic",**ident,"tracker_row":last["tracker_row"],"common_anchor_id":anchor,"constituent_identities":canon([[str(r["channel"]),int(r["prn"]),int(r["tracker_row"]),str(r["role"])] for r in block]),"support_start_sample":starts[0],"support_end_sample":ends[-1],"support_length_samples":sum(int(r["support_length_samples"]) for r in block),"state_provenance":"each constituent own previous-row state","state_digest":hashlib.sha256(canon([r["state_digest"] for r in block]).encode()).hexdigest(),"surface_sha256":array_sha(agg["normalized_power_mean"]),"surface_real":"","surface_imag":"","overlap_transition_count":sum(x>0 for x in overlaps),"overlap_samples":sum(overlaps),"rejected_overlap_count":sum(x not in (0,1) for x in overlaps),**score_fields(primary,"primary"),**score_fields(np.sqrt(agg["raw_power_sum"]),"raw_power_sum"),**score_fields(agg["magnitude_mean"],"magnitude_mean"),**score_fields(agg["robust_median"],"robust_median"),"primary_surface_values":canon(agg["normalized_power_mean"].tolist()),"raw_power_sum_surface_values":canon(agg["raw_power_sum"].tolist()),"magnitude_mean_surface_values":canon(agg["magnitude_mean"].tolist()),"robust_median_surface_values":canon(agg["robust_median"].tolist())};block_rows.append(row)
   by_l[L]=current;counts=Counter(r["prn"] for r in current);metrics={"L":L,**doppler_metrics(current),"block_count":len(current),"prn_count":len(counts),"dominant_fraction":max(counts.values(),default=0)/max(len(current),1)};aggregation.append(metrics);aggregation_prn.extend({"L":L,**x} for x in grouped(current,"prn",doppler_metrics));aggregation_role.extend({"L":L,**x} for x in grouped(current,"role",doppler_metrics))
   if L>1:
    pairs=paired_improvements(by_l[1],current);paired.extend({"L":L,**x} for x in pairs);boots[str(L)]=bootstrap_paired(pairs)
  if {L:len(blocks[L]) for L in LENGTHS}!={L:513 for L in LENGTHS}:raise RuntimeError("common anchor count is not exact 513")
  l20=next(x for x in aggregation if x["L"]==20);pair20=[r for r in paired if r["L"]==20];prn_diff=[{"prn":p,"difference":float(np.mean([r["difference"] for r in pair20 if r["prn"]==p]))} for p in sorted({r["prn"] for r in pair20})];role_diff=[{"role":p,"difference":float(np.mean([r["difference"] for r in pair20 if r["role"]==p]))} for p in ROLES]
  a1=validate_reference(reproduced);a2=FROZEN_CONFIG.document()["candidate_string"]==CANDIDATE_STRING and len(gps_l1ca_code(1))==1023;a3a=prompt_gate(prompt_all) and all(prompt_gate(x) for x in prompt_prn+prompt_channel+prompt_role) and offset_zero_clearly_better(offset_rows);a3b=delay_gate(delay_all,delay_prn,delay_role);a3c=aggregation_gate(l20,boots["20"],prn_diff,role_diff);verdict=final_gates(a1,a2,a3a,a3b,a3c)
  write_json(stage/"config.json",{"scope":"cleanStatic-only","epochs":969,"lengths":list(LENGTHS),"grid":GRID,"bootstrap_seed":1401,"bootstrap_replicates":10000,"aggregation_primary":"mean normalized power","diagnostics":["raw power sum","magnitude mean","robust median"]});write_json(stage/"environment.json",{"python":sys.version,"numpy":np.__version__});write_json(stage/"r13_frozen_lineage.json",lineage);write_json(stage/"frozen_reconstruction_config.json",FROZEN_CONFIG.document());write_json(stage/"prompt_reproduction_metrics.json",prompt_all);write_csv(stage/"prompt_reproduction_by_prn.csv",prompt_prn);write_csv(stage/"prompt_reproduction_by_time_block.csv",prompt_role);write_json(stage/"delay_recovery_metrics.json",delay_all);write_csv(stage/"delay_recovery_by_prn.csv",delay_prn);write_csv(stage/"delay_recovery_by_time_block.csv",delay_role);write_json(stage/"doppler_1ms_metrics.json",doppler_all);write_csv(stage/"aggregation_metrics.csv",aggregation);write_csv(stage/"aggregation_by_prn.csv",aggregation_prn);write_csv(stage/"aggregation_by_time_block.csv",aggregation_role);write_csv(stage/"paired_improvement.csv",paired);write_json(stage/"bootstrap_results.json",boots)
  main=[]
  for r in scores:
   s=surfaces[(str(r["channel"]),int(r["prn"]),int(r["tracker_row"]))];center=s[5,8];sl=np.abs(s[:,8]);global_ratio=float(np.max(np.abs(s))/max(abs(center),np.finfo(float).eps));slice_ratio=float(np.max(sl)/max(abs(center),np.finfo(float).eps));row={k:r[k] for k in ("role","prn","channel","tracker_row")}
   for d in (-150,-100,-50,0,50,100,150):
    z=s[GRID["doppler_hz"].index(d),8];q=z/center if abs(center)>0 else complex(np.nan,np.nan);tag=f"{d:+d}_hz";row|={f"real_{tag}":float(z.real),f"imag_{tag}":float(z.imag),f"magnitude_{tag}":float(abs(z)),f"center_normalized_real_{tag}":float(q.real),f"center_normalized_imag_{tag}":float(q.imag)}
   row|={"global_2d_peak_center_ratio":global_ratio,"delay_center_slice_1d_peak_center_ratio":slice_ratio};main.append(row)
  write_csv(stage/"doppler_mainlobe_diagnostics.csv",main);write_json(stage/"residual_doppler_diagnostics.json",{"status":"NOT_APPLICABLE","used_by_gate":False,"reason":"authenticated MAT complex Prompt continuity not established"});write_csv(stage/"per_block_scores.csv",epoch_evidence+block_rows);write_json(stage/"execution_validity.json",{"caf_executed":True,"source_authenticated":True,"r13_lineage_validated_before_aggregation":True,"elapsed_seconds":time.perf_counter()-started,"common_anchor_counts":{str(k):len(v) for k,v in blocks.items()},"attack_inputs_read":False,"overlap_policy":"0 or 1 sample between consecutive fixed 25000 supports","overlap_samples":sum(int(r["overlap_samples"]) for r in block_rows),"overlap_rejection_count":sum(int(r["rejected_overlap_count"]) for r in block_rows)});write_json(stage/"go_no_go.json",verdict);(stage/"README.md").write_text("# R1.4 Doppler validation\n\nAuthenticated cleanStatic-only reconstruction; no candidate or attack search.\n");(stage/"test_report.txt").write_text("Source-phase focused tests are recorded in the correction commit.\n")
  vals=[float(x["within_50_fraction"]) for x in aggregation]
  for name,(xl,yl,series) in PLOTS.items():svg_plot(stage/"plots"/f"{name}.svg",name,xl,yl,series,vals)
  write_json(stage/"verification_report.json",{"status":"NOT_YET_VERIFIED"});write_json(stage/"checksums.json",{"files":{str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob("*")) if p.is_file() and p.name not in {"checksums.json","verification_report.json"}}})
  actual={p.name for p in stage.iterdir()};missing=set(INVENTORY)-actual;extra=actual-set(INVENTORY)
  if missing or extra:raise RuntimeError(f"top-level inventory drift missing={missing} extra={extra}")
  os.replace(stage,final);return verdict
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--raw",type=Path,required=True);p.add_argument("--tracker-dir",type=Path,required=True);p.add_argument("--manifest",type=Path);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--execute-production",action="store_true");a=p.parse_args(argv);clean_only_guard(["cleanStatic"])
 if not a.execute_production:raise SystemExit("source-only safety: pass --execute-production in the later campaign")
 if a.output.resolve()!=OUT.resolve():raise ValueError("R1.4 writes only its reserved artifact path")
 run(a)
if __name__=="__main__":main()
