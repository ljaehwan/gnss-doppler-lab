#!/usr/bin/env python3
"""Independent fail-closed verifier; imports no R1.4 producer helpers."""
from __future__ import annotations
import argparse,ast,csv,hashlib,json,math
import xml.etree.ElementTree as ET
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
REQUIRED=("README.md config.json environment.json r13_frozen_lineage.json frozen_reconstruction_config.json prompt_reproduction_metrics.json prompt_reproduction_by_prn.csv prompt_reproduction_by_channel.csv prompt_reproduction_by_time_block.csv delay_recovery_metrics.json delay_recovery_by_prn.csv delay_recovery_by_time_block.csv doppler_1ms_metrics.json aggregation_metrics.csv aggregation_by_prn.csv aggregation_by_time_block.csv paired_improvement.csv bootstrap_results.json doppler_mainlobe_diagnostics.csv residual_doppler_diagnostics.json per_block_scores.csv execution_validity.json go_no_go.json test_report.txt verification_report.json checksums.json plots").split()
PLOTS={"l-histograms":("Doppler offset (Hz)","Count"),"l-recovery":("Integration length L","Recovery fraction"),"prn-l1-l20":("PRN","Within 50 Hz fraction"),"role-comparison":("Integration length L","Within 50 Hz fraction"),"prompt-scatter":("MAT Prompt magnitude","Reconstructed center magnitude"),"delay-histogram":("Delay offset (chips)","Count"),"peak-center-distribution":("Peak / center ratio","Empirical cumulative fraction")}
R13_REPORT_KEYS={"ca_code_independent_validation","candidate_physical_applications","cross_role_nonoverlap","errors","gates","global_offsets_independent","recomputed","recursive_checksums","status","verdict"}
SOURCE_SHA="9889a5e5007c92d6016e5ef0d38a03cea96cdd40eded3cea91df1e4276d16e42";CHECKSUMS_SHA="04b5395b311641b4ab3f3a58a1a5cbb54d4249068f8252659049ea4386a95abb";REPORT_SHA="4a4177a51b2fcd1d552155e5714efbb69afcfa2fa3da51bd3594201bda884591";REPORT_BYTES=20868;CENTER_SHA="cb07c2b3d192c6bd30e6eeca6ffae6d615523f1ba4569d4259ccb01d866ba198";IDENTITY_SHA="65933645102b7a05087f0d9991ad1c55c822b4b83090cb57a4e6f74e17675e5c";ROLES=("train","calibration","holdout");LENGTHS=(1,5,10,20);DOP=list(range(-250,251,50));DEL=[round(-1+.125*i,3) for i in range(17)]
R13={"n":969,"prn_count":8,"pooled_spearman":0.9999965049269979,"median_prn_spearman":0.9999652753663446,"boundary_fraction":0.006191950464396285,"within_tolerance_fraction":0.8565531475748194,"exact_center_fraction":0.42105263157894735}
CANDIDATE="nco_row=previous_aux_row=previous_remnant_sign=-1_carrier_sign=-1_global_offset=0"
def loadj(p):return json.loads(p.read_text())
def loadc(p):
 with p.open(newline="") as f:return list(csv.DictReader(f))
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)
def b(v):return v is True or str(v).lower() in {"true","1"}
def checksums(root):return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in {"checksums.json","verification_report.json"}}
def exact_inventory(root):
 expected_files=(set(REQUIRED)-{"plots"})|{f"plots/{name}.svg" for name in PLOTS};expected_dirs={"plots"};files=set();dirs=set();links=set();other=set()
 for p in root.rglob("*"):
  rel=str(p.relative_to(root))
  if p.is_symlink():links.add(rel)
  elif p.is_file():files.add(rel)
  elif p.is_dir():dirs.add(rel)
  else:other.add(rel)
 return files==expected_files and dirs==expected_dirs and not links and not other
def valid_r13_report(report):
 return (isinstance(report,dict) and set(report)==R13_REPORT_KEYS and report.get("status")=="PASS" and report.get("errors")==[] and isinstance(report.get("gates"),dict) and isinstance(report.get("recomputed"),dict) and isinstance(report.get("recursive_checksums"),dict) and isinstance(report.get("ca_code_independent_validation"),dict))
def exact_r13_inventory(root,manifest_files):
 expected_files=set(manifest_files)|{"checksums.json","verification_report.json"};expected_dirs={"plots"};files=set();dirs=set();links=set();other=set()
 for p in root.rglob("*"):
  rel=p.relative_to(root).as_posix()
  if p.is_symlink():links.add(rel)
  elif p.is_file():files.add(rel)
  elif p.is_dir():dirs.add(rel)
  else:other.add(rel)
 manifest_plots={n for n in manifest_files if n.startswith("plots/")}
 return (root.is_dir() and not root.is_symlink() and len(manifest_files)==25 and len(expected_files)==27 and manifest_plots=={"plots/center-recovery.svg"} and files==expected_files and dirs==expected_dirs and not links and not other)
def plot_payloads(aggregation,aggregation_prn,aggregation_role,epochs):
 by_l={int(r["L"]):r for r in aggregation};ratios=sorted(float(r["peak_center_ratio"]) for r in epochs);delays=sorted({float(r["peak_delay_offset_chips"]) for r in epochs})
 return {
  "l-histograms":{"kind":"grouped_histogram","series":[{"label":f"L{L}","points":[[float(x),int(n)] for x,n in sorted(by_l[L]["histogram"].items(),key=lambda z:float(z[0]))]} for L in LENGTHS]},
  "l-recovery":{"kind":"line","series":[{"label":f"within {hz} Hz","points":[[L,float(by_l[L][f"within_{hz}_fraction"])] for L in LENGTHS]} for hz in (50,100,150)]},
  "prn-l1-l20":{"kind":"line","series":[{"label":f"L{L}","points":[[int(r["prn"]),float(r["within_50_fraction"])] for r in aggregation_prn if int(r["L"])==L]} for L in (1,20)]},
  "role-comparison":{"kind":"line","series":[{"label":role,"points":[[int(r["L"]),float(r["within_50_fraction"])] for r in aggregation_role if r["role"]==role]} for role in ROLES]},
  "prompt-scatter":{"kind":"scatter","series":[{"label":"969 epochs","points":[[float(r["mat_prompt_magnitude"]),float(r["center_magnitude"])] for r in epochs]}]},
  "delay-histogram":{"kind":"histogram","series":[{"label":"epochs","points":[[x,sum(float(r["peak_delay_offset_chips"])==x for r in epochs)] for x in delays]}]},
  "peak-center-distribution":{"kind":"distribution","series":[{"label":"epochs","points":[[x,(i+1)/len(ratios)] for i,x in enumerate(ratios)]}]}}
def plot_semantic_ok(path,xlabel,ylabel,payload):
 tree=ET.parse(path);root=tree.getroot();meta=next((x for x in root.iter() if x.tag.endswith("metadata") and x.attrib.get("id")=="canonical-data"),None);data=canon(payload)
 return meta is not None and meta.text==data and meta.attrib.get("data-sha256")==hashlib.sha256(data.encode()).hexdigest() and xlabel in "".join(root.itertext()) and ylabel in "".join(root.itertext()) and any(x.tag.endswith(("path","rect","circle")) for x in root.iter())
def close(a,z,tol=1e-11):
 if isinstance(a,(dict,list)) and isinstance(z,str):
  try:z=ast.literal_eval(z)
  except (ValueError,SyntaxError):return False
 if isinstance(a,dict):return isinstance(z,dict) and set(a)==set(z) and all(close(v,z[k],tol) for k,v in a.items())
 if isinstance(a,list):return isinstance(z,list) and len(a)==len(z) and all(close(x,y,tol) for x,y in zip(a,z))
 if isinstance(a,bool):return b(z)==a
 if isinstance(a,(int,float)):return isinstance(z,(int,float,str)) and math.isfinite(float(z)) and abs(float(a)-float(z))<=tol
 return a==z
def rho(rows):
 if len(rows)<3:return 0.
 x=float(spearmanr([float(r["center_magnitude"]) for r in rows],[float(r["mat_prompt_magnitude"]) for r in rows]).statistic);return x if math.isfinite(x) else 0.
def prompt(rows):
 e=np.asarray([abs(float(r["center_magnitude"])/float(r["mat_prompt_magnitude"])-1) for r in rows]);ps=[rho([r for r in rows if int(r["prn"])==p]) for p in sorted({int(r["prn"]) for r in rows})]
 return {"n":len(rows),"pooled_spearman":rho(rows),"median_prn_spearman":float(np.median(ps)),"median_relative_error":float(np.median(e)),"p95_relative_error":float(np.quantile(e,.95)),"p99_relative_error":float(np.quantile(e,.99)),"max_relative_error":float(np.max(e))}
def delay(rows):
 x=np.asarray([float(r["peak_delay_offset_chips"]) for r in rows]);bd=np.asarray([b(r["delay_boundary"]) for r in rows]);return {"n":len(rows),"exact_center_fraction":float(np.mean(x==0)),"within_0_125_fraction":float(np.mean(abs(x)<=.125)),"boundary_fraction":float(np.mean(bd)),"histogram":{str(v):int(np.sum(x==v)) for v in sorted(set(x))}}
def doppler(rows):
 x=np.asarray([float(r["peak_doppler_offset_hz"]) for r in rows]);q=np.asarray([float(r["peak_center_ratio"]) for r in rows]);out={"n":len(rows),"exact_center_fraction":float(np.mean(x==0)),"boundary_fraction":float(np.mean([b(r["doppler_boundary"]) for r in rows])),"median_abs_offset_hz":float(np.median(abs(x))),"p95_abs_offset_hz":float(np.quantile(abs(x),.95)),"median_peak_center_ratio":float(np.median(q)),"histogram":{str(v):int(np.sum(x==v)) for v in sorted(set(x))}}
 for hz in (50,100,150):out[f"within_{hz}_fraction"]=float(np.mean(abs(x)<=hz))
 return out
def score(surface):
 mag=np.abs(surface);di,ci=np.unravel_index(int(np.argmax(mag)),mag.shape);center=float(mag[5,8]);peak=float(mag[di,ci]);return {"peak_delay_offset_chips":float(DEL[ci]),"peak_doppler_offset_hz":float(DOP[di]),"peak_magnitude":peak,"center_magnitude":center,"peak_center_ratio":peak/max(center,np.finfo(float).eps),"delay_boundary":ci in (0,16),"doppler_boundary":di in (0,10)}
def r13_metrics(rows):
 prns=sorted({int(r["prn"]) for r in rows});return {"n":len(rows),"prn_count":len(prns),"role_counts":dict(sorted(Counter(str(r["role"]) for r in rows).items())),"pooled_spearman":rho(rows),"median_prn_spearman":float(np.median([rho([r for r in rows if int(r["prn"])==p]) for p in prns])),"boundary_fraction":float(np.mean([b(r["delay_boundary"]) or b(r["doppler_boundary"]) for r in rows])),"within_tolerance_fraction":float(np.mean([abs(float(r["peak_delay_offset_chips"]))<=.125 and abs(float(r["peak_doppler_offset_hz"]))<=50 for r in rows])),"exact_center_fraction":float(np.mean([float(r["peak_delay_offset_chips"])==0 and float(r["peak_doppler_offset_hz"])==0 for r in rows]))}
def valid_ref(m):return m.get("role_counts")=={x:323 for x in ROLES} and all(k in m and abs(float(m[k])-float(v))<=1e-6 for k,v in R13.items())
def grouped(rows,key,metric):return [{key:v,**metric([r for r in rows if r[key]==v])} for v in sorted({r[key] for r in rows},key=str)]
def boot(rows):
 by=defaultdict(list)
 for r in rows:by[int(r["prn"])].append(float(r["difference"]))
 rng=np.random.default_rng(1401);prns=sorted(by);vals=np.empty(10000)
 for i in range(10000):vals[i]=np.mean([x for p in rng.choice(prns,len(prns),replace=True) for x in by[int(p)]])
 lo=float(np.quantile(vals,.025));return {"seed":1401,"replicates":10000,"observed_difference":float(np.mean([float(r["difference"]) for r in rows])),"ci95_low":lo,"ci95_high":float(np.quantile(vals,.975)),"sign_consistent":lo>0}
def gate(a1,a2,a3a,a3b,a3c):
 v="RECONSTRUCTION_IMPLEMENTATION_INVALID" if not a1 or not a2 else "TRACKER_RAW_RECONSTRUCTION_UNRESOLVED" if not a3a or not a3b else "PHYSICAL_RECONSTRUCTION_VALID_DOPPLER_RESOLUTION_LIMITED" if not a3c else "PHYSICAL_CENTER_VALID"
 return {"A1_SOURCE_BINDING":"PASS" if a1 else "FAIL","A2_RECONSTRUCTION_IMPLEMENTATION":"PASS" if a2 else "FAIL","A3a_PROMPT_REPRODUCTION":"PASS" if a3a else "FAIL","A3b_CODE_DELAY":"PASS" if a3b else "FAIL","A3c_DOPPLER_AGGREGATION":"PASS" if a3c else "FAIL","verdict":v}
def verify(root:Path):
 errors=[]
 if not root.is_dir() or root.is_symlink() or not exact_inventory(root):return {"status":"FAIL","errors":["exact_recursive_inventory"]}
 try:
  if loadj(root/"checksums.json")!={"files":checksums(root)}:errors.append("recursive_checksums")
  source=ROOT/"src/gnss_doppler_lab/acaf_nf_stage0_r13_reconstruction.py";art=ROOT/"artifacts/acaf_nf_stage0_static_r13_reconstruction";mp=art/"checksums.json";centerp=art/"center_validation.csv";reportp=art/"verification_report.json"
  if digest(source)!=SOURCE_SHA or digest(mp)!=CHECKSUMS_SHA or digest(centerp)!=CENTER_SHA:errors.append("r13_trust_anchor")
  if reportp.stat().st_size!=REPORT_BYTES or digest(reportp)!=REPORT_SHA:errors.append("r13_verification_report_trust_anchor")
  manifest=loadj(mp);manifest_files=manifest.get("files",{}) if isinstance(manifest,dict) else {}
  if set(manifest)!={"files"} or not isinstance(manifest_files,dict) or not exact_r13_inventory(art,manifest_files) or any(digest(art/n)!=d for n,d in manifest_files.items()):errors.append("r13_manifest_inventory")
  if not valid_r13_report(loadj(reportp)):errors.append("r13_verification_report")
  frozen=loadc(centerp);fids=[(str(r["channel"]),int(r["prn"]),int(r["tracker_row"]),str(r["role"])) for r in frozen]
  if hashlib.sha256(canon(fids).encode()).hexdigest()!=IDENTITY_SHA:errors.append("r13_identity_order")
  fr=[dict(r,delay_boundary=b(r["grid_boundary"]),doppler_boundary=b(r["grid_boundary"])) for r in frozen]
  if not valid_ref(r13_metrics(fr)):errors.append("r13_reference_recompute")
  lineage=loadj(root/"r13_frozen_lineage.json");required_lineage={"r13_source_sha256":SOURCE_SHA,"r13_artifact_checksums_sha256":CHECKSUMS_SHA,"center_validation_sha256":CENTER_SHA,"identity_order_sha256":IDENTITY_SHA,"verification_status":"PASS","inventory_verified":True,"metrics_tolerance":1e-6,"reference_metrics":R13}
  if any(lineage.get(k)!=v for k,v in required_lineage.items()):errors.append("r13_lineage_document")
  cfg=loadj(root/"frozen_reconstruction_config.json");expected_cfg={"signal":"gps_l1ca_code","fs_hz":25000000.0,"raw_format":"interleaved_signed_int16_iq","global_offset_samples":0,"nco_row":"previous","aux_row":"previous","remnant_sign":-1,"carrier_sign":-1,"replica_direction":"forward","prompt_row":"current","support_samples":25000,"candidate_string":CANDIDATE}
  if cfg!=expected_cfg:errors.append("frozen_config")
  evidence=loadc(root/"per_block_scores.csv");epochs=[r for r in evidence if r["record_type"]=="epoch"];agrows=[r for r in evidence if r["record_type"]=="aggregate"]
  eids=[(r["channel"],int(r["prn"]),int(r["tracker_row"]),r["role"]) for r in epochs]
  if eids!=fids or len(set(eids))!=969:errors.append("epoch_identity_order")
  surfaces={};raw=[]
  for r in epochs:
   state_doc={"code_freq_chips":float(r["code_freq_chips"]),"carrier_doppler_hz":float(r["carrier_doppler_hz"]),"aux1":float(r["aux1"]),"state_mat_row":int(r["state_mat_row"]),"state_mat_path":r["state_mat_path"]}
   if r["state_provenance"]!="previous tracker row" or hashlib.sha256(canon(state_doc).encode()).hexdigest()!=r["state_digest"] or len(r["raw_interval_sha256"])!=64:errors.append("epoch_state_provenance")
   s=np.asarray(json.loads(r["surface_real"]),float)+1j*np.asarray(json.loads(r["surface_imag"]),float);key=(r["channel"],int(r["prn"]),int(r["tracker_row"]));surfaces[key]=s;sc=score(s)
   if s.shape!=(11,17) or array_sha(s)!=r["surface_sha256"] or any(not close(sc[k],r[k]) for k in sc):errors.append("epoch_surface_score");break
   ratio=float(sc["center_magnitude"])/float(r["mat_prompt_magnitude"]);raw.append({**r,**sc,"prompt_ratio":ratio,"prompt_abs_relative_error":abs(ratio-1)})
  ref=r13_metrics(raw)
  if not valid_ref(ref) or not close(ref,lineage.get("r14_raw_recomputed_reference_metrics",{}),1e-6):errors.append("r14_reference_recompute")
  if not close(prompt(raw),loadj(root/"prompt_reproduction_metrics.json")):errors.append("prompt_overall")
  if not close(grouped(raw,"prn",prompt),loadc(root/"prompt_reproduction_by_prn.csv")):errors.append("prompt_prn")
  if not close(grouped(raw,"channel",prompt),loadc(root/"prompt_reproduction_by_channel.csv")):errors.append("prompt_channel")
  if not close(grouped(raw,"role",prompt),loadc(root/"prompt_reproduction_by_time_block.csv")):errors.append("prompt_role")
  if not close(delay(raw),loadj(root/"delay_recovery_metrics.json")):errors.append("delay_overall")
  if not close(grouped(raw,"prn",delay),loadc(root/"delay_recovery_by_prn.csv")):errors.append("delay_prn")
  if not close(grouped(raw,"role",delay),loadc(root/"delay_recovery_by_time_block.csv")):errors.append("delay_role")
  if not close(doppler(raw),loadj(root/"doppler_1ms_metrics.json")):errors.append("doppler_1ms")
  recomputed={L:[] for L in LENGTHS};anchor_sets={};total_overlap=0
  for r in agrows:
   L=int(r["L"]);ids=json.loads(r["constituent_identities"])
   if L not in LENGTHS or len(ids)!=L or len({tuple(x[:3]) for x in ids})!=L or {x[0] for x in ids}!={r["channel"]} or {int(x[1]) for x in ids}!={int(r["prn"])} or {x[3] for x in ids}!={r["role"]}:errors.append("constituent_identity");continue
   if ids[-1][:3]!=[r["channel"],int(r["prn"]),int(r["anchor_tracker_row"])]:errors.append("anchor_identity")
   ss=[surfaces[(x[0],int(x[1]),int(x[2]))] for x in ids];constituents=[next(x for x in epochs if x["channel"]==i[0] and int(x["prn"])==int(i[1]) and int(x["tracker_row"])==int(i[2])) for i in ids]
   expected_state_sha=hashlib.sha256(canon([x["state_digest"] for x in constituents]).encode()).hexdigest()
   if r["state_provenance"]!="each constituent own previous-row state" or r["state_digest"]!=expected_state_sha:errors.append("aggregate_state_provenance")
   power=np.abs(np.asarray(ss))**2;norm=np.mean(power/(np.sum(power,axis=(1,2),keepdims=True)+1e-15),axis=0);rawsum=np.sum(power,axis=0);mean=np.mean(np.sqrt(power),axis=0);median=np.median(np.sqrt(power),axis=0)
   vals={"primary_surface_values":norm,"raw_power_sum_surface_values":rawsum,"magnitude_mean_surface_values":mean,"robust_median_surface_values":median}
   if any(not np.allclose(np.asarray(json.loads(r[k]),float),v,rtol=0,atol=1e-13) for k,v in vals.items()) or array_sha(norm)!=r["surface_sha256"]:errors.append("aggregate_surface")
   for prefix,surf in (("primary",np.sqrt(norm)),("raw_power_sum",np.sqrt(rawsum)),("magnitude_mean",mean),("robust_median",median)):
    sc=score(surf)
    if any(not close(sc[k],r[f"{prefix}_{k}"]) for k in ("peak_magnitude","peak_delay_offset_chips","peak_doppler_offset_hz","center_magnitude","peak_center_ratio")):errors.append("aggregate_score_"+prefix)
   starts=[int(next(x["support_start_sample"] for x in epochs if x["channel"]==i[0] and int(x["prn"])==int(i[1]) and int(x["tracker_row"])==int(i[2]))) for i in ids];ends=[int(next(x["support_end_sample"] for x in epochs if x["channel"]==i[0] and int(x["prn"])==int(i[1]) and int(x["tracker_row"])==int(i[2]))) for i in ids];ov=[max(0,a-bb) for a,bb in zip(ends,starts[1:])]
   if any(x not in (0,1) for x in ov) or int(r["overlap_samples"])!=sum(ov) or int(r["overlap_transition_count"])!=sum(x>0 for x in ov) or int(r["rejected_overlap_count"])!=0:errors.append("overlap_audit")
   total_overlap+=sum(ov);sc=score(np.sqrt(norm));recomputed[L].append({"role":r["role"],"prn":int(r["prn"]),"channel":r["channel"],"anchor_tracker_row":int(r["anchor_tracker_row"]),**sc})
  for L in LENGTHS:anchor_sets[L]=[(x["channel"],x["prn"],x["anchor_tracker_row"],x["role"]) for x in recomputed[L]]
  if any(len(recomputed[L])!=513 for L in LENGTHS) or any(anchor_sets[L]!=anchor_sets[20] for L in LENGTHS):errors.append("common_anchors")
  ag=[];agp=[];agr=[]
  for L in LENGTHS:
   rows=recomputed[L];cnt=Counter(x["prn"] for x in rows);ag.append({"L":L,**doppler(rows),"block_count":len(rows),"prn_count":len(cnt),"dominant_fraction":max(cnt.values())/len(rows)});agp.extend({"L":L,**x} for x in grouped(rows,"prn",doppler));agr.extend({"L":L,**x} for x in grouped(rows,"role",doppler))
  if not close(ag,loadc(root/"aggregation_metrics.csv")):errors.append("aggregation_overall")
  if not close(agp,loadc(root/"aggregation_by_prn.csv")):errors.append("aggregation_prn")
  if not close(agr,loadc(root/"aggregation_by_time_block.csv")):errors.append("aggregation_role")
  payloads=plot_payloads(ag,agp,agr,raw)
  for name,(xl,yl) in PLOTS.items():
   if not plot_semantic_ok(root/"plots"/(name+".svg"),xl,yl,payloads[name]):errors.append("plot_semantic_"+name)
  expected_pairs=[];expected_boot={}
  left={(x["channel"],x["prn"],x["anchor_tracker_row"],x["role"]):x for x in recomputed[1]}
  for L in (5,10,20):
   right={(x["channel"],x["prn"],x["anchor_tracker_row"],x["role"]):x for x in recomputed[L]};group=[]
   if set(left)!=set(right):errors.append("pair_anchor_"+str(L))
   for k in sorted(left):
    a=abs(float(left[k]["peak_doppler_offset_hz"]))<=50;z=abs(float(right[k]["peak_doppler_offset_hz"]))<=50;group.append({"L":L,"channel":k[0],"prn":k[1],"anchor_tracker_row":k[2],"role":k[3],"l1_success":a,"aggregated_success":z,"difference":int(z)-int(a)})
   expected_pairs+=group;expected_boot[str(L)]=boot(group)
  saved_pairs=loadc(root/"paired_improvement.csv")
  if not close(expected_pairs,saved_pairs):errors.append("pair_math")
  if not close(expected_boot,loadj(root/"bootstrap_results.json")):errors.append("bootstrap")
  main=loadc(root/"doppler_mainlobe_diagnostics.csv");mm={(r["channel"],int(r["prn"]),int(r["tracker_row"])):r for r in main}
  if set(mm)!=set(surfaces):errors.append("mainlobe_identity")
  else:
   for key,s in surfaces.items():
    r=mm[key];center=s[5,8]
    for d in (-150,-100,-50,0,50,100,150):
     z=s[DOP.index(d),8];q=z/center;tag=f"{d:+d}_hz"
     if not all(close(v,r[k]) for k,v in ((f"real_{tag}",z.real),(f"imag_{tag}",z.imag),(f"magnitude_{tag}",abs(z)),(f"center_normalized_real_{tag}",q.real),(f"center_normalized_imag_{tag}",q.imag))):errors.append("mainlobe_complex");break
    if not close(float(np.max(np.abs(s))/abs(center)),r["global_2d_peak_center_ratio"]) or not close(float(np.max(np.abs(s[:,8]))/abs(center)),r["delay_center_slice_1d_peak_center_ratio"]):errors.append("mainlobe_ratio")
  residual=loadj(root/"residual_doppler_diagnostics.json")
  if residual.get("status")!="NOT_APPLICABLE" or residual.get("used_by_gate") is not False:errors.append("optional_phase_gate")
  execution=loadj(root/"execution_validity.json")
  if execution.get("caf_executed") is not True or execution.get("source_authenticated") is not True or execution.get("r13_lineage_validated_before_aggregation") is not True or execution.get("attack_inputs_read") is not False or execution.get("common_anchor_counts")!={str(L):513 for L in LENGTHS} or int(execution.get("overlap_samples",-1))!=total_overlap or int(execution.get("overlap_rejection_count",-1))!=0:errors.append("execution_validity")
  pm=prompt(raw);bypp=grouped(raw,"prn",prompt);bypc=grouped(raw,"channel",prompt);bypr=grouped(raw,"role",prompt);offset=loadc(art/"global_offset_sensitivity.csv");off={int(x["global_offset_samples"]):float(x["pooled_spearman"]) for x in offset};offsetpass=set(off)=={-1000,-500,0,500,1000} and all(off[0]>off[x] for x in off if x)
  a1=not any(x.startswith("r13_") or x in {"epoch_identity_order","r14_reference_recompute"} for x in errors);a2=cfg==expected_cfg;a3a=pm["pooled_spearman"]>=.999 and pm["median_prn_spearman"]>=.99 and pm["median_relative_error"]<=.001 and pm["p99_relative_error"]<=.01 and all(x["pooled_spearman"]>=.999 and x["median_prn_spearman"]>=.99 and x["median_relative_error"]<=.001 and x["p99_relative_error"]<=.01 for x in bypp+bypc+bypr) and offsetpass
  da=delay(raw);dp=grouped(raw,"prn",delay);dr=grouped(raw,"role",delay);a3b=da["within_0_125_fraction"]>=.95 and da["boundary_fraction"]<=.01 and len(dp)==8 and sum(x["within_0_125_fraction"]>=.95 for x in dp)>=7 and {x["role"] for x in dr}==set(ROLES) and all(x["within_0_125_fraction"]>=.95 for x in dr)
  p20=[x for x in expected_pairs if x["L"]==20];pd={p:np.mean([x["difference"] for x in p20 if x["prn"]==p]) for p in {x["prn"] for x in p20}};rd={q:np.mean([x["difference"] for x in p20 if x["role"]==q]) for q in {x["role"] for x in p20}};l20=ag[-1];a3c=l20["within_50_fraction"]>=.95 and l20["boundary_fraction"]<=.01 and expected_boot["20"]["ci95_low"]>0 and len(pd)==8 and sum(x>0 for x in pd.values())>=7 and set(rd)==set(ROLES) and all(x>0 for x in rd.values())
  if loadj(root/"go_no_go.json")!=gate(a1,a2,a3a,a3b,a3c):errors.append("gate_verdict")
 except Exception as exc:errors.append(f"exception:{type(exc).__name__}:{exc}")
 return {"status":"PASS" if not errors else "FAIL","errors":errors}
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("artifact",type=Path);p.add_argument("--write-report",action="store_true");a=p.parse_args(argv);r=verify(a.artifact)
 if a.write_report:(a.artifact/"verification_report.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
 print(json.dumps(r,indent=2,sort_keys=True));raise SystemExit(0 if r["status"]=="PASS" else 1)
if __name__=="__main__":main()
