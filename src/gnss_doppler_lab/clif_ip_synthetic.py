"""CLIF-IP synthetic-normal R4 target-matched, leakage-safe primitives.

R3 estimators remain frozen/read-only.  This module adds run-disjoint indexing,
s16le format-aware M1 extraction, per-run publication contracts, multi-recording
history reset, domain-gap diagnostics, and deterministic region-local tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

DOMAINS=("SYN-OAK","SYN-TEX")
TAP_ORDER=("E4","E3","E2","E","P","L","L2","L3","L4")
IMPAIRMENT_AXES=("awgn_cn0_dbhz","cfo_hz","cfo_drift_hz_s","phase_noise_std_rad",
                 "frontend_bw_hz","iq_gain_imbalance_db","iq_phase_imbalance_deg",
                 "dc_i","dc_q","agc_gain_db","quantization_bits","clipping_fullscale")
ROOT=Path(__file__).resolve().parents[2]


def target_spec(domain:str)->dict[str,object]:
    specs={"SYN-OAK":{"sample_rate_hz":5_000_000,"sample_format":"s16le_iq","gnss_sdr_item_type":"ishort"},
           "SYN-TEX":{"sample_rate_hz":25_000_000,"sample_format":"s16le_iq","gnss_sdr_item_type":"ishort"}}
    if domain not in specs: raise ValueError(f"unknown target domain: {domain}")
    return dict(specs[domain])


def exact_iq_bytes(domain:str,duration_s:float)->int:
    samples=target_spec(domain)["sample_rate_hz"]*float(duration_s)
    if samples != int(samples) or duration_s<=0: raise ValueError("duration must produce an integer positive sample count")
    return int(samples)*4


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def _candidate_locations()->pd.DataFrame:
    paths=(ROOT/"configs/generated/normal_v3_large_300/run_index.csv",ROOT/"configs/experiments/normal_v3_synthetic_only_30.csv")
    for p in paths:
        if p.exists():
            d=pd.read_csv(p).drop_duplicates("capital")
            if len(d)>=30:return d.iloc[:30].reset_index(drop=True)
    raise FileNotFoundError("need at least 30 normal-v3 location candidates")


def _impairments(seed:int,domain:str)->dict[str,object]:
    r=np.random.default_rng(seed); fs=int(target_spec(domain)["sample_rate_hz"])
    return {"awgn_cn0_dbhz":float(r.uniform(43,50)),"cfo_hz":float(r.uniform(-35,35)),
      "cfo_drift_hz_s":float(r.uniform(-.08,.08)),"phase_noise_std_rad":float(r.uniform(2e-5,1.5e-4)),
      "frontend_bw_hz":float(r.uniform(.32,.46)*fs),"iq_gain_imbalance_db":float(r.uniform(-.25,.25)),
      "iq_phase_imbalance_deg":float(r.uniform(-1.2,1.2)),"dc_i":float(r.uniform(-12,12)),
      "dc_q":float(r.uniform(-12,12)),"agc_gain_db":float(r.uniform(-.8,.8)),
      "quantization_bits":16,"clipping_fullscale":30000,"attack":False,"spoofing":False,
      "profile":"open_sky_normal"}


def rinex_nav_validity(path:Path)->tuple[datetime,datetime]:
    """Return the epoch bounds gps-sdr-sim reports for a RINEX-2 NAV file.

    gps-sdr-sim derives its accepted ``-t`` interval from the first and last
    broadcast ephemeris epochs.  Reading those epochs from the selected NAV
    file keeps index construction tied to the actual simulator input instead
    of a campaign-date assumption.
    """
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    epoch=re.compile(r"^\s*\d{1,2}\s+(\d{2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d+(?:\.\d*)?)")
    times=[];in_body=False
    with p.open(errors="replace") as f:
        for line in f:
            if not in_body:
                in_body="END OF HEADER" in line
                continue
            m=epoch.match(line)
            if not m:continue
            yy,month,day,hour,minute,second=m.groups();year=2000+int(yy) if int(yy)<80 else 1900+int(yy)
            sec=float(second);whole=int(sec)
            times.append(datetime(year,int(month),int(day),int(hour),int(minute),whole,
                                  round((sec-whole)*1_000_000),tzinfo=timezone.utc))
    if not times:raise ValueError(f"no RINEX-2 navigation epochs: {p}")
    return min(times),max(times)


def _paired_unused_utc_slots(loc:pd.DataFrame,base:datetime,duration_s:float,reserved:set[datetime],count:int)->list[datetime]:
    """Choose deterministic paired 10-minute starts inside all selected NAVs."""
    bounds={str(rel):rinex_nav_validity(ROOT/str(rel)) for rel in loc.rinex_nav.unique()}
    tmin=max(x[0] for x in bounds.values());tmax=min(x[1] for x in bounds.values())
    earliest=max(base,tmin);earliest=earliest.replace(second=0,microsecond=0)
    if earliest<max(base,tmin):earliest+=timedelta(minutes=1)
    earliest+=timedelta(minutes=(-earliest.minute)%10)
    latest=tmax-timedelta(seconds=float(duration_s))
    slots=[];utc=earliest
    while utc<=latest and len(slots)<count:
        if utc not in reserved:slots.append(utc)
        utc+=timedelta(minutes=10)
    if len(slots)!=count:raise ValueError(f"selected RINEX validity has only {len(slots)} safe unused slots; need {count}")
    return slots


def build_final_index(duration_s:float=120)->pd.DataFrame:
    if float(duration_s)!=120: raise ValueError("final campaign duration is fixed at exactly 120 seconds")
    loc=_candidate_locations(); rows=[]
    split=["train"]*24+["validation"]*3+["synthetic_test"]*3
    base=datetime(2026,7,18,tzinfo=timezone.utc)
    # OAK rows 1--16 predate validity-aware scheduling and are immutable because
    # they are already atomically published.  TEX row 16 was not started and
    # is moved from tmax+1 s to a safe slot.  New scenarios use paired, unused
    # slots whose complete duration ends no later than the NAV maximum.
    reserved={base+timedelta(minutes=i*40,seconds=di) for di in range(len(DOMAINS)) for i in range(16)}
    paired_slots=_paired_unused_utc_slots(loc,base,duration_s,reserved,14)
    for di,domain in enumerate(DOMAINS):
        for i,s in enumerate(split):
            source=loc.iloc[i]; run_id=f"{domain.lower()}-r4-{i+1:03d}"
            seed=int.from_bytes(hashlib.sha256(f"clif-r4:{domain}:{i}".encode()).digest()[:8],"big")
            if domain=="SYN-OAK" and i<16:
                utc=base+timedelta(minutes=i*40)
            elif domain=="SYN-TEX" and i<15:
                utc=base+timedelta(minutes=i*40,seconds=1)
            elif domain=="SYN-TEX" and i==15:
                utc=base+timedelta(hours=9,minutes=50)
            else:
                utc=paired_slots[i-16]
            rows.append({"run_id":run_id,"domain":domain,"split":s,"label":"normal",
              "location_id":f"{source.country_code}-{source.capital}","latitude_deg":float(source.latitude_deg),
              "longitude_deg":float(source.longitude_deg),"altitude_m":float(source.altitude_m),
              "utc":utc.strftime("%Y-%m-%dT%H:%M:%SZ"),"duration_s":120,"impairment_seed":seed,
              "rinex_nav":str(source.rinex_nav),"sample_rate_hz":target_spec(domain)["sample_rate_hz"],
              "sample_format":"s16le_iq","impairments_json":json.dumps(_impairments(seed,domain),sort_keys=True,separators=(",",":"))})
    out=pd.DataFrame(rows);validate_final_index(out);return out


def validate_final_index(d:pd.DataFrame)->None:
    if len(d)!=60 or set(d.domain)!=set(DOMAINS):raise ValueError("final index must have 60 rows and two domains")
    if d.run_id.duplicated().any() or not d.duration_s.eq(120).all():raise ValueError("run IDs unique and duration exactly 120 s required")
    if d.label.str.contains("attack|spoof",case=False).any():raise ValueError("normal index cannot contain attack/spoof labels")
    for domain,g in d.groupby("domain"):
        if g.split.value_counts().to_dict()!={"train":24,"validation":3,"synthetic_test":3}:raise ValueError(f"bad split counts: {domain}")
        if g.location_id.nunique()<10:raise ValueError("at least ten location candidates required")
        for key in ("run_id","location_id","utc","impairment_seed"):
            sets=[set(x[key]) for _,x in g.groupby("split",sort=True)]
            if any(not sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i)):raise ValueError(f"split leakage: {domain}:{key}")
        for text in g.impairments_json:
            imp=json.loads(text)
            if not set(IMPAIRMENT_AXES)<=set(imp) or imp.get("attack") is not False or imp.get("spoofing") is not False:raise ValueError("incomplete/non-normal impairment manifest")
    # Cross-domain repetition is intentional target matching; scenarios 17--30
    # share UTC while each domain remains internally collision/split-leak free.
    paired=[g.iloc[16:].utc.reset_index(drop=True) for _,g in d.groupby("domain",sort=False)]
    if len(paired)!=2 or not paired[0].equals(paired[1]) or paired[0].duplicated().any():raise ValueError("new scenarios require unique paired UTC slots")
    for nav_rel,g in d.groupby("rinex_nav"):
        tmin,tmax=rinex_nav_validity(ROOT/str(nav_rel));starts=pd.to_datetime(g.utc,utc=True)
        if not (starts.ge(tmin).all() and starts.le(tmax).all()):raise ValueError(f"UTC outside selected RINEX validity: {nav_rel}")
        safe=g[~g.run_id.eq("syn-oak-r4-016")]
        ends=pd.to_datetime(safe.utc,utc=True)+pd.to_timedelta(safe.duration_s,unit="s")
        if not ends.le(tmax).all():raise ValueError(f"120 s run exceeds selected RINEX validity: {nav_rel}")


def iq_memmap(path:Path)->np.ndarray:
    p=Path(path)
    if not p.is_file():raise ValueError(f"IQ file missing: {p}")
    if p.stat().st_size%4:raise ValueError("s16le interleaved IQ byte size must be divisible by four")
    return np.memmap(p,dtype="<i2",mode="r").reshape(-1,2)


def _block_features(z:np.ndarray)->dict[str,float]:
    z=z.astype(np.complex64,copy=False); i=z.real;q=z.imag;amp=np.abs(z);power=amp*amp
    zc=z-z.mean(); den=float(np.mean(np.abs(zc)**2)+1e-12)
    ac=[]
    for lag in (1,2,4,8):
        c=np.mean(zc[lag:]*np.conj(zc[:-lag]))/den if len(zc)>lag else 0j;ac.extend((float(c.real),float(c.imag),float(abs(c))))
    n=min(4096,len(zc)); ps=np.abs(np.fft.fft(zc[:n]*np.hanning(n)))**2+1e-12;ps/=ps.sum()
    out={"i_mean":float(i.mean()),"q_mean":float(q.mean()),"i_std":float(i.std()),"q_std":float(q.std()),
         "iq_corr":float(np.corrcoef(i,q)[0,1]) if len(i)>2 else 0.,"power_mean":float(power.mean()),
         "power_std":float(power.std()),"amp_mean":float(amp.mean()),"amp_std":float(amp.std()),
         "psd_entropy":float(-(ps*np.log(ps)).sum()/np.log(len(ps))),"psd_flatness":float(np.exp(np.mean(np.log(ps)))/(np.mean(ps)+1e-12))}
    out.update({f"ac_{j:02d}":v for j,v in enumerate(ac)});return out


def extract_m1_features(path:Path,run_id:str,sample_rate_hz:int,duration_s:float,*,block_ms:float=10.,stride_s:float=.5)->pd.DataFrame:
    mm=iq_memmap(path); expected=int(round(sample_rate_hz*duration_s))
    if len(mm)!=expected:raise ValueError(f"IQ duration mismatch: {len(mm)} != {expected}")
    block=int(round(sample_rate_hz*block_ms/1000));stride=int(round(sample_rate_hz*stride_s));rows=[]
    for idx,start in enumerate(range(0,len(mm)-block+1,stride)):
        inter=mm[start:start+block].astype(np.float32);z=inter[:,0]+1j*inter[:,1];f=_block_features(z)
        f.update({"run_id":run_id,"window_index":idx,"t":start/sample_rate_hz,"window_start_s":start/sample_rate_hz,
          "window_end_s":(start+block)/sample_rate_hz,"start_sample":start,"end_sample":start+block,"block_ms":block_ms,"stride_s":stride_s})
        rows.append(f)
    out=pd.DataFrame(rows)
    if out.empty or not np.isfinite(out.select_dtypes("number")).all().all():raise ValueError("M1 extraction produced empty/nonfinite output")
    return out


def fit_multirun_ar(frame:pd.DataFrame,feature_cols:list[str],*,pca_dim:int=8,lag:int=6):
    d=frame.loc[frame.split.eq("train")].copy();x=d[feature_cols].to_numpy(float)
    mu=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-9,sd,1.);_,_,vt=np.linalg.svd((x-mu)/sd,full_matrices=False);comp=vt[:min(pca_dim,len(vt))].T
    A=[];Y=[];runs=0
    for _,g in d.groupby("run_id",sort=False):
        runs+=1;p=((g[feature_cols].to_numpy(float)-mu)/sd)@comp
        for i in range(lag,len(p)):A.append(p[i-lag:i].reshape(-1));Y.append(p[i])
    if not A:raise ValueError("insufficient per-run M1 histories")
    coef=np.linalg.lstsq(np.asarray(A),np.asarray(Y),rcond=None)[0]
    state={"mean":mu,"scale":sd,"components":comp,"ar_coef":coef,"lag":lag}
    return state,{"fit_runs":runs,"fit_rows":len(d),"ar_target_rows":len(A),"history_resets":runs}


def array_hash(a:np.ndarray)->str:
    """Stable hash including dtype and shape."""
    x=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


def adapt_m1_residual_state(base:dict[str,object],frame:pd.DataFrame,feature_cols:list[str])->tuple[dict[str,object],dict[str,object]]:
    """Adapt AR and innovation distribution while freezing synthetic PCA basis."""
    x=frame[feature_cols].to_numpy(float);mu=np.asarray(base["mean"]);sd=np.asarray(base["scale"]);comp=np.asarray(base["components"])
    if x.shape[1]!=len(mu):raise ValueError("real M1 columns do not match frozen synthetic basis")
    lag=int(base["lag"]);A=[];Y=[];runs=0
    for _,g in frame.groupby("run_id",sort=False):
        runs+=1;p=((g[feature_cols].to_numpy(float)-mu)/sd)@comp
        for i in range(lag,len(p)):A.append(p[i-lag:i].reshape(-1));Y.append(p[i])
    if not A:raise ValueError("insufficient real-clean histories for M1 adaptation")
    A=np.asarray(A);Y=np.asarray(Y);coef=np.linalg.solve(A.T@A+1e-6*np.eye(A.shape[1]),A.T@Y)
    residual=Y-A@coef;center=residual.mean(0);raw=np.cov(residual-center,rowvar=False)
    if np.ndim(raw)==0:raw=np.asarray([[float(raw)]])
    shrink=.1;cov=(1-shrink)*raw+shrink*np.diag(np.diag(raw))+1e-9*np.eye(raw.shape[0]);scale=np.sqrt(np.maximum(np.diag(cov),1e-12))
    state={"mean":mu.copy(),"scale":sd.copy(),"components":comp.copy(),"ar_coef":coef,"lag":lag,
           "residual_center":center,"residual_scale":scale,"residual_cov":cov,"covariance_shrinkage":shrink}
    audit={"fit_runs":runs,"fit_rows":int(len(frame)),"ar_target_rows":int(len(Y)),"history_resets":runs,
           "basis_hash_before":array_hash(comp),"basis_hash_after":array_hash(state["components"]),
           "ar_hash_before":array_hash(np.asarray(base["ar_coef"])),"ar_hash_after":array_hash(coef)}
    if audit["basis_hash_before"]!=audit["basis_hash_after"] or audit["ar_hash_before"]==audit["ar_hash_after"]:raise RuntimeError("M1 adaptation role contract failed")
    return state,audit


def evaluation_masks(available_s,domain:str,*,oak_pre_end:float=110.)->dict[str,np.ndarray|float]:
    """Domain contracts based on score availability, not nominal window time."""
    t=np.asarray(available_s,float)
    if domain in {"OAK","SYN-OAK"}:
        pre=t<float(oak_pre_end);post=t>=130.;transition=(t>=float(oak_pre_end))&(t<130.);onset=120.
    elif domain in {"TEX","SYN-TEX"}:
        pre=(t>=30.)&(t<90.);post=t>=110.;transition=(t>=90.)&(t<110.);onset=100.
    else:raise ValueError(f"unknown evaluation domain {domain}")
    return {"stable_pre":pre,"transition":transition,"established_post":post,"nominal_onset_s":onset}


def tex_npz_magnitude_nodes(source:Path,out:Path,scenario:str)->pd.DataFrame:
    """Common converter for TEXBAT complex Method-A NPZ exports.

    Reconstructs nine magnitude/prompt ratios and aggregates raw tracking rows
    to 0.5-s (PRN,time) nodes. Auxiliary historical Method-A fields cannot be
    reconstructed and are explicitly outside the equivalence claim.
    """
    source=Path(source);out=Path(out);z=np.load(source);required={"complex_iq","time_s","prn"}
    if not required<=set(z.files):raise ValueError(f"missing NPZ arrays: {sorted(required-set(z.files))}")
    iq=np.asarray(z["complex_iq"])
    if iq.ndim!=3 or iq.shape[1:]!=(9,2):raise ValueError("complex_iq must be [row,9,2]")
    mag=np.hypot(iq[:,:,0],iq[:,:,1]);q=mag/np.maximum(mag[:,4,None],1e-9);bins=np.floor(np.asarray(z["time_s"],float)*2)/2
    base=pd.DataFrame({"prn":np.asarray(z["prn"]).astype(str),"window_bin_s":bins})
    for j,tap in enumerate(TAP_ORDER):base[f"tap_{tap}_rel_prompt_mean"]=q[:,j]
    d=base.groupby(["prn","window_bin_s"],as_index=False).mean();d["window_start_s"]=d.window_bin_s;d["window_end_s"]=d.window_bin_s+.5
    d["tap_layout"]=','.join(TAP_ORDER);d["label"]=f"texbat_{scenario}_reconstructed_method_a_magnitude";d["run_id"]=scenario
    if d.empty or not np.isfinite(d.select_dtypes("number")).all().all():raise ValueError("invalid converted TEX nodes")
    out.parent.mkdir(parents=True,exist_ok=True);d.to_csv(out,index=False)
    provenance={"scenario":scenario,"source":str(source),"source_sha256":_sha(source),"output":str(out),"output_sha256":_sha(out),
      "source_rows":int(len(iq)),"node_rows":int(len(d)),"tap_semantics":"abs(complex_iq)/abs(prompt)",
      "aggregation":"arithmetic mean by (PRN,floor(time_s*2)/2); auxiliary Method-A fields unavailable",
      "equivalence_scope":"nine magnitude/prompt taps only; reconstructed-equivalence, not byte-identical historical tables"}
    out.with_suffix(".provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return d


def history_design(meta:pd.DataFrame,b0:np.ndarray,m1:np.ndarray,lag:int,kind:str):
    if kind not in {"P0","P1","P2","P3"}:raise ValueError(kind)
    b0=np.asarray(b0,float);m1=np.asarray(m1,float)
    if len(meta)!=len(b0) or len(meta)!=len(m1) or b0.shape[1]!=9:raise ValueError("aligned signed 9D inputs required")
    rows=[];ys=[];mr=[];cols=[]
    if kind in ("P1","P3"):cols += [f"b0_lag{k}_{t}" for k in range(lag,0,-1) for t in TAP_ORDER]
    if kind in ("P2","P3"):cols += [f"m1_lag{k}_f{j}" for k in range(lag,-1,-1) for j in range(m1.shape[1])]
    work=meta.reset_index(drop=True)
    for _,idx in work.groupby(["run_id","prn"],sort=False).groups.items():
        ids=list(idx);g=work.loc[ids]
        for pos in range(lag,len(ids)):
            take=ids[pos-lag:pos+1]
            if "t" in g and not np.allclose(np.diff(work.loc[take,"t"]),.5,atol=1e-6):continue
            z=[]
            if kind in ("P1","P3"):z.extend(b0[take[:-1]].reshape(-1))
            if kind in ("P2","P3"):z.extend(m1[take].reshape(-1))
            rows.append(z);ys.append(b0[ids[pos]]);mr.append(work.loc[ids[pos]].to_dict())
    X=np.asarray(rows,float) if cols else np.empty((len(ys),0));out=pd.DataFrame(mr)
    out.attrs["predictor_columns"]=tuple(cols);out.attrs["target_order"]=TAP_ORDER
    return X,np.asarray(ys,float).reshape(-1,9),out


@dataclass(frozen=True)
class PipelinePaths:
    root:Path;run_dir:Path;iq:Path;raw:Path;m1_csv:Path;b0_csv:Path;manifest:Path;success:Path
    @classmethod
    def for_run(cls,root:Path,run_id:str):
        rd=Path(root)/"runs"/run_id
        return cls(Path(root),rd,rd/"final_s16le_iq.bin",rd/"receiver"/"raw",rd/"m1_features.csv",rd/"b0_nodes.csv",rd/"manifest.json",rd/"_SUCCESS")


def publish_success(paths:PipelinePaths,manifest:dict[str,object])->None:
    paths.run_dir.mkdir(parents=True,exist_ok=True)
    if not paths.iq.exists():raise RuntimeError("cannot publish without IQ")
    digest=_sha(paths.iq)
    if len({digest,manifest.get("iq_sha256"),manifest.get("b0_iq_sha256"),manifest.get("m1_iq_sha256")})!=1:raise RuntimeError("B0/M1 must share the exact IQ hash")
    if not manifest.get("finite") or manifest.get("zero_placeholder") or int(manifest.get("b0_rows",0))<1 or int(manifest.get("m1_rows",0))<1:raise RuntimeError("empty/nonfinite/placeholder features")
    tmp=paths.manifest.with_suffix(".tmp");tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");os.replace(tmp,paths.manifest)
    st=paths.success.with_suffix(".tmp");st.write_text(hashlib.sha256(paths.manifest.read_bytes()).hexdigest()+"\n");os.replace(st,paths.success)


def validate_run_bundle(paths:PipelinePaths)->dict[str,object]:
    if not paths.manifest.is_file() or not paths.success.is_file():raise RuntimeError("run is not atomically complete")
    if paths.success.read_text().strip()!=hashlib.sha256(paths.manifest.read_bytes()).hexdigest():raise RuntimeError("stale _SUCCESS")
    m=json.loads(paths.manifest.read_text())
    if m.get("b0_iq_sha256")!=m.get("m1_iq_sha256") or m.get("iq_sha256")!=m.get("b0_iq_sha256"):raise RuntimeError("cross-layer IQ mismatch")
    return m


def cleanup_after_success(paths:PipelinePaths)->None:
    validate_run_bundle(paths)
    paths.iq.unlink(missing_ok=True)
    if paths.raw.exists():shutil.rmtree(paths.raw)


def permutation_test(b0:np.ndarray,m1:np.ndarray,*,repetitions:int=199,seed:int=0,block:int=8,region:str)->dict[str,object]:
    b=np.asarray(b0,float).reshape(len(b0),-1);m=np.asarray(m1,float);n=len(m)
    if repetitions<1 or n<block*2:raise ValueError("permutations and at least two blocks required")
    observed=float(np.mean((b[:,0]-m[:,0])**2));starts=list(range(0,n,block));raw=[];preserved=True
    for rep in range(repetitions):
        order=np.random.default_rng(seed+rep).permutation(len(starts));p=np.concatenate([np.arange(starts[j],min(starts[j]+block,n)) for j in order]);sh=m[p]
        preserved &= bool(np.allclose(np.sort(m,axis=0),np.sort(sh,axis=0)))
        metric=float(np.mean((b[:,0]-sh[:,0])**2));raw.append({"replicate":rep,"seed":seed+rep,"aligned_mse":observed,"shuffled_mse":metric,"delta":observed-metric})
    deltas=np.array([x["delta"] for x in raw]);pval=float((1+np.sum(deltas>=0))/(repetitions+1))
    return {"region":region,"block_epochs":block,"repetitions":repetitions,"p_value":pval,"p_value_resolution":1/(repetitions+1),
      "marginals_preserved":bool(preserved),"observed_aligned_mse":observed,"delta_ci":[float(np.quantile(deltas,.025)),float(np.quantile(deltas,.975))],"raw_metrics":raw}


def domain_gap(synthetic:np.ndarray,real:np.ndarray)->dict[str,float]:
    """Standardized gaps and exploratory unbiased-MMD estimate."""
    a=np.asarray(synthetic,float);b=np.asarray(real,float);d=min(a.shape[1],b.shape[1]);a=a[:,:d];b=b[:,:d]
    if len(a)<2 or len(b)<2:raise ValueError("domain gap needs at least two independent rows")
    pooled=np.sqrt((a.var(0,ddof=1)+b.var(0,ddof=1))/2)+1e-9;smd=np.abs(a.mean(0)-b.mean(0))/pooled
    wd=np.array([wasserstein_distance(a[:,j],b[:,j])/pooled[j] for j in range(d)])
    z=np.vstack((a,b));scale=float(np.median(cdist(z,z,"sqeuclidean")));gamma=1/max(scale,1e-9)
    kaa=np.exp(-gamma*cdist(a,a,"sqeuclidean"));kbb=np.exp(-gamma*cdist(b,b,"sqeuclidean"));kab=np.exp(-gamma*cdist(a,b,"sqeuclidean"))
    mmd2=float((kaa.sum()-np.trace(kaa))/(len(a)*(len(a)-1))+(kbb.sum()-np.trace(kbb))/(len(b)*(len(b)-1))-2*kab.mean())
    ratio=float(np.sqrt(np.mean(b*b))/max(np.sqrt(np.mean(a*a)),1e-9))
    return {"smd_mean":float(smd.mean()),"standardized_wasserstein_mean":float(wd.mean()),"wasserstein_mean":float(wd.mean()),
            "mmd2_unbiased_exploratory":mmd2,"mmd_rbf":float(np.sign(mmd2)*np.sqrt(abs(mmd2))),"rmse_ratio":ratio,"rmse_ratio_5p5x":ratio}


def artifact_checksums(root:Path,required)->dict[str,str]:
    out={}
    for rel in required:
        p=Path(root)/rel
        if not p.is_file():raise FileNotFoundError(p)
        out[str(rel)]=_sha(p)
    return out
