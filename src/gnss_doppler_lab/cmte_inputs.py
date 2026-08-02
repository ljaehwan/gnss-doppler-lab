"""CMTE canonical node preparation and split-reset B0 extraction adapters."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

TAPS=("E4","E3","E2","E","P","L","L2","L3","L4")
FEATURES=[f"tap_{x}_rel_prompt_mean" for x in TAPS]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def scorer_module(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[2]
    path=root/"scripts"/"score_tap_residual_common_drive.py"
    spec=importlib.util.spec_from_file_location("cmte_hardened_b0",path); module=importlib.util.module_from_spec(spec)
    if spec.loader is None: raise RuntimeError("cannot import hardened B0 loader")
    sys.modules[spec.name]=module; spec.loader.exec_module(module); return module


def load_checkpoint(checkpoint,expected_sha,device="cpu",root=None):
    """Use the existing hardened architecture/feature/scaler/hash verifier."""
    return scorer_module(root).load_frozen_b0(Path(checkpoint),device,expected_sha256=expected_sha)


def _rename_innovations(frame):
    return frame.rename(columns={f"innovation_{i}":f"residual_{i:03d}" for i in range(9)})


def split_contiguous_history_chunks(nodes, *, seq_len=12, cadence_s=.5, tolerance_s=1e-7):
    """Create deterministic frozen-B0 identities for contiguous cadence chunks.

    The identity contract is ``run_id/prn/segment/channel``.  No rows are
    interpolated or moved across a gap.  Chunks with too little history to
    produce a target are omitted and represented explicitly in the audit.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    local=nodes.copy()
    for column,default in (("segment","0"),("channel","0")):
        if column not in local: local[column]=default
    required=("run_id","prn","segment","channel","window_bin_s")
    missing=[c for c in required if c not in local]
    if missing: raise ValueError(f"cadence chunk input missing {missing}")
    if local[list(required)].isna().any().any(): raise ValueError("cadence chunk identity/timing must be non-null")
    local["window_bin_s"]=pd.to_numeric(local.window_bin_s,errors="raise")
    if not np.isfinite(local.window_bin_s.to_numpy(float)).all(): raise ValueError("cadence chunk timing must be finite")
    identity=["run_id","prn","segment","channel"]
    if local.duplicated([*identity,"window_bin_s"]).any(): raise ValueError("duplicate canonical node window in cadence identity")
    parts=[]; chunks=[]; gaps=0
    ordered=local.sort_values([*identity,"window_bin_s","window_start_s"],kind="mergesort")
    channel_counts=ordered.groupby(["run_id","prn","segment"],dropna=False)["channel"].nunique(dropna=False).to_dict()
    for key,group in ordered.groupby(identity,sort=True,dropna=False):
        group=group.reset_index(drop=True); bins=group.window_bin_s.to_numpy(float)
        breaks=np.flatnonzero(~np.isclose(np.diff(bins),cadence_s,atol=tolerance_s,rtol=0))+1
        gaps += int(len(breaks)); cuts=np.r_[0,breaks,len(group)]
        for chunk_index,(start,stop) in enumerate(zip(cuts[:-1],cuts[1:])):
            piece=group.iloc[int(start):int(stop)].copy(); rows=int(len(piece)); predictions=max(0,rows-seq_len)
            base=str(key[0]); channel_suffix=""
            if channel_counts[(key[0],key[1],key[2])]>1:
                channel_suffix=f"::channel-{hashlib.sha256(str(key[3]).encode()).hexdigest()[:12]}"
            chunk_run=f"{base}{channel_suffix}::cadence-chunk-{chunk_index:04d}"
            reason=None if predictions else "insufficient_history_rows_le_seq_len"
            record={"identity":{"run_id":str(key[0]),"prn":str(key[1]),"segment":str(key[2]),"channel":str(key[3])},
                    "chunk_index":int(chunk_index),"chunk_run_id":chunk_run,"first_window_bin_s":float(bins[int(start)]),
                    "last_window_bin_s":float(bins[int(stop)-1]),"input_rows":rows,"prediction_rows":predictions,
                    "status":"scored" if predictions else "dropped","reason":reason}
            chunks.append(record)
            if predictions:
                piece["run_id"]=chunk_run; parts.append(piece)
    scored=pd.concat(parts,ignore_index=True) if parts else ordered.iloc[:0].copy()
    dropped=[x for x in chunks if x["status"]=="dropped"]
    audit={"schema":"cmte-cadence-chunk-audit-v1","cadence_seconds":float(cadence_s),"seq_len":int(seq_len),
           "identity_columns":identity,"identity_groups":int(ordered.groupby(identity,dropna=False).ngroups),
           "input_rows":int(len(local)),"gaps_detected":int(gaps),"chunks_total":int(len(chunks)),
           "chunks_scored":int(len(chunks)-len(dropped)),"chunks_dropped":int(len(dropped)),
           "rows_dropped":int(sum(x["input_rows"] for x in dropped)),
           "dropped_reasons":{"insufficient_history_rows_le_seq_len":int(len(dropped))} if dropped else {},
           "never_interpolate_fill_or_bridge":True,"chunks":chunks}
    return scored,audit


def extract_innovations(nodes,model,features,mean,std,*,seq_len=12,device="cpu",extractor=None):
    if extractor is None:
        from gnss_doppler_lab.tap_residual_common_drive import extract_b0_innovations
        extractor=extract_b0_innovations
    return _rename_innovations(extractor(nodes,model,features,mean,std,seq_len=seq_len,device=device))


def _extract_chunked(nodes,model,features,mean,std,*,seq_len,device,extractor):
    chunked,audit=split_contiguous_history_chunks(nodes,seq_len=seq_len)
    residual=extract_innovations(chunked,model,features,mean,std,seq_len=seq_len,device=device,extractor=extractor)
    residual.attrs["cadence_chunk_audit"]=audit
    return residual


def extract_role_innovations(nodes,model,features,mean,std,*,seq_len=12,device="cpu",extractor=None):
    """Partition canonical node windows, then reset B0 at roles and gaps.

    Rewritten run IDs force role-local state.  Each role is then split by
    ``run_id/prn/segment/channel`` at every non-0.5 s cadence boundary before
    invoking frozen B0.  No history window crosses either kind of boundary.
    """
    bounds={"train":(0.,240.),"validation":(250.,330.),"test":(340.,None)}; result={}
    for role,(lo,hi) in bounds.items():
        mask=nodes.window_start_s.astype(float).ge(lo)
        if hi is not None: mask &= nodes.window_end_s.astype(float).le(hi)
        local=nodes.loc[mask].copy()
        local["source_run_id"]=local.run_id.astype(str)
        segment=local["segment"].astype(str) if "segment" in local else local.get("segment_index",pd.Series("0",index=local.index)).astype(str)
        local["run_id"]=local.run_id.astype(str)+f"::cmte-{role}-reset::segment-"+segment
        result[role]=_extract_chunked(local,model,features,mean,std,seq_len=seq_len,device=device,extractor=extractor)
    return result


def extract_recording_innovations(nodes,model,features,mean,std,*,scenario,seq_len=12,device="cpu",extractor=None):
    local=nodes.copy(); local["source_run_id"]=local.run_id.astype(str)
    segment=local["segment"].astype(str) if "segment" in local else local.get("segment_index",pd.Series("0",index=local.index)).astype(str)
    local["run_id"]=f"cmte-{scenario}-fresh::"+local.run_id.astype(str)+"::segment-"+segment
    return _extract_chunked(local,model,features,mean,std,seq_len=seq_len,device=device,extractor=extractor)


def validate_node_table(frame):
    if "segment" not in frame and "segment_index" in frame: frame["segment"]=frame["segment_index"]
    required={"run_id","prn","channel","segment","window_start_s","window_end_s","window_mid_s","window_bin_s",*FEATURES}
    missing=sorted(required-set(frame))
    if missing: raise ValueError(f"canonical node table missing {missing}")
    if frame.duplicated(["run_id","prn","segment","channel","window_bin_s"]).any(): raise ValueError("duplicate canonical node window")
    if not np.isfinite(frame[["window_start_s","window_end_s","window_mid_s","window_bin_s",*FEATURES]].to_numpy(float)).all(): raise ValueError("nonfinite canonical node value")
    for _,g in frame.groupby(["run_id","prn","segment","channel"],sort=False):
        b=np.sort(g.window_bin_s.to_numpy(float))
        if len(b)>1 and not np.allclose(np.diff(b),.5,atol=1e-7): raise ValueError("node table bridges gap or violates 0.5 s cadence")
    return frame


def convert_complex_npz(source,out_csv,manifest_path,*,scenario,source_metadata=None):
    """Convert explicit complex NPZ samples to 1 s/0.5 s Method-A node windows."""
    source=Path(source).resolve(strict=True); z=np.load(source,allow_pickle=False)
    required={"complex_iq","prn","time_s"}; missing=sorted(required-set(z.files))
    if missing: raise ValueError(f"NPZ missing {missing}")
    if "segment" not in z.files and "segment_index" not in z.files: raise ValueError("NPZ missing segment/segment_index")
    iq=np.asarray(z["complex_iq"])
    if iq.ndim!=3 or iq.shape[1:]!=(9,2): raise ValueError("complex_iq must have shape [N,9,2]")
    n=len(iq); raw_prn=np.asarray(z["prn"]); prn=np.asarray([f"G{int(x):02d}" if str(x).lstrip("+-").isdigit() else str(x) for x in raw_prn]); segment=np.asarray(z["segment"] if "segment" in z.files else z["segment_index"]); times=np.asarray(z["time_s"],float)
    channel=np.asarray(z["channel"]) if "channel" in z.files else np.zeros(n,int)
    if any(len(x)!=n for x in (prn,segment,times,channel)) or not np.isfinite(iq).all() or not np.isfinite(times).all(): raise ValueError("invalid NPZ arrays")
    mag=np.hypot(iq[:,:,0],iq[:,:,1]); rows=[]; run=f"texbat-{scenario}-cmte-reconstructed"
    keys=pd.DataFrame({"prn":prn,"segment":segment,"channel":channel}).drop_duplicates().itertuples(index=False,name=None)
    for p,s,c in keys:
        idx=np.flatnonzero((prn==p)&(segment==s)&(channel==c)); order=idx[np.argsort(times[idx],kind="mergesort")]; tt=times[order]; mm=mag[order]
        if len(tt)<2: continue
        cadence=float(np.median(np.diff(tt)))
        # split again at timestamp gaps; segment metadata alone is not trusted.
        cuts=np.r_[0,np.flatnonzero(np.diff(tt)>max(1e-9,1.5*cadence))+1,len(tt)]
        for piece,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
            t=tt[a:b]; m=mm[a:b]
            if not len(t): continue
            start=float(t.min())
            window_index=0
            while start+1<=t.max()+cadence+1e-9:
                mask=(t>=start)&(t<start+1)
                if mask.any() and t[mask].min()<=start+cadence+1e-7 and t[mask].max()>=start+1-cadence-1e-7:
                    rel=m[mask]/(m[mask,4,None]+1e-8)
                    row={"run_id":run,"prn":p,"channel":c,"segment":f"{s}:{piece}","segment_index":s,
                         "window_start_s":start,"window_end_s":start+1,"window_mid_s":start+.5,"window_bin_s":np.floor((start+.5)*2+.5)/2,
                         "window_index":window_index,
                         "epoch_count":int(mask.sum()),"tap_count":9,"tap_layout":",".join(TAPS),"label":f"texbat_{scenario}_9tap_w1.0_s0.5"}
                    row.update({name:float(rel[:,j].mean()) for j,name in enumerate(FEATURES)}); rows.append(row)
                start+=.5; window_index+=1
    frame=pd.DataFrame(rows); validate_node_table(frame); out=Path(out_csv); out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(out,index=False)
    doc={"schema":"cmte-input-v1","scenario":scenario,"role":"normal_clean" if scenario=="cleanStatic" else "evaluation_only",
      "producer_grade":"reconstructed_equivalence","source_path":str(source),"source_sha256":sha256(source),"node_path":str(out.resolve()),
      "node_sha256":sha256(out),"checkpoint_sha256":"0"*64,"window_seconds":1.,"stride_seconds":.5,"tap_layout":list(TAPS),
      "complex_semantics":"magnitude=hypot(I,Q); per-epoch tap/prompt then window mean","never_bridge_segment_or_gap":True,
      "source_metadata":source_metadata or {}}
    Path(manifest_path).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); return frame


def copy_verified_ds4(node_csv,out_csv,manifest_path,*,source_manifest=None):
    src=Path(node_csv).resolve(strict=True); frame=validate_node_table(pd.read_csv(src)); out=Path(out_csv); out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(out,index=False)
    doc={"schema":"cmte-input-v1","scenario":"DS4","role":"evaluation_only","producer_grade":"verified_node_artifact",
      "source_path":str(src),"source_sha256":sha256(src),"node_path":str(out.resolve()),"node_sha256":sha256(out),"checkpoint_sha256":"0"*64,
      "upstream_manifest":str(Path(source_manifest).resolve(strict=True)) if source_manifest else None}
    Path(manifest_path).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); return frame
