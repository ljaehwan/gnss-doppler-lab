"""Atomic, provenance-rich export of the current three-tap E/P/L dataset."""
from __future__ import annotations
import csv,hashlib,json,os,platform,subprocess,tempfile
from pathlib import Path
from typing import Sequence
import numpy as np
from .tracking_feature_windows import TrackingWindowFeatureRecord,collect_receiver_run_tracking_feature_records
SCHEMA_VERSION=3

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _git(root):
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
        dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip())
        return sha,dirty
    except Exception: return "unknown",True

def _source(run):
    mp=run/"manifest.json"; data=json.loads(mp.read_text()); raw=run/str(data.get("tracking",{}).get("raw_directory","raw")); mats=sorted(raw.glob("epl_tracking_ch_*.mat"))
    if not mats: raise ValueError(f"receiver run has no source tracking MAT files: {run.name}")
    entries=[{"path":str(p),"sha256":_sha(p)} for p in mats]
    fp=hashlib.sha256("".join(sorted(x["sha256"] for x in entries)).encode()).hexdigest()
    return {"run_id":run.name,"receiver_manifest":{"path":str(mp),"sha256":_sha(mp)},"source_mat_files":entries,"source_fingerprint":fp}

def _temp(path):
    fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent); os.close(fd); return Path(name)

def export_tracking_feature_dataset(receiver_run_dirs: Sequence[str|Path], *, output_path: str|Path, manifest_output_path: str|Path|None=None,window_s=1.0,stride_s=0.5,min_epochs=4,label="normal",prns=None):
    runs=[Path(x) for x in receiver_run_dirs]
    if not runs: raise ValueError("receiver_run_dirs must contain at least one run")
    ids=[x.name for x in runs]; dup=sorted(x for x in set(ids) if ids.count(x)>1)
    if dup: raise ValueError(f"duplicate run_id values are not allowed: {', '.join(dup)}")
    output=Path(output_path); manifest=Path(manifest_output_path) if manifest_output_path else output.with_suffix(".manifest.json")
    if output.resolve()==manifest.resolve(): raise ValueError("dataset CSV and manifest path alias")
    output.parent.mkdir(parents=True,exist_ok=True); manifest.parent.mkdir(parents=True,exist_ok=True)
    fields=list(TrackingWindowFeatureRecord.__dataclass_fields__); rows=[]; sources=[]
    for run in sorted(runs,key=lambda x:x.name):
        src=_source(run); sources.append(src); records=collect_receiver_run_tracking_feature_records(run,window_s=window_s,stride_s=stride_s,min_epochs=min_epochs,label=label,prns=prns)
        if not records: raise ValueError(f"zero rows generated for run {run.name}")
        for record in records:
            row=record.to_row()
            if list(row)!=fields: raise ValueError(f"feature schema mismatch for {run.name}")
            if row["run_id"]!=run.name or row["source_fingerprint"]!=src["source_fingerprint"]: raise ValueError(f"run provenance mismatch for {run.name}")
            rows.append(row)
    if not rows: raise ValueError("zero rows generated for dataset")
    rows.sort(key=lambda r:(str(r["run_id"]),str(r["prn"]),int(r["channel"]),int(r["segment_index"]),int(r["window_index"])))
    tc,tm=_temp(output),_temp(manifest)
    try:
        with tc.open("w",encoding="utf-8",newline="") as h:
            w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)
        root=Path(__file__).resolve().parents[2]; gsha,dirty=_git(root)
        doc={"schema":"gnss-doppler-lab.tracking-feature-dataset-epl","schema_version":SCHEMA_VERSION,"tap_count":3,"tap_layout":["E","P","L"],"gnss_sdr_note":"GNSS-SDR 0.0.19 GPS_L1_CA_DLL_PLL_Tracking computes real E/P/L; MAT abs_VE/abs_VL are all-zero format placeholders and are not used.","parameters":{"window_s":window_s,"stride_s":stride_s,"min_epochs":min_epochs,"label":label},"feature_schema":fields,"feature_columns":fields[fields.index("near_sym_mean"): ],"row_count":len(rows),"sources":sources,"project":{"git_sha":gsha,"dirty":dirty},"runtime":{"python":platform.python_version(),"numpy":np.__version__},"output_csv":{"path":str(output),"sha256":_sha(tc)}}
        tm.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
        manifest.unlink(missing_ok=True); os.replace(tc,output); os.replace(tm,manifest)
    finally: tc.unlink(missing_ok=True); tm.unlink(missing_ok=True)
    return output
