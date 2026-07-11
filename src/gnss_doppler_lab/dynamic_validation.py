"""Dynamic truth versus GNSS-SDR PVT/velocity/Doppler validation."""
from __future__ import annotations
import argparse, csv, hashlib, hmac, json, math, os, re, tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .trajectory import MIN_PHYSICAL_ECEF_RADIUS_M, ecef_to_llh, llh_to_enu

POS_RE = re.compile(r"Position at (.+?) UTC using (\d+) observations is Lat = ([+-]?[\d.]+) \[deg\], Long = ([+-]?[\d.]+) \[deg\], Height = ([+-]?[\d.]+) \[m\]")
VEL_RE = re.compile(r"Velocity: East: ([+-]?[\d.]+) \[m/s\], North: ([+-]?[\d.]+) \[m/s\], Up = ([+-]?[\d.]+) \[m/s\]")
NAV_MESSAGE_RE = re.compile(
    r"New GPS NAV message received in channel \d+: subframe \d+ "
    r"from satellite GPS PRN (\d{1,2})(?=\s|$)"
)
FIELDS = ["utc","trajectory_time_s","truth_lat_deg","truth_lon_deg","truth_h_m","pvt_lat_deg","pvt_lon_deg","pvt_h_m","truth_e_m","truth_n_m","truth_u_m","pvt_e_m","pvt_n_m","pvt_u_m","truth_ve_mps","truth_vn_mps","truth_vu_mps","pvt_ve_mps","pvt_vn_mps","pvt_vu_mps","horizontal_position_error_m","vertical_position_error_m","position_3d_error_m","horizontal_velocity_error_mps","velocity_3d_error_mps","observation_count"]

def parse_pvt_log(path):
    lines=Path(path).read_text(errors="replace").splitlines(); out=[]
    for i,line in enumerate(lines):
        m=POS_RE.search(line)
        if not m: continue
        if i+1>=len(lines) or not (v:=VEL_RE.search(lines[i+1])): raise ValueError(f"position line {i+1} has no following velocity line")
        dt=datetime.strptime(m[1],"%Y-%b-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        out.append({"utc":dt,"observations":int(m[2]),"llh":tuple(map(float,m.groups()[2:])),"vel":tuple(map(float,v.groups()))})
    if not out: raise ValueError(f"no GNSS-SDR PVT Position/Velocity lines found in {path}")
    return out

def parse_nav_decoded_prns(path):
    """Return GPS PRNs with decoded NAV messages, in first-seen order."""
    seen=set(); out=[]
    for line in Path(path).read_text(errors="replace").splitlines():
        for match in NAV_MESSAGE_RE.finditer(line):
            prn=f"G{int(match.group(1)):02d}"
            if prn not in seen:
                seen.add(prn); out.append(prn)
    return out

def finite_difference(t, xyz):
    t=np.asarray(t,float); x=np.asarray(xyz,float)
    if len(t)<2 or np.any(np.diff(t)<=0): raise ValueError("truth timestamps require at least two strictly increasing values")
    return np.gradient(x,t,axis=0,edge_order=1)

def metrics(a):
    a=np.asarray(a,float); return {"median":float(np.median(a)),"p95":float(np.percentile(a,95)),"max":float(np.max(a))}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def atomic(path, writer, overwrite=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not overwrite: raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    fd,tmp=tempfile.mkstemp(prefix="."+path.name+".",suffix=".tmp",dir=path.parent); os.close(fd)
    try: writer(Path(tmp)); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def read_truth(path, coordinate_system):
    if coordinate_system not in ("llh", "ecef"):
        raise ValueError("scenario.position.coordinate_system must be 'llh' or 'ecef'")
    rows=[]
    with open(path,newline="") as f:
        for n,r in enumerate(csv.reader(f),1):
            try: rows.append(tuple(map(float,r)))
            except Exception as e: raise ValueError(f"invalid truth row {n}") from e
    if not rows or any(len(r)!=4 for r in rows): raise ValueError("truth CSV must contain exactly four columns")
    result=np.asarray(rows)
    if not np.all(np.isfinite(result)):
        raise ValueError("truth CSV values must be finite")
    if coordinate_system == "llh":
        if np.any((result[:,1] < -90) | (result[:,1] > 90) | (result[:,2] < -180) | (result[:,2] > 180)):
            raise ValueError("truth CSV LLH coordinates out of range")
        return result
    radii=np.linalg.norm(result[:,1:4],axis=1)
    if np.any(radii < MIN_PHYSICAL_ECEF_RADIUS_M):
        raise ValueError(f"truth CSV ECEF geocentric radius must be at least {MIN_PHYSICAL_ECEF_RADIUS_M:.0f} m")
    llh=np.asarray([ecef_to_llh(*xyz) for xyz in result[:,1:4]])
    return np.column_stack((result[:,0],llh))

def doppler_stats(path):
    d={}; series={}
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            p=r["prn"]; series.setdefault(p,[[],[]]); series[p][0].append(float(r["time_s"])); series[p][1].append(float(r["carrier_doppler_hz"]))
    for p,(_,v) in series.items():
        a=np.asarray(v); d[p]={"sample_count":len(v),"median_hz":float(np.median(a)),"min_hz":float(a.min()),"max_hz":float(a.max()),"range_hz":float(np.ptp(a)),"std_hz":float(a.std())}
    return d,series

def validate_run(receiver_run, rf_manifest=None, overwrite=False, legacy_gps_utc_offset_seconds=None):
    rr=Path(receiver_run); rm=json.loads((rr/"manifest.json").read_text())
    rfp=Path(rf_manifest or rm["source"]["rf_manifest"]); rf=json.loads(rfp.read_text())
    position=rf["scenario"]["position"]
    truthp=Path(position["path"])
    expected_truth_sha=position.get("sha256")
    if not isinstance(expected_truth_sha,str) or re.fullmatch(r"[0-9a-f]{64}",expected_truth_sha) is None:
        raise ValueError("scenario.position.sha256 must be a lowercase 64-character SHA-256 digest")
    current_truth_sha=sha(truthp)
    if not hmac.compare_digest(current_truth_sha,expected_truth_sha):
        raise ValueError("truth CSV SHA-256 mismatch with RF manifest scenario.position.sha256")
    tr=read_truth(truthp,position.get("coordinate_system")); start=datetime.fromisoformat(rf["scenario"]["utc"].replace("Z","+00:00"))
    tm=rf["scenario"].get("time")
    if tm is None:
        if legacy_gps_utc_offset_seconds is None:
            raise ValueError("legacy/ambiguous RF manifest has no scenario.time timescale metadata; pass --legacy-gps-utc-offset-seconds explicitly")
        alignment=float(legacy_gps_utc_offset_seconds)
        alignment_mode="explicit_legacy_override"
    else:
        required={"requested_utc","simulator_input_calendar","simulator_input_time_scale","gps_minus_utc_seconds"}
        if not required.issubset(tm) or tm["simulator_input_time_scale"] != "GPST" or tm["requested_utc"] != rf["scenario"]["utc"]:
            raise ValueError("invalid or inconsistent scenario.time timescale metadata")
        if legacy_gps_utc_offset_seconds is not None:
            raise ValueError("legacy GPS-UTC override is not allowed for a corrected manifest")
        alignment=0.0; alignment_mode="corrected_manifest_direct_utc"
    origin=tr[0,1:4]; tenu=np.array([llh_to_enu(*q,*origin) for q in tr[:,1:4]]); tvel=finite_difference(tr[:,0],tenu)
    fixes=parse_pvt_log(rr/"receiver.log"); rows=[]
    for f in fixes:
        ts=(f["utc"]-start).total_seconds()+alignment
        if ts<tr[0,0] or ts>tr[-1,0]: continue
        llh=np.array([np.interp(ts,tr[:,0],tr[:,j]) for j in range(1,4)])
        te=np.array([np.interp(ts,tr[:,0],tenu[:,j]) for j in range(3)]); tv=np.array([np.interp(ts,tr[:,0],tvel[:,j]) for j in range(3)])
        pe=np.array(llh_to_enu(*f["llh"],*origin)); pv=np.array(f["vel"]); de=pe-te; dv=pv-tv
        vals=[f["utc"].isoformat().replace("+00:00","Z"),ts,*llh,*f["llh"],*te,*pe,*tv,*pv,np.hypot(*de[:2]),abs(de[2]),np.linalg.norm(de),np.hypot(*dv[:2]),np.linalg.norm(dv),f["observations"]]
        rows.append(dict(zip(FIELDS,vals)))
    if not rows: raise ValueError("no PVT fixes overlap the 10 Hz truth time range")
    ds,series=doppler_stats(rr/rm["tracking"]["csv"]); last=max(max(x[0]) for x in series.values()); final=sum(max(x[0])>=last-1 for x in series.values())
    nav_decoded_prns=parse_nav_decoded_prns(rr/"receiver.log")
    nav_set=set(nav_decoded_prns)
    nav_ds={p:ds[p] for p in nav_decoded_prns if p in ds}
    nav_series={p:series[p] for p in nav_decoded_prns if p in series}
    acquisition_only_prns=[p for p in series if p not in nav_set]
    def col(k): return [r[k] for r in rows]
    summary={"schema_version":3,"receiver_run_id":rm.get("receiver_run_id"),"rf_run_id":rf.get("run_id"),"time_alignment":{"mode":alignment_mode,"correction_seconds":alignment},"fix_count":len(rows),"first_fix_utc":rows[0]["utc"],"last_fix_utc":rows[-1]["utc"],"horizontal_position_error_m":metrics(col("horizontal_position_error_m")),"position_3d_error_m":metrics(col("position_3d_error_m")),"horizontal_velocity_error_mps":metrics(col("horizontal_velocity_error_mps")),"prn_counts":{"acquired":rm.get("acquisition",{}).get("tracked_prn_count",0),"final_tracking":final,"pvt_used":max(col("observation_count")),"nav_decoded":len(nav_decoded_prns)},"nav_decoded_prns":nav_decoded_prns,"acquisition_only_prns":acquisition_only_prns,"doppler_by_prn_all":ds,"doppler_by_prn_nav_decoded":nav_ds,"doppler_by_prn":ds,"doppler_by_prn_compatibility":"legacy alias of doppler_by_prn_all","artifacts":{"truth":{"path":str(truthp),"sha256":expected_truth_sha},"receiver_log":{"path":str(rr/"receiver.log"),"sha256":sha(rr/"receiver.log")},"tracking":{"path":str(rr/rm["tracking"]["csv"]),"sha256":sha(rr/rm["tracking"]["csv"])},"rf_manifest":{"path":str(rfp),"sha256":sha(rfp)},"receiver_manifest":{"path":str(rr/"manifest.json"),"sha256":sha(rr/"manifest.json")}}}
    out=rr/"validation"; atomic(out/"per_fix.csv",lambda p:_write_csv(p,rows),overwrite); atomic(out/"summary.json",lambda p:p.write_text(json.dumps(summary,indent=2)+"\n"),overwrite); atomic(out/"dashboard.png",lambda p:_plot(p,rows,nav_series,rr.name),overwrite)
    return summary

def _write_csv(p,rows):
    with p.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
def _plot(p,rows,series,title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(13,9)); t=np.array([r["trajectory_time_s"] for r in rows])
    ax[0,0].plot([r["truth_e_m"] for r in rows],[r["truth_n_m"] for r in rows],label="truth"); ax[0,0].plot([r["pvt_e_m"] for r in rows],[r["pvt_n_m"] for r in rows],label="PVT"); ax[0,0].set_title("ENU path"); ax[0,0].set_xlabel("East [m]"); ax[0,0].set_ylabel("North [m]"); ax[0,0].set_aspect("equal",adjustable="datalim"); ax[0,0].legend()
    ax[0,1].plot(t,[r["horizontal_position_error_m"] for r in rows]); ax[0,1].set_title("Horizontal position error [m]")
    for key,lab,style in [("truth_ve_mps","truth E","-"),("truth_vn_mps","truth N","-"),("pvt_ve_mps","PVT E","--"),("pvt_vn_mps","PVT N","--")]: ax[1,0].plot(t,[r[key] for r in rows],style,label=lab)
    ax[1,0].set_title("EN velocity [m/s]"); ax[1,0].legend()
    for prn,(x,y) in sorted(series.items()): ax[1,1].plot(x,y,lw=.6,label=prn)
    ax[1,1].set_title("NAV-decoded carrier Doppler [Hz]")
    if series: ax[1,1].legend(ncol=3,fontsize=7)
    for a in ax.flat: a.grid(alpha=.25)
    for a in (ax[0,1],ax[1,0],ax[1,1]): a.set_xlabel("trajectory/receiver time [s]")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(p,dpi=150,format="png"); plt.close(fig)
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("receiver_run"); ap.add_argument("--rf-manifest"); ap.add_argument("--overwrite",action="store_true",help="atomically replace existing validation outputs (default: refuse)"); ap.add_argument("--legacy-gps-utc-offset-seconds",type=float,help="explicit truth-time correction for a legacy manifest lacking timescale metadata"); a=ap.parse_args(argv)
    s=validate_run(a.receiver_run,a.rf_manifest,a.overwrite,a.legacy_gps_utc_offset_seconds); print(json.dumps(s,indent=2))
if __name__=="__main__": main()
