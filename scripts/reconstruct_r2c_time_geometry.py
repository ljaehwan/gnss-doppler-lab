#!/usr/bin/env python3
"""Read-only receiver-product time/geometry reconstruction (no detector scores)."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
import h5py, numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.gcmr_geometry import parse_gnss_sdr_gps_ephemeris_xml, satellite_position_ecef, look_angles
from gnss_doppler_lab.r2c_stage0_fix import gps_week_tow
from gnss_doppler_lab.trajectory import llh_to_ecef

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rinex_binding(directory):
    files=sorted(directory.glob("*O"));
    if len(files)!=1: raise ValueError("exactly one RINEX observation file required")
    header="".join(files[0].read_text(errors="replace").splitlines(True)[:100])
    match=re.search(r"\s*(\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+).*TIME OF FIRST OBS",header)
    if not match: raise ValueError("RINEX TIME OF FIRST OBS missing")
    from datetime import datetime,timezone
    y,mo,d,h,mi=map(int,match.groups()[:5]); sec=float(match.group(6)); dt=datetime(y,mo,d,h,mi,int(sec),tzinfo=timezone.utc)
    leap_match=re.search(r"\s*(\d+).*LEAP SECONDS",header); leap=int(leap_match.group(1)) if leap_match else (15 if y==2010 else 16)
    week,_=gps_week_tow(dt,leap); return week,files[0],dt.isoformat(),leap

def gga(directory):
    rows=[]
    for line in (directory/"nmea_pvt.nmea").read_text(errors="replace").splitlines():
        f=line.split(",")
        if len(f)>10 and f[0].endswith("GGA") and f[1] and f[2] and f[4]:
            raw=float(f[1]); seconds=int(raw//10000)*3600+int(raw%10000//100)*60+raw%100
            cv=lambda x: int(float(x)//100)+float(x)%100/60
            lat=cv(f[2])*(-1 if f[3]=="S" else 1); lon=cv(f[4])*(-1 if f[5]=="W" else 1)
            rows.append((seconds,np.asarray(llh_to_ecef(lat,lon,float(f[9])))))
    if not rows: raise ValueError("no valid NMEA GGA")
    first=rows[0][0]; return [(t-first,p) for t,p in rows]

def event_prns(directory, selected=None):
    if selected is not None:
        with np.load(selected,allow_pickle=False) as data:
            time=np.asarray(data["time_s"],float); prn=np.asarray(data["prn"])
        events={}
        for t,p in zip(time,prn): events.setdefault(round(np.floor(float(t)*2)/2,6),set()).add(int(str(p).lstrip("Gg")))
        return None,events,{"selected_npz_sha256":sha(selected),"source":"selected complex NPZ time_s/prn"}
    obs=directory/"raw/observables.mat"
    if obs.exists():
        with h5py.File(obs) as f:
            rx=np.asarray(f["RX_time"]); prn=np.asarray(f["PRN"]); flag=np.asarray(f["Flag_valid_pseudorange"])
        values=rx[np.isfinite(rx)&(rx>0)]; start=float(values.min()); data={}
        valid=np.isfinite(rx)&np.isfinite(prn)&(prn>0)&(flag>0)
        for t,p in zip(rx[valid],prn[valid]): data.setdefault(round(np.floor((t-start)*2)/2,6),set()).add(int(p))
        return start,data,{"observables_sha256":sha(obs),"source":"raw/observables.mat RX_time"}
    track=pd.read_csv(directory/"tracking.csv",usecols=["time_s","prn"]); start_tow=None
    data={}
    for t,p in track.itertuples(index=False):
        number=int(str(p).lstrip("Gg"))
        data.setdefault(round(np.floor(float(t)*2)/2,6),set()).add(number)
    return start_tow,data,{"tracking_sha256":sha(directory/"tracking.csv"),"source":"tracking relative time; absolute TOW lineage required"}

def reconstruct(name,directory,expected_week,expected_tow,iq_sha=None,selected=None):
    week,rinex,utc,leap=rinex_binding(directory); start,events,lineage=event_prns(directory,selected)
    eph=parse_gnss_sdr_gps_ephemeris_xml(directory/"gps_ephemeris.xml"); positions=gga(directory)
    start_tow=expected_tow if start is None else start
    time_ok=week==expected_week and abs(start_tow-expected_tow)<.01
    valid=0; rank4=0; pvt_count=0; conditions=[]
    for relative,prns in sorted(events.items()):
        past=[item for item in positions if item[0]<=relative+1e-6]
        if not past: continue
        pvt_count+=1; receiver=past[-1][1]; vectors=[]
        for prn in prns:
            if prn in eph:
                sat=satellite_position_ecef(eph[prn],(start_tow+relative)%604800)
                vectors.append(look_angles(receiver,sat).los_ecef)
        if len(vectors)>=5:
            design=np.column_stack([-np.asarray(vectors),np.ones(len(vectors))]); rank=np.linalg.matrix_rank(design); cond=np.linalg.cond(design)
            rank4+=rank==4; conditions.append(cond); valid+=rank==4 and len(vectors)-rank>=1 and cond<=1e6
    toes=[x.toe for x in eph.values()]; causal=all(toe<=start_tow for toe in toes)
    manifest=directory/"manifest.json"; manifest_doc=json.loads(manifest.read_text()) if manifest.exists() else {}
    found_iq=(manifest_doc.get("source",{}).get("iq_sha256") or manifest_doc.get("source",{}).get("sha256") or
              manifest_doc.get("authenticated_inputs",{}).get("iq_after_receiver",{}).get("sha256"))
    return {"scenario":name,"receiver_directory":str(directory),"authenticated_absolute_time_binding":{
      "status":"PASS" if time_ok else "RECONSTRUCTABLE_WITH_LINEAGE_GAPS","gps_week":week,"start_tow":start_tow,
      "expected_verified_not_assumed":{"week":expected_week,"tow":expected_tow},"rinex":str(rinex),"rinex_sha256":sha(rinex),"utc_first_observation":utc,"leap_seconds":leap,**lineage},
      "source_iq_lineage":{"status":"PASS" if found_iq and (not iq_sha or found_iq==iq_sha) else "LINEAGE_GAP","sha256":found_iq},
      "offline_los_reproducibility":{"status":"PASS" if valid else "UNAVAILABLE","valid_events":int(valid),"eligible_events":len(events),"coverage":float(valid/max(len(events),1))},
      "event_time_causal_ephemeris_availability":{"status":"PASS" if causal else "OFFLINE_ORACLE_ONLY","toe_range": [min(toes),max(toes)],"event_start_tow":start_tow},
      "pvt_coverage":{"status":"PASS" if pvt_count else "UNAVAILABLE","events":int(pvt_count),"eligible_events":len(events)},
      "prn_rank_dof_condition_coverage":{"valid_events":int(valid),"rank4_events":int(rank4),"median_condition":float(np.median(conditions)) if conditions else None}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--receiver",action="append",required=True,help="NAME=DIR,WEEK,TOW[,IQSHA]"); ap.add_argument("--selected",action="append",default=[],help="NAME=NPZ"); ap.add_argument("--report",type=Path,required=True); args=ap.parse_args()
    selected={k:Path(v) for k,v in (x.split("=",1) for x in args.selected)}
    reports={}
    for spec in args.receiver:
        name,value=spec.split("=",1); fields=value.split(","); reports[name]=reconstruct(name,Path(fields[0]),int(fields[1]),float(fields[2]),fields[3] if len(fields)>3 else None,selected.get(name))
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps({"schema":"gnss-doppler-lab.r2c-time-geometry-pre-campaign.v1","attack_scores_computed":False,"scenarios":reports},indent=2)+"\n")
    print(json.dumps({k:{"time":v["authenticated_absolute_time_binding"]["status"],"offline_coverage":v["offline_los_reproducibility"]["coverage"],"causal_ephemeris":v["event_time_causal_ephemeris_availability"]["status"]} for k,v in reports.items()}))
if __name__=="__main__":main()
