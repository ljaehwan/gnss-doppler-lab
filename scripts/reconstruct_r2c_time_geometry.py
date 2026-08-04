#!/usr/bin/env python3
"""Strict read-only source-derived Stage-0 time/geometry reconstruction."""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.gcmr_geometry import parse_gnss_sdr_gps_ephemeris_xml,satellite_position_ecef,look_angles
from gnss_doppler_lab.trajectory import llh_to_ecef
GPS_EPOCH=datetime(1980,1,6,tzinfo=timezone.utc)
LEAP_EFFECTIVE=((datetime(1981,7,1,tzinfo=timezone.utc),1),(datetime(1982,7,1,tzinfo=timezone.utc),2),(datetime(1983,7,1,tzinfo=timezone.utc),3),(datetime(1985,7,1,tzinfo=timezone.utc),4),(datetime(1988,1,1,tzinfo=timezone.utc),5),(datetime(1990,1,1,tzinfo=timezone.utc),6),(datetime(1991,1,1,tzinfo=timezone.utc),7),(datetime(1992,7,1,tzinfo=timezone.utc),8),(datetime(1993,7,1,tzinfo=timezone.utc),9),(datetime(1994,7,1,tzinfo=timezone.utc),10),(datetime(1996,1,1,tzinfo=timezone.utc),11),(datetime(1997,7,1,tzinfo=timezone.utc),12),(datetime(1999,1,1,tzinfo=timezone.utc),13),(datetime(2006,1,1,tzinfo=timezone.utc),14),(datetime(2009,1,1,tzinfo=timezone.utc),15),(datetime(2012,7,1,tzinfo=timezone.utc),16),(datetime(2015,7,1,tzinfo=timezone.utc),17),(datetime(2017,1,1,tzinfo=timezone.utc),18))
def gps_utc_offset(stamp):return max((value for moment,value in LEAP_EFFECTIVE if stamp>=moment),default=0)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def valid_sha(value):return isinstance(value,str) and bool(re.fullmatch(r"[0-9a-f]{64}",value))
def checksum(line):
    if not line.startswith("$") or "*" not in line:return False
    body,given=line[1:].split("*",1);value=0
    for char in body:value^=ord(char)
    try:return value==int(given[:2],16)
    except ValueError:return False
def rinex(directory):
    files=sorted(directory.glob("*O"))
    if len(files)!=1:raise ValueError("exactly one RINEX observation file required")
    header=[]
    for line in files[0].read_text(errors="replace").splitlines():
        header.append(line)
        if "END OF HEADER" in line:break
    match=next((re.match(r"\s*(\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)",x) for x in header if "TIME OF FIRST OBS" in x),None)
    if not match:raise ValueError("RINEX TIME OF FIRST OBS absent")
    line=next(x for x in header if "TIME OF FIRST OBS" in x);system=line[48:51].strip() or "UNKNOWN"
    y,mo,d,h,mi=map(int,match.groups()[:5]);sec=float(match.group(6));stamp=datetime(y,mo,d,h,mi,tzinfo=timezone.utc)+timedelta(seconds=sec)
    total=(stamp-GPS_EPOCH).total_seconds()+(gps_utc_offset(stamp) if system in {"UTC","GLO"} else 0)
    week=int(total//604800);tow=total%604800
    return {"path":str(files[0]),"sha256":sha(files[0]),"gps_week":week,"first_epoch_tow_s":tow,"time_system":system,"fractional_second":sec-int(sec),"recording_utc_date":stamp.date().isoformat(),"gps_utc_offset_s":gps_utc_offset(stamp)}
def nmea_pvt(directory,start_tow,recording_date):
    leap=gps_utc_offset(datetime.fromisoformat(recording_date).replace(tzinfo=timezone.utc))
    rows=[];invalid_checksum=0;invalid_fix=0;previous=None;roll=0
    for line in (directory/"nmea_pvt.nmea").read_text(errors="replace").splitlines():
        if not line.startswith("$GPGGA"):continue
        if not checksum(line):invalid_checksum+=1;continue
        f=line.split(",")
        try:
            raw=float(f[1]);quality=int(f[6]);h=int(raw//10000);m=int(raw%10000//100);s=raw%100;seconds=h*3600+m*60+s
            if quality<=0:invalid_fix+=1;continue
            cv=lambda x:int(float(x)//100)+float(x)%100/60
            lat=cv(f[2])*(-1 if f[3]=="S" else 1);lon=cv(f[4])*(-1 if f[5]=="W" else 1);height=float(f[9])+float(f[11])
        except (ValueError,IndexError):invalid_fix+=1;continue
        if previous is not None and seconds<previous-43200:roll+=86400
        elif previous is not None and seconds<previous:continue
        previous=seconds;start_utc_sod=(start_tow-leap)%86400;absolute_day=start_tow-start_utc_sod+seconds+roll
        rows.append((absolute_day-start_tow,np.asarray(llh_to_ecef(lat,lon,height))))
    if not rows:raise ValueError("no checksum-valid fixed NMEA GGA")
    return rows,{"valid_fixes":len(rows),"invalid_checksum":invalid_checksum,"invalid_fix":invalid_fix,"time_system":"UTC","gps_utc_offset_s":leap,"height_datum":"ellipsoid_msl_plus_geoid"}
def raw_observables(directory):
    path=directory/"raw/observables.mat"
    with h5py.File(path) as f:
        rx=np.asarray(f["RX_time"],float);tow=np.asarray(f["TOW_at_current_symbol_s"],float);prn=np.asarray(f["PRN"],float);flag=np.asarray(f["Flag_valid_pseudorange"],float);pseudorange=np.asarray(f["Pseudorange_m"],float)
    base=np.isfinite(rx)&(rx>0)&np.isfinite(prn)&(prn>=1)&(prn<=32)&(flag>0)&np.isfinite(pseudorange)&(pseudorange>0)
    supported=np.zeros_like(base,bool)
    for value in np.unique(np.floor(rx[base]*10)/10):
        group=base&(rx>=value)&(rx<value+.1+1e-9)
        if len(set(prn[group].astype(int)))>=5:supported|=group
    start_values=rx[supported]
    if not len(start_values):raise ValueError("RX_time absent")
    # The minimum must be a supported lower edge, not a singleton outlier.
    candidate=float(start_values.min())
    if int(np.sum((start_values>=candidate)&(start_values<=candidate+.1)))<5:raise ValueError("RX_time start lacks robust multi-channel support")
    start=round(candidate,2)
    consistent=np.isfinite(rx)&np.isfinite(tow)&(np.abs(rx-tow)<=1.)
    valid=consistent&supported
    bins={}
    for t,p in zip(rx[valid],prn[valid]):
        rel=float(t-start)
        if rel>=0:bins.setdefault(int(np.floor(rel/.5)),set()).add(int(p))
    return start,bins,{"path":str(path),"sha256":sha(path),"valid_rows":int(valid.sum()),"tow_outlier_rows":int((base&~consistent).sum()),"rinex_clock_nearest_delta_s":None},rx[valid]
def selected_bins(path,directory):
    with np.load(path,allow_pickle=False) as z:times=np.asarray(z["time_s"],float)
    digest=sha(path);manifest=path.with_suffix(".manifest.json")
    if not manifest.is_file():raise ValueError("selected NPZ export manifest missing")
    doc=json.loads(manifest.read_text());output=doc.get("output",{})
    if output.get("sha256")!=digest or int(output.get("row_count",-1))!=len(times):raise ValueError("selected NPZ/export manifest mismatch")
    receiver_iq=source_iq_hash(directory);export_iq=doc.get("source_iq_sha256")
    if not export_iq:
        historical=path.parents[1]/"receiver/manifest.json";declared=doc.get("receiver_manifest",{}).get("sha256")
        if historical.is_file() and declared==sha(historical):export_iq=source_iq_hash(historical.parent)
    if receiver_iq and receiver_iq!=export_iq:raise ValueError("selected NPZ/receiver source-IQ lineage mismatch")
    lineage_status="HASH_BOUND" if valid_sha(receiver_iq) and valid_sha(export_iq) and receiver_iq==export_iq else "LINEAGE_GAP"
    return sorted(set(np.floor(times/.5).astype(int))),{"path":str(path),"sha256":digest,"rows":len(times),"manifest_path":str(manifest),"manifest_sha256":sha(manifest),"source_iq_sha256":export_iq,"receiver_source_iq_sha256":receiver_iq,"lineage_status":lineage_status}
def source_iq_hash(directory):
    path=directory/"manifest.json"
    if not path.is_file():return None
    doc=json.loads(path.read_text());return (doc.get("source",{}).get("iq_sha256") or doc.get("source",{}).get("sha256") or doc.get("authenticated_inputs",{}).get("iq_after_receiver",{}).get("sha256"))
def reconstruct(name,directory,selected,config,expected_week=None,expected_tow=None):
    binding=rinex(directory);start,raw,lineage,valid_rx=raw_observables(directory);pvt,pvt_report=nmea_pvt(directory,start,binding["recording_utc_date"])
    eph=parse_gnss_sdr_gps_ephemeris_xml(directory/"gps_ephemeris.xml");eligible,selected_report=selected_bins(selected,directory)
    modulo={x.WN for x in eph.values()};week_ok=all(binding["gps_week"]%1024==x for x in modulo)
    rinex_rx_delta=float(np.min(np.abs(valid_rx-binding["first_epoch_tow_s"])));lineage["rinex_clock_nearest_delta_s"]=rinex_rx_delta
    assertions={"week":{"expected":expected_week,"derived":binding["gps_week"],"status":"NOT_PROVIDED" if expected_week is None else "PASS" if expected_week==binding["gps_week"] else "FAIL"},
      "tow":{"expected":expected_tow,"derived":start,"status":"NOT_PROVIDED" if expected_tow is None else "PASS" if abs(expected_tow-start)<=.05 else "FAIL"}}
    reasons={};valid=0;los_by_bin={};conditions=[];valid_bins=[]
    for bin_id in eligible:
        event_t=bin_id*.5+.5;event_tow=start+event_t;why=[];prns=sorted(raw.get(bin_id,set()))
        if not prns:why.append("no_raw_valid_pseudorange")
        past=[x for x in pvt if x[0]<=event_t+1e-9]
        if not past or event_t-past[-1][0]>config["maximum_pvt_age_s"]:why.append("no_causal_pvt_within_1s");receiver=None
        else:receiver=past[-1][1]
        vectors={}
        for prn in prns:
            item=eph.get(prn)
            if item is None:continue
            if item.SV_health!=0:continue
            age=abs((event_tow-item.toe+302400)%604800-302400)
            if age>config["maximum_toe_age_s"]:continue
            if receiver is not None:
                sat=satellite_position_ecef(item,event_tow%604800);vectors[prn]=look_angles(receiver,sat).los_ecef
        if len(vectors)<config["minimum_prns"]:why.append("fewer_than_5_healthy_geometry_prns")
        if len(vectors)>=config["minimum_prns"]:
            design=np.column_stack([-np.asarray(list(vectors.values())),np.ones(len(vectors))]);rank=np.linalg.matrix_rank(design);dof=len(vectors)-rank;condition=float(np.linalg.cond(design));conditions.append(condition)
            if rank<config["minimum_rank"]:why.append("rank_below_4")
            if dof<config["minimum_residual_dof"]:why.append("residual_dof_below_1")
            if condition>config["maximum_condition_number"]:why.append("condition_limit")
        if why:
            for reason in why:reasons[reason]=reasons.get(reason,0)+1
        else:
            valid+=1;valid_bins.append(bin_id);los_by_bin[str(bin_id)]={str(p):list(map(float,u)) for p,u in vectors.items()}
    blocks={b//20 for b in valid_bins if all(x in valid_bins for x in range((b//20)*20,(b//20+1)*20))}
    coverage=valid/max(len(eligible),1);coverage_pass=coverage>=config["minimum_coverage"] and len(blocks)>=config["minimum_complete_10s_blocks"]
    decoded=[x.decoded_tow for x in eph.values() if x.decoded_tow is not None]
    causal=False
    return {"scenario":name,"derived_time":{"status":"PASS" if week_ok and abs(rinex_rx_delta)<=1 else "FAIL","gps_week":binding["gps_week"],"start_tow_s":start,"rinex_rx_delta_s":rinex_rx_delta,"ephemeris_week_modulo":sorted(modulo),"assertions":assertions},
      "lineage":{"rinex":binding,"observables":lineage,"selected":selected_report,"receiver_manifest":{"path":str((directory/"manifest.json").resolve()),"sha256":sha(directory/"manifest.json")} if (directory/"manifest.json").is_file() else {"status":"LINEAGE_GAP"},"receiver_source_iq_sha256":source_iq_hash(directory),"export_source_iq_sha256":selected_report.get("source_iq_sha256"),"source_iq_binding_status":selected_report.get("lineage_status"),"ephemeris":{"path":str((directory/"gps_ephemeris.xml").resolve()),"sha256":sha(directory/"gps_ephemeris.xml")},"nmea":{"path":str((directory/"nmea_pvt.nmea").resolve()),"sha256":sha(directory/"nmea_pvt.nmea")}},
      "nmea":pvt_report,"event_time_causal_ephemeris_availability":{"status":"PASS" if causal else "OFFLINE_ORACLE_ONLY","decoded_history_authenticated":causal},
      "offline_geometry_coverage":{"status":"PASS" if coverage_pass else "FAIL","valid_events":valid,"eligible_events":len(eligible),"coverage":coverage,"complete_10s_blocks":len(blocks),"rejection_reasons":reasons,"maximum_condition":max(conditions) if conditions else None},"los_by_bin":los_by_bin}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--receiver",action="append",required=True,help="NAME=DIR");ap.add_argument("--selected",action="append",required=True,help="NAME=NPZ");ap.add_argument("--assert-week",action="append",default=[]);ap.add_argument("--assert-tow",action="append",default=[]);ap.add_argument("--config",type=Path,default=ROOT/"configs/r2c_gnss_stage0_fix.json");ap.add_argument("--report",type=Path,required=True);args=ap.parse_args()
    receivers={k:Path(v) for k,v in (x.split("=",1) for x in args.receiver)};selected={k:Path(v) for k,v in (x.split("=",1) for x in args.selected)};weeks={k:int(v) for k,v in (x.split("=",1) for x in args.assert_week)};tows={k:float(v) for k,v in (x.split("=",1) for x in args.assert_tow)};config=json.loads(args.config.read_text())["geometry"]
    reports={n:reconstruct(n,d,selected[n],config,weeks.get(n),tows.get(n)) for n,d in receivers.items()}
    target=args.report.resolve();production=(ROOT/"artifacts/r2c_gnss_stage0_fix").resolve()
    if target==production or production in target.parents:raise ValueError("geometry report cannot enter campaign artifact in pre-campaign phase")
    subset_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps({"schema":"gnss-doppler-lab.r2c-strict-time-geometry.v2","attack_scores_computed":False,"geometry_config":{"values":config,"sha256":subset_hash},"scenarios":reports},indent=2)+"\n")
    print(json.dumps({n:{"week":r["derived_time"]["gps_week"],"tow":r["derived_time"]["start_tow_s"],"coverage":r["offline_geometry_coverage"]} for n,r in reports.items()}))
if __name__=="__main__":main()
