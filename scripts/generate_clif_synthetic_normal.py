#!/usr/bin/env python3
"""Generate/extract target-matched CLIF-IP R4 runs with bounded storage."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import butter,sosfilt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.clif_ip_synthetic import (DOMAINS,PipelinePaths,build_final_index,exact_iq_bytes,
 extract_m1_features,publish_success,target_spec,validate_final_index,validate_run_bundle)
from gnss_doppler_lab.gnss_sdr import export_tracking_csv,parse_acquired_prns,parse_receiver_reported_prns
from gnss_doppler_lab.tracking_feature_windows import export_receiver_run_tap_feature_csv
from gnss_doppler_lab.normal_multi_prn_dataset import export_tap_multi_prn_dataset

DEFAULT_OUT=Path("artifacts/clif_ip_synthetic_normal_r4")
DEFAULT_SIM=Path("/home/ubuntu/projects/gnss-doppler-lab/.tools/gps-sdr-sim-src/gps-sdr-sim")
DEFAULT_RECEIVER=Path("/home/ubuntu/build-gnss-sdr-complex9/build-complex/src/main/gnss-sdr")

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()

def atomic_csv(d,p):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");d.to_csv(t,index=False);os.replace(t,p)
def atomic_json(d,p):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n");os.replace(t,p)

def smoke_index(duration):
 d=build_final_index().groupby("domain",sort=False).head(1).copy();d["duration_s"]=float(duration)
 d["run_id"]=[f"smoke-{x.lower()}" for x in d.domain];return d

def receiver_config(iq,run_dir,fs,channels=11):
 prefix=(run_dir/"raw"/"epl_tracking_ch_").resolve()
 return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={fs}
SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq.resolve()}
SignalSource.item_type=ishort
SignalSource.sampling_frequency={fs}
SignalSource.samples=0
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false
SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ishort_To_Complex
InputFilter.implementation=Pass_Through
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
Resampler.implementation=Pass_Through
Resampler.item_type=gr_complex
Channels_1C.count={channels}
Channels.in_acquisition={channels}
Channel.signal=1C
Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.threshold=2.5
Acquisition_1C.doppler_max=10000
Acquisition_1C.doppler_step=100
Tracking_1C.implementation=GPS_L1_CA_DLL_PLL_Tracking
Tracking_1C.item_type=gr_complex
Tracking_1C.pll_bw_hz=20.0
Tracking_1C.dll_bw_hz=1.5
Tracking_1C.order=3
Tracking_1C.dump=true
Tracking_1C.dump_filename={prefix}
Tracking_1C.tap_count=9
Tracking_1C.tap_spacing_chips=0.125
TelemetryDecoder_1C.implementation=GPS_L1_CA_Telemetry_Decoder
TelemetryDecoder_1C.dump=false
Observables.implementation=Hybrid_Observables
Observables.dump=true
Observables.dump_filename={(run_dir/'raw'/'observables.dat').resolve()}
PVT.implementation=RTKLIB_PVT
PVT.positioning_mode=Single
PVT.output_rate_ms=100
PVT.display_rate_ms=500
PVT.flag_rtcm_server=false
PVT.flag_rtcm_tty_port=false
PVT.dump=false
"""

def impair(clean,final,fs,imp,seed):
 """Streaming physical frontend model; no duplication, padding, or resampling."""
 rng=np.random.default_rng(seed);sos=butter(4,min(.999,float(imp["frontend_bw_hz"])/(fs/2)),output="sos");zi=np.zeros((len(sos),2),complex)
 nall=Path(clean).stat().st_size//4;phase=0.;gain=10**(float(imp["agc_gain_db"])/20);ig=10**(float(imp["iq_gain_imbalance_db"])/20);phi=np.deg2rad(float(imp["iq_phase_imbalance_deg"]));clip=float(imp["clipping_fullscale"])
 # C/N0 equation: E|n|^2 = measured clean signal power * Fs / 10^(CN0/10).
 probe=np.memmap(clean,dtype="<i2",mode="r",shape=(min(nall,fs),2));power=float(np.mean(probe.astype(float)**2).sum());noise_complex=power*fs/(10**(float(imp["awgn_cn0_dbhz"])/10));sigma=np.sqrt(noise_complex/2)
 src=np.memmap(clean,dtype="<i2",mode="r",shape=(nall,2));tmp=Path(str(final)+".tmp")
 with tmp.open("wb") as out:
  chunk=1_000_000
  for start in range(0,nall,chunk):
   a=src[start:min(start+chunk,nall)].astype(np.float64);z=a[:,0]+1j*a[:,1];idx=start+np.arange(len(z));t=idx/fs
   freq=float(imp["cfo_hz"])+float(imp["cfo_drift_hz_s"])*t
   increments=2*np.pi*freq/fs+rng.normal(0,float(imp["phase_noise_std_rad"]),len(z));ph=phase+np.cumsum(increments);phase=float(ph[-1]%(2*np.pi));z*=np.exp(1j*ph)
   z,zi=sosfilt(sos,z,zi=zi);i=z.real*ig;q=(z.imag*np.cos(phi)+z.real*np.sin(phi))/ig
   z=(i+float(imp["dc_i"]))+1j*(q+float(imp["dc_q"]));z=gain*z+sigma*(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))
   o=np.c_[np.clip(np.rint(z.real),-clip,clip),np.clip(np.rint(z.imag),-clip,clip)].astype("<i2");o.tofile(out)
 os.replace(tmp,final)
 return {"equation":"z1=LPF((I*g)+j(Q*cos(phi)+I*sin(phi))/g)*exp(j*cumsum(2pi*(CFO+drift*t)/Fs+Wiener)); z2=AGC*z1+DC+n; E|n|^2=Pclean*Fs/10^(CN0/10); round+clip to s16le",
  "source_sha256":sha(clean),"target_sha256":sha(final),"signal_power_probe":power,"noise_sigma_per_component":sigma,"direct_target_generation":True,"resampling":False}

def run_receiver(iq,run_dir,run_id,fs,exe,timeout):
 rec=run_dir/"receiver";raw=rec/"raw";raw.mkdir(parents=True,exist_ok=True);cfg=rec/"receiver.conf";cfg.write_text(receiver_config(iq,rec,fs));log=rec/"receiver.log"
 cmd=[str(Path(exe).resolve()),f"--config_file={cfg.resolve()}","--keyboard=false"]
 with log.open("w") as h:r=subprocess.run(cmd,cwd=rec,stdout=h,stderr=subprocess.STDOUT,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError(f"GNSS-SDR rc={r.returncode}; {log}")
 mats=sorted(raw.glob("epl_tracking_ch_*.mat"))
 if not mats:raise RuntimeError(f"Method-A produced no tracking MAT: {log}")
 report=export_tracking_csv(mats,rec/"tracking.csv",rec/"tracking_summary.csv",sample_rate_hz=fs);text=log.read_text(errors="replace")
 if report["row_count"]<1 or not report["prns"]:raise RuntimeError("Method-A produced zero valid tracking")
 manifest={"schema_version":4,"status":"complete","receiver_run_id":run_id,"source_rf_run_id":run_id,
  "source":{"iq":str(iq.resolve()),"iq_sha256":sha(iq),"sample_rate_hz":fs,"sample_format":"little-endian interleaved int16 IQ"},
  "receiver":{"name":"GNSS-SDR Method-A","executable":str(Path(exe).resolve()),"config":cfg.name,"command":cmd,"exit_code":r.returncode},
  "acquisition":{"tracked_prns":parse_acquired_prns(text),"receiver_reported_prns":parse_receiver_reported_prns(text)},
  "tracking":{"tap_count":9,"tap_spacing_chips":.125,"raw_directory":"raw",**report}}
 atomic_json(manifest,rec/"manifest.json");return rec,manifest

def process_row(row,out,sim,receiver,timeout,keep):
 paths=PipelinePaths.for_run(out,str(row.run_id))
 if paths.success.exists():validate_run_bundle(paths);return {"run_id":row.run_id,"status":"resumed_valid"}
 paths.run_dir.mkdir(parents=True,exist_ok=True);clean=paths.run_dir/"gpssim_clean_s16le.bin";imp=json.loads(row.impairments_json);fs=int(row.sample_rate_hz);started=time.time()
 nav=(ROOT/str(row.rinex_nav)).resolve();utc=pd.Timestamp(row.utc);stamp=utc.strftime("%Y/%m/%d,%H:%M:%S")
 # This gps-sdr-sim writes d-0.1 s (its first 0.1 s update is not emitted).
 # Request d+0.1 and verify exact target bytes; use -t with matching broadcast
 # ephemeris.  Probe showed -T produced an all-zero stream for this RINEX.
 cmd=[str(Path(sim).resolve()),"-e",str(nav),"-l",f"{row.latitude_deg},{row.longitude_deg},{row.altitude_m}","-t",stamp,"-d",str(float(row.duration_s)+.1),"-s",str(fs),"-b","16","-o",str(clean)]
 genlog=paths.run_dir/"generator.log"
 with genlog.open("w") as h:r=subprocess.run(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,text=True,timeout=max(300,int(row.duration_s)*60))
 if r.returncode:raise RuntimeError(f"gps-sdr-sim rc={r.returncode}; {genlog}")
 if clean.stat().st_size!=exact_iq_bytes(row.domain,float(row.duration_s)):raise RuntimeError("gps-sdr-sim target byte contract failed")
 transform=impair(clean,paths.iq,fs,imp,int(row.impairment_seed));clean.unlink()
 if paths.iq.stat().st_size!=exact_iq_bytes(row.domain,float(row.duration_s)):raise RuntimeError("final target byte contract failed")
 iqhash=sha(paths.iq);m1=extract_m1_features(paths.iq,str(row.run_id),fs,float(row.duration_s));atomic_csv(m1,paths.m1_csv)
 rec,rmanifest=run_receiver(paths.iq,paths.run_dir,str(row.run_id),fs,receiver,timeout)
 tap=paths.run_dir/"b0_9tap_features.csv";export_receiver_run_tap_feature_csv(rec,output_path=tap,tap_count=9,window_s=1.,stride_s=.5,min_epochs=4,label="normal")
 node,graph,multi=export_tap_multi_prn_dataset(tap,output_dir=paths.run_dir/"b0_multi",stride_s=.5,min_prns_per_graph=1,feature_mode="normalized_dmcpd")
 shutil.copy2(node,paths.b0_csv);b0=pd.read_csv(paths.b0_csv)
 manifest={"schema":"clif-ip.synthetic-normal.r4.run.v1","run_id":row.run_id,"domain":row.domain,"split":row.split,"label":"normal",
  "duration_s":float(row.duration_s),"sample_rate_hz":fs,"sample_format":"little-endian int16 interleaved IQ (ishort)","iq_bytes":paths.iq.stat().st_size,
  "iq_sha256":iqhash,"m1_iq_sha256":iqhash,"b0_iq_sha256":iqhash,"m1_rows":len(m1),"b0_rows":len(b0),"finite":bool(np.isfinite(m1.select_dtypes("number")).all().all() and np.isfinite(b0.select_dtypes("number")).all().all()),"zero_placeholder":False,
  "impairments":imp,"generator":{"command":cmd,"binary_sha256":sha(Path(sim)),"native_options":"arbitrary -s and -b 16 verified","transform":transform},
  "receiver":{"manifest":str(rec/"manifest.json"),"tracked_prns":rmanifest["tracking"]["prns"],"tracking_rows":rmanifest["tracking"]["row_count"]},
  "b0_semantics":"Method-A receiver-backed 9-tap magnitudes; signed 9D innovation is x-xhat in shared PRN-local model, not raw complex signed taps",
  "elapsed_s":time.time()-started,"completed_utc":datetime.now(timezone.utc).isoformat()}
 publish_success(paths,manifest)
 if not keep:
  paths.iq.unlink();shutil.rmtree(paths.raw)
 return {"run_id":row.run_id,"status":"ok","elapsed_s":manifest["elapsed_s"],"tracked_prns":len(rmanifest["tracking"]["prns"]),"b0_rows":len(b0),"m1_rows":len(m1)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=DEFAULT_OUT);ap.add_argument("--simulator",type=Path,default=DEFAULT_SIM);ap.add_argument("--receiver",type=Path,default=DEFAULT_RECEIVER)
 ap.add_argument("--limit",type=int);ap.add_argument("--resume",action="store_true");ap.add_argument("--domains",nargs="*",choices=DOMAINS);ap.add_argument("--duration",type=float,default=120);ap.add_argument("--smoke",action="store_true");ap.add_argument("--index-only",action="store_true");ap.add_argument("--keep-transients",action="store_true");ap.add_argument("--receiver-timeout",type=int,default=1800);a=ap.parse_args()
 if a.smoke:
  if not 2<=a.duration<=30:raise SystemExit("smoke duration must be 2--30 seconds")
  out=a.out;d=smoke_index(a.duration)
 else:
  if a.duration!=120:raise SystemExit("final campaign duration is fixed at 120 seconds")
  out=a.out;d=build_final_index();validate_final_index(d)
 if a.domains:d=d[d.domain.isin(a.domains)]
 index=out/"synthetic_run_manifest.csv";atomic_csv(d,index)
 atomic_json({"schema":"clif-ip.synthetic-normal.r4.generation.v1","campaign_kind":"smoke" if a.smoke else "final-60x120s","indexed_rows":len(d),"generator_direct_s16le":True,"reports":[]},out/"generation_summary.json")
 if a.index_only:return
 rows=d.iloc[:a.limit] if a.limit else d;reports=[]
 for row in rows.itertuples():
  try:reports.append(process_row(row,out,a.simulator,a.receiver,a.receiver_timeout,a.keep_transients))
  except Exception as e:
   reports.append({"run_id":row.run_id,"status":"failed","error":f"{type(e).__name__}: {e}"});atomic_json({"reports":reports},out/"generation_summary.json");raise
 atomic_json({"schema":"clif-ip.synthetic-normal.r4.generation.v1","campaign_kind":"smoke" if a.smoke else "final-60x120s","indexed_rows":len(d),"processed_rows":len(reports),"reports":reports},out/"generation_summary.json");print(json.dumps(reports,indent=2))
if __name__=="__main__":main()
