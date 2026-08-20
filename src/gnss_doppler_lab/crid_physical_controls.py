"""Streaming raw-IQ physical controls fixed before attack evaluation."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

def transform_ishort(source:Path,target:Path,kind:str,*,seed:int=20260820,gain:float=1.,phase_rad:float=0.,noise_sigma:float=0.,delay_samples:float=0.,duplicate_db:float=-6.,sample_rate_hz:float=1.,ramp_hz_per_s:float=0.,start_sample:int=0,max_samples:int|None=None)->dict:
 """Apply whole-waveform controls without editing derived receiver fields."""
 rng=np.random.default_rng(seed);target.parent.mkdir(parents=True,exist_ok=True)
 digest=hashlib.sha256();peak=0.;count=0;keep=max(int(np.ceil(delay_samples))+2,2);previous=np.zeros(keep,np.complex64)
 with source.open("rb") as src,target.open("wb") as dst:
  src.seek(start_sample*4)
  while True:
   remaining=None if max_samples is None else max_samples-count
   if remaining is not None and remaining<=0:break
   raw=np.fromfile(src,dtype="<i2",count=2*min(1_000_000,remaining if remaining is not None else 1_000_000))
   if not len(raw):break
   if len(raw)%2:raise ValueError("odd scalar IQ count")
   z=raw.reshape(-1,2).astype(np.float32);x=z[:,0]+1j*z[:,1]
   if kind in ("byte_identical","single_source_code_ramp"):y=x
   elif kind in ("gain","clock_drift"):y=x*gain
   elif kind=="phase":y=x*np.exp(1j*phase_rad)
   elif kind=="doppler_ramp":
    n=np.arange(count,count+len(x));t=n/sample_rate_hz
    y=x*np.exp(1j*2*np.pi*.5*ramp_hz_per_s*t*t)
   elif kind in ("awgn","cn0_reduction"):y=x+(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))*noise_sigma
   elif kind=="nav_sign":y=-x
   elif kind in ("duplicate","zero_delay_duplicate"):
    amp=10**(duplicate_db/20)
    if kind=="zero_delay_duplicate" or delay_samples==0:y=x*(1+amp*np.exp(1j*phase_rad))
    else:
     joined=np.concatenate([previous,x]);position=np.arange(len(x))+keep-delay_samples
     lo=np.floor(position).astype(int);frac=position-lo
     delayed=joined[lo]*(1-frac)+joined[lo+1]*frac;previous=joined[-keep:]
     y=x+amp*np.exp(1j*phase_rad)*delayed
   else:raise ValueError(kind)
   out=np.column_stack([y.real,y.imag]);peak=max(peak,float(np.max(np.abs(out))))
   packed=np.clip(np.rint(out),-32768,32767).astype("<i2").tobytes();dst.write(packed);digest.update(packed);count+=len(x)
 return {"kind":kind,"seed":seed,"complex_samples":count,"peak_preclip":peak,"sha256":digest.hexdigest(),
  "parameters":{"gain":gain,"phase_rad":phase_rad,"noise_sigma":noise_sigma,"delay_samples":delay_samples,"duplicate_db":duplicate_db,"sample_rate_hz":sample_rate_hz,"ramp_hz_per_s":ramp_hz_per_s,"start_sample":start_sample,"max_samples":max_samples}}
