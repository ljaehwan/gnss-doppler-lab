"""Hash-bound epoch truth writer for frozen CRID R3 controls."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .crid_control_generator import amplitude_envelope, frozen_phase, pull_delay, pull_integral, pull_rate, sha256_file

TRUTH_DTYPE = np.dtype([
    ("absolute_sample", "<i8"), ("prn", "<i4"), ("authentic_amplitude", "<f8"),
    ("counterfeit_amplitude", "<f8"), ("code_delay_chips", "<f8"),
    ("code_rate_chips_s", "<f8"), ("carrier_phase_rad", "<f8"),
    ("doppler_hz", "<f8"), ("envelope", "<f8"), ("nav_sign", "i1"), ("padding", "V7")])

def write_truth_epochs(ctx, case, alpha: dict[int, complex], path: Path) -> dict[str, object]:
    samples=np.arange(ctx.start,ctx.end,ctx.fs//1000,dtype=np.int64);targets=case.targets or (0,)
    rows=np.zeros(len(samples)*len(targets),dtype=TRUTH_DTYPE);duration=ctx.count/ctx.fs;k=0
    for absolute in samples:
        t=(absolute-ctx.start)/ctx.fs;env=float(amplitude_envelope(np.array([t]),duration)[0])
        for p in targets:
            row=rows[k];row["absolute_sample"]=absolute;row["prn"]=p
            if p:row["authentic_amplitude"]=abs(alpha[p]);row["nav_sign"]=int(ctx.replicas[p].nav(int(absolute),1)[0])
            if case.family=="positive":
                row["counterfeit_amplitude"]=abs(alpha[p])*10**(case.power_db/20)*env
                row["code_delay_chips"]=pull_delay(np.array([t]),case.delay_chips)[0]
                row["code_rate_chips_s"]=pull_rate(np.array([t]),case.delay_chips*np.pi/8)[0]
                row["carrier_phase_rad"]=frozen_phase(ctx.spec["positive_controls"]["phase_seed"],case.case_id,p);row["envelope"]=env
            elif case.kind=="single_source_code_ramp" and p:
                row["code_delay_chips"]=pull_delay(np.array([t]),.05)[0];row["code_rate_chips_s"]=pull_rate(np.array([t]),.05*np.pi/8)[0]
            elif case.kind=="single_source_doppler_ramp" and p:
                row["doppler_hz"]=pull_rate(np.array([t]),5.)[0];row["carrier_phase_rad"]=2*np.pi*pull_integral(np.array([t]),5.)[0]
            elif case.kind=="independent_multipath" and p:
                i=list(ctx.prns[:4]).index(p);row["counterfeit_amplitude"]=abs(alpha[p])*10**(-9/20)
                row["code_delay_chips"]=[.07,.11,.17,.23][i];row["carrier_phase_rad"]=[.2,-.7,1.1,-1.9][i];row["envelope"]=1
            elif case.kind=="zero_delay_collapsed_duplicate" and p:
                row["counterfeit_amplitude"]=abs(alpha[p])*10**(-3/20);row["envelope"]=1
            k+=1
    path.parent.mkdir(parents=True,exist_ok=True);rows.tofile(path)
    return {"path":str(path),"sha256":sha256_file(path),"record_count":len(rows),"record_bytes":TRUTH_DTYPE.itemsize,
            "schema":"<q i d d d d d d d b 7x","cadence_ms":1}
