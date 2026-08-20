import json
from pathlib import Path
import numpy as np
from gnss_doppler_lab.crid_control_generator import amplitude_envelope,pull_delay,pull_integral,enumerate_cases,frozen_phase,encode,decode

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'artifacts/crid_stage0_r3_control_generator_foundation/control_spec.json').read_text())

def test_exact_grid_and_assignment():
 for domain in ('OAK','TEX'):
  cases=enumerate_cases(SPEC,domain);pos=[c for c in cases if c.family=='positive'];neg=[c for c in cases if c.family=='negative']
  assert len(pos)==18 and len(neg)==15 and len({c.case_id for c in cases})==33
  assert all(len(c.targets)==(1 if c.mode=='single' else 4) for c in pos)
  assert set().union(*(set(c.targets) for c in pos))==set(SPEC['datasets'][domain]['validated_prns_sorted'])

def test_smooth_pull_off_and_envelope():
 t=np.linspace(0,11.98,10001);d=pull_delay(t,.3);e=amplitude_envelope(t,11.98)
 assert d[0]==0 and np.all(np.diff(d)>=-1e-14) and abs(d[-1]-.3)<1e-12
 assert e[0]==0 and e[-1]==0 and e.max()==1 and np.all((e>=0)&(e<=1))
 assert np.max(np.abs(np.diff(d)))<.001

def test_phase_and_integral_deterministic():
 assert frozen_phase('x','y',3)==frozen_phase('x','y',3)
 assert frozen_phase('x','y',3)!=frozen_phase('x','y',4)
 t=np.array([0,.5,4.5,5.5]);v=pull_integral(t,5)
 assert np.allclose(v,[0,0,10,15])

def test_int16_roundtrip_and_clipping():
 raw=np.array([1+2j,-32768+32767j]);payload,cs,cc=encode(raw)
 assert np.array_equal(decode(payload),raw) and cs==cc==0
 _,cs,cc=encode(np.array([40000+0j]));assert cs==1 and cc==1

def test_attack_paths_absent_from_spec():
 text=json.dumps(SPEC).lower()
 for token in ('ds1.bin','ds2.bin','ds3.bin','ds4.bin','ds7.bin','os1.bin','os2.bin','os3.bin','os4.bin'):
  assert token not in text
