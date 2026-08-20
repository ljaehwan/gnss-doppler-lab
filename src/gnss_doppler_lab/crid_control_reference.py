"""Independent CRID R3 control checks; deliberately imports no generator code."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

def file_sha(path: Path, chunk: int = 8*1024*1024) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk),b''):h.update(b)
    return h.hexdigest()

def byte_difference(source: Path, output: Path, start_sample: int, sample_count: int, chunk: int=8*1024*1024)->dict:
    changed=total=0; hs=hashlib.sha256(); ho=hashlib.sha256()
    with source.open('rb') as a,output.open('rb') as b:
        a.seek(start_sample*4);remaining=sample_count*4
        while remaining:
            n=min(chunk,remaining);x=a.read(n);y=b.read(n)
            if len(x)!=n or len(y)!=n:raise EOFError('independent bounds failure')
            changed+=sum(i!=j for i,j in zip(x,y));total+=n;hs.update(x);ho.update(y);remaining-=n
        if b.read(1):raise ValueError('output longer than frozen window')
    return {'source_window_sha256':hs.hexdigest(),'output_sha256':ho.hexdigest(),'changed_bytes':changed,'bytes':total}

def verify_manifest(row:dict, spec:dict)->dict:
    ds=spec['datasets'][row['domain']]; expected=(ds['absolute_end_sample_exclusive']-ds['absolute_start_sample'])*4
    failures=[]
    if row['size_bytes']!=expected:failures.append('byte_count')
    if row['complex_samples']*4!=expected:failures.append('sample_count')
    if row['absolute_start_sample']!=ds['absolute_start_sample'] or row['absolute_end_sample_exclusive']!=ds['absolute_end_sample_exclusive']:failures.append('absolute_mapping')
    if row['family']=='positive' and len(row['targets']) not in (1,4):failures.append('target_count')
    if row['kind']=='byte_identical' and row['changed_bytes']!=0:failures.append('identity')
    if row['kind']!='byte_identical' and row['changed_bytes']==0:failures.append('non_identity')
    if row['clipping_fraction']>spec['sample_contract']['clipping_fail_closed']['maximum_total_clip_fraction']:failures.append('clipping')
    return {'case_id':row['case_id'],'domain':row['domain'],'status':'PASS' if not failures else 'FAIL','failures':failures}

def measure_gain_phase(source:Path,output:Path,start:int,count:int=250000)->dict:
    with source.open('rb') as a:a.seek(start*4);ra=np.frombuffer(a.read(count*4),dtype='<i2').reshape(-1,2).astype(float)
    rb=np.fromfile(output,dtype='<i2',count=count*2).reshape(-1,2).astype(float)
    x=ra[:,0]+1j*ra[:,1];y=rb[:,0]+1j*rb[:,1];alpha=np.vdot(x,y)/np.vdot(x,x)
    return {'complex_gain_real':float(alpha.real),'complex_gain_imag':float(alpha.imag),'magnitude':float(abs(alpha)),'phase_rad':float(np.angle(alpha))}
