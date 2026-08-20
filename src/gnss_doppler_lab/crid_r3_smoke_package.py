"""Build a full clean prefix with only the frozen R0c window replaced."""
from __future__ import annotations
import hashlib
from pathlib import Path

def compose_prefix(source:Path,control:Path,target:Path,*,absolute_start:int,control_samples:int,total_samples:int)->dict:
    if absolute_start+control_samples>total_samples:raise ValueError('control outside smoke prefix')
    target.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();remaining=total_samples*4
    with source.open('rb') as src,target.open('wb') as dst:
        while remaining:
            payload=src.read(min(8*1024*1024,remaining))
            if not payload:raise EOFError('clean prefix truncated')
            dst.write(payload);remaining-=len(payload)
    with control.open('rb') as patch,target.open('r+b') as dst:
        dst.seek(absolute_start*4);remaining=control_samples*4
        while remaining:
            payload=patch.read(min(8*1024*1024,remaining))
            if not payload:raise EOFError('control window truncated')
            dst.write(payload);remaining-=len(payload)
        if patch.read(1):raise ValueError('control window too long')
    with target.open('rb') as stream:
        for payload in iter(lambda:stream.read(8*1024*1024),b''):digest.update(payload)
    return {'path':str(target),'size_bytes':target.stat().st_size,'complex_samples':total_samples,'sha256':digest.hexdigest(),
      'replacement_start_sample':absolute_start,'replacement_end_sample_exclusive':absolute_start+control_samples}
