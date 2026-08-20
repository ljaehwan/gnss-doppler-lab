"""R3-only C0 smoke wrapper around the unchanged R1 terminal supervisor."""
from __future__ import annotations
import json,resource,time
from pathlib import Path
from .crid import receiver_configurations,render_receiver_config
from .crid_receiver_replay import _supervised_run,sha256_file

def run_c0_smoke(*,receiver:Path,base_config:Path,raw:Path,out:Path,scenario:str,fs:int,
                 absolute_start:int,complex_samples:int,expected_dumps:int,raw_sha256:str)->dict:
    out.mkdir(parents=True,exist_ok=True)
    for p in out.glob('trace_native_1ms_ch_*.bin'):p.unlink()
    values={'SignalSource.filename':str(raw),'SignalSource.seconds_to_skip':0.0,'SignalSource.samples':complex_samples*2,
      'SignalSource.repeat':'false','Tracking_1C.dump':'false','Tracking_1C.dump_mat':'false','Tracking_1C.trace_dump':'true',
      'Tracking_1C.trace_dump_filename':'trace_native_1ms_ch_','Tracking_1C.trace_scenario_id':scenario,
      'Tracking_1C.trace_raw_sample_offset':absolute_start,'Observables.dump':'false'}
    cfg=out/'receiver.conf';cfg.write_text(render_receiver_config(base_config.read_text(),receiver_configurations()['C0'],values))
    command=[str(receiver),f'--config_file={cfg}','--keyboard=false'];began=time.monotonic()
    with (out/'receiver.log').open('wb') as log:
        rc,termination=_supervised_run(command,cwd=out,log=log,raw=raw,expected_end_byte=complex_samples*4,expected_dump_count=expected_dumps)
    dumps=sorted(out.glob('trace_native_1ms_ch_*.bin'))
    manifest={'schema':'gnss-doppler-lab.crid-r3-c0-smoke.v1','scenario':scenario,'config':'C0','command':command,'exit_code':rc,
      'elapsed_s':time.monotonic()-began,'peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
      'receiver':{'path':str(receiver),'sha256':sha256_file(receiver)},'raw':{'path':str(raw),'sha256':raw_sha256,
      'sample_rate_hz':fs,'absolute_start_sample':absolute_start,'complex_samples':complex_samples},
      'config':{'path':str(cfg),'sha256':sha256_file(cfg)},'termination':termination,
      'dumps':[{'path':str(p),'size':p.stat().st_size,'sha256':sha256_file(p)} for p in dumps],
      'status':'PASS' if rc==0 and termination['status']=='PASS' and len(dumps)==expected_dumps else 'FAIL'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest
