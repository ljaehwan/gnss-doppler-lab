"""Leakage-safe clean-only GCMR experiment primitives.

The split API is intentionally explicit: training, epoch selection, clean score
reference, threshold calibration and sealed held normal are separate temporal
roles. Attack access is guarded until all clean artifacts and held results have
been frozen. Scores are causal and become available at each window's end.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import copy, csv, hashlib, json, math, os, random
from pathlib import Path
from typing import Iterable
import h5py
import numpy as np
import torch
from .gcmr_model import (CONDITION_DIM, OBSERVATION_DIM, CleanReferenceScoreCalibrator,
 GcmrNet, collate_gcmr_events, event_reconstruction_errors, gcmr_loss)
from .gcmr_relations import (CONDITION_FEATURES, OBSERVATION_FEATURES,
 GcmrPairRelationEvent)
from .trajectory import llh_to_ecef

CACHE_SCHEMA_VERSION=4
RELATION_CONTRACT_VERSION=4
CHECKPOINT_SCHEMA_VERSION=2
GPS_UTC_LEAP_OFFSET_S=18
@dataclass(frozen=True)
class TemporalRole:
 name:str; start_s:float; end_s:float
 def __post_init__(self):
  if not self.name or not math.isfinite(self.start_s) or not math.isfinite(self.end_s) or self.start_s>=self.end_s: raise ValueError('invalid temporal role')
DEFAULT_ROLES=(TemporalRole('train',30,180),TemporalRole('selection_val',190,260),TemporalRole('clean_reference',270,330),TemporalRole('event_calibration',340,400),TemporalRole('sealed_held',410,470))

def validate_roles(roles=DEFAULT_ROLES):
 roles=tuple(roles)
 if len({r.name for r in roles})!=len(roles):raise ValueError('role names must be unique')
 for left,right in zip(roles,roles[1:]):
  if left.end_s>=right.start_s:raise ValueError('roles overlap or have no strict purge')
 return roles

def select_role_events(events,role):
 """Select only windows wholly contained in a role (no boundary leakage)."""
 return [e for e in events if e.window_start_s>=role.start_s and e.window_end_s<=role.end_s]

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def source_hashes(paths):return {str(Path(p).resolve()):_sha(p) for p in sorted(map(Path,paths),key=lambda x:str(x.resolve()))}
def _canonical(value):return json.loads(json.dumps(value,sort_keys=True,separators=(',',':')))

_IMPLEMENTATION_REQUIRED = (
 'pyproject.toml','src/gnss_doppler_lab/gcmr_model.py','src/gnss_doppler_lab/gcmr_relations.py',
 'src/gnss_doppler_lab/gcmr_geometry.py','src/gnss_doppler_lab/gcmr_experiment.py',
 'src/gnss_doppler_lab/trajectory.py',
 'scripts/run_gcmr_oakbat_poc.py','tests/test_gcmr_model.py','tests/test_gcmr_relations.py',
 'tests/test_gcmr_geometry.py','tests/test_gcmr_experiment.py')

def implementation_manifest(anchor=None):
 """Hash the exact repository GCMR implementation, independent of caller CWD."""
 start=Path(anchor or __file__).resolve(); start=start if start.is_dir() else start.parent
 root=next((p for p in (start,*start.parents) if (p/'pyproject.toml').is_file() and (p/'src/gnss_doppler_lab').is_dir()),None)
 if root is None:raise ValueError('GCMR repository root unresolved')
 missing=[rel for rel in _IMPLEMENTATION_REQUIRED if not (root/rel).is_file()]
 if missing:raise ValueError(f'missing expected GCMR implementation files: {missing}')
 paths=set(root/rel for rel in _IMPLEMENTATION_REQUIRED)
 for pattern in ('src/gnss_doppler_lab/*gcmr*.py','scripts/*gcmr*.py','tests/*gcmr*.py','config/**/*gcmr*'):
  paths.update(p for p in root.glob(pattern) if p.is_file())
 records=[];aggregate=hashlib.sha256()
 for path in sorted(paths,key=lambda p:p.relative_to(root).as_posix()):
  rel=path.relative_to(root).as_posix();digest=_sha(path);records.append({'path':rel,'sha256':digest})
  aggregate.update(rel.encode());aggregate.update(b'\0');aggregate.update(bytes.fromhex(digest))
 return {'files':records,'aggregate_sha256':aggregate.hexdigest()}

def cache_events(path,events,*,source_paths,metadata):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);events=list(events)
 counts=np.asarray([len(e.pair_prns) for e in events],dtype=np.int64); offsets=np.r_[0,np.cumsum(counts)]
 def join(name,shape,dtype):
  values=[np.asarray(getattr(e,name),dtype=dtype) for e in events]
  return np.concatenate(values) if values else np.empty(shape,dtype=dtype)
 meta={**_canonical(metadata),'schema_version':CACHE_SCHEMA_VERSION,'relation_contract_version':RELATION_CONTRACT_VERSION,'observation_features':list(OBSERVATION_FEATURES),'condition_features':list(CONDITION_FEATURES),'source_sha256':source_hashes(source_paths)}
 tmp=path.with_name(path.name+'.tmp.npz')
 np.savez_compressed(tmp,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)),window_start_s=np.asarray([e.window_start_s for e in events]),window_end_s=np.asarray([e.window_end_s for e in events]),offsets=offsets,pair_prns=join('pair_prns',(0,2),np.int64),observations=join('observations',(0,OBSERVATION_DIM),np.float64),observation_mask=join('observation_mask',(0,OBSERVATION_DIM),bool),conditions=join('conditions',(0,CONDITION_DIM),np.float64))
 os.replace(tmp,path);return meta

def load_event_cache(path,*,source_paths,expected_metadata):
 try:
  with np.load(path,allow_pickle=False) as z:
   meta=json.loads(str(z['metadata_json'])); arrays={k:z[k].copy() for k in ('window_start_s','window_end_s','offsets','pair_prns','observations','observation_mask','conditions')}
 except (OSError,KeyError,ValueError,json.JSONDecodeError) as exc:raise ValueError(f'incompatible event cache: {exc}') from exc
 if meta.get('schema_version')!=CACHE_SCHEMA_VERSION or meta.get('relation_contract_version')!=RELATION_CONTRACT_VERSION or meta.get('observation_features')!=list(OBSERVATION_FEATURES) or meta.get('condition_features')!=list(CONDITION_FEATURES):raise ValueError('incompatible event cache schema/features')
 if meta.get('source_sha256')!=source_hashes(source_paths):raise ValueError('stale event cache: source hash mismatch')
 for key,value in _canonical(expected_metadata).items():
  if meta.get(key)!=value:raise ValueError(f'incompatible event cache metadata: {key}')
 o=arrays['offsets'];n=len(arrays['window_start_s'])
 if o.ndim!=1 or not np.issubdtype(o.dtype,np.integer) or len(o)!=n+1 or o[0]!=0 or np.any(o<0) or np.any(np.diff(o)<=0):raise ValueError('incompatible event cache offsets')
 total=int(o[-1])
 expected_shapes={'pair_prns':(total,2),'observations':(total,OBSERVATION_DIM),'observation_mask':(total,OBSERVATION_DIM),'conditions':(total,CONDITION_DIM)}
 if any(arrays[k].shape != shape for k,shape in expected_shapes.items()):raise ValueError('incompatible event cache concatenated length or dimensions')
 if arrays['window_end_s'].shape!=(n,) or arrays['window_start_s'].shape!=(n,):raise ValueError('incompatible event cache window lengths')
 events=[]
 for i in range(n):
  s=slice(int(o[i]),int(o[i+1]));events.append(GcmrPairRelationEvent(float(arrays['window_start_s'][i]),float(arrays['window_end_s'][i]),arrays['pair_prns'][s],arrays['observations'][s],arrays['observation_mask'][s],arrays['conditions'][s]))
 return events,meta

def _valid_sentence(line):
 line=line.strip()
 if not line.startswith('$') or '*' not in line:return None
 body,raw=line[1:].rsplit('*',1)
 try: expected=int(raw[:2],16)
 except ValueError:return None
 check=0
 for char in body:check^=ord(char)
 return body.split(',') if check==expected else None

def _hms(value):
 if len(value)<6:raise ValueError
 return int(value[:2]),int(value[2:4]),float(value[4:])
def _degree(value,hemisphere):
 width=2 if hemisphere in ('N','S') else 3
 result=float(value[:width])+float(value[width:])/60
 return -result if hemisphere in ('S','W') else result

def _utc_tow(date,hms):
 hh,mm,ss=hms;dt=datetime(date.year,date.month,date.day,hh,mm,int(ss),tzinfo=timezone.utc)
 return (((dt.weekday()+1)%7)*86400+hh*3600+mm*60+ss+GPS_UTC_LEAP_OFFSET_S)%604800


def _gps_week_and_tow(date, hms):
 hh,mm,ss=hms
 dt=datetime(date.year,date.month,date.day,hh,mm,int(ss),tzinfo=timezone.utc)
 seconds=(dt-datetime(1980,1,6,tzinfo=timezone.utc)).total_seconds()+GPS_UTC_LEAP_OFFSET_S+(ss-int(ss))
 return int(seconds//604800),seconds%604800

def preflight_oakbat_geometry(observables_path,nmea_path,ephemerides,*,configured_tow0_s,
                              max_toe_age_s,tow_tolerance_s=.05,onset_s=120.,tracked_prns=None,min_prns=None):
 """Verify OAKBAT recording, UTC date, and ephemeris alignment before geometry."""
 tow0=float(configured_tow0_s);tolerance=float(tow_tolerance_s)
 if not math.isfinite(tow0) or not math.isfinite(tolerance) or tolerance < 0:
  raise ValueError('tow0 and tolerance must be finite; tolerance must be nonnegative')
 try:
  with h5py.File(observables_path,'r') as h:
   if 'RX_time' not in h: raise ValueError('observables MAT is missing RX_time')
   rx=np.asarray(h['RX_time']).reshape(-1).astype(float)
 except OSError as exc: raise ValueError(f'invalid observables MAT: {exc}') from exc
 usable=rx[np.isfinite(rx)&(rx>0)]
 if not len(usable): raise ValueError('observables RX_time contains no finite positive recording times')
 start=float(np.min(usable))
 if abs(start-tow0)>tolerance+1e-12:
  raise ValueError(f'configured tow0 does not match recording start RX_time: {tow0} vs {start}')
 weeks=[]
 for line in Path(nmea_path).read_text(errors='replace').splitlines():
  f=_valid_sentence(line)
  if not f or f[0][-3:]!='RMC' or len(f)<=9 or f[2]!='A': continue
  try:
   week,tow=_gps_week_and_tow(datetime.strptime(f[9],'%d%m%y').date(),_hms(f[1]))
   rel=(tow-tow0+302400)%604800-302400
   if 0 <= rel < float(onset_s): weeks.append(week)
  except (ValueError,IndexError): continue
 if not weeks: raise ValueError('no checksum-valid pre-onset RMC date available')
 if len(set(weeks))!=1: raise ValueError('pre-onset RMC dates disagree on full GPS week')
 from .gcmr_geometry import ephemeris_health_selection, validate_ephemeris_time_alignment
 alignment=validate_ephemeris_time_alignment(ephemerides,full_gps_week=weeks[0],
  recording_start_tow_s=tow0,max_toe_age_s=max_toe_age_s)
 _,health=ephemeris_health_selection(ephemerides,tracked_prns=ephemerides.keys() if tracked_prns is None else tracked_prns,min_prns=min_prns)
 return {'recording_start_rx_time_s':start,'configured_tow0_s':tow0,
  'ephemeris_health':health,
  'tow_tolerance_s':tolerance,'full_gps_week':weeks[0],'ephemeris_alignment':alignment,
  'decoded_snapshot':{'available':alignment['decoded_snapshot_available'],
   'relation':alignment['decoded_snapshot_relation'],
   'required_at_or_before_recording_start':False,
   'limitation':'saved map may be an end-of-run offline oracle snapshot'},
  'geometry_contract':{'classification':'offline_trusted_static_receive_time_approximation',
   'time_basis':'receive_time','receiver_position':'trusted static pre-onset NMEA oracle held fixed',
   'omitted_corrections':['transmit_time','Sagnac','satellite_clock'],
   'operational_precise_doppler':False}}

def parse_preonset_nmea_position(path,*,gps_tow_at_time_zero_s,onset_s=120,position_window_s=(20,90)):
 """Derive one trusted static receiver position from valid pre-onset GGA only.

 RMC supplies the absolute UTC date; GGA supplies fix time and LLH. UTC is
 converted to GPS TOW with the explicit fixed 18 s experiment-era leap offset.
 The resulting robust median position is held fixed for the entire scenario.
 """
 current_date=None;points=[];lo,hi=map(float,position_window_s)
 for line in Path(path).read_text(errors='replace').splitlines():
  f=_valid_sentence(line)
  if not f:continue
  kind=f[0][-3:]
  try:
   if kind=='RMC' and len(f)>9 and f[2]=='A':
    d=f[9];current_date=datetime.strptime(d,'%d%m%y').date()
   elif kind=='GGA' and current_date is not None and len(f)>11 and int(f[6])>0 and f[9] and f[10]=='M':
    tow=_utc_tow(current_date,_hms(f[1]));rel=(tow-float(gps_tow_at_time_zero_s)+302400)%604800-302400
    # Explicitly enforce both robust-position interval and pre-onset oracle boundary.
    if lo<=rel<=hi and rel<float(onset_s):points.append((rel,_degree(f[2],f[3]),_degree(f[4],f[5]),float(f[9])))
  except (ValueError,IndexError):continue
 if not points:raise ValueError('no valid pre-onset NMEA GGA positions in requested interval')
 a=np.asarray(points,float);llh=tuple(np.median(a[:,i]).item() for i in (1,2,3))
 return {'llh':llh,'ecef':tuple(map(float,llh_to_ecef(*llh))),'sample_count':len(points),'relative_times_s':a[:,0].tolist(),'timing':{'source':'absolute UTC RMC/GGA','gps_utc_leap_offset_s':GPS_UTC_LEAP_OFFSET_S,'gps_tow_at_time_zero_s':float(gps_tow_at_time_zero_s),'position_window_s':[lo,hi],'onset_s':float(onset_s)},'assumption':'trusted pre-onset static NMEA GGA oracle; position held fixed; post-onset PVT prohibited'}

@dataclass
class TrainingResult:
 model:GcmrNet; history:list; best_epoch:int; config:dict

def _device(value):return torch.device(value if value else ('cuda' if torch.cuda.is_available() else 'cpu'))
def train_clean_model(train_events,val_events,*,seed=7,max_epochs=40,patience=6,device=None,learning_rate=1e-3,compactness_weight=.01,warmup_epochs=5,pair_hidden=32,event_hidden=64,latent_dim=32):
 if not train_events or not val_events:raise ValueError('nonempty train and selection validation events required')
 if not 1<=int(max_epochs)<=40:raise ValueError('max_epochs must be in [1, 40]')
 if not 0<=compactness_weight<=.01:raise ValueError('compactness weight must be in [0, .01]')
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.use_deterministic_algorithms(True);dev=_device(device)
 tb=collate_gcmr_events(train_events,device=dev);vb=collate_gcmr_events(val_events,device=dev)
 model=GcmrNet(pair_hidden=pair_hidden,event_hidden=event_hidden,latent_dim=latent_dim).to(dev);model.fit_scaler(**tb)
 opt=torch.optim.Adam(model.parameters(),lr=learning_rate);history=[];best=None;best_loss=float('inf');best_epoch=0;bad=0;center=None
 for epoch in range(int(max_epochs)):
  model.train();opt.zero_grad();recon,z=model(**tb)
  if epoch==warmup_epochs:
   with torch.no_grad():center=z.mean(0).detach()
  weight=compactness_weight if epoch>=warmup_epochs and center is not None else 0.
  loss=gcmr_loss(recon,tb['observations'],tb['observation_mask'],tb['pair_mask'],observation_scale=model.scaler.observation_scale,latent=z,compactness_weight=weight,compactness_center=center);loss.backward();opt.step()
  model.eval()
  with torch.no_grad():vr,vz=model(**vb);vl=gcmr_loss(vr,vb['observations'],vb['observation_mask'],vb['pair_mask'],observation_scale=model.scaler.observation_scale).item()
  row={'epoch':epoch+1,'train_loss':float(loss.item()),'val_reconstruction_loss':float(vl)};history.append(row)
  if vl<best_loss-1e-12:best_loss=vl;best_epoch=epoch+1;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
  else:
   bad+=1
   if bad>=patience:break
 model.load_state_dict(best);model.eval()
 config={'seed':seed,'max_epochs':max_epochs,'patience':patience,'learning_rate':learning_rate,'compactness_weight':compactness_weight,'warmup_epochs':warmup_epochs,'pair_hidden':pair_hidden,'event_hidden':event_hidden,'latent_dim':latent_dim,'device':str(dev)}
 return TrainingResult(model,history,best_epoch,config)

def score_events(model,events,calibrator=None,*,device=None):
 events=list(events)
 if not events: return {'window_start_s':np.empty(0),'window_end_s':np.empty(0),'availability_s':np.empty(0),'reconstruction':np.empty(0),'latent':np.empty((0,0)),'latent_score':np.empty(0),'combined_score':np.empty(0)}
 dev=_device(device) if device else next(model.parameters()).device;b=collate_gcmr_events(events,device=dev);model.eval()
 with torch.no_grad():r,z=model(**b);err=event_reconstruction_errors(r,b['observations'],b['observation_mask'],b['pair_mask'],observation_scale=model.scaler.observation_scale).cpu().numpy();latent=z.cpu().numpy()
 if calibrator is None:rz=np.full(len(events),np.nan);lz=np.full(len(events),np.nan);score=np.full(len(events),np.nan)
 else:rz,lz=calibrator.components(err,latent);score=(rz+lz)/2
 end=np.asarray([e.window_end_s for e in events],float)
 return {'window_start_s':np.asarray([e.window_start_s for e in events],float),'window_end_s':end,'availability_s':end.copy(),'reconstruction':err,'latent':latent,'reconstruction_score':rz,'latent_score':lz,'combined_score':score}

def calibration_threshold(calibration_scores,*,quantile=.99):
 x=np.asarray(calibration_scores,float)
 if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('finite calibration scores required')
 if not 0<quantile<1:raise ValueError('quantile must be in (0,1)')
 return float(np.quantile(x,quantile))

def ablated_events(events,*,mode,seed):
 rng=np.random.default_rng(seed);out=[]
 for e in events:
  if mode=='geometry_channels_permutation':
   c=e.conditions.copy();c[:,:4]=e.conditions[rng.permutation(len(e.conditions)),:4]
  elif mode=='geometry_channels_zero':
   c=e.conditions.copy();c[:,:4]=0
  else:raise ValueError('unknown inference ablation')
  out.append(replace(e,conditions=c))
 return out

class ExperimentGate:
 def __init__(self):self.frozen=False;self.held_evaluated=False;self.attacks_open=False
 def freeze(self):self.frozen=True
 def mark_held_evaluated(self):
  if not self.frozen:raise RuntimeError('freeze clean artifacts before held evaluation')
  self.held_evaluated=True
 def open_attacks(self,*,explicit):
  if not self.frozen or not self.held_evaluated:raise RuntimeError('sealed held normal must be evaluated after freeze before attacks')
  if not explicit:raise PermissionError('attack evaluation requires explicit --open-attacks')
  self.attacks_open=True

def _jsonable(x):
 if isinstance(x,(np.integer,np.floating)):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def write_summary(path,results,**extra):
 value={'contract':{'training':'clean-only normal','score_availability':'window_end_s','attack_use':'evaluation only after sealed held freeze'},'results':results,**extra};p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,sort_keys=True,default=_jsonable)+'\n');return value

def save_score_csv(path,scored,threshold):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 alarm=np.asarray(scored['combined_score'])>threshold
 with p.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['window_start_s','window_end_s','score_available_s','reconstruction','latent_score','combined_score','threshold','alarm'])
  for row in zip(scored['window_start_s'],scored['window_end_s'],scored['availability_s'],scored['reconstruction'],scored['latent_score'],scored['combined_score'],np.full(len(alarm),threshold),alarm):w.writerow(row)

def calibrator_state(c):
 return {k:(torch.as_tensor(v) if isinstance(v,np.ndarray) else v) for k,v in vars(c).items()}

def _restore_calibrator(state):
 c=CleanReferenceScoreCalibrator(shrinkage=float(state['shrinkage']),minimum_scale=float(state['minimum_scale']))
 for key,value in state.items():
  if key in ('shrinkage','minimum_scale','fitted_'):continue
  setattr(c,key,value.cpu().numpy() if torch.is_tensor(value) else value)
 c.fitted_=bool(state.get('fitted_',False))
 if not c.fitted_:raise ValueError('checkpoint calibrator is not fitted')
 return c

@dataclass(frozen=True)
class LoadedCheckpoint:
 model:GcmrNet; calibrator:CleanReferenceScoreCalibrator; threshold:float; config:dict; best_epoch:int; provenance:dict

def save_checkpoint(path,training,calibrator,threshold,*,provenance):
 if not isinstance(provenance,dict) or provenance.get('implementation')!=implementation_manifest():raise ValueError('checkpoint provenance implementation manifest mismatch')
 payload={'format':'gcmr-clean-v2','schema_version':CHECKPOINT_SCHEMA_VERSION,'config':training.config,'best_epoch':training.best_epoch,'feature_contract':{'observation':list(OBSERVATION_FEATURES),'condition':list(CONDITION_FEATURES)},'model_state':{k:v.detach().cpu() for k,v in training.model.state_dict().items()},'calibrator':calibrator_state(calibrator),'threshold':float(threshold),'provenance':_canonical(provenance)}
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(payload,tmp);os.replace(tmp,p)

def load_checkpoint(path,*,expected_provenance=None,device='cpu'):
 try: payload=torch.load(path,map_location='cpu',weights_only=True)
 except (OSError,RuntimeError,ValueError,KeyError) as exc:raise ValueError(f'invalid GCMR checkpoint: {exc}') from exc
 if not isinstance(payload,dict) or payload.get('format')!='gcmr-clean-v2' or payload.get('schema_version')!=CHECKPOINT_SCHEMA_VERSION:raise ValueError('incompatible checkpoint schema')
 expected_features={'observation':list(OBSERVATION_FEATURES),'condition':list(CONDITION_FEATURES)}
 if payload.get('feature_contract')!=expected_features:raise ValueError('checkpoint feature contract mismatch')
 saved_provenance=payload.get('provenance')
 if not isinstance(saved_provenance,dict) or saved_provenance.get('implementation')!=implementation_manifest():raise ValueError('checkpoint provenance implementation manifest mismatch')
 if expected_provenance is not None and saved_provenance!=_canonical(expected_provenance):raise ValueError('checkpoint provenance mismatch')
 config=payload.get('config',{}); required=('pair_hidden','event_hidden','latent_dim')
 if any(k not in config for k in required):raise ValueError('checkpoint architecture config missing')
 model=GcmrNet(**{k:int(config[k]) for k in required})
 try:model.load_state_dict(payload['model_state'],strict=True)
 except (KeyError,RuntimeError) as exc:raise ValueError(f'checkpoint model state mismatch: {exc}') from exc
 model.to(_device(device)).eval();calibrator=_restore_calibrator(payload['calibrator']);threshold=float(payload['threshold'])
 if not math.isfinite(threshold):raise ValueError('checkpoint threshold must be finite')
 return LoadedCheckpoint(model,calibrator,threshold,dict(config),int(payload['best_epoch']),dict(payload['provenance']))
