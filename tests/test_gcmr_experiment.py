import json, numpy as np, pytest, torch
from pathlib import Path
from gnss_doppler_lab.gcmr_experiment import *
from gnss_doppler_lab.gcmr_relations import GcmrPairRelationEvent

def ev(s,p=3):
 r=np.random.default_rng(int(s*10)+3);o=r.normal(size=(p,10)).astype('f');c=r.normal(size=(p,8)).astype('f')
 return GcmrPairRelationEvent(s,s+1,np.c_[np.arange(1,p+1),np.arange(2,p+2)],o,np.ones_like(o,dtype=bool),c)
def events(n=20):return [ev(i*.5,3+i%3) for i in range(n)]
def ck(b):
 x=0
 for c in b:x^=ord(c)
 return f'${b}*{x:02X}'

def test_roles_strict_purges_and_boundaries():
 validate_roles(DEFAULT_ROLES);assert [r.name for r in DEFAULT_ROLES]==['train','selection_val','clean_reference','event_calibration','sealed_held']
 assert all(a.end_s<b.start_s for a,b in zip(DEFAULT_ROLES,DEFAULT_ROLES[1:]))
 assert [e.window_start_s for e in select_role_events([ev(29.5),ev(30),ev(179.5),ev(180)],DEFAULT_ROLES[0])]==[30]
 with pytest.raises(ValueError,match='overlap|purge'):validate_roles((TemporalRole('a',0,10),TemporalRole('b',10,20)))

def test_nmea_absolute_mapping_excludes_post_onset(tmp_path):
 lines=[ck('GPRMC,095940.00,A,4500.0000,N,07300.0000,W,0,0,260726,,,A'),ck('GPGGA,100000.00,4500.0000,N,07300.0000,W,1,10,.8,100.0,M,0,M,,'),ck('GPGGA,100100.00,4500.0060,N,07300.0060,W,1,10,.8,102.0,M,0,M,,'),ck('GPGGA,100210.00,5000.0000,N,08000.0000,W,1,10,.8,999,M,0,M,,')]
 p=tmp_path/'x';p.write_text('\n'.join(lines));x=parse_preonset_nmea_position(p,gps_tow_at_time_zero_s=35998,onset_s=120,position_window_s=(20,90))
 assert x['sample_count']==2 and x['relative_times_s']==pytest.approx([20,80]);assert x['llh']==pytest.approx((45.00005,-73.00005,101));assert x['timing']['gps_utc_leap_offset_s']==18

def test_cache_roundtrip_hash_mismatch(tmp_path):
 s=tmp_path/'s';s.write_bytes(b'a');p=tmp_path/'c.npz';m={'scenario':'x','timing':{'window_s':1.,'stride_s':.5}}
 cache_events(p,events(3),source_paths=[s],metadata=m);got,meta=load_event_cache(p,source_paths=[s],expected_metadata=m);assert len(got)==3 and meta['schema_version']>=1
 s.write_bytes(b'b')
 with pytest.raises(ValueError,match='stale|hash'):load_event_cache(p,source_paths=[s],expected_metadata=m)

def test_deterministic_variable_pair_training():
 x=events();a=train_clean_model(x[:12],x[12:],seed=9,max_epochs=3,device='cpu');b=train_clean_model(x[:12],x[12:],seed=9,max_epochs=3,device='cpu')
 assert a.history==b.history and all(torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items())

def test_attack_gate():
 g=ExperimentGate()
 with pytest.raises(RuntimeError,match='held'):g.open_attacks(explicit=True)
 g.freeze();g.mark_held_evaluated()
 with pytest.raises(PermissionError,match='explicit'):g.open_attacks(explicit=False)
 g.open_attacks(explicit=True);assert g.attacks_open

def test_calibration_threshold_and_availability():
 assert calibration_threshold(np.arange(100.),quantile=.99)==pytest.approx(np.quantile(np.arange(100.),.99))
 model=train_clean_model(events(14)[:10],events(14)[10:],seed=2,max_epochs=2,device='cpu').model
 ref=score_events(model,events(6));from gnss_doppler_lab.gcmr_model import CleanReferenceScoreCalibrator
 cal=CleanReferenceScoreCalibrator().fit(ref['reconstruction'],ref['latent']);assert score_events(model,[ev(4.5)],calibrator=cal)['availability_s'].tolist()==[5.5]

def test_geometry_channel_ablation_is_deterministic_and_preserves_quality_bytes():
 x=events(4);a=ablated_events(x,mode='geometry_channels_permutation',seed=4);b=ablated_events(x,mode='geometry_channels_permutation',seed=4)
 assert all(np.array_equal(i.conditions,j.conditions) for i,j in zip(a,b))
 assert all(np.array_equal(i.conditions[:,4:].view(np.uint8),j.conditions[:,4:].view(np.uint8)) for i,j in zip(x,a))
 assert all(sorted(map(tuple,i.conditions[:,:4]))==sorted(map(tuple,j.conditions[:,:4])) for i,j in zip(x,a))
 zero=ablated_events(x,mode='geometry_channels_zero',seed=1)
 assert all(not i.conditions[:,:4].any() for i in zero)
 assert all(np.array_equal(i.conditions[:,4:].view(np.uint8),j.conditions[:,4:].view(np.uint8)) for i,j in zip(x,zero))

def test_summary_normal_context(tmp_path):
 p=tmp_path/'s.json';write_summary(p,{'held':{'normal_context':'same_recording_held_normal'},'os1':{'normal_context':'ood_normal_pre_attack'}});x=json.loads(p.read_text());assert x['results']['held']['normal_context']!=x['results']['os1']['normal_context']


def test_relation_contract_rejects_old_cache_even_with_current_schema(tmp_path):
 src=tmp_path/'s';src.write_bytes(b'a');p=tmp_path/'cache.npz';m={'scenario':'x'}
 meta=cache_events(p,events(1),source_paths=[src],metadata=m)
 assert meta['relation_contract_version'] >= 2
 with np.load(p,allow_pickle=False) as z: payload={k:z[k].copy() for k in z.files}
 stale=json.loads(str(payload['metadata_json']));stale['relation_contract_version']=1
 payload['metadata_json']=np.asarray(json.dumps(stale,sort_keys=True));np.savez_compressed(p,**payload)
 with pytest.raises(ValueError,match='contract|schema'):load_event_cache(p,source_paths=[src],expected_metadata=m)

def test_oakbat_geometry_preflight_reads_rx_time_and_absolute_rmc_week(tmp_path):
 raw=tmp_path/'raw';raw.mkdir()
 import h5py
 with h5py.File(raw/'observables.mat','w') as h:h.create_dataset('RX_time',data=np.asarray([0.,381618.02,381618.04]))
 nmea=tmp_path/'nmea';nmea.write_text(ck('GPRMC,100030.19,A,3555.8,N,08418.6,W,0,0,190320,,,A')+chr(10))
 from gnss_doppler_lab.gcmr_geometry import GpsEphemeris
 eph={7:GpsEphemeris(7,1.,4e-9,.01,5153.,.7,.94,-.3,-8e-9,1e-10,1e-6,2e-6,200.,-80.,3e-8,-2e-8,388800.,49,toc=388800.,decoded_tow=382068.,SV_health=0,fit_interval_flag=0)}
 report=preflight_oakbat_geometry(raw/'observables.mat',nmea,eph,configured_tow0_s=381618.,max_toe_age_s=7200.,tow_tolerance_s=.05)
 assert report['recording_start_rx_time_s']==pytest.approx(381618.02) and report['full_gps_week']==2097
 assert report['geometry_contract']['classification']=='offline_trusted_static_receive_time_approximation'
 assert report['geometry_contract']['omitted_corrections']==['transmit_time','Sagnac','satellite_clock']
 assert report['decoded_snapshot']['required_at_or_before_recording_start'] is False
 with pytest.raises(ValueError,match='tow0|recording start'):preflight_oakbat_geometry(raw/'observables.mat',nmea,eph,configured_tow0_s=381600.,max_toe_age_s=7200.,tow_tolerance_s=.05)


def test_oakbat_preflight_documents_unhealthy_tracked_prns_and_cache_metadata(tmp_path):
 raw=tmp_path/'raw';raw.mkdir()
 import h5py
 with h5py.File(raw/'observables.mat','w') as h:h.create_dataset('RX_time',data=np.asarray([381618.02]))
 nmea=tmp_path/'nmea';nmea.write_text(ck('GPRMC,100030.19,A,3555.8,N,08418.6,W,0,0,190320,,,A')+'\n')
 from gnss_doppler_lab.gcmr_geometry import GpsEphemeris
 def eph(prn,health):return GpsEphemeris(prn,1.,4e-9,.01,5153.,.7,.94,-.3,-8e-9,1e-10,1e-6,2e-6,200.,-80.,3e-8,-2e-8,388800.,49,SV_health=health)
 ephemerides={p:eph(p,63 if p==18 else 0) for p in (7,8,10,11,18)}
 report=preflight_oakbat_geometry(raw/'observables.mat',nmea,ephemerides,
  configured_tow0_s=381618.,max_toe_age_s=7200.,tracked_prns=(7,8,10,11,18),min_prns=4)
 health=report['ephemeris_health']
 assert health['healthy_tracked_prns']==[7,8,10,11]
 assert health['excluded_tracking_prns']==[18]
 assert health['excluded_ephemeris_health_by_prn']=={18:63}
 src=tmp_path/'source';src.write_bytes(b'x');cache=tmp_path/'events.npz'
 meta=cache_events(cache,events(1),source_paths=[src],metadata={'geometry_preflight':report})
 assert meta['geometry_preflight']['ephemeris_health']['excluded_ephemeris_health_by_prn']=={'18':63}

def test_oakbat_preflight_rejects_too_few_healthy_tracked_prns(tmp_path):
 raw=tmp_path/'raw';raw.mkdir()
 import h5py
 with h5py.File(raw/'observables.mat','w') as h:h.create_dataset('RX_time',data=np.asarray([381618.02]))
 nmea=tmp_path/'nmea';nmea.write_text(ck('GPRMC,100030.19,A,3555.8,N,08418.6,W,0,0,190320,,,A')+'\n')
 from gnss_doppler_lab.gcmr_geometry import GpsEphemeris
 def eph(prn,health):return GpsEphemeris(prn,1.,4e-9,.01,5153.,.7,.94,-.3,-8e-9,1e-10,1e-6,2e-6,200.,-80.,3e-8,-2e-8,388800.,49,SV_health=health)
 ephemerides={p:eph(p,63 if p==4 else 0) for p in (1,2,3,4)}
 with pytest.raises(ValueError,match='healthy tracked PRNs'):
  preflight_oakbat_geometry(raw/'observables.mat',nmea,ephemerides,configured_tow0_s=381618.,
   max_toe_age_s=7200.,tracked_prns=(1,2,3,4),min_prns=4)


def _rewrite_cache(path, mutate):
 with np.load(path,allow_pickle=False) as z: payload={k:z[k].copy() for k in z.files}
 mutate(payload)
 with Path(path).open("wb") as f:np.savez_compressed(f,**payload)

@pytest.mark.parametrize("mutation",[lambda x:x["offsets"].__setitem__(1,-1),lambda x:x["offsets"].__setitem__(1,x["offsets"][0])])
def test_cache_rejects_negative_or_nonmonotone_offsets(tmp_path,mutation):
 src=tmp_path/"s";src.write_bytes(b"x");p=tmp_path/"c";cache_events(p,events(2),source_paths=[src],metadata={});_rewrite_cache(p,mutation)
 with pytest.raises(ValueError,match="offset"):load_event_cache(p,source_paths=[src],expected_metadata={})

def test_cache_rejects_truncated_concatenated_array(tmp_path):
 src=tmp_path/"s";src.write_bytes(b"x");p=tmp_path/"c";cache_events(p,events(2),source_paths=[src],metadata={})
 _rewrite_cache(p,lambda x:x.__setitem__("conditions",x["conditions"][:-1]))
 with pytest.raises(ValueError,match="length|offset"):load_event_cache(p,source_paths=[src],expected_metadata={})

def test_implementation_manifest_cwd_independent_and_content_sensitive(tmp_path,monkeypatch):
 root=tmp_path/"repo";(root/"src/gnss_doppler_lab").mkdir(parents=True);(root/"scripts").mkdir();(root/"tests").mkdir()
 files={"pyproject.toml":"[project]","src/gnss_doppler_lab/gcmr_model.py":"a","src/gnss_doppler_lab/gcmr_relations.py":"b","src/gnss_doppler_lab/gcmr_geometry.py":"c","src/gnss_doppler_lab/gcmr_experiment.py":"d","src/gnss_doppler_lab/trajectory.py":"t","scripts/run_gcmr_oakbat_poc.py":"e","tests/test_gcmr_model.py":"f","tests/test_gcmr_relations.py":"g","tests/test_gcmr_geometry.py":"h","tests/test_gcmr_experiment.py":"i"}
 for rel,data in files.items():(root/rel).write_text(data)
 anchor=root/"src/gnss_doppler_lab/gcmr_model.py";a=implementation_manifest(anchor);monkeypatch.chdir(tmp_path);assert a==implementation_manifest(anchor)
 (root/"tests/test_gcmr_model.py").write_text("F");assert implementation_manifest(anchor)["aggregate_sha256"]!=a["aggregate_sha256"]
 (root/"tests/test_gcmr_model.py").write_text("f");before=implementation_manifest(anchor)["aggregate_sha256"]
 trajectory=root/"src/gnss_doppler_lab/trajectory.py";trajectory.write_bytes(trajectory.read_bytes()+b"!")
 assert implementation_manifest(anchor)["aggregate_sha256"]!=before
 (root/"pyproject.toml").unlink()
 with pytest.raises(ValueError,match="root|missing"):implementation_manifest(anchor)

def test_checkpoint_validated_roundtrip_and_contract_rejection(tmp_path):
 x=events(14);tr=train_clean_model(x[:10],x[10:],seed=2,max_epochs=2,device="cpu")
 raw=score_events(tr.model,x[:6]);cal=CleanReferenceScoreCalibrator().fit(raw["reconstruction"],raw["latent"]);sc=score_events(tr.model,x[6:8],cal);th=calibration_threshold(sc["combined_score"])
 prov={"implementation":implementation_manifest()};p=tmp_path/"model.pt"
 with pytest.raises(ValueError,match="implementation"):save_checkpoint(p,tr,cal,th,provenance={"implementation":{"files":[],"aggregate_sha256":"bad"}})
 save_checkpoint(p,tr,cal,th,provenance=prov)
 loaded=load_checkpoint(p,expected_provenance=prov);actual=score_events(loaded.model,x[6:8],loaded.calibrator)
 assert np.array_equal(sc["combined_score"],actual["combined_score"]);assert np.array_equal(sc["combined_score"]>th,actual["combined_score"]>loaded.threshold)
 original=torch.load(p,weights_only=True)
 payload=dict(original);payload["feature_contract"]={**original["feature_contract"],"observation":["bad",*original["feature_contract"]["observation"][1:]]};torch.save(payload,p)
 with pytest.raises(ValueError,match="feature"):load_checkpoint(p,expected_provenance=prov)
 payload=dict(original);payload["schema_version"]=-1;torch.save(payload,p)
 with pytest.raises(ValueError,match="schema"):load_checkpoint(p,expected_provenance=prov)
 payload=dict(original);payload["provenance"]={"implementation":{"aggregate_sha256":"bad","files":[]}};torch.save(payload,p)
 with pytest.raises(ValueError,match="provenance"):load_checkpoint(p,expected_provenance=prov)


def test_checkpoint_rejects_saved_implementation_stale_against_runtime(tmp_path):
 x=events(14);tr=train_clean_model(x[:10],x[10:],seed=2,max_epochs=1,device="cpu")
 raw=score_events(tr.model,x[:6]);cal=CleanReferenceScoreCalibrator().fit(raw["reconstruction"],raw["latent"])
 current={"implementation":implementation_manifest()};p=tmp_path/"model.pt"
 save_checkpoint(p,tr,cal,1.0,provenance=current)
 payload=torch.load(p,weights_only=True);stale={**payload["provenance"],"implementation":{**payload["provenance"]["implementation"],"aggregate_sha256":"0"*64}}
 payload["provenance"]=stale;torch.save(payload,p)
 with pytest.raises(ValueError,match="implementation|provenance|stale"):
  load_checkpoint(p,expected_provenance=stale)


def test_torch_is_gcmr_optional_extra_not_base_dependency():
 import tomllib
 pyproject=tomllib.loads((Path(__file__).parents[1]/"pyproject.toml").read_text())
 assert not any(x.startswith("torch") for x in pyproject["project"]["dependencies"])
 assert any(x.startswith("torch") for x in pyproject["project"]["optional-dependencies"]["gcmr"])
