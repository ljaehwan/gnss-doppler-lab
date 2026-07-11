import math
import hashlib
import json
from datetime import timezone
from pathlib import Path
import numpy as np, pytest
from gnss_doppler_lab.dynamic_validation import parse_pvt_log, parse_nav_decoded_prns, finite_difference, metrics, validate_run, _plot
from gnss_doppler_lab.trajectory import enu_to_llh, llh_to_enu, llh_to_ecef

def test_parser_and_utc(tmp_path):
 p=tmp_path/'r.log'; p.write_text('Position at 2022-Jan-01 00:00:01.500000 UTC using 7 observations is Lat = 37.0 [deg], Long = 127.0 [deg], Height = 10.0 [m]\nVelocity: East: 1.0 [m/s], North: -2.0 [m/s], Up = 0.5 [m/s]\n')
 x=parse_pvt_log(p)[0]; assert x['utc'].tzinfo==timezone.utc and x['utc'].timestamp()==1640995201.5; assert x['vel']==(1,-2,.5)
def test_missing_log_lines(tmp_path):
 p=tmp_path/'r'; p.write_text('noise\n');
 with pytest.raises(ValueError,match='no GNSS-SDR PVT'): parse_pvt_log(p)

def test_nav_parser_matches_exact_message_and_order_dedupes(tmp_path):
 p=tmp_path/'receiver.log'; p.write_text(
  'Acquisition success for GPS PRN 31\nPosition using GPS PRN 29\n'
  'New GPS NAV message received in channel 4: subframe 3 from satellite GPS PRN 05 (Block IIR-M) with CN0=49 dB-Hz\n'
  'New GPS NAV message received in channel 1: subframe 4 from satellite GPS PRN 24 (Block IIF) with CN0=58 dB-Hz\n'
  'New GPS NAV message received in channel 4: subframe 4 from satellite GPS PRN 05 (Block IIR-M) with CN0=50 dB-Hz\n'
  'New GPS NAV message received for GPS PRN 31\n')
 assert parse_nav_decoded_prns(p)==['G05','G24']
def test_position_and_timestamp_aware_velocity():
 o=(37.5,127.,100.); q=enu_to_llh(30,-20,4,*o); assert np.allclose(llh_to_enu(*q,*o),(30,-20,4),atol=1e-4)
 t=np.array([0.,1.,3.]); xyz=np.column_stack((2*t, -3*t, .5*t)); assert np.allclose(finite_difference(t,xyz),[ [2,-3,.5]]*3)
def test_metrics_percentiles():
 m=metrics(range(1,101)); assert m['median']==50.5 and m['p95']==pytest.approx(95.05) and m['max']==100


def _validation_fixture(tmp_path, *, corrected, pvt_time):
 truth=tmp_path/'truth.csv'; truth.write_text('0,37,127,10\n20,37,127,10\n')
 digest=hashlib.sha256(truth.read_bytes()).hexdigest()
 rf={"schema_version":2 if corrected else 1,"run_id":"rf","scenario":{"utc":"2022-01-01T00:00:00Z","position":{"path":str(truth),"sha256":digest,"coordinate_system":"llh"}}}
 if corrected:
  rf['scenario']['time']={"requested_utc":"2022-01-01T00:00:00Z","simulator_input_calendar":"2022/01/01,00:00:18","simulator_input_time_scale":"GPST","gps_minus_utc_seconds":18}
 rfp=tmp_path/'rf.json'; rfp.write_text(json.dumps(rf))
 rr=tmp_path/'receiver'; rr.mkdir(); (rr/'manifest.json').write_text(json.dumps({"receiver_run_id":"rx","source":{"rf_manifest":str(rfp)},"tracking":{"csv":"tracking.csv"},"acquisition":{"tracked_prn_count":1}}))
 (rr/'tracking.csv').write_text('time_s,prn,carrier_doppler_hz\n0,G01,1\n')
 (rr/'receiver.log').write_text(f'Position at {pvt_time}.000000 UTC using 7 observations is Lat = 37.0 [deg], Long = 127.0 [deg], Height = 10.0 [m]\nVelocity: East: 0.0 [m/s], North: 0.0 [m/s], Up = 0.0 [m/s]\n')
 return rr,rfp


def test_truth_hash_mutation_is_rejected_before_validation_outputs(tmp_path):
 rr,rfp=_validation_fixture(tmp_path,corrected=True,pvt_time='2022-Jan-01 00:00:01')
 truth=Path(json.loads(rfp.read_text())['scenario']['position']['path'])
 truth.write_text(truth.read_text()+'21,37,127,10\n')
 with pytest.raises(ValueError,match='truth.*SHA-256.*mismatch'):
  validate_run(rr,rfp)
 assert not (rr/'validation').exists()


@pytest.mark.parametrize('bad_hash',[None,'not-a-sha256','A'*64])
def test_truth_hash_missing_or_invalid_fails_closed(tmp_path,bad_hash):
 rr,rfp=_validation_fixture(tmp_path,corrected=True,pvt_time='2022-Jan-01 00:00:01')
 rf=json.loads(rfp.read_text())
 if bad_hash is None: del rf['scenario']['position']['sha256']
 else: rf['scenario']['position']['sha256']=bad_hash
 rfp.write_text(json.dumps(rf))
 with pytest.raises(ValueError,match='scenario.position.sha256'):
  validate_run(rr,rfp)


def test_ecef_truth_matches_equivalent_llh_enu_and_metrics(tmp_path):
 llh_dir=tmp_path/'llh'; llh_dir.mkdir()
 ecef_dir=tmp_path/'ecef'; ecef_dir.mkdir()
 llh_rr,llh_rfp=_validation_fixture(llh_dir,corrected=True,pvt_time='2022-Jan-01 00:00:01')
 ecef_rr,ecef_rfp=_validation_fixture(ecef_dir,corrected=True,pvt_time='2022-Jan-01 00:00:01')
 ecef_rf=json.loads(ecef_rfp.read_text()); truth=Path(ecef_rf['scenario']['position']['path'])
 xyz=llh_to_ecef(37,127,10)
 truth.write_text(''.join(f'{t},{xyz[0]:.9f},{xyz[1]:.9f},{xyz[2]:.9f}\n' for t in (0,20)))
 ecef_rf['scenario']['position'].update(coordinate_system='ecef',sha256=hashlib.sha256(truth.read_bytes()).hexdigest())
 ecef_rfp.write_text(json.dumps(ecef_rf))
 llh=validate_run(llh_rr,llh_rfp); ecef=validate_run(ecef_rr,ecef_rfp)
 for key in ('horizontal_position_error_m','position_3d_error_m','horizontal_velocity_error_mps'):
  assert ecef[key]==pytest.approx(llh[key],abs=1e-6)
 assert ecef['artifacts']['truth']['sha256']==ecef_rf['scenario']['position']['sha256']
 llh_rows=list(__import__('csv').DictReader((llh_rr/'validation/per_fix.csv').open()))
 ecef_rows=list(__import__('csv').DictReader((ecef_rr/'validation/per_fix.csv').open()))
 for key in ('truth_e_m','truth_n_m','truth_u_m'):
  assert float(ecef_rows[0][key])==pytest.approx(float(llh_rows[0][key]),abs=1e-6)


def test_unknown_truth_coordinate_system_is_rejected(tmp_path):
 rr,rfp=_validation_fixture(tmp_path,corrected=True,pvt_time='2022-Jan-01 00:00:01')
 rf=json.loads(rfp.read_text()); rf['scenario']['position']['coordinate_system']='utm'; rfp.write_text(json.dumps(rf))
 with pytest.raises(ValueError,match='coordinate_system'):
  validate_run(rr,rfp)


def test_corrected_manifest_uses_direct_utc_zero_alignment(tmp_path):
 rr,rfp=_validation_fixture(tmp_path,corrected=True,pvt_time='2022-Jan-01 00:00:01')
 summary=validate_run(rr,rfp)
 assert summary['time_alignment']=={'mode':'corrected_manifest_direct_utc','correction_seconds':0.0}
 assert summary['horizontal_position_error_m']['median'] < 1e-6


def test_legacy_manifest_requires_and_records_explicit_plus_18(tmp_path):
 rr,rfp=_validation_fixture(tmp_path,corrected=False,pvt_time='2021-Dec-31 23:59:42')
 with pytest.raises(ValueError,match='legacy/ambiguous'): validate_run(rr,rfp)
 summary=validate_run(rr,rfp,legacy_gps_utc_offset_seconds=18)
 assert summary['time_alignment']=={'mode':'explicit_legacy_override','correction_seconds':18.0}


def test_summary_separates_all_tracking_from_nav_decoded(tmp_path):
 rr,rfp=_validation_fixture(tmp_path,corrected=True,pvt_time='2022-Jan-01 00:00:01')
 (rr/'tracking.csv').write_text('time_s,prn,carrier_doppler_hz\n0,G01,1\n0,G31,999\n')
 with (rr/'receiver.log').open('a') as f:
  f.write('New GPS NAV message received in channel 0: subframe 3 from satellite GPS PRN 01 (Block IIF) with CN0=45 dB-Hz\n')
 summary=validate_run(rr,rfp)
 assert summary['schema_version']==3
 assert set(summary['doppler_by_prn_all'])=={'G01','G31'}
 assert set(summary['doppler_by_prn_nav_decoded'])=={'G01'}
 assert summary['doppler_by_prn']==summary['doppler_by_prn_all']
 assert summary['doppler_by_prn_compatibility']=='legacy alias of doppler_by_prn_all'
 assert summary['prn_counts']['nav_decoded']==1
 assert summary['nav_decoded_prns']==['G01']
 assert summary['acquisition_only_prns']==['G31']


def test_dashboard_enu_axis_labels_and_equal_aspect(tmp_path, monkeypatch):
 import matplotlib.figure
 figures=[]
 monkeypatch.setattr(matplotlib.figure.Figure,'savefig',lambda self,*a,**k: figures.append(self))
 row={k:0.0 for k in ('trajectory_time_s','truth_e_m','truth_n_m','pvt_e_m','pvt_n_m',
                       'horizontal_position_error_m','truth_ve_mps','truth_vn_mps',
                       'pvt_ve_mps','pvt_vn_mps')}
 _plot(tmp_path/'unused.png',[row],{'G01':([0],[1])},'test')
 enu=figures[0].axes[0]
 assert enu.get_xlabel()=='East [m]'
 assert enu.get_ylabel()=='North [m]'
 assert enu.get_aspect()==1.0
 assert figures[0].axes[3].get_title()=='NAV-decoded carrier Doppler [Hz]'
