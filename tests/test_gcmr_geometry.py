import math
import pytest
from gnss_doppler_lab.gcmr_geometry import (GPS_L1_WAVELENGTH_M, GpsEphemeris, common_clock_removed_residuals, look_angles, parse_gnss_sdr_gps_ephemeris_xml, predict_static_l1_doppler, satellite_observation, satellite_position_ecef, satellite_velocity_ecef, _solve_kepler, validate_ephemeris_time_alignment)
from gnss_doppler_lab.trajectory import llh_to_ecef

VALID_FIELDS={"PRN":7,"toc":604790.,"tow":25.,"SV_health":0,"SV_accuracy":2,"fit_interval_flag":0,"M_0":1.0,"delta_n":4.5e-9,"ecc":.01,"sqrtA":5153.7955,"OMEGA_0":.7,"i_0":.94,"omega":-.3,"OMEGAdot":-8e-9,"idot":1e-10,"Cuc":1e-6,"Cus":2e-6,"Crc":200.,"Crs":-80.,"Cic":3e-8,"Cis":-2e-8,"toe":604790.,"WN":2300}
def ephemeris(**updates):
 values=dict(VALID_FIELDS); values.update(updates); values["decoded_tow"]=values.pop("tow"); return GpsEphemeris(**values)
def boost_xml(fields):
 body="\n".join(f"<{n}>{v}</{n}>" for n,v in fields.items())
 return f'''<?xml version="1.0"?><boost_serialization><GNSS-SDR_ephemeris_map><count>1</count><item_version>0</item_version><item><first>{fields.get("PRN",7)}</first><second>{body}</second></item></GNSS-SDR_ephemeris_map></boost_serialization>'''

def test_parses_gnss_sdr_boost_xml_fixture(tmp_path):
 p=tmp_path/"gps_ephemeris.xml"; p.write_text(boost_xml(VALID_FIELDS)); parsed=parse_gnss_sdr_gps_ephemeris_xml(p)
 assert set(parsed)=={7}; assert parsed[7].prn==7; assert parsed[7].sqrt_a==pytest.approx(5153.7955); assert parsed[7].omega_dot==pytest.approx(-8e-9); assert parsed[7].week==2300

def test_broadcast_position_has_plausible_gps_orbit_radius():
 radius=math.dist((0,0,0),satellite_position_ecef(ephemeris(),5.)); assert 25_000_000 < radius < 28_000_000

def test_week_crossover_position_and_velocity_are_continuous():
 eph=ephemeris(toe=604790.); before=satellite_position_ecef(eph,604799.9); after=satellite_position_ecef(eph,.1); velocity=satellite_velocity_ecef(eph,0.,difference_s=.5)
 assert math.dist(before,after)<1000.; assert 1000.<math.dist((0,0,0),velocity)<5000.

def test_look_angles_obey_local_enu_invariant():
 receiver=llh_to_ecef(0.,0.,0.); overhead=(receiver[0]+20e6,receiver[1],receiver[2]); east_horizon=(receiver[0],receiver[1]+20e6,receiver[2]); up=look_angles(receiver,overhead); east=look_angles(receiver,east_horizon)
 assert up.elevation_deg==pytest.approx(90.); assert up.azimuth_deg==pytest.approx(0.); assert east.elevation_deg==pytest.approx(0.); assert east.azimuth_deg==pytest.approx(90.); assert math.dist((0,0,0),east.los_ecef)==pytest.approx(1.)

def test_static_receiver_doppler_uses_negative_range_rate_convention():
 receiver=llh_to_ecef(45.,-73.,100.); obs=satellite_observation(receiver,ephemeris(),100.); velocity=satellite_velocity_ecef(ephemeris(),100.); expected=sum(a*b for a,b in zip(velocity,obs.los_ecef))
 assert obs.range_rate_mps==pytest.approx(expected); assert obs.predicted_l1_doppler_hz==pytest.approx(-expected/GPS_L1_WAVELENGTH_M)

def test_common_clock_removed_residuals_use_visible_prn_median():
 observed={1:110.,2:212.,3:309.,4:9999.}; predicted={1:100.,2:200.,3:300.,4:0.}; residuals=common_clock_removed_residuals(observed,predicted,visible_prns=[1,2,3]); assert residuals=={1:0.,2:2.,3:-1.}

@pytest.mark.parametrize("bad",[float("nan"),float("inf"),"not-a-number"])
def test_nonfinite_or_invalid_inputs_fail_closed(bad):
 with pytest.raises(ValueError): satellite_position_ecef(ephemeris(),bad)
 with pytest.raises(ValueError): look_angles((bad,0.,0.),(20e6,0.,0.))

def test_missing_or_nonfinite_ephemeris_fails_closed(tmp_path):
 missing=dict(VALID_FIELDS); missing.pop("sqrtA"); p=tmp_path/"missing.xml"; p.write_text(boost_xml(missing))
 with pytest.raises(ValueError,match="sqrtA"): parse_gnss_sdr_gps_ephemeris_xml(p)
 invalid=dict(VALID_FIELDS,ecc="nan"); p=tmp_path/"invalid.xml"; p.write_text(boost_xml(invalid))
 with pytest.raises(ValueError,match="ecc"): parse_gnss_sdr_gps_ephemeris_xml(p)
 with pytest.raises(ValueError,match="WN"):
  satellite_position_ecef(ephemeris(WN=-1), 0.)
 with pytest.raises(ValueError,match="PRN 9"): common_clock_removed_residuals({7:1.},{7:1.},visible_prns=[7,9])


def test_predicted_doppler_convenience_returns_observation_value():
 receiver=llh_to_ecef(10.,20.,0.); eph=ephemeris(); assert predict_static_l1_doppler(receiver,eph,200.)==pytest.approx(satellite_observation(receiver,eph,200.).predicted_l1_doppler_hz)


def test_parses_epoch_snapshot_health_and_fit_metadata(tmp_path):
 p=tmp_path/'gps_ephemeris.xml';p.write_text(boost_xml(VALID_FIELDS));e=parse_gnss_sdr_gps_ephemeris_xml(p)[7]
 assert e.toc==pytest.approx(604790.) and e.decoded_tow==pytest.approx(25.)
 assert (e.SV_health,e.SV_accuracy,e.fit_interval_flag)==(0,2,0)

def test_ephemeris_alignment_checks_week_and_toe_age_but_reports_snapshot_separately():
 e=ephemeris(WN=49,toe=388800.,toc=388800.,tow=382068.)
 report=validate_ephemeris_time_alignment({7:e},full_gps_week=2097,recording_start_tow_s=381618.,max_toe_age_s=7200.)
 assert report['week_modulus']==1024 and report['decoded_snapshot_available'] is True
 assert report['decoded_snapshot_relation']=='after_recording_start_allowed_offline_oracle'
 with pytest.raises(ValueError,match='week'): validate_ephemeris_time_alignment({7:e},full_gps_week=2098,recording_start_tow_s=381618.,max_toe_age_s=7200.)
 with pytest.raises(ValueError,match='toe age'): validate_ephemeris_time_alignment({7:e},full_gps_week=2097,recording_start_tow_s=380000.,max_toe_age_s=100.)

def test_missing_optional_decoded_snapshot_is_reported_not_rejected():
 report=validate_ephemeris_time_alignment({7:ephemeris(WN=49,toe=381600.,toc=381600.,tow=None)},full_gps_week=2097,recording_start_tow_s=381618.,max_toe_age_s=60.)
 assert report['decoded_snapshot_available'] is False


def test_kepler_solver_fails_closed_on_invalid_eccentricity_and_nonconvergence():
 with pytest.raises(ValueError,match="eccentricity"):_solve_kepler(1.,.2)
 with pytest.raises(RuntimeError,match="converge"):_solve_kepler(1.,.01,max_iterations=1,tolerance=1e-30)

def test_oakbat_g10_hard_coded_geometry_sanity():
 e=GpsEphemeris(10,-0.7553282304526149,4.343395205558577e-09,0.0054967796895653,5153.670192718506,-0.9196452474416399,0.9650487489197218,-2.6465066741058054,-7.960331579762318e-09,2.0000833114980698e-11,-5.023553967475891e-06,8.769333362579346e-06,210.84375,-99.03125,9.499490261077881e-08,6.51925802230835e-08,388800.,49,SV_health=0)
 rx=llh_to_ecef(35+55.8321729/60,-(84+18.6390752/60),256.509);o=satellite_observation(rx,e,381648.19)
 assert o.azimuth_deg == pytest.approx(344.,abs=1.)
 assert o.elevation_deg == pytest.approx(67.,abs=1.)
 # Observed +1188 Hz; tolerance covers the documented receive-time approximation.
 assert o.predicted_l1_doppler_hz == pytest.approx(1188.,abs=10.)
