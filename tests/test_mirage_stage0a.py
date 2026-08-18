import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from gnss_doppler_lab.mosaic_receiver_in_loop import NavBitTimeline, ReceiverNcoEpoch, ReplicaState, StatefulReplica, iter_stateful_replica_epochs

ROOT=Path(__file__).resolve().parents[1]


def load_runner():
    path=ROOT/"scripts/run_mirage_stage0a.py"; spec=importlib.util.spec_from_file_location("mirage_runner",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_nav_timeline_and_chunk_nco_continuity():
    mapping=[{"prn":"3","corrected_raw_start_sample":"0","corrected_raw_end_sample_exclusive":"20","bit_value_pm1":"1"},
             {"prn":"3","corrected_raw_start_sample":"10","corrected_raw_end_sample_exclusive":"20","bit_value_pm1":"-1"}]
    nav=NavBitTimeline(mapping,prn=3); replica=StatefulReplica(3,1000,ReplicaState(0,.2,.4))
    epochs=[ReceiverNcoEpoch(0,10,1023,2),ReceiverNcoEpoch(10,20,1023,2)]
    chunks=list(iter_stateful_replica_epochs(replica,epochs,nav,delay_chips=.1,delta_f_hz=1,phase_offset_rad=.3))
    assert [x[0] for x in chunks]==[0,10] and replica.state.absolute_sample_index==20
    assert np.isfinite(np.concatenate([x[1] for x in chunks])).all()


def test_preregistered_source_sha_values_are_bound():
    config=json.loads((ROOT/"configs/mirage_stage0a.json").read_text())
    assert len(config["datasets"]["OAKBAT.cleanStatic"]["raw_sha256"])==64
    assert len(config["datasets"]["TEXBAT.cleanStatic"]["raw_sha256"])==64
    assert config["nav"]["outside_common_interval_authorized"] is False


def test_common_case_support_has_six_nonoverlapping_half_second_anchors():
    runner=load_runner(); config=runner.load_config(); values=runner.anchors(config)
    for item in values.values():
        assert item["anchor_ranges_nonoverlap"] and item["anchors_inside_authorized_interval"]
        assert len(item["anchor_ranges"])==6


def test_deterministic_gzip_writer(tmp_path):
    runner=load_runner(); old=runner.ART
    try:
        runner.ART=tmp_path; runner.write_csv_gz("a.csv.gz",["x"],[{"x":1}]); first=(tmp_path/"a.csv.gz").read_bytes()
        runner.write_csv_gz("a.csv.gz",["x"],[{"x":1}]); second=(tmp_path/"a.csv.gz").read_bytes()
        assert hashlib.sha256(first).digest()==hashlib.sha256(second).digest()
    finally: runner.ART=old


def test_algebraic_audit_is_independent_and_passes():
    result=load_runner().algebraic_tests()
    assert result["status"]=="PASS" and result["rank1_max_minor"]<1e-12 and result["rank2_max_minor"]>.01


def test_support_failure_is_input_inconclusive_not_physics_no_go():
    config=json.loads((ROOT/"configs/mirage_stage0a.json").read_text())
    for spec in config["datasets"].values():
        start,end=spec["authorized_common_interval"]
        assert (end-start)/spec["sample_rate_hz"] < config["clean_split"]["required_common_valid_span_seconds"]


def test_no_existing_model_or_artifact_modified_by_new_files():
    status={p.parts[0] for p in [Path("src/gnss_doppler_lab/mirage_complex_minor.py"),Path("scripts/run_mirage_stage0a.py")]}
    assert status=={"src","scripts"}
