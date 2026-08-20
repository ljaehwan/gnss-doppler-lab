import csv
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

from gnss_doppler_lab.cora_cross_cumulant import cross_cumulant_matrix

ROOT=Path(__file__).parents[1]
ART=ROOT/"artifacts/cora_stage0_cross_prn_common_origin"


def test_complex_gain_phase_and_nav_sign_invariance():
    rng=np.random.default_rng(4801)
    tokens=(rng.normal(size=(64,5,4))+1j*rng.normal(size=(64,5,4)))
    tokens+=rng.laplace(size=(64,1,1))
    base=cross_cumulant_matrix(tokens)
    transforms=np.asarray([2*np.exp(.3j),-.7*np.exp(-1.1j),3j,-2.2,.4*np.exp(2j)])
    changed=tokens*transforms[None,:,None]
    assert np.allclose(cross_cumulant_matrix(changed),base,rtol=1e-12,atol=1e-12)


def test_configuration_freeze_and_attack_access_order():
    freeze=json.loads((ART/"configuration_freeze.json").read_text())
    assert freeze["statement"]=="Configuration frozen before this CORA evaluation."
    assert freeze["remote_freeze_sha"]=="c226b942a82dbd63c6682e76e44b2aefe1c60156"
    assert freeze["remote_freeze_verified"] and freeze["frozen_before_attack_payload_read"]
    inventory=json.loads((ART/"data_inventory.json").read_text())
    assert inventory["configuration_freeze_sha"]==freeze["remote_freeze_sha"]
    assert inventory["attack_payload_access_started_after_remote_freeze_verification"]


def test_chronological_split_guards_normal_only_calibration():
    audit=json.loads((ART/"clean_split_audit.json").read_text())
    assert audit["status"]=="PASS"
    assert audit["raw_sample_or_target_window_overlap"] is False
    assert audit["guard_intervals_s"]==[[211.0,221.0],[321.0,331.0]]
    assert audit["attack_preonset_used_for_fit_or_calibration"] is False
    assert audit["calibration_score_count_per_domain"]==50


def test_official_onsets_common_cadence_and_ten_second_blocks():
    config=json.loads((ART/"config.json").read_text())
    expected={"oakbat_os3":120.0,"oakbat_os4":120.0,"texbat_ds1":125.0,"texbat_ds3":118.9,"texbat_ds7":110.0,"texbat_ds8":110.0}
    assert {k:config["datasets"][k]["onset_s"] for k in expected}==expected
    with gzip.open(ART/"per_block_scores.csv.gz","rt",newline="") as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        assert float(row["window_end_s"])-float(row["window_start_s"])==2.0
        assert int(float(row["bootstrap_block"]))==int(float(row["window_start_s"])//10)
        assert int(float(row["prn_count"]))>=4


def test_raw_lineage_bounds_hashes_and_ds7_ds8_overlap_audit():
    binding=json.loads((ART/"raw_source_binding.json").read_text())
    assert len(binding["datasets"])==8
    for item in binding["datasets"].values():
        lo,hi=item["selected_raw_sample_interval"]
        assert 0<=lo<hi<=item["raw_sample_count"]
        assert len(item["full_sha256"])==64 and len(item["cache_sha256"])==64
    overlap=binding["ds7_ds8_pre110_overlap_audit"]
    assert overlap["identical"] and not overlap["counted_as_independent_normal_evidence"]


def test_independent_artifact_verifier_recomputes_metrics_and_verdict():
    path=ROOT/"scripts/verify_cora_stage0.py"
    spec=importlib.util.spec_from_file_location("cora_verifier",path);module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;spec.loader.exec_module(module)
    result=module.verify(ART)
    assert result["status"]=="PASS"
    assert result["checks"]["matrix_score_recomputation"]["mismatches"]==0
    assert result["checks"]["relation_recomputation"]["mismatches"]==0
    assert result["checks"]["verdict"]["recomputed"]=="NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS"
