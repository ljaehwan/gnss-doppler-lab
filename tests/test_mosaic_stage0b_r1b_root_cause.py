import ast, hashlib, json
from pathlib import Path
import numpy as np
import pytest
from gnss_doppler_lab.mosaic_stage0b_r1b_root_cause import (
    decide_recommendation, decide_root_cause, diagnostic_projection,
    receiver_frame_coordinates, segment_indices, select_single_comparators,
    triangular_template,
)

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/mosaic_stage0b_r1b_multiprn_root_cause"

def test_receiver_frame_coordinate_unit_and_sign():
    delay,doppler=receiver_frame_coordinates(.1,50,[.02,.03],[.07,.01],[100,110],[130,90])
    assert np.allclose(delay,[.15,.08]); assert np.allclose(doppler,[20,70])

def test_fixed_and_oracle_template_calculation():
    taps=np.arange(-.5,.5001,.125); q=triangular_template(taps,[.1,.2],[0,np.pi/2])
    assert q.shape==(2,9); assert np.iscomplexobj(q); assert np.all(np.abs(q)<=1)

def test_projection_ratio_bounds_and_perfect_template():
    clean=np.tile(np.arange(1,10),(20,1)).astype(complex); q=triangular_template(np.arange(-.5,.5001,.125),np.full(20,.1),np.zeros(20)); observed=2*clean+3j*q
    result=diagnostic_projection(clean,observed,q); assert 0<=result["projection_ratio"]<=1; assert result["projection_ratio"]==pytest.approx(1)

def test_comparator_selection_priority():
    four={"case":{"dataset":"D","case_id":"D.four.03","rho_db":0,"delta_tau_chips":.1,"delta_f_hz":50,"delta_phi_rad":1}}
    exact={"case":{"dataset":"D","case_id":"D.single.11","rho_db":0,"delta_tau_chips":.1,"delta_f_hz":50,"delta_phi_rad":1}}
    near={"case":{"dataset":"D","case_id":"D.single.03","rho_db":-3,"delta_tau_chips":.1,"delta_f_hz":50,"delta_phi_rad":1}}
    tier,rows=select_single_comparators(four,[near,exact]); assert tier=="exact_parameter_match" and rows==[exact]

def test_window_segmentation_half_open_and_deterministic():
    t=[4,4.099999,4.1,4.2]; first=segment_indices(t,4,.1); second=segment_indices(t,4,.1)
    assert [x.tolist() for x in first]==[[0,1],[2],[3]]; assert all(np.array_equal(a,b) for a,b in zip(first,second))

@pytest.mark.parametrize("available,supported,oracle,expected",[(False,[],False,"ROOT_CAUSE_EVIDENCE_UNAVAILABLE"),(True,["H1"],True,"SCORER_RECEIVER_FRAME_MISMATCH_SUPPORTED"),(True,["H2"],False,"RELATIVE_PHASE_CANCELLATION_SUPPORTED"),(True,["H1","H3"],False,"MIXED_OR_UNIDENTIFIED_ROOT_CAUSE")])
def test_final_verdict_truth_table(available,supported,oracle,expected): assert decide_root_cause(available,supported,oracle)==expected

def test_recommendation_requires_every_condition():
    assert decide_recommendation(oracle_restores=True,consistent_improvement=True,comparators_not_degraded=True,controls_separated=True)=="Frozen corrected observer confirmation"
    assert decide_recommendation(oracle_restores=True,consistent_improvement=True,comparators_not_degraded=False,controls_separated=True)=="Terminate MOSAIC"

def test_no_iq_injection_receiver_replay_or_source_raw_modification():
    paths=[ROOT/"src/gnss_doppler_lab/mosaic_stage0b_r1b_root_cause.py",ROOT/"scripts/run_mosaic_stage0b_r1b_root_cause_audit.py",ROOT/"scripts/verify_mosaic_stage0b_r1b_root_cause.py"]
    forbidden={"subprocess","generate_injected_prefix","run_receiver","inject_payload","decode_interleaved_int16"}
    for p in paths:
        tree=ast.parse(p.read_text()); names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}; attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        assert forbidden.isdisjoint(names|attrs)
    assert all("raw_path).open" not in p.read_text() for p in paths)

def test_prior_r1a_verdict_unchanged():
    prior=json.loads((ROOT/"artifacts/mosaic_stage0b_r1a_frozen_analysis/final_verdict.json").read_text()); assert prior["verdict"]=="NO_GO_MOSAIC_MULTI_PRN_RECOVERY"

def test_artifact_checksum_and_fresh_clone_verifier():
    if not (ART/"artifact_manifest_sha256.json").exists(): pytest.skip("results generated only after ROOT_CAUSE_ANALYSIS_FREEZE")
    import importlib.util
    p=ROOT/"scripts/verify_mosaic_stage0b_r1b_root_cause.py"; spec=importlib.util.spec_from_file_location("r1bverify",p); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); assert module.verify()>=18
