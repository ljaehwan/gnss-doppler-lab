import importlib.util
from pathlib import Path
import numpy as np

from gnss_doppler_lab.chord_identifiability import (
    TAP_OFFSETS_CHIPS, assign_split, block_bootstrap, ca_single_source_template,
    complex_shrinkage_whitener, fingerprint, fit_tangent_residual,
    projective_similarity, raw_projective_fingerprint, select_matched_negative,
    split_nonoverlap_audit,
)

ROOT=Path(__file__).resolve().parents[1]

def test_canonical_nine_tap_template_hand_computed():
    assert np.array_equal(ca_single_source_template(0), np.array([.5,.625,.75,.875,1,.875,.75,.625,.5]))

def test_known_vector_projective_similarity_hand_computed():
    a=np.array([1+0j,0]); b=np.array([1+0j,1+0j])/np.sqrt(2)
    assert projective_similarity(a,b)==np.float64(.5)

def test_complex_gain_phase_and_nav_sign_invariance():
    x=np.arange(1,10)+1j*np.arange(9)
    for gain in (3.2, np.exp(1j*.71), -1):
        assert abs(projective_similarity(x,gain*x)-1)<1e-14

def test_delay_fit_and_tangent_projection_orthogonality():
    grid=np.arange(-.125,.1251,.0025); template=ca_single_source_template(.05)
    perturb=np.array([1,-2,1,-1,2,-1,1,-2,1],dtype=float)*1e-3j
    fit=fit_tangent_residual((2+3j)*template+perturb,np.eye(9),grid)
    assert abs(fit.tau_chips-.05)<1e-9
    assert np.isfinite(fit.tangent_residual).all()
    assert abs(np.vdot(np.r_[ca_single_source_template(fit.tau_chips),np.zeros(9)],np.r_[fit.tangent_residual.real,fit.tangent_residual.imag]))<1e-8

def test_covariance_whitener_is_hermitian_and_phase_equivariant():
    rng=np.random.default_rng(4); r=rng.normal(size=(30,9))+1j*rng.normal(size=(30,9))
    c,w=complex_shrinkage_whitener(r); c2,w2=complex_shrinkage_whitener(r*np.exp(1j*.6))
    assert np.allclose(c,c.conj().T); assert np.allclose(c,c2); assert np.allclose(w,w2)

def test_low_energy_direction_unavailable():
    assert fingerprint(np.zeros(9,dtype=complex),1e-4) is None
    assert fingerprint(np.ones(9,dtype=complex),1e-4) is not None

def test_prn_permutation_and_variable_count():
    profiles={p:raw_projective_fingerprint(np.arange(1,10)*(1+1j*p)) for p in (3,7,9)}
    before=sorted(projective_similarity(profiles[a],profiles[b]) for a in profiles for b in profiles if a<b)
    perm={9:profiles[3],3:profiles[7],7:profiles[9]}
    after=sorted(projective_similarity(perm[a],perm[b]) for a in perm for b in perm if a<b)
    assert np.allclose(before,after); assert len({k:v for k,v in profiles.items() if k!=7})==2

def test_chronological_split_and_raw_nonoverlap():
    assert [assign_split(t) for t in (30,219.9,220,229.9,230,305.9,306,315.9,316,429.9)]==["fit","fit","guard_1","guard_1","calibration","calibration","guard_2","guard_2","holdout","holdout"]
    rows=[{"split":"fit","timestamp_s":31,"raw_sample_start":1,"raw_sample_end":2},
          {"split":"calibration","timestamp_s":231,"raw_sample_start":3,"raw_sample_end":4},
          {"split":"holdout","timestamp_s":317,"raw_sample_start":5,"raw_sample_end":6}]
    audit=split_nonoverlap_audit(rows); assert not audit["raw_sample_overlap"]; assert not audit["ten_second_block_overlap"]

def test_pair_matching_is_deterministic_and_different_prn():
    a={"prn":1,"cn0_db_hz":40,"residual_norm":2}; p={"prn":1,"cn0_db_hz":39,"residual_norm":2.1}
    candidates=[{"prn":2,"cn0_db_hz":39,"residual_norm":2.1},{"prn":3,"cn0_db_hz":30,"residual_norm":4}]
    assert select_matched_negative(a,candidates,p,1,1,{})["prn"]==2

def test_block_bootstrap_resamples_blocks_not_rows():
    y=np.array([0,1,0,1]); s=np.array([0,1,0,1]); blocks=np.array([1,1,2,2])
    out=block_bootstrap(y,s,blocks,resamples=20,seed=3); assert len(out)==20; assert np.all(out==1)

def test_attack_paths_are_rejected_without_access():
    spec=importlib.util.spec_from_file_location("runner",ROOT/"scripts/run_chord_stage0a.py"); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for path in (Path("/x/DS1/file"),Path("/x/OS4/file"),Path("/x/not-clean/file")):
        try: module.safe_clean_path(path)
        except RuntimeError: pass
        else: raise AssertionError("forbidden path accepted")

def test_deterministic_similarity():
    rng=np.random.default_rng(9); a=rng.normal(size=9)+1j*rng.normal(size=9); b=rng.normal(size=9)+1j*rng.normal(size=9)
    assert projective_similarity(a,b)==projective_similarity(a,b)

def test_checksum_verifier_detects_tamper(tmp_path):
    spec=importlib.util.spec_from_file_location("verifier",ROOT/"scripts/verify_chord_stage0a.py"); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path/"x").write_text("a"); recorded=module.actual_manifest(tmp_path); (tmp_path/"x").write_text("b")
    assert recorded!=module.actual_manifest(tmp_path)
