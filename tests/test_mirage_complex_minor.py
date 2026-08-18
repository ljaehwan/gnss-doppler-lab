import numpy as np
import pytest

from gnss_doppler_lab.mirage_complex_minor import (
    clean_calibration_threshold, deterministic_design, design_sha256,
    doppler_grid_hz, full_score, magnitude_minors, nav_wipeoff,
    normalized_complex_minors, raw_ranges_nonoverlap, split_support_audit,
    temporal_desynchronize, validate_clean_source_path,
)


def test_canonical_literal_rank_one_known_vector():
    # Literal independent matrix: every adjacent determinant is hand-zero.
    matrix=np.array([[1,2j,-1,3,-2j],[2,4j,-2,6,-4j],[-1,-2j,1,-3,2j],
                     [3,6j,-3,9,-6j],[.5,1j,-.5,1.5,-1j],[2j,-4, -2j,6j,4],
                     [1+1j,-2+2j,-1-1j,3+3j,2-2j],[-2,-4j,2,-6,4j],[4,8j,-4,12,-8j]],complex)
    assert np.max(normalized_complex_minors(matrix)) < 1e-14


def test_literal_rank_two_known_vector_has_nonzero_minor():
    matrix=np.ones((9,5),complex);matrix[1,1]=2+3j
    assert normalized_complex_minors(matrix)[0,0] > .1


def test_common_complex_gain_and_global_phase_invariance():
    rng=np.random.default_rng(2);matrix=rng.normal(size=(9,5))+1j*rng.normal(size=(9,5))
    base=normalized_complex_minors(matrix)
    assert np.allclose(base,normalized_complex_minors((3-4j)*matrix),atol=1e-14)
    assert np.allclose(base,normalized_complex_minors(np.exp(1.7j)*matrix),atol=1e-14)


def test_low_energy_has_no_nan_or_inf():
    assert np.all(normalized_complex_minors(np.zeros((9,5),complex))==0)


def test_equal_magnitude_different_complex_phase_is_distinguished():
    a=np.ones((9,5),complex)
    b=np.fromfunction(lambda i,j:np.exp(.3j*i*j),(9,5),dtype=float)
    assert np.max(np.abs(normalized_complex_minors(a)-normalized_complex_minors(b)))>.1
    assert np.max(magnitude_minors(b)) < 1e-14


def test_frozen_normalized_doppler_grids():
    assert doppler_grid_hz(.02).tolist()==[-100,-50,0,50,100]
    assert doppler_grid_hz(.1).tolist()==[-20,-10,0,10,20]
    assert doppler_grid_hz(.5).tolist()==[-4,-2,0,2,4]


def test_nav_bit_wipeoff_requires_authenticated_pm_one():
    signal=np.array([1+2j,-3+4j]); signs=np.array([1,-1])
    assert np.array_equal(nav_wipeoff(signal,signs),np.array([1+2j,3-4j]))
    with pytest.raises(ValueError): nav_wipeoff(signal,np.array([1,0]))


def test_deterministic_design_and_hash():
    datasets={"A":{"prns":[1,2,3,4,5],"anchor_start_samples":[10,20,30,40,50,60]},
              "B":{"prns":[6,7,8,9,10],"anchor_start_samples":[70,80,90,100,110,120]}}
    a=deterministic_design(7,datasets);b=deterministic_design(7,datasets)
    assert a==b and design_sha256(a)==design_sha256(b) and len(a)==72
    assert sum(x["mode"]=="single_prn" for x in a)==60


def test_chronological_support_and_raw_nonoverlap():
    assert split_support_audit(29)["status"]=="PASS"
    assert split_support_audit(11.99)["status"]=="FAIL"
    assert raw_ranges_nonoverlap([(0,5),(5,10),(20,30)])
    assert not raw_ranges_nonoverlap([(0,6),(5,10)])


def test_clean_only_threshold_has_frozen_quantiles():
    assert clean_calibration_threshold(range(100),.99)==99
    with pytest.raises(ValueError): clean_calibration_threshold(range(100),.9)


def test_attack_paths_rejected_before_access():
    validate_clean_source_path("/safe/oakbat/cleanStatic_gps.bin")
    for path in ("/data/DS1/raw.bin","/data/OS4/raw.bin","/data/other.bin"):
        with pytest.raises(ValueError): validate_clean_source_path(path)


def test_prn_permutation_and_variable_count():
    assert full_score([1,8,3,5,2])==full_score([5,2,8,1,3])
    assert full_score([1,2,3]) is None and full_score([1,2,3,4])==2.5


def test_temporal_destruction_preserves_each_prn_distribution():
    x=np.arange(40,dtype=float).reshape(10,4)
    y=temporal_desynchronize(x,[0,1,3,5])
    for p in range(4): assert np.array_equal(np.sort(x[:,p]),np.sort(y[:,p]))
    assert np.array_equal(np.sum(np.isfinite(x),axis=1),np.sum(np.isfinite(y),axis=1))
