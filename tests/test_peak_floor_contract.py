from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from gnss_doppler_lab.peak_floor_contract import (MAX_PRNS, MORPH_FEATURES, FLOOR_FEATURES,
    AlignedData, align_modalities, partition_aligned, fit_robust_scalers, apply_scalers,
    make_causal_pairs, validate_normal_only_inputs)

def frames(n=16, prns=("G01","G02","G03")):
    mr=[]; fr=[]
    for i in range(n):
        t=.5*(i+1)
        for k,p in enumerate(prns):
            r={"label":"clean","run_id":"r","source_fingerprint":"s","tap_count":9,"prn":p,
               "window_bin_s":t,"window_start_s":t-.48,"window_end_s":t+.02}
            r.update({c:np.sin(t+j*.01)+k*.1 for j,c in enumerate(MORPH_FEATURES)}); mr.append(r)
        q={"scenario":"clean","window_start_s":t,"window_end_s":t+.04}
        q.update({c:np.cos(t+j*.01) for j,c in enumerate(FLOOR_FEATURES)}); fr.append(q)
    return pd.DataFrame(mr),pd.DataFrame(fr)

def test_contract_dimensions_alignment_mask_and_scaler_padding():
    assert (len(MORPH_FEATURES),len(FLOOR_FEATURES),MAX_PRNS)==(35,57,32)
    m,f=frames(8); a=align_modalities(m,f)
    assert a.morph.shape==(8,32,35) and a.prn_mask.sum()==24
    s=fit_robust_scalers(a); z=apply_scalers(a,s)
    assert np.all(z.morph[~z.prn_mask]==0) and np.isfinite(z.floor).all()

def test_split_before_pairs_no_boundary_or_gap_crossing_and_full_support_available_time():
    m,f=frames(20); a=align_modalities(m,f)
    # create a cadence gap inside validation
    keep=np.arange(len(a.times))!=11
    a=AlignedData(a.times[keep],a.available_times[keep],a.support_start_times[keep],a.morph[keep],a.floor[keep],a.prn_mask[keep],a.morph_features,a.floor_features)
    parts=partition_aligned(a,{"train":(None,5.),"validation":(5.5,8.),"calibration":(8.5,9.),"held_clean":(9.5,None)})
    p=make_causal_pairs(parts["validation"],context_len=3)
    assert np.all(p.context_times[:,0]>=5.5) and np.all(p.target_times<=8.)
    assert np.allclose(np.diff(p.context_times,axis=1),.5)
    expected=np.array([max(parts["validation"].available_times[np.where(parts["validation"].times==x)[0][0]] for x in [*ct,t]) for ct,t in zip(p.context_times,p.target_times)])
    assert np.allclose(p.available_times,expected)

def test_clean_only_rejects_attack():
    m,f=frames(); m.loc[0,"label"]="attack"
    with pytest.raises(ValueError,match="clean"): align_modalities(m,f,validate_clean=True)

@pytest.mark.parametrize("bad", ["cleanDynamic", "unclean", "not_clean", "jammed", "cleanStatic_attack", "cleanStaticFoo"])
def test_clean_static_contract_rejects_non_static_or_substring_labels(bad):
    m,f=frames(); m["label"]=bad
    with pytest.raises(ValueError,match="cleanStatic|clean static"):
        validate_normal_only_inputs(m,f)
    m,f=frames(); f["scenario"]=bad
    with pytest.raises(ValueError,match="cleanStatic|clean static"):
        validate_normal_only_inputs(m,f)

def test_clean_static_contract_requires_source_identity():
    m,f=frames()
    for column in ("run_id", "source_fingerprint", "tap_count"):
        bad=m.drop(columns=[column])
        with pytest.raises(ValueError,match="identity|tap_count"):
            validate_normal_only_inputs(bad,f)

def test_clean_static_contract_rejects_missing_values_in_required_rows():
    for column in ("label", "run_id", "source_fingerprint", "tap_count"):
        m,f=frames();m.loc[m.index[0],column]=np.nan
        with pytest.raises(ValueError,match="missing|null|NaN"):
            validate_normal_only_inputs(m,f)
    m,f=frames();f.loc[f.index[0],"scenario"]=np.nan
    with pytest.raises(ValueError,match="missing|null|NaN"):
        validate_normal_only_inputs(m,f)

def test_causal_pair_exposes_true_full_support_start():
    m,f=frames(8)
    a=align_modalities(m,f)
    p=make_causal_pairs(a,context_len=3)
    assert np.allclose(p.support_start_times, p.context_times[:,0]-.48)
