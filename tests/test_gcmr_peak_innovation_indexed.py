import numpy as np
import pytest
from types import SimpleNamespace
from gnss_doppler_lab.gcmr_peak_innovation_adapter import index_peak_windows, aggregate_peak_windows


def series():
    return SimpleNamespace(tap_names=("E", "P", "L"), time_s=np.array([0., 1., 2., 3., 4.]),
        magnitudes=np.array([[1.,2.,3.],[2.,3.,4.],[3.,4.,5.],[4.,5.,6.],[5.,6.,7.]]),
        cn0_db_hz=np.array([10.,11.,12.,13.,14.]))


def test_indexed_peak_windows_match_legacy_and_are_half_open():
    windows=[(0.,2.),(1.,3.),(2.,4.)]
    indexed=index_peak_windows(series(),windows)
    legacy={(r.window_start_s,r.window_end_s):r for r in aggregate_peak_windows(series(),windows)}
    assert set(indexed.records)==set(legacy)
    for key, record in indexed.records.items():
        np.testing.assert_allclose(record.epl,legacy[key].epl)
        assert record.cn0 == legacy[key].cn0
    # Epoch t=2 belongs to [2,4), not [0,2).
    np.testing.assert_allclose(indexed.target(0.,2.).epl,[1.5,2.5,3.5])
    np.testing.assert_allclose(indexed.target(2.,4.).epl,[3.5,4.5,5.5])


def test_index_history_is_causal_at_exact_boundary():
    index=index_peak_windows(series(),[(0.,1.),(1.,2.),(2.,3.),(3.,4.)])
    prior=index.history_before(2.,2)
    assert [(x.window_start_s,x.window_end_s) for x in prior]==[(0.,1.),(1.,2.)]
    assert index.history_before(1.,2) is None
