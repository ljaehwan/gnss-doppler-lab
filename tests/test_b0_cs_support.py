import pandas as pd
import pytest

from gnss_doppler_lab.b0_cs_support import common_epoch_prn_support


def test_common_epoch_prn_support_is_exact_sorted_and_audited():
    first = pd.DataFrame({
        "physical_recording_id": ["r", "r", "r"],
        "window_bin_s": [1., 0., 0.], "prn": ["G01", "G02", "G01"],
        "score": [3., 2., 1.],
    })
    second = pd.DataFrame({
        "physical_recording_id": ["r", "r", "r"],
        "window_bin_s": [0., 1., 2.], "prn": ["G01", "G01", "G01"],
        "score": [4., 5., 6.],
    })
    aligned, audit = common_epoch_prn_support(first, second)
    assert audit["common_rows"] == 2 and audit["exact_common_epoch_prn_support"]
    assert list(zip(aligned[0].window_bin_s, aligned[0].prn)) == [(0., "G01"), (1., "G01")]
    assert list(zip(aligned[1].window_bin_s, aligned[1].prn)) == [(0., "G01"), (1., "G01")]


def test_common_support_fails_closed_on_duplicate_or_empty_intersection():
    duplicate = pd.DataFrame({
        "physical_recording_id": ["r", "r"], "window_bin_s": [0., 0.], "prn": ["G01", "G01"]
    })
    other = pd.DataFrame({"physical_recording_id": ["x"], "window_bin_s": [1.], "prn": ["G02"]})
    with pytest.raises(ValueError, match="duplicate"):
        common_epoch_prn_support(duplicate, other)
    with pytest.raises(ValueError, match="no common"):
        common_epoch_prn_support(duplicate.iloc[:1], other)
