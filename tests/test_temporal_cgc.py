import pytest

from gnss_doppler_lab.temporal_cgc import causal_prn_median


def _rows(values, *, prn="G01", pair="p1", condition="locked"):
    return [
        {
            "pair_id": pair,
            "condition": condition,
            "prn": prn,
            "bin_index": index,
            "estimated_delay_chips": value,
        }
        for index, value in enumerate(values)
    ]


def test_causal_median_uses_no_future_samples():
    rows = _rows([0.0, 100.0, 0.0, 0.0, 0.0])
    filtered = causal_prn_median(rows, window_bins=3)

    assert [row["stabilized_delay_chips"] for row in filtered] == [0.0, 50.0, 0.0, 0.0, 0.0]
    assert [row["temporal_support_bins"] for row in filtered] == [1, 2, 3, 3, 3]


def test_causal_median_is_independent_across_prns_and_streams():
    rows = [
        *_rows([0.0, 2.0], prn="G01", pair="p1"),
        *_rows([10.0, 20.0], prn="G02", pair="p1"),
        *_rows([100.0, 200.0], prn="G01", pair="p2"),
    ]
    filtered = causal_prn_median(reversed(rows), window_bins=2)
    lookup = {
        (row["pair_id"], row["prn"], row["bin_index"]): row["stabilized_delay_chips"]
        for row in filtered
    }

    assert lookup[("p1", "G01", 1)] == 1.0
    assert lookup[("p1", "G02", 1)] == 15.0
    assert lookup[("p2", "G01", 1)] == 150.0


def test_causal_median_uses_wall_clock_window_across_gaps():
    rows = _rows([0.0, 2.0])
    rows[1]["bin_index"] = 5
    filtered = causal_prn_median(rows, window_bins=3)

    assert filtered[1]["stabilized_delay_chips"] == 2.0
    assert filtered[1]["temporal_support_bins"] == 1


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_causal_median_rejects_invalid_window(window):
    with pytest.raises(ValueError):
        causal_prn_median(_rows([0.0]), window_bins=window)


def test_causal_median_rejects_duplicate_stream_prn_bin():
    row = _rows([0.0])[0]
    with pytest.raises(ValueError, match="at most one"):
        causal_prn_median([row, dict(row)], window_bins=3)
