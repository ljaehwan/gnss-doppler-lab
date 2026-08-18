import numpy as np

from gnss_doppler_lab.mirage_r1 import nav_signs_for_epochs
from gnss_doppler_lab.mirage_r1_executor import raised_cosine_envelope, render_receiver_config


def test_raised_cosine_transient_contract():
    t = np.array([-.1, 0, .125, .25, 1.0, 1.75, 1.875, 2.0, 2.1])
    observed = raised_cosine_envelope(t)
    np.testing.assert_allclose(observed, [0, 0, .5, 1, 1, 1, .5, 0, 0], atol=1e-14)
    dense = raised_cosine_envelope(np.arange(2000) / 1000)
    assert np.count_nonzero(dense == 1) >= 1500


def test_authenticated_nav_lookup_has_no_plus_one_fallback():
    rows = [
        {"dataset": "D", "prn": "3", "raw_start_sample": "100", "raw_end_sample_exclusive": "120", "bit_value_pm1": "-1"},
        {"dataset": "D", "prn": "3", "raw_start_sample": "120", "raw_end_sample_exclusive": "140", "bit_value_pm1": "1"},
    ]
    np.testing.assert_array_equal(nav_signs_for_epochs(rows, "D", 3, [100, 119, 120, 139]), [-1, -1, 1, 1])
    try:
        nav_signs_for_epochs(rows, "D", 3, [99])
    except ValueError as error:
        assert "gap" in str(error)
    else:
        raise AssertionError("NAV gap must fail closed")


def test_nav_boundary_is_next_epoch_start():
    rows = [
        {"dataset": "D", "prn": "7", "raw_start_sample": "10", "raw_end_sample_exclusive": "30", "bit_value_pm1": "-1"},
        {"dataset": "D", "prn": "7", "raw_start_sample": "30", "raw_end_sample_exclusive": "50", "bit_value_pm1": "1"},
    ]
    assert nav_signs_for_epochs(rows, "D", 7, [29])[0] == -1
    assert nav_signs_for_epochs(rows, "D", 7, [30])[0] == 1


def test_receiver_replay_binding_changes_only_filename(tmp_path):
    base = tmp_path / "base.conf"
    base.write_text("SignalSource.filename=/old.raw\nSignalSource.sampling_frequency=5000000\nTracking.foo=bar\n")
    rendered = render_receiver_config(base, tmp_path / "new.raw")
    assert f"SignalSource.filename={tmp_path / 'new.raw'}" in rendered
    assert "SignalSource.sampling_frequency=5000000" in rendered
    assert "Tracking.foo=bar" in rendered


def test_receiver_replay_binding_fails_closed_without_filename(tmp_path):
    base = tmp_path / "base.conf"; base.write_text("SignalSource.sampling_frequency=5000000\n")
    try:
        render_receiver_config(base, tmp_path / "new.raw")
    except ValueError as error:
        assert "allowlist" in str(error)
    else:
        raise AssertionError("missing filename binding must fail")
