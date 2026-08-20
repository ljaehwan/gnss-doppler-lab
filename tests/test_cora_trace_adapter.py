from pathlib import Path

import h5py
import numpy as np
import pytest

from gnss_doppler_lab.cora_trace_adapter import LegacyTraceIndex, NativeTraceIndex, validate_epoch_sequence
from gnss_doppler_lab.mosaic_raw_recorrelation import (
    correlate_nine_taps,
    fit_complex_amplitude,
    normalized_complex_cosine,
    read_ishort_complex_window,
)
from gnss_doppler_lab.trace_native_1ms import TAPS


DS3_LEGACY = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9/raw")
DS3_RAW = Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds3.bin")
TEX_CLEAN_NATIVE = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/texbat_cleanstatic/rep1")


def test_native_index_selects_receiver_interval_and_bounds():
    if not TEX_CLEAN_NATIVE.exists():
        pytest.skip("native clean trace absent")
    index = NativeTraceIndex(TEX_CLEAN_NATIVE)
    rows = [index.select(prn, 200.0) for prn in (3, 13, 19, 23)]
    audit = validate_epoch_sequence(rows, DS3_RAW.stat().st_size // 4)
    assert audit["status"] == "PASS"
    assert all(abs(row.receiver_timestamp_s - 200.0) < 0.00075 for row in rows)
    assert all(row.raw_end_sample - row.raw_start_sample in (24_999, 25_000, 25_001) for row in rows)


def test_legacy_ds3_endpoint_and_action_reconstruct_logged_taps():
    if not DS3_LEGACY.exists() or not DS3_RAW.exists():
        pytest.skip("legacy DS3 validation source absent")
    index = LegacyTraceIndex(DS3_LEGACY, 25_000_000.0)
    selected = index.select(19, 200.0)
    iq = read_ishort_complex_window(
        DS3_RAW, selected.raw_start_sample, selected.raw_end_sample - selected.raw_start_sample
    )
    reconstructed = correlate_nine_taps(
        iq, prn=selected.prn, action=selected.action, tap_offsets_chips=np.arange(-4, 5) * 0.125
    )
    with h5py.File(selected.source_path, "r") as handle:
        native = np.asarray([
            np.asarray(handle[f"I_{tap}"]).reshape(-1)[selected.source_row]
            + 1j * np.asarray(handle[f"Q_{tap}"]).reshape(-1)[selected.source_row]
            for tap in TAPS
        ])
    _, fitted = fit_complex_amplitude(reconstructed, native)
    assert normalized_complex_cosine(fitted, native) > 0.9999


def test_legacy_mapping_is_exclusive_endpoint_and_monotone():
    if not DS3_LEGACY.exists():
        pytest.skip("legacy DS3 validation source absent")
    index = LegacyTraceIndex(DS3_LEGACY, 25_000_000.0)
    rows = [index.select(19, value) for value in (120.0, 160.0, 200.0)]
    audit = validate_epoch_sequence(rows, DS3_RAW.stat().st_size // 4)
    assert audit == {"status": "PASS", "failures": [], "epoch_count": 3, "prns": [19]}
    assert all(row.raw_end_sample == round(row.receiver_timestamp_s * 25_000_000) for row in rows)
