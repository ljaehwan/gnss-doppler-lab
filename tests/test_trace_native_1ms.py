from __future__ import annotations

from pathlib import Path

import numpy as np

from gnss_doppler_lab.trace_equivariance import action_shuffle_indices, robust_epoch_blocks
from gnss_doppler_lab.trace_native_1ms import (
    ACTION_VALUE_FIELDS,
    ENDIAN_MARKER,
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC,
    RECORD_DTYPE,
    RECORD_SIZE,
    SCHEMA_VERSION,
    TAPS,
    load_native_trace_pairs,
    read_records,
    validate_dump_files,
)


def _write_dump(path: Path, *, channel: int, prn: int, rows: int = 6) -> Path:
    offsets = tuple(np.arange(-0.5, 0.5001, 0.125, dtype=np.float32))
    header = HEADER_STRUCT.pack(
        MAGIC,
        SCHEMA_VERSION,
        HEADER_SIZE,
        RECORD_SIZE,
        ENDIAN_MARKER,
        1_000_000.0,
        0.125,
        0.001,
        b"SYNTHETIC",
        b"1ddd4562723040fd66cb334b578a5b69455625f4",
        *offsets,
        0,
    )
    records = np.zeros(rows, dtype=RECORD_DTYPE)
    for row in range(rows):
        records["loop_sequence"][row] = row
        records["tracking_session_id"][row] = 1
        records["action_used_source_loop_sequence"][row] = np.iinfo(np.uint64).max if row == 0 else row - 1
        records["raw_interval_start_sample"][row] = 10_000 + row * 1_000
        records["raw_interval_end_sample"][row] = 11_000 + row * 1_000
        records["receiver_timestamp_s"][row] = 0.010 + row * 0.001
        records["integration_duration_s"][row] = 0.001
        records["channel"][row] = channel
        records["prn"][row] = prn
        records["valid_tracking"][row] = 1
        records["valid_lock"][row] = 1
        records["loop_update_boundary"][row] = 1
        records["receiver_state"][row] = 4
        records["cn0_db_hz"][row] = 42.0
        records["carrier_lock_test"][row] = 1.0
        records["coherent_integration_s"][row] = 0.001
        for tap_index, tap in enumerate(TAPS):
            records[f"{tap}_i"][row] = 1.0 + row * 0.01 - abs(tap_index - 4) * 0.04
            records[f"{tap}_q"][row] = 0.02 * tap_index + row * 0.001
        for field_index, field in enumerate(ACTION_VALUE_FIELDS):
            next_value = float(100 + 10 * row + field_index)
            if field == "code_nco_rate_chips_s":
                next_value = 1_023_000.0
            elif field == "carrier_doppler_hz":
                next_value = 0.0
            records[f"action_next_{field}"][row] = next_value
            if row:
                records[f"action_used_{field}"][row] = records[f"action_next_{field}"][row - 1]
        records["action_next_interval_length_samples"][row] = 1_000
        records["action_used_interval_length_samples"][row] = 1_000
    with path.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)
    return path


def test_schema_sizes_and_header_round_trip(tmp_path):
    path = _write_dump(tmp_path / "trace_native_1ms_ch_0.bin", channel=0, prn=7)
    header, records = read_records(path, mmap=False)
    assert HEADER_STRUCT.size == HEADER_SIZE == 192
    assert RECORD_DTYPE.itemsize == RECORD_SIZE == 416
    assert header.scenario_id == "SYNTHETIC"
    assert header.tap_offsets_chips == tuple(np.arange(-0.5, 0.5001, 0.125))
    assert len(records) == 6


def test_causal_tuple_and_multi_prn_validation(tmp_path):
    files = [
        _write_dump(tmp_path / f"trace_native_1ms_ch_{channel}.bin", channel=channel, prn=7 + channel)
        for channel in range(4)
    ]
    result = validate_dump_files(files, expected_scenario_id="SYNTHETIC", minimum_prns=4)
    assert result["status"] == "PASS"
    assert result["causal_pair_count"] == 20
    assert result["causal_value_mismatch_count"] == 0
    assert result["consume_span_mismatch_count"] == 0
    assert result["maximum_valid_prns_same_rounded_ms_epoch"] == 4


def test_causal_value_corruption_fails_closed(tmp_path):
    path = _write_dump(tmp_path / "trace_native_1ms_ch_0.bin", channel=0, prn=7)
    _, records = read_records(path)
    records = np.array(records)
    records["action_used_code_nco_rate_chips_s"][2] += 1.0
    with path.open("r+b") as stream:
        stream.seek(HEADER_SIZE)
        records.tofile(stream)
    result = validate_dump_files([path], minimum_prns=1)
    assert result["status"] == "FAIL"
    assert "ACTION_MAPPING_UNRESOLVED" in result["failure_labels"]


def test_native_adapter_known_zero_action_vector(tmp_path):
    _write_dump(tmp_path / "trace_native_1ms_ch_0.bin", channel=0, prn=7)
    pairs = load_native_trace_pairs(tmp_path, cn0_min_db_hz=0.0, lock_min=0.0)
    assert len(pairs.current) == 5
    assert np.allclose(pairs.code_action, 0.0)
    assert np.allclose(pairs.carrier_action, 0.0)
    common = pairs.valid_support
    assert np.array_equal(common, np.broadcast_to(common[0], common.shape))
    assert np.allclose(pairs.warped[common], pairs.current[common])
    assert pairs.source_row.tolist() == [0, 1, 2, 3, 4]


def test_prn_permutation_variable_count_and_common_support(tmp_path):
    for channel in range(4):
        _write_dump(tmp_path / f"trace_native_1ms_ch_{channel}.bin", channel=channel, prn=10 + channel)
    pairs = load_native_trace_pairs(tmp_path, cn0_min_db_hz=0.0, lock_min=0.0)
    scores = np.abs(pairs.target[:, 0] - pairs.current[:, 0])
    blocks = robust_epoch_blocks(pairs, scores, block_s=0.005, minimum_prns=4)
    permutation = np.arange(len(scores))[::-1]
    permuted = pairs.take(permutation)
    permuted_blocks = robust_epoch_blocks(permuted, scores[permutation], block_s=0.005, minimum_prns=4)
    assert np.array_equal(blocks, permuted_blocks)
    assert len(robust_epoch_blocks(pairs, scores, block_s=0.005, minimum_prns=5)) == 0


def test_action_shuffle_preserves_marginals_and_is_deterministic(tmp_path):
    for channel in range(2):
        _write_dump(tmp_path / f"trace_native_1ms_ch_{channel}.bin", channel=channel, prn=20 + channel)
    first = load_native_trace_pairs(tmp_path, cn0_min_db_hz=0.0, lock_min=0.0)
    second = load_native_trace_pairs(tmp_path, cn0_min_db_hz=0.0, lock_min=0.0)
    assert np.array_equal(first.current, second.current)
    shuffle = action_shuffle_indices(first.prn, first.cn0_db_hz, seed=23017)
    assert np.array_equal(np.sort(first.code_action), np.sort(first.code_action[shuffle]))
    assert np.array_equal(first.prn, first.prn[shuffle])


def test_normal_only_threshold_contract(tmp_path):
    _write_dump(tmp_path / "trace_native_1ms_ch_0.bin", channel=0, prn=7)
    normal = load_native_trace_pairs(tmp_path, cn0_min_db_hz=0.0, lock_min=0.0)
    normal_score = np.square(np.abs(normal.target[:, 0] - normal.current[:, 0]))
    threshold = float(np.quantile(normal_score, 0.99))
    attack_score = normal_score + 100.0
    assert threshold == float(np.quantile(normal_score, 0.99))
    assert np.all(attack_score > threshold)
