from pathlib import Path

import numpy as np

from gnss_doppler_lab.trace_native_1ms import RECORD_DTYPE
from gnss_doppler_lab.trace_reproducibility import (
    CanonicalReplay,
    canonical_join,
    canonical_semantic_hash,
    common_epoch_count,
    exact_equal,
)


def records(prns=(3, 6), order=(0, 1)):
    value = np.zeros(2, dtype=RECORD_DTYPE)
    value["prn"] = prns
    value["raw_interval_start_sample"] = (25_001, 25_010)
    value["raw_interval_end_sample"] = (50_001, 50_010)
    value["P_i"] = (1.0, 2.0)
    value["action_used_interval_length_samples"] = 25_000
    value["action_next_interval_length_samples"] = 25_000
    return value[list(order)]


def replay(value):
    return CanonicalReplay("clean", Path("."), value, np.asarray(["a", "b"]))


def test_canonical_join_ignores_channel_order_and_uses_prn_raw_interval():
    first = records()
    second = records(order=(1, 0))
    second["channel"] = (9, 8)
    joined = canonical_join(replay(first), replay(second))
    assert len(joined) == 2
    assert list(joined["prn"]) == [3, 6]


def test_semantic_hash_is_prn_permutation_invariant_but_physical_sensitive():
    first = replay(records())
    second = replay(records(order=(1, 0)))
    assert canonical_semantic_hash(first) == canonical_semantic_hash(second)
    changed = records(order=(1, 0))
    changed["P_i"][0] = np.nextafter(changed["P_i"][0], np.float32(np.inf))
    assert canonical_semantic_hash(first) != canonical_semantic_hash(replay(changed))


def test_exact_equal_is_bitwise_for_floats():
    positive_zero = np.asarray([0.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0], dtype=np.float32)
    assert not exact_equal(positive_zero, negative_zero)[0]


def test_common_epoch_uses_rounded_millisecond_and_variable_prn_count():
    joined = canonical_join(replay(records()), replay(records(order=(1, 0))))
    assert common_epoch_count(joined, 25_000_000, minimum_prns=2) == 1
    assert common_epoch_count(joined, 25_000_000, minimum_prns=3) == 0
