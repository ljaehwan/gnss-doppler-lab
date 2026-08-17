import numpy as np

from gnss_doppler_lab.mosaic_navbit_provenance import (
    GPS_PREAMBLE,
    decode_word,
    find_valid_subframe_pairs,
    gps_word_parity_ok,
    recover_bits,
)


def _encode_word(data_bits, previous_d29, previous_d30):
    data = 0
    for bit in data_bits:
        data = (data << 1) | bit
    transmitted_data = data ^ (0xFFFFFF if previous_d30 else 0)
    for parity in range(64):
        transmitted = (transmitted_data << 6) | parity
        extended = transmitted | (previous_d30 << 30) | (previous_d29 << 31)
        if previous_d30:
            extended ^= 0x3FFFFFC0
        if gps_word_parity_ok(extended):
            return [(transmitted >> shift) & 1 for shift in range(29, -1, -1)]
    raise AssertionError("no parity encoding")


def _two_subframes(tow1=123456, sf1=2):
    previous_d29 = previous_d30 = 0
    bits = [previous_d29, previous_d30]
    for sf_index in range(2):
        for word_index in range(10):
            data = [0] * 24
            if word_index == 0:
                data[:8] = GPS_PREAMBLE
            if word_index == 1:
                tow_count = tow1 // 6 + sf_index
                data[:17] = [(tow_count >> shift) & 1 for shift in range(16, -1, -1)]
                subframe_id = (sf1 - 1 + sf_index) % 5 + 1
                data[19:22] = [(subframe_id >> shift) & 1 for shift in range(2, -1, -1)]
            word = _encode_word(data, previous_d29, previous_d30)
            bits.extend(word)
            previous_d29, previous_d30 = word[-2:]
    return np.asarray(bits, dtype=np.uint8)


def test_gps_nav_preamble_and_parity_known_vector():
    # TLM data 0x8b0000 with D29*=D30*=0 has parity 0x12.
    transmitted = 0x22C00012
    assert gps_word_parity_ok(transmitted)
    assert not gps_word_parity_ok(transmitted ^ (1 << 12))
    bits = [(transmitted >> shift) & 1 for shift in range(29, -1, -1)]
    decoded = decode_word(bits, 0, 0)
    assert decoded.parity_ok
    assert decoded.decoded_data[:8] == GPS_PREAMBLE


def test_global_polarity_inversion_has_same_decoded_structure():
    bits = _two_subframes()
    assert len(find_valid_subframe_pairs(bits)) == 1
    inverted = 1 - bits
    assert len(find_valid_subframe_pairs(inverted)) == 1
    direct = find_valid_subframe_pairs(bits)[0]
    inverse = find_valid_subframe_pairs(inverted)[0]
    assert direct.tow_seconds == inverse.tow_seconds
    assert direct.subframe_ids == inverse.subframe_ids


def test_20ms_boundary_recovery_and_single_epoch_dropout():
    bits = _two_subframes()[2:]
    prefix = 7
    epochs = np.zeros(prefix + 20 * len(bits) + 4, dtype=np.complex128)
    for index, bit in enumerate(bits):
        epochs[prefix + 20 * index:prefix + 20 * (index + 1)] = (1 if bit else -1) * (4 + 0.2j)
    flags = np.zeros(len(epochs), dtype=np.uint8)
    flags[prefix::20] = 1
    lock = np.ones(len(epochs), dtype=np.uint8)
    recovered = recover_bits(epochs, flags, lock)
    assert recovered.epoch_phase == prefix
    assert np.array_equal(recovered.logical_bits, bits)
    dropped = epochs.copy()
    dropped[prefix + 20 * 11 + 3] = np.nan + 1j * np.nan
    recovered_drop = recover_bits(dropped, flags, lock)
    assert np.array_equal(recovered_drop.logical_bits, bits)


def test_tow_discontinuity_is_rejected_and_decode_is_deterministic():
    valid = _two_subframes()
    first = find_valid_subframe_pairs(valid)
    second = find_valid_subframe_pairs(valid.copy())
    assert first == second
    broken = _two_subframes()
    # A bit flip must prevent a structurally valid all-parity pair.
    broken[2 + 10 * 30 + 40] ^= 1
    assert find_valid_subframe_pairs(broken) == []
