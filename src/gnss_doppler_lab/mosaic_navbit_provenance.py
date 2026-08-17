"""GPS L1 C/A NAV-bit decoding and exact TRACE-to-raw-sample binding.

The decoder is deliberately non-adaptive: receiver symbol-boundary flags bind
the 20 ms timing, the receiver Prompt real-axis convention fixes the carrier
representative, and GPS preamble/parity/TOW structure only validates the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

GPS_PREAMBLE = (1, 0, 0, 0, 1, 0, 1, 1)
EPOCHS_PER_BIT = 20
BITS_PER_WORD = 30
BITS_PER_SUBFRAME = 300


def _rotl32(value: int, count: int) -> int:
    value &= 0xFFFFFFFF
    return ((value << count) & 0xFFFFFFFF) | (value >> (32 - count))


def gps_word_parity_ok(extended_word: int) -> bool:
    """Port of the authenticated receiver's IS-GPS-200 parity checker."""
    word = extended_word & 0xFFFFFFFF
    terms = (
        word & 0xFBFFBF00,
        _rotl32(word, 1) & 0x07FFBF01,
        _rotl32(word, 2) & 0xFC0F8100,
        _rotl32(word, 3) & 0xF81FFE02,
        _rotl32(word, 4) & 0xFC00000E,
        _rotl32(word, 5) & 0x07F00001,
        _rotl32(word, 6) & 0x00003000,
    )
    folded = 0
    for term in terms:
        folded ^= term
    parity = folded
    for count in (6, 12, 18, 24):
        parity ^= _rotl32(folded, count)
    return (parity & 0x3F) == (word & 0x3F)


def _bits_to_int(bits: Iterable[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


@dataclass(frozen=True)
class DecodedWord:
    word_index: int
    transmitted_word: int
    decoded_data: tuple[int, ...]
    parity_ok: bool
    previous_d29: int
    previous_d30: int
    d29: int
    d30: int


def decode_word(bits: Iterable[int], previous_d29: int, previous_d30: int, word_index: int = 0) -> DecodedWord:
    bit_tuple = tuple(int(v) for v in bits)
    if len(bit_tuple) != BITS_PER_WORD or any(v not in (0, 1) for v in bit_tuple):
        raise ValueError("a GPS word must contain exactly 30 binary bits")
    transmitted = _bits_to_int(bit_tuple)
    extended = transmitted | (int(previous_d30) << 30) | (int(previous_d29) << 31)
    if previous_d30:
        extended ^= 0x3FFFFFC0
    data = tuple((extended >> shift) & 1 for shift in range(29, 5, -1))
    return DecodedWord(
        word_index=word_index,
        transmitted_word=transmitted,
        decoded_data=data,
        parity_ok=gps_word_parity_ok(extended),
        previous_d29=int(previous_d29),
        previous_d30=int(previous_d30),
        d29=(transmitted >> 1) & 1,
        d30=transmitted & 1,
    )


def decode_words(bits: np.ndarray, start: int, count: int) -> list[DecodedWord]:
    if start < 2 or start + count * BITS_PER_WORD > len(bits):
        raise ValueError("word range lacks preceding D29*/D30* or input coverage")
    d29, d30 = int(bits[start - 2]), int(bits[start - 1])
    words: list[DecodedWord] = []
    for index in range(count):
        lo = start + index * BITS_PER_WORD
        decoded = decode_word(bits[lo:lo + BITS_PER_WORD], d29, d30, index)
        words.append(decoded)
        d29, d30 = decoded.d29, decoded.d30
    return words


def word_tow_seconds(how: DecodedWord) -> int:
    return _bits_to_int(how.decoded_data[:17]) * 6


def word_subframe_id(how: DecodedWord) -> int:
    return _bits_to_int(how.decoded_data[19:22])


@dataclass(frozen=True)
class SubframePair:
    start_bit: int
    words: tuple[DecodedWord, ...]
    tow_seconds: tuple[int, int]
    subframe_ids: tuple[int, int]
    initial_d29: int
    initial_d30: int


def find_valid_subframe_pairs(bits: np.ndarray) -> list[SubframePair]:
    """Return only two-subframe candidates satisfying all frozen structure."""
    candidates: list[SubframePair] = []
    for start in range(0, len(bits) - 2 * BITS_PER_SUBFRAME + 1):
        states = ([(int(bits[start - 2]), int(bits[start - 1]))] if start >= 2
                  else [(d29, d30) for d29 in (0, 1) for d30 in (0, 1)])
        for initial_d29, initial_d30 in states:
            d29, d30 = initial_d29, initial_d30
            words = []
            for index in range(20):
                lo = start + index * BITS_PER_WORD
                decoded = decode_word(bits[lo:lo + BITS_PER_WORD], d29, d30, index)
                words.append(decoded)
                d29, d30 = decoded.d29, decoded.d30
            if not all(word.parity_ok for word in words):
                continue
            if tuple(words[0].decoded_data[:8]) != GPS_PREAMBLE:
                continue
            if tuple(words[10].decoded_data[:8]) != GPS_PREAMBLE:
                continue
            tows = (word_tow_seconds(words[1]), word_tow_seconds(words[11]))
            ids = (word_subframe_id(words[1]), word_subframe_id(words[11]))
            if tows[1] - tows[0] != 6:
                continue
            if ids[0] not in range(1, 6) or ids[1] != ids[0] % 5 + 1:
                continue
            candidates.append(SubframePair(start, tuple(words), tows, ids, initial_d29, initial_d30))
    return candidates


def prompt_decision_axis(prompt: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Project Prompt on the receiver's deterministic squared-phase axis."""
    values = np.asarray(prompt, dtype=np.complex128)
    mask = np.isfinite(values.real) & np.isfinite(values.imag)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if not np.any(mask):
        raise ValueError("no finite valid Prompt samples")
    phase = 0.5 * float(np.angle(np.sum(values[mask] ** 2)))
    return (values * np.exp(-1j * phase)).real, phase


def receiver_boundary_phase(boundary_flags: np.ndarray) -> tuple[int, np.ndarray]:
    indices = np.flatnonzero(np.asarray(boundary_flags, dtype=bool))
    if len(indices) < 2:
        raise ValueError("fewer than two receiver data-symbol boundary flags")
    if not np.all(np.diff(indices) == EPOCHS_PER_BIT):
        raise ValueError("receiver data-symbol boundary cadence is not exactly 20 epochs")
    residues = np.unique(indices % EPOCHS_PER_BIT)
    if len(residues) != 1:
        raise ValueError("receiver data-symbol boundary phase is ambiguous")
    return int(residues[0]), indices


@dataclass(frozen=True)
class RecoveredBits:
    epoch_phase: int
    epoch_starts: np.ndarray
    logical_bits: np.ndarray
    values_pm1: np.ndarray
    metrics: np.ndarray
    confidence: np.ndarray
    carrier_axis_phase_rad: float
    boundary_flag_indices: np.ndarray


def recover_bits(prompt: np.ndarray, boundary_flags: np.ndarray, valid_lock: np.ndarray) -> RecoveredBits:
    """Recover bit decisions using timing fixed solely by receiver flags."""
    decision, axis_phase = prompt_decision_axis(prompt, valid_lock)
    epoch_phase, flag_indices = receiver_boundary_phase(boundary_flags)
    starts = np.arange(epoch_phase, len(decision) - EPOCHS_PER_BIT + 1, EPOCHS_PER_BIT, dtype=np.int64)
    metrics = np.empty(len(starts), dtype=np.float64)
    confidence = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        window = decision[start:start + EPOCHS_PER_BIT]
        finite = np.isfinite(window)
        if finite.sum() < EPOCHS_PER_BIT - 1:
            metrics[index] = np.nan
            confidence[index] = 0.0
            continue
        values = window[finite]
        metrics[index] = float(np.sum(values))
        denominator = float(np.sum(np.abs(values)))
        confidence[index] = abs(metrics[index]) / denominator if denominator else 0.0
    if not np.isfinite(metrics).all():
        raise ValueError("unrecoverable navigation bit window")
    # This is the receiver Prompt real-axis convention, fixed before GPS checks.
    logical = (metrics > 0.0).astype(np.uint8)
    return RecoveredBits(
        epoch_phase=epoch_phase,
        epoch_starts=starts,
        logical_bits=logical,
        values_pm1=np.where(logical == 1, 1, -1).astype(np.int8),
        metrics=metrics,
        confidence=confidence,
        carrier_axis_phase_rad=axis_phase,
        boundary_flag_indices=flag_indices,
    )


def phase_candidate_score(prompt: np.ndarray, phase: int, valid_lock: np.ndarray) -> dict[str, int]:
    """Diagnostic only; never used to select the receiver-fixed phase."""
    decision, _ = prompt_decision_axis(prompt, valid_lock)
    starts = np.arange(phase, len(decision) - EPOCHS_PER_BIT + 1, EPOCHS_PER_BIT)
    bits = np.asarray([np.sum(decision[s:s + EPOCHS_PER_BIT]) > 0 for s in starts], dtype=np.uint8)
    pairs = find_valid_subframe_pairs(bits)
    return {"valid_pair_count": len(pairs), "maximum_valid_words": 20 if pairs else 0}
