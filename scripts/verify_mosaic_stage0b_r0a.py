#!/usr/bin/env python3
"""Independent, self-contained verifier for MOSAIC Stage-0B R0a hardening.

This module intentionally does not import the production NAV-bit decoder and
does not trust PASS booleans from the frozen R0 derived CSV files.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"
AUDIT = ROOT / "artifacts/mosaic_stage0b_r0a_provenance_hardening"
PREAMBLE = (1, 0, 0, 0, 1, 0, 1, 1)


class VerificationFailure(RuntimeError):
    def __init__(self, label: str, detail: str = ""):
        super().__init__(f"{label}: {detail}")
        self.label = label
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, compressed: bool = False) -> list[dict[str, str]]:
    opener = gzip.open if compressed else open
    kwargs = {"mode": "rt", "encoding": "utf-8", "newline": ""} if compressed else {"mode": "r", "encoding": "utf-8", "newline": ""}
    with opener(path, **kwargs) as stream:
        return list(csv.DictReader(stream))


def xor_bits(values) -> int:
    value = 0
    for item in values:
        value ^= int(item)
    return value


def expected_parity(data: list[int], previous_d29: int, previous_d30: int) -> list[int]:
    """Explicit IS-GPS-200 D25..D30 equations; data is de-whitened d1..d24."""
    if len(data) != 24:
        raise ValueError("GPS data word must contain 24 bits")
    d = [0] + data
    equations = (
        (previous_d29, (1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23)),
        (previous_d30, (2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24)),
        (previous_d29, (1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22)),
        (previous_d30, (2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23)),
        (previous_d30, (1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24)),
        (previous_d29, (3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24)),
    )
    return [base ^ xor_bits(d[index] for index in indices) for base, indices in equations]


def bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def decode_chain(bits: list[int], previous_d29: int, previous_d30: int) -> tuple[list[dict[str, object]], int]:
    words = []
    chain_errors = 0
    for word_index in range(20):
        transmitted = bits[word_index * 30:(word_index + 1) * 30]
        if len(transmitted) != 30:
            raise VerificationFailure("BIT_COUNT_MISMATCH", "incomplete 30-bit word")
        decoded_data = [bit ^ previous_d30 for bit in transmitted[:24]]
        parity_expected = expected_parity(decoded_data, previous_d29, previous_d30)
        parity_observed = transmitted[24:]
        parity_ok = parity_expected == parity_observed
        words.append({
            "word_index": word_index,
            "subframe_index": word_index // 10,
            "word_position": word_index % 10 + 1,
            "transmitted_word_hex": f"0x{bits_to_int(transmitted):08x}",
            "previous_d29": previous_d29,
            "previous_d30": previous_d30,
            "decoded_data": decoded_data,
            "parity_expected": parity_expected,
            "parity_observed": parity_observed,
            "parity_valid": parity_ok,
        })
        next_d29, next_d30 = transmitted[-2], transmitted[-1]
        if word_index and (previous_d29, previous_d30) != tuple(bits[word_index * 30 - 2:word_index * 30]):
            chain_errors += 1
        previous_d29, previous_d30 = next_d29, next_d30
    return words, chain_errors


def recompute_from_rows(decoded_rows: list[dict[str, str]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    all_grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in decoded_rows:
        key = (row["dataset"], int(row["prn"]))
        all_grouped[key].append(row)
        if row["validated_navbit"] == "True":
            grouped[key].append(row)
    if len(grouped) != 10:
        raise VerificationFailure("MULTI_PRN_COUNT_MISMATCH", str(len(grouped)))
    reports = []
    word_rows: list[dict[str, object]] = []
    distribution = Counter()
    sequence_hashes = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["bit_index"]))
        indices = [int(row["bit_index"]) for row in rows]
        if len(rows) != 600 or indices != list(range(indices[0], indices[0] + 600)):
            raise VerificationFailure("BIT_INDEX_CONTINUITY_FAIL", str(key))
        bits = [int(row["transmitted_logical_bit"]) for row in rows]
        if any(bit not in (0, 1) for bit in bits):
            raise VerificationFailure("NON_BINARY_TRANSMITTED_BIT", str(key))
        if len(set(bits)) == 1:
            raise VerificationFailure("CONSTANT_PLUS_ONE_SEQUENCE", str(key))
        distribution.update(bits)
        sequence_sha = hashlib.sha256(bytes(bits)).hexdigest()
        sequence_hashes.append(sequence_sha)
        all_rows = sorted(all_grouped[key], key=lambda row: int(row["bit_index"]))
        preceding = [int(row["transmitted_logical_bit"]) for row in all_rows if int(row["bit_index"]) < indices[0]]
        if len(preceding) >= 2:
            states = [(preceding[-2], preceding[-1])]
        elif len(preceding) == 1:
            states = [(d29, preceding[-1]) for d29 in (0, 1)]
        else:
            states = [(d29, d30) for d29 in (0, 1) for d30 in (0, 1)]
        structurally_valid = []
        failures = []
        for initial_d29, initial_d30 in states:
            words, chain_errors = decode_chain(bits, initial_d29, initial_d30)
            preambles = [tuple(words[index]["decoded_data"][:8]) == PREAMBLE for index in (0, 10)]
            hows = [words[index]["decoded_data"] for index in (1, 11)]
            tows = [bits_to_int(how[:17]) * 6 for how in hows]
            subframe_ids = [bits_to_int(how[19:22]) for how in hows]
            checks = {
                "parity": all(bool(word["parity_valid"]) for word in words),
                "preamble": all(preambles),
                "tow": tows[1] - tows[0] == 6,
                "subframe": subframe_ids[0] in range(1, 6) and subframe_ids[1] == subframe_ids[0] % 5 + 1,
                "chain": chain_errors == 0,
            }
            if all(checks.values()):
                structurally_valid.append((initial_d29, initial_d30, words, chain_errors, tows, subframe_ids))
            else:
                failures.append(checks)
        if len(structurally_valid) != 1:
            # Prefer the most informative negative label for tamper tests.
            if failures and not any(item["preamble"] for item in failures):
                label = "PREAMBLE_RECOMPUTATION_FAIL"
            elif failures and not any(item["parity"] for item in failures):
                label = "PARITY_RECOMPUTATION_FAIL"
            elif failures and not any(item["tow"] for item in failures):
                label = "TOW_RECOMPUTATION_FAIL"
            else:
                label = "SCIENTIFIC_RECOMPUTATION_FAIL"
            raise VerificationFailure(label, f"{key}: valid initial states={len(structurally_valid)}")
        initial_d29, initial_d30, words, chain_errors, tows, subframe_ids = structurally_valid[0]
        for word in words:
            word_rows.append({
                "dataset": key[0], "prn": key[1], "validated_start_bit_index": indices[0],
                "initial_d29": initial_d29, "initial_d30": initial_d30,
                **{name: value for name, value in word.items() if name != "decoded_data"},
            })
        reports.append({
            "dataset": key[0], "prn": key[1], "validated_start_bit_index": indices[0],
            "validated_bits": len(bits), "words": len(words), "parity_valid_words": sum(bool(w["parity_valid"]) for w in words),
            "preambles_valid": 2, "tow_seconds": tows, "subframe_ids": subframe_ids,
            "initial_d29": initial_d29, "initial_d30": initial_d30, "d29_d30_chain_errors": chain_errors,
            "sequence_sha256": sequence_sha,
        })
    if len(set(sequence_hashes)) != len(sequence_hashes):
        raise VerificationFailure("DUPLICATE_PRN_SEQUENCE", "full 600-bit sequence hash collision")
    result = {
        "validated_prns": len(reports), "validated_bits_per_prn": 600,
        "total_validated_bits": sum(item["validated_bits"] for item in reports),
        "words_per_prn": 20, "total_words": len(word_rows),
        "parity_valid_words": sum(bool(item["parity_valid"]) for item in word_rows),
        "preambles_valid": sum(item["preambles_valid"] for item in reports),
        "tow_continuity_valid_prns": sum(item["tow_seconds"][1] - item["tow_seconds"][0] == 6 for item in reports),
        "d29_d30_chain_errors": sum(item["d29_d30_chain_errors"] for item in reports),
        "transmitted_bit_distribution": {"0": distribution[0], "1": distribution[1]},
        "unique_prn_sequence_hashes": len(set(sequence_hashes)), "per_prn": reports,
    }
    return result, word_rows


def cross_validate_derived_rows(
    recomputation: dict[str, object],
    words: list[dict[str, object]],
    derived_words: list[dict[str, str]],
    preambles: list[dict[str, str]],
    tow: list[dict[str, str]],
) -> None:
    lookup = {(r["dataset"], int(r["prn"]), int(r["global_word_index"])): r for r in derived_words}
    for word in words:
        row = lookup.get((word["dataset"], int(word["prn"]), int(word["word_index"])))
        if row is None or row["transmitted_word_hex"] != word["transmitted_word_hex"] or row["parity_valid"] != str(word["parity_valid"]):
            raise VerificationFailure("DERIVED_WORD_MISMATCH", str((word["dataset"], word["prn"], word["word_index"])))
    if len(preambles) != 20 or any(row["observed_decoded_preamble"] != "10001011" for row in preambles):
        raise VerificationFailure("DERIVED_PREAMBLE_MISMATCH")
    expected = {(p["dataset"], p["prn"]): p["tow_seconds"] for p in recomputation["per_prn"]}
    for row in tow:
        actual = [int(row["first_how_tow_s"]), int(row["second_how_tow_s"])]
        if expected[(row["dataset"], int(row["prn"]))] != actual:
            raise VerificationFailure("DERIVED_TOW_MISMATCH", str((row["dataset"], row["prn"])))


def cross_validate_derived(frozen: Path, recomputation: dict[str, object], words: list[dict[str, object]]) -> None:
    cross_validate_derived_rows(
        recomputation, words, read_csv(frozen / "parity_validation.csv"),
        read_csv(frozen / "preamble_detections.csv"), read_csv(frozen / "tow_continuity.csv"),
    )


def canonical_mapping_digest(rows: list[dict[str, str]]) -> str:
    keys = ("dataset", "prn", "bit_index", "raw_start_sample", "raw_end_sample_exclusive", "code_epoch_start", "code_epoch_end_inclusive")
    payload = "\n".join("|".join(row[key] for key in keys) for row in sorted(rows, key=lambda r: (r["dataset"], int(r["prn"]), int(r["bit_index"]))))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_manifest(root: Path) -> int:
    manifest = json.loads((root / "artifact_manifest_sha256.json").read_text())
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise VerificationFailure("ARTIFACT_INTEGRITY_FAIL", item["path"])
    return len(manifest["files"])


def main() -> None:
    checksums = verify_manifest(AUDIT)
    decoded = read_csv(FROZEN / "decoded_nav_bits.csv.gz", compressed=True)
    recomputation, words = recompute_from_rows(decoded)
    cross_validate_derived(FROZEN, recomputation, words)
    committed = json.loads((AUDIT / "independent_bit_recomputation.json").read_text())
    if recomputation != committed["recomputation"]:
        raise VerificationFailure("SCIENTIFIC_RECOMPUTATION_MISMATCH", "committed recomputation differs")
    mapping = read_csv(FROZEN / "navbit_sample_mapping.csv.gz", compressed=True)
    transcription = json.loads((AUDIT / "trace_endpoint_transcription.json").read_text())
    if canonical_mapping_digest(mapping) != transcription["frozen_mapping_canonical_sha256"]:
        raise VerificationFailure("RAW_MAPPING_MISMATCH", "frozen mapping content changed")
    scope = json.loads((AUDIT / "scope_limitations.json").read_text())
    if scope["two_separated_intervals"] is not False or scope["distant_interval_validation"] != "NOT_PERFORMED":
        raise VerificationFailure("SCOPE_LIMITATION_MISSING")
    tamper = json.loads((AUDIT / "tamper_negative_tests.json").read_text())
    if not tamper["all_expected_failures_observed"]:
        raise VerificationFailure("TAMPER_NEGATIVE_TEST_FAIL")
    verdict = json.loads((AUDIT / "final_verdict.json").read_text())
    if verdict["verdict"] != "RAW_MAPPING_MISMATCH" or verdict["stage0b_injection_authorized"]:
        raise VerificationFailure("VERDICT_MISMATCH")
    print(f"PASS: independently recomputed {recomputation['total_words']}/200 words and {recomputation['total_validated_bits']}/6000 bits; {checksums} audit checksums; fail-closed RAW_MAPPING_MISMATCH")


if __name__ == "__main__":
    try:
        main()
    except VerificationFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
