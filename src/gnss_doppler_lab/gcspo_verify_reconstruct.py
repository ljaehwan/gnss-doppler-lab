"""Independent semantic reconstruction used by the final verifier."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from .gcspo_statistics import compute_scientific_gates


def _record_hash(row):
    payload = {key: value for key, value in row.items() if key != "record_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _utc_timestamp(value):
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        raise ValueError("access timestamp is not canonical UTC")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo != timezone.utc: raise ValueError("access timestamp timezone is not UTC")
    return parsed


def _validate_access_identity(row):
    if row.get("actor") != "gnss_doppler_lab.gcspo.AccessGate": raise ValueError("access actor identity mismatch")
    run = row.get("run_identity")
    if not isinstance(run, str) or not _HEX40.fullmatch(run) or row.get("authorization_sha") != run:
        raise ValueError("access run identity mismatch")
    counter = row.get("access_counter")
    if isinstance(counter, bool) or not isinstance(counter, int) or counter < 1:
        raise ValueError("access counter is invalid")
    path = row.get("canonical_path")
    if not isinstance(path, str) or not Path(path).is_absolute(): raise ValueError("access target path identity mismatch")
    if row.get("record_type") in {"PRE", "POST"}:
        expected_sha, expected_size = row.get("expected_sha256"), row.get("expected_size")
        if not isinstance(expected_sha, str) or not _HEX64.fullmatch(expected_sha):
            raise ValueError("access target hash identity mismatch")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError("access target size identity mismatch")
        if row.get("byte_range") != f"[0,{expected_size})" or row.get("row_range") != "ALL_ROWS_IN_BYTE_RANGE":
            raise ValueError("access target range identity mismatch")
        if not row.get("operation") or not row.get("kind") or not row.get("identity_source"):
            raise ValueError("access target provenance is incomplete")
    return run, counter, _utc_timestamp(row.get("timestamp_utc"))


def validate_access_ledger(records):
    if not isinstance(records, list) or not records: raise ValueError("access ledger is empty")
    previous, pending, successful = "0" * 64, None, 0
    run_identity, last_timestamp, completed_counter = None, None, 0
    for sequence, row in enumerate(records, start=1):
        if row.get("sequence") != sequence or row.get("previous_record_sha256") != previous:
            raise ValueError("access ledger sequence/hash chain mismatch")
        if row.get("record_sha256") != _record_hash(row): raise ValueError("access ledger record hash mismatch")
        current_run, counter, timestamp = _validate_access_identity(row)
        if run_identity is None: run_identity = current_run
        if current_run != run_identity: raise ValueError("access ledger mixes run identities")
        if last_timestamp is not None and timestamp < last_timestamp: raise ValueError("access timestamps are out of order")
        last_timestamp = timestamp; previous = row["record_sha256"]
        kind = row.get("record_type")
        if kind == "DENIED":
            if pending is not None or counter != completed_counter + 1:
                raise ValueError("denial interleaves or misnumbers an access pair")
            completed_counter = counter; continue
        if kind == "PRE":
            if pending is not None or row.get("outcome") != "OPEN_PENDING" or counter != completed_counter + 1:
                raise ValueError("unpaired or misnumbered access PRE")
            pending = row; continue
        if kind != "POST" or pending is None: raise ValueError("unpaired access POST")
        common = ("actor", "canonical_path", "scenario", "phase", "purpose", "authorization_sha", "run_identity",
                  "access_counter", "expected_sha256", "expected_size", "byte_range", "row_range", "operation",
                  "kind", "identity_source")
        if counter != pending["access_counter"] or any(row.get(key) != pending.get(key) for key in common):
            raise ValueError("PRE/POST access identity mismatch")
        if timestamp < _utc_timestamp(pending["timestamp_utc"]): raise ValueError("POST predates PRE")
        if row.get("outcome") == "SUCCESS":
            if row.get("observed_sha256") != pending.get("expected_sha256") or row.get("observed_size") != pending.get("expected_size"):
                raise ValueError("successful access observed identity mismatch")
            successful += 1
        completed_counter = counter; pending = None
    if pending is not None: raise ValueError("access ledger ends with unpaired PRE")
    if successful == 0: raise ValueError("access ledger contains no successful science file")
    return {"records": len(records), "successful_files": successful, "chain_head": previous,
            "run_identity": run_identity, "access_count": completed_counter}


def _reject_placeholders(value):
    if isinstance(value, dict):
        for child in value.values(): _reject_placeholders(child)
    elif isinstance(value, list):
        for child in value: _reject_placeholders(child)
    elif isinstance(value, str) and value.upper() in {"NA", "N/A", "PLACEHOLDER", "UNIMPLEMENTED"}:
        raise ValueError("placeholder evidence is forbidden")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite evidence is forbidden")


def verify_evidence_document(document):
    if document.get("scientific_status") != "VALID_SCIENCE" or document.get("protected_run_count") != 1:
        raise ValueError("scientific status/count mismatch")
    evidence = document.get("evidence"); _reject_placeholders(evidence)
    recomputed = compute_scientific_gates(evidence)
    if document.get("gates") != recomputed: raise ValueError("reported gates differ from recomputed gates")
    verdict = "GO_FOR_NEURAL_STAGE1" if all(row["status"] == "PASS" for row in recomputed) else "NO_GO_PHYSICAL_HYPOTHESIS"
    if document.get("verdict") != verdict: raise ValueError("reported verdict differs from recomputed verdict")
    return {"verdict": verdict, "gates": recomputed}
