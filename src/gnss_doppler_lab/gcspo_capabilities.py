"""Outcome-free capability, source-field, and timeline sidecar validation."""
from __future__ import annotations

import math
from pathlib import PurePath
import re


SCHEMA = "gnss-doppler-lab.gcspo-stage0.capability-sidecar.v1"
SCENARIOS = {"DS3", "DS4", "DS7", "DS8"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sha(value, label):
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} SHA-256 is invalid")
    return value


def _size(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} size is invalid")
    return value


def _absolute(value, label):
    if not isinstance(value, str) or not PurePath(value).is_absolute() or ".." in PurePath(value).parts:
        raise ValueError(f"{label} canonical path is invalid")
    return value


def adapt_manifest_children(document):
    """Normalize explicitly supported receiver-manifest layouts.

    No sibling discovery or payload probing is permitted.  Every returned row
    comes from a named authenticated container in the manifest itself.
    """
    doc = _mapping(document, "receiver manifest")
    layouts = []
    if isinstance(doc.get("files"), list):
        layouts.append(("files", doc["files"]))
    if isinstance(doc.get("artifacts"), list):
        layouts.append(("artifacts", doc["artifacts"]))
    nested = doc.get("manifest")
    if isinstance(nested, dict) and isinstance(nested.get("files"), list):
        layouts.append(("manifest.files", nested["files"]))
    tracking = doc.get("tracking")
    if isinstance(tracking, dict) and isinstance(tracking.get("raw_mats"), list):
        raw_directory = tracking.get("raw_directory")
        if not isinstance(raw_directory, str) or not raw_directory:
            raise ValueError("tracking.raw_mats lacks an explicit raw directory")
        rows = []
        for item in tracking["raw_mats"]:
            if not isinstance(item, dict):
                raise ValueError("tracking.raw_mats child is malformed")
            rows.append({**item, "path": f"{raw_directory}/{item.get('name', '')}"})
        layouts.append(("tracking.raw_mats", rows))
    if len(layouts) != 1 or not layouts[0][1]:
        raise ValueError("receiver manifest has no unique supported child binding layout")
    adapter, raw_rows = layouts[0]
    rows = []
    for raw in raw_rows:
        row = _mapping(raw, "manifest child")
        path = row.get("path", row.get("relative_path", row.get("name")))
        sha = row.get("sha256", row.get("checksum_sha256"))
        size = row.get("size_bytes", row.get("bytes", row.get("size")))
        if not isinstance(path, str) or not path or PurePath(path).is_absolute() or ".." in PurePath(path).parts:
            raise ValueError("manifest child path is not canonical relative")
        rows.append({"path": path, "sha256": _sha(sha, "manifest child identity"),
                     "size_bytes": _size(size, "manifest child identity")})
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest child paths are duplicated")
    return {"adapter": adapter, "files": rows}


def validate_capability_sidecar(document):
    doc = _mapping(document, "capability sidecar")
    required = {"schema", "scenario", "purpose", "producer", "root_manifest",
                "children", "fields", "timeline"}
    if set(doc) != required or doc.get("schema") != SCHEMA:
        raise ValueError("capability sidecar schema/keys mismatch")
    scenario = doc.get("scenario")
    if scenario not in SCENARIOS or not isinstance(doc.get("purpose"), str) or not doc["purpose"]:
        raise ValueError("capability scenario/purpose is invalid")
    producer = _mapping(doc["producer"], "producer")
    if not isinstance(producer.get("identity"), str) or not producer["identity"]:
        raise ValueError("producer identity is invalid")
    producer_sha = _sha(producer.get("source_sha256"), "producer source")
    root = _mapping(doc["root_manifest"], "root manifest")
    _absolute(root.get("canonical_path"), "root manifest")
    _sha(root.get("sha256"), "root manifest"); _size(root.get("size_bytes"), "root manifest")
    if not isinstance(root.get("adapter"), str) or not root["adapter"]:
        raise ValueError("root manifest adapter is invalid")
    children = doc["children"]
    if not isinstance(children, list) or not children:
        raise ValueError("capability children are absent")
    child_paths = []
    for child in children:
        child = _mapping(child, "capability child")
        child_paths.append(_absolute(child.get("canonical_path"), "capability child"))
        _sha(child.get("sha256"), "capability child"); _size(child.get("size_bytes"), "capability child")
        if child.get("scenario") != scenario or not isinstance(child.get("purpose"), str) or not child["purpose"]:
            raise ValueError("capability child scenario/purpose mismatch")
    if len(child_paths) != len(set(child_paths)):
        raise ValueError("capability child path is duplicated")
    fields = doc["fields"]
    if not isinstance(fields, list) or not fields:
        raise ValueError("source-verified field inventory is absent")
    names = []
    for field in fields:
        field = _mapping(field, "field inventory")
        keys = {"name", "producer", "source_sha256", "unit", "sign_proof",
                "cadence_s", "role", "direct_loading"}
        if set(field) != keys:
            raise ValueError("field inventory keys/sign proof mismatch")
        names.append(field["name"])
        if field["producer"] != producer["identity"] or _sha(field["source_sha256"], "field source") != producer_sha:
            raise ValueError("field producer/source binding mismatch")
        if not all(isinstance(field[key], str) and field[key] for key in
                   ("name", "unit", "sign_proof", "role", "direct_loading")):
            raise ValueError("field unit/sign/role binding is invalid")
        if not isinstance(field["cadence_s"], (int, float)) or not math.isfinite(field["cadence_s"]) or field["cadence_s"] <= 0:
            raise ValueError("field cadence is invalid")
    if len(names) != len(set(names)):
        raise ValueError("field inventory names are duplicated")
    timeline = _mapping(doc["timeline"], "timeline")
    expected_timeline = {"official_document", "raw_iq", "processed_sample_count",
                         "recording_relative_seconds", "rx_time_s", "nmea_time",
                         "gps_time", "onset_s", "pull_off_s"}
    if set(timeline) != expected_timeline:
        raise ValueError("timeline linkage keys mismatch")
    official = _mapping(timeline["official_document"], "official document")
    if not isinstance(official.get("identity"), str) or not official["identity"]:
        raise ValueError("official document identity is absent")
    _sha(official.get("sha256"), "official document")
    iq = _mapping(timeline["raw_iq"], "raw IQ")
    _absolute(iq.get("canonical_path"), "raw IQ"); _sha(iq.get("sha256"), "raw IQ")
    _size(iq.get("size_bytes"), "raw IQ")
    if iq.get("byte_zero") != 0 or not isinstance(iq.get("sample_rate_hz"), int) or iq["sample_rate_hz"] <= 0:
        raise ValueError("raw IQ byte-zero/sample-rate linkage is invalid")
    count = timeline["processed_sample_count"]
    relative = timeline["recording_relative_seconds"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("processed sample count is invalid")
    if not isinstance(relative, (int, float)) or not math.isfinite(relative) or relative < 0:
        raise ValueError("recording-relative time is invalid")
    if not math.isclose(relative, count / iq["sample_rate_hz"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("sample-count/recording-relative timeline mismatch")
    for key in ("rx_time_s", "onset_s", "pull_off_s"):
        if not isinstance(timeline[key], (int, float)) or not math.isfinite(timeline[key]):
            raise ValueError(f"timeline {key} is invalid")
    gps = _mapping(timeline["gps_time"], "GPS time")
    if not isinstance(gps.get("week"), int) or not isinstance(gps.get("tow_s"), (int, float)):
        raise ValueError("GPS time linkage is invalid")
    if timeline["pull_off_s"] < timeline["onset_s"]:
        raise ValueError("onset/pull-off ordering is invalid")
    if scenario == "DS8" and not any("observables" in path.lower() for path in child_paths):
        return {"status": "UNAVAILABLE", "reason": "DS8_MISSING_AUTHENTICATED_OBSERVABLES"}
    if scenario == "DS4" and producer["identity"] == "complex9":
        return {"status": "UNAVAILABLE", "reason": "DS4_METHOD_A_SOURCE_IDENTITY_NOT_INDEPENDENTLY_PROVEN"}
    return {"status": "AVAILABLE", "scenario": scenario, "child_count": len(children),
            "field_count": len(fields), "manifest_adapter": root["adapter"]}
