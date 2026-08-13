"""Descriptor-bound, tamper-evident protected-file capability reader."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Callable, TypeVar

import h5py

T = TypeVar("T")
PRIOR_RESULT_NAMES = {"scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv",
                      "shared_state_estimates.csv", "final_verdict.json"}
SCENARIOS = {"DS1", "DS2", "DS3", "DS4", "DS7", "DS8", "cleanDynamic", "DS5", "DS6"}


class AccessGate:
    """Only object permitted to expose authenticated protected science bytes."""

    def __init__(self, ledger_path):
        self.ledger_path = Path(ledger_path)
        self.state = "PREREGISTERED_UNVALIDATED"
        self._preflight = self._remote = None
        self._allow: dict[str, dict[str, object]] = {}
        self._sequence = 0
        self._access_counter = 0
        self._previous = "0" * 64
        self._last_timestamp = None
        if self.ledger_path.is_file():
            lines = [line for line in self.ledger_path.read_text().splitlines() if line.strip()]
            if lines:
                records = [json.loads(line) for line in lines]; last = records[-1]
                self._sequence = int(last["sequence"]); self._previous = last["record_sha256"]
                self._access_counter = max(int(row.get("access_counter", 0)) for row in records)
                if last.get("timestamp_utc"):
                    self._last_timestamp = datetime.fromisoformat(last["timestamp_utc"].replace("Z", "+00:00"))

    def set_preflight(self, *, clean_only_pass, reviews_pass, freeze_sha, frozen_hashes):
        if not clean_only_pass or not reviews_pass or len(freeze_sha) != 40 or not frozen_hashes or any(len(x) != 64 for x in frozen_hashes.values()):
            raise ValueError("incomplete implementation freeze preflight")
        self._preflight = {"freeze_sha": freeze_sha, "frozen_hashes": dict(frozen_hashes)}
        self._refresh()

    def set_remote_sync(self, *, local_sha, remote_sha, ahead, behind, clean):
        self._remote = {"local_sha": local_sha, "remote_sha": remote_sha, "ahead": ahead, "behind": behind, "clean": clean}
        self._refresh()

    def _refresh(self):
        if self._preflight and self._remote:
            freeze = self._preflight["freeze_sha"]
            if self._remote == {"local_sha": freeze, "remote_sha": freeze, "ahead": 0, "behind": 0, "clean": True}:
                self.state = "VALID_FOR_PROTECTED_ACCESS"

    def _next_access_counter(self):
        self._access_counter += 1
        return self._access_counter

    def _base_record(self, record_type, *, path, scenario, phase, purpose, access_counter,
                     operation="READ", byte_range="UNRESOLVED"):
        run_identity = self._preflight["freeze_sha"] if self._preflight else None
        return {"record_type": record_type, "actor": "gnss_doppler_lab.gcspo.AccessGate", "canonical_path": str(path),
                "operation": operation, "byte_range": byte_range, "row_range": "ALL_ROWS_IN_BYTE_RANGE",
                "scenario": scenario, "phase": phase, "purpose": purpose, "access_counter": access_counter,
                "run_identity": run_identity, "authorization_sha": run_identity}

    def _timestamp(self):
        observed = datetime.now(timezone.utc)
        if self._last_timestamp is not None and observed <= self._last_timestamp:
            observed = self._last_timestamp + timedelta(microseconds=1)
        self._last_timestamp = observed
        return observed.isoformat(timespec="microseconds").replace("+00:00", "Z")

    def _append(self, payload):
        self._sequence += 1
        record = {**payload, "timestamp_utc": self._timestamp(), "sequence": self._sequence,
                  "previous_record_sha256": self._previous}
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
        data = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.ledger_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, data); os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent = os.open(self.ledger_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(parent)
        finally: os.close(parent)
        self._previous = record["record_sha256"]
        return record

    def _deny(self, path, reason, *, scenario="UNCLASSIFIED", phase="UNCLASSIFIED", purpose="capability registration"):
        candidate = Path(path)
        self._append({**self._base_record("DENIED", path=candidate.resolve(strict=False), scenario=scenario, phase=phase,
                                         purpose=purpose, access_counter=self._next_access_counter()),
                      "outcome": "DENIED", "reason": reason})

    def _validate_candidate(self, path):
        candidate = Path(path)
        text = str(candidate)
        if any(token in text for token in ("*", "?", "[")):
            self._deny(candidate, "GLOB_FORBIDDEN"); raise ValueError("protected path globs are forbidden")
        if candidate.name in PRIOR_RESULT_NAMES:
            self._deny(candidate, "PRIOR_RESULT_FORBIDDEN"); raise PermissionError("prior-result paths are forbidden")
        try: info = os.lstat(candidate)
        except OSError as exc:
            self._deny(candidate, "PATH_NOT_REGULAR"); raise PermissionError("protected path is not an existing regular file") from exc
        if stat.S_ISLNK(info.st_mode):
            self._deny(candidate, "SYMLINK_FORBIDDEN"); raise PermissionError("protected symlinks are forbidden")
        if not stat.S_ISREG(info.st_mode):
            self._deny(candidate, "PATH_NOT_REGULAR"); raise PermissionError("protected directories/non-regular files are forbidden")
        return candidate.resolve(strict=True), info

    def register_pinned(self, path, *, expected_sha256, expected_size, kind,
                        preclaim_dev=None, preclaim_ino=None):
        if self.state != "VALID_FOR_PROTECTED_ACCESS":
            self._deny(path, "GATE_NOT_READY"); raise PermissionError("remote implementation freeze is not exactly synchronized" if self._preflight else "VALID_FOR_PROTECTED_ACCESS not reached")
        canonical, info = self._validate_candidate(path)
        if len(expected_sha256) != 64 or isinstance(expected_size, bool) or expected_size <= 0 or not kind:
            self._deny(canonical, "INVALID_EXPECTED_IDENTITY"); raise ValueError("expected protected identity is incomplete")
        if info.st_size != int(expected_size):
            raise ValueError("protected declared size mismatch before claim")
        if (preclaim_dev is not None or preclaim_ino is not None) and (
                info.st_dev != preclaim_dev or info.st_ino != preclaim_ino):
            raise RuntimeError("protected pinned identity replaced after preclaim check")
        self._allow[str(canonical)] = {"expected_sha256": expected_sha256.lower(), "expected_size": int(expected_size),
                                       "kind": str(kind), "registered_dev": info.st_dev, "registered_ino": info.st_ino,
                                       "source": "PINNED"}
        return canonical

    def _capability(self, path):
        candidate = Path(path)
        registered_path = Path(os.path.abspath(candidate))
        capability = self._allow.get(str(registered_path))
        if capability is None:
            self._deny(candidate, "UNREGISTERED_PATH"); raise PermissionError("manifest-derived identity is absent")
        return registered_path, capability

    @staticmethod
    def _metadata_precheck(canonical, capability):
        """Authenticate child identity metadata without opening protected bytes."""
        try:
            info = os.lstat(canonical)
        except OSError as exc:
            raise RuntimeError("protected registered identity missing before claim") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("protected registered identity type changed before claim")
        try:
            resolved = canonical.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError("protected registered identity missing before claim") from exc
        if resolved != canonical:
            raise RuntimeError("protected registered identity path changed before claim")
        registered = (int(capability["registered_dev"]), int(capability["registered_ino"]))
        if (info.st_dev, info.st_ino) != registered:
            raise RuntimeError("protected registered identity replaced before claim")
        if info.st_size != int(capability["expected_size"]):
            raise ValueError("protected declared size mismatch before claim")
        return info

    def consume(self, path, *, scenario, phase, purpose, consumer: Callable[[object], T], operation="READ") -> T:
        if self.state != "VALID_FOR_PROTECTED_ACCESS":
            self._deny(path, "GATE_NOT_READY", scenario=scenario, phase=phase, purpose=purpose)
            raise PermissionError("remote implementation freeze is not exactly synchronized" if self._preflight else "VALID_FOR_PROTECTED_ACCESS not reached")
        if scenario not in SCENARIOS or not phase or not purpose:
            self._deny(path, "INVALID_CLASSIFICATION", scenario=scenario, phase=phase, purpose=purpose)
            raise ValueError("protected access classification is incomplete")
        canonical, capability = self._capability(path)
        registered = self._metadata_precheck(canonical, capability)
        expected_size = int(capability["expected_size"]); byte_range = f"[0,{expected_size})"
        common = self._base_record("PRE", path=canonical, scenario=scenario, phase=phase, purpose=purpose,
                                   access_counter=self._next_access_counter(), operation=operation,
                                   byte_range=byte_range)
        common.update({"expected_sha256": capability["expected_sha256"], "expected_size": expected_size,
                       "identity_source": capability["source"], "kind": capability["kind"]})
        self._append({**common, "outcome": "OPEN_PENDING"})
        descriptor = None; observed_sha = None; observed_size = None
        try:
            before = os.lstat(canonical)
            if (before.st_dev, before.st_ino, before.st_size) != (registered.st_dev, registered.st_ino, expected_size):
                self._append({**{**common, "record_type": "POST"}, "observed_sha256": None,
                              "observed_size": before.st_size, "outcome": "PATH_REPLACED"})
                raise RuntimeError("protected path replaced after preclaim check")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(canonical, flags)
            opened = os.fstat(descriptor)
            if (not stat.S_ISREG(opened.st_mode) or
                    (before.st_dev, before.st_ino, expected_size) !=
                    (opened.st_dev, opened.st_ino, opened.st_size)):
                self._append({**{**common, "record_type": "POST"}, "observed_sha256": None,
                              "observed_size": opened.st_size, "outcome": "PATH_REPLACED"})
                raise RuntimeError("protected path replaced before open")
            digest = hashlib.sha256(); observed_size = 0
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block: break
                digest.update(block); observed_size += len(block)
            observed_sha = digest.hexdigest()
            if observed_sha != capability["expected_sha256"] or observed_size != expected_size:
                self._append({**{**common, "record_type": "POST"}, "observed_sha256": observed_sha,
                              "observed_size": observed_size, "outcome": "IDENTITY_MISMATCH"})
                raise ValueError("protected identity mismatch before exposure")
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
                try: result = consumer(handle)
                except Exception as exc:
                    self._append({**{**common, "record_type": "POST"}, "observed_sha256": observed_sha,
                                  "observed_size": observed_size, "outcome": "PARSE_ERROR", "error_type": type(exc).__name__})
                    raise
            after_fd = os.fstat(descriptor)
            try: after_path = os.lstat(canonical)
            except OSError: after_path = None
            if after_path is None or (after_path.st_dev, after_path.st_ino) != (after_fd.st_dev, after_fd.st_ino):
                self._append({**{**common, "record_type": "POST"}, "observed_sha256": observed_sha,
                              "observed_size": observed_size, "outcome": "PATH_REPLACED"})
                raise RuntimeError("protected path replaced during read")
            if after_fd.st_size != expected_size:
                self._append({**{**common, "record_type": "POST"}, "observed_sha256": observed_sha,
                              "observed_size": after_fd.st_size,
                              "outcome": "IDENTITY_MISMATCH"})
                raise RuntimeError("protected descriptor size changed during read")
            self._append({**{**common, "record_type": "POST"}, "observed_sha256": observed_sha,
                          "observed_size": observed_size, "outcome": "SUCCESS"})
            return result
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            self._append({**{**common, "record_type": "POST"}, "observed_sha256": observed_sha,
                          "observed_size": observed_size, "outcome": "READ_ERROR", "error_type": type(exc).__name__})
            raise
        finally:
            if descriptor is not None: os.close(descriptor)

    def read_text(self, path, *, scenario, phase, purpose, encoding="utf-8"):
        return self.consume(path, scenario=scenario, phase=phase, purpose=purpose,
                            consumer=lambda handle: handle.read().decode(encoding), operation="READ_TEXT")

    def read_json(self, path, *, scenario, phase, purpose):
        return self.consume(path, scenario=scenario, phase=phase, purpose=purpose,
                            consumer=lambda handle: json.load(handle), operation="READ_JSON")

    def read_h5(self, path, *, datasets, scenario, phase, purpose):
        names = tuple(datasets)
        def parse(handle):
            with h5py.File(handle, "r") as source:
                missing = set(names) - set(source.keys())
                if missing: raise ValueError(f"HDF5 datasets missing: {sorted(missing)}")
                return {name: source[name][()] for name in names}
        return self.consume(path, scenario=scenario, phase=phase, purpose=purpose, consumer=parse, operation="READ_HDF5")

    def authenticate_manifest(self, path, *, scenario, phase, purpose):
        from .gcspo_capabilities import adapt_manifest_children

        canonical, capability = self._capability(path)
        if capability["kind"] not in {"RECEIVER_MANIFEST", "TRACKING_MANIFEST"} or capability["source"] != "PINNED":
            raise PermissionError("manifest root must be inventory-pinned")
        payload = self.read_json(canonical, scenario=scenario, phase=phase, purpose=purpose)
        adapted = adapt_manifest_children(payload)
        rows = adapted["files"]
        parsed = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or len(str(row.get("sha256", ""))) != 64 or not isinstance(row.get("size_bytes"), int) or row["size_bytes"] <= 0:
                raise ValueError("manifest child identity is incomplete")
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("manifest child path escapes authenticated root")
            child = (canonical.parent / relative).resolve(strict=True)
            if canonical.parent not in child.parents:
                raise ValueError("manifest child path escapes authenticated root")
            info = os.lstat(child)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise PermissionError("manifest child must be a regular non-symlink file")
            if info.st_size != row["size_bytes"]:
                raise ValueError("manifest child declared size mismatch before claim")
            self._allow[str(child)] = {"expected_sha256": row["sha256"].lower(), "expected_size": row["size_bytes"],
                                       "kind": Path(row["path"]).suffix.upper().lstrip(".") or "FILE",
                                       "registered_dev": info.st_dev, "registered_ino": info.st_ino,
                                       "source": f"AUTHENTICATED_MANIFEST:{canonical}"}
            parsed.append(child)
        return {**payload, "_normalized_files": rows,
                "_manifest_adapter": adapted["adapter"]}

    def register_sidecar_children(self, root_manifest, children):
        """Register immutable sidecar children without opening their bytes."""
        canonical, capability = self._capability(root_manifest)
        if capability["kind"] != "RECEIVER_MANIFEST" or capability["source"] != "PINNED":
            raise PermissionError("sidecar root manifest must be inventory-pinned")
        parsed = []
        for row in children:
            if (not isinstance(row, dict) or not isinstance(row.get("canonical_path"), str) or
                    len(str(row.get("sha256", ""))) != 64 or
                    isinstance(row.get("size_bytes"), bool) or
                    not isinstance(row.get("size_bytes"), int) or row["size_bytes"] <= 0):
                raise ValueError("sidecar child identity is incomplete")
            candidate = Path(row["canonical_path"])
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise PermissionError("sidecar child must be a regular non-symlink file")
            child = candidate.resolve(strict=True)
            if canonical.parent not in child.parents:
                raise ValueError("sidecar child escapes authenticated root")
            if info.st_size != row["size_bytes"]:
                raise ValueError("sidecar child declared size mismatch before claim")
            if (info.st_dev, info.st_ino) != (row.get("_preclaim_dev", info.st_dev),
                                              row.get("_preclaim_ino", info.st_ino)):
                raise RuntimeError("sidecar child replaced after preclaim check")
            self._allow[str(child)] = {"expected_sha256": row["sha256"].lower(),
                                       "expected_size": int(row["size_bytes"]),
                                       "kind": child.suffix.upper().lstrip(".") or "FILE",
                                       "registered_dev": info.st_dev, "registered_ino": info.st_ino,
                                       "source": f"IMMUTABLE_CAPABILITY_SIDECAR:{canonical}"}
            parsed.append(child)
        return tuple(parsed)

    def authorize(self, path, *, scenario, phase, expected_sha256, expected_size):
        """Compatibility shim: register a pinned identity; bytes remain unexposed."""
        canonical = self.register_pinned(path, expected_sha256=expected_sha256, expected_size=expected_size, kind="PINNED_FILE")
        return {"canonical_path": str(canonical), "scenario": scenario, "phase": phase, "outcome": "REGISTERED_NOT_READ"}
