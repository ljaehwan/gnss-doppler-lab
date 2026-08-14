"""Externally controller-witnessed clean-A5 execution provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .gcspo_provenance import (
    BOUNDED_TRUST_ASSUMPTION, WITNESS_IDENTITY, WITNESS_NAMESPACE,
    _check_identity, _git, _identity, _timestamp, canonical_json_bytes,
)


PREPARED_KEYS = {
    "schema", "status", "run_id", "nonce", "source_commit", "challenge_path",
    "challenge", "source_identities", "argv", "environment", "backend",
    "scratch_root", "output_root", "evidence_root", "prelaunch_assertions",
    "scientific_output_names", "controller", "prepared_signed_utc",
    "bounded_trust_assumption",
}
COMPLETED_KEYS = {
    "schema", "status", "run_id", "nonce", "source_commit", "challenge_path",
    "challenge", "prepared_envelope", "prepared_signature", "argv", "environment",
    "backend", "observed_exit_code", "process_identity",
    "observed_process_start_utc", "observed_process_end_utc", "observed_elapsed_ns",
    "stdout", "stderr", "execution_receipt", "backend_truth", "scientific_outputs",
    "post_run_source", "controller", "completed_signed_utc",
    "bounded_trust_assumption",
}
RECEIPT_KEYS = {
    "schema", "run_id", "nonce", "process_identity", "child_started_utc",
    "child_finished_utc", "backend_truth", "source_commit", "challenge", "argv",
    "scientific_outputs", "transcript_state",
}


def _require_keys(document, expected, label):
    if not isinstance(document, dict) or set(document) != set(expected):
        missing = sorted(set(expected) - set(document)) if isinstance(document, dict) else sorted(expected)
        extra = sorted(set(document) - set(expected)) if isinstance(document, dict) else []
        raise ValueError(f"provenance {label} schema keys mismatch: missing={missing} extra={extra}")


def _canonical_document(path, label):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"provenance signed {label} envelope is absent")
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
        expected = canonical_json_bytes(document)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"provenance signed {label} envelope is not canonical JSON") from exc
    if payload != expected:
        raise ValueError(f"provenance signed {label} envelope is not exact canonical JSON")
    return document


def _public_fingerprint(path):
    try:
        result = subprocess.run(["ssh-keygen", "-lf", str(path)], check=True,
                                capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("provenance pinned controller public key is invalid") from exc
    fields = result.stdout.split()
    if len(fields) < 2 or not fields[1].startswith("SHA256:"):
        raise ValueError("provenance pinned controller public key fingerprint is invalid")
    return fields[1]


def _verify_signature(envelope_path, signature_path, *, public_key_path,
                      fingerprint, identity, namespace, label):
    envelope = Path(envelope_path)
    signature = Path(signature_path)
    if not signature.is_file() or not signature.read_bytes():
        raise ValueError(f"provenance controller {label} signature is absent")
    public_line = Path(public_key_path).read_text().strip().split()
    if len(public_line) < 2 or public_line[0] != "ssh-ed25519":
        raise ValueError("provenance pinned controller public key is malformed")
    if _public_fingerprint(public_key_path) != fingerprint:
        raise ValueError("provenance pinned controller public key fingerprint mismatch")
    allowed = None
    try:
        with tempfile.NamedTemporaryFile("w", prefix="gcspo-witness-", suffix=".allowed",
                                         dir="/tmp", delete=False) as handle:
            handle.write(f"{identity} {public_line[0]} {public_line[1]}\n")
            handle.flush()
            os.fsync(handle.fileno())
            allowed = Path(handle.name)
        result = subprocess.run(
            ["ssh-keygen", "-Y", "verify", "-f", str(allowed), "-I", identity,
             "-n", namespace, "-s", str(signature)], input=envelope.read_bytes(),
            capture_output=True)
    except OSError as exc:
        raise ValueError("provenance deterministic Ed25519 verifier is unavailable") from exc
    finally:
        if allowed is not None:
            allowed.unlink(missing_ok=True)
    if result.returncode != 0:
        raise ValueError(f"provenance controller {label} signature verification failed")


def _challenge(repo_root, source_commit, challenge_path):
    repo = Path(repo_root).resolve(strict=True)
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("provenance witness source commit is invalid")
    current = Path(challenge_path)
    if not current.is_absolute():
        current = repo / current
    relative = current.resolve(strict=True).relative_to(repo).as_posix()
    committed = _git(repo, "show", f"{source_commit}:{relative}", binary=True)
    if current.read_bytes() != committed:
        raise ValueError("provenance witness challenge source mismatch")
    challenge = json.loads(committed)
    if challenge.get("schema") != "gnss-doppler-lab.gcspo-stage0.a5-controller-challenges.v2":
        raise ValueError("provenance witness challenge schema mismatch")
    trust = challenge.get("trust")
    _require_keys(trust, {"public_key_path", "public_key_fingerprint", "identity",
                          "namespace", "bounded_assumption"}, "witness trust")
    if (trust["identity"] != WITNESS_IDENTITY or trust["namespace"] != WITNESS_NAMESPACE or
            trust["bounded_assumption"] != BOUNDED_TRUST_ASSUMPTION):
        raise ValueError("provenance pinned controller identity/namespace/trust mismatch")
    key_relative = trust["public_key_path"]
    if (not isinstance(key_relative, str) or Path(key_relative).is_absolute() or
            ".." in Path(key_relative).parts):
        raise ValueError("provenance pinned controller public key path is invalid")
    key_path = (repo / key_relative).resolve(strict=True)
    if key_path.relative_to(repo) != Path(key_relative):
        raise ValueError("provenance pinned controller public key path escapes repository")
    key_committed = _git(repo, "show", f"{source_commit}:{key_relative}", binary=True)
    if key_path.read_bytes() != key_committed:
        raise ValueError("provenance pinned controller public key source mismatch")
    if _public_fingerprint(key_path) != trust["public_key_fingerprint"]:
        raise ValueError("provenance pinned controller public key fingerprint mismatch")
    tolerance = challenge.get("tolerance")
    if (not isinstance(tolerance, dict) or set(tolerance) != {"absolute", "relative"} or
            any(isinstance(tolerance[name], bool) or
                not isinstance(tolerance[name], (int, float)) or tolerance[name] <= 0
                for name in tolerance)):
        raise ValueError("provenance preregistered parity tolerance is invalid")
    allowlist = challenge.get("environment_allowlist")
    if (not isinstance(allowlist, list) or not allowlist or
            len(allowlist) != len(set(allowlist)) or
            any(not isinstance(name, str) or not name for name in allowlist)):
        raise ValueError("provenance environment allowlist is invalid")
    output_names = challenge.get("scientific_output_names")
    if (not isinstance(output_names, list) or not output_names or
            len(output_names) != len(set(output_names)) or
            any(Path(name).name != name for name in output_names)):
        raise ValueError("provenance scientific output contract is invalid")
    runs = challenge.get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise ValueError("provenance witnessed run set must contain exactly three runs")
    ids, nonces, roots, commands, controller_commands = [], [], [], [], []
    for row in runs:
        _require_keys(row, {"id", "nonce", "backend", "argv", "scratch_root",
                            "output_root", "evidence_root", "controller_argv", "environment"}, "witness run")
        ids.append(row["id"]); nonces.append(row["nonce"]); commands.append(tuple(row["argv"]))
        controller_commands.append(tuple(row["controller_argv"]))
        roots.extend([row["scratch_root"], row["output_root"], row["evidence_root"]])
        if (row["backend"] not in {"cpu", "cuda"} or
                not isinstance(row["nonce"], str) or len(row["nonce"]) != 64 or
                not isinstance(row["argv"], list) or not row["argv"] or
                not isinstance(row["controller_argv"], list) or not row["controller_argv"] or
                any(not isinstance(value, str) or not value for value in row["argv"]) or
                any(not isinstance(value, str) or not value for value in row["controller_argv"])):
            raise ValueError("provenance witnessed run challenge is invalid")
        if (not isinstance(row["environment"], dict) or set(row["environment"]) != set(allowlist) or
                any(not isinstance(value, str) for value in row["environment"].values())):
            raise ValueError("provenance witnessed run environment is invalid")
        argv = row["argv"]
        try:
            argv_ok = (argv[argv.index("--run-id") + 1] == row["id"] and
                       argv[argv.index("--nonce") + 1] == row["nonce"] and
                       argv[argv.index("--backend") + 1] == row["backend"])
        except (ValueError, IndexError):
            argv_ok = False
        if not argv_ok:
            raise ValueError("provenance nonce/run/backend absent from actual child argv")
        scratch, output, evidence = map(Path, (row["scratch_root"], row["output_root"],
                                               row["evidence_root"]))
        if (not scratch.is_absolute() or not output.is_absolute() or not evidence.is_absolute() or
                output != scratch / "output" or scratch == evidence or
                scratch in evidence.parents or evidence in scratch.parents):
            raise ValueError("provenance witnessed fresh roots are invalid")
    if (len(ids) != len(set(ids)) or len(nonces) != len(set(nonces)) or
            len(commands) != len(set(commands)) or len(roots) != len(set(roots)) or
            len(controller_commands) != len(set(controller_commands)) or
            sorted(row["backend"] for row in runs) != ["cpu", "cuda", "cuda"]):
        raise ValueError("provenance witnessed run independence challenge is invalid")
    source = {}
    for item in challenge.get("source_files", []):
        if not isinstance(item, str) or Path(item).is_absolute() or ".." in Path(item).parts:
            raise ValueError("provenance witnessed source path is invalid")
        payload = _git(repo, "show", f"{source_commit}:{item}", binary=True)
        path = repo / item
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError("provenance witnessed source mismatch")
        source[item] = {"sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload)}
    if not source:
        raise ValueError("provenance witnessed source identities are absent")
    return {"repo": repo, "document": challenge, "relative": relative,
            "identity": _identity(current, allow_empty=False), "source": source,
            "trust": trust, "public_key": key_path, "source_commit": source_commit}


def _controller_fields(trust):
    return {"public_key_fingerprint": trust["public_key_fingerprint"],
            "identity": trust["identity"], "namespace": trust["namespace"]}


def _validate_prepared(document, context, row):
    _require_keys(document, PREPARED_KEYS, "PREPARED")
    if (document["schema"] != "gnss-doppler-lab.gcspo-stage0.a5-controller-prepared.v2" or
            document["status"] != "PREPARED"):
        raise ValueError("provenance PREPARED schema/state mismatch")
    if (document["source_commit"] != context["source_commit"] or
            document["challenge_path"] != context["relative"] or
            document["challenge"] != context["identity"] or
            document["source_identities"] != context["source"]):
        raise ValueError("provenance PREPARED source/challenge binding mismatch")
    for source_name, prepared_name in (("id", "run_id"), ("nonce", "nonce"),
                                       ("backend", "backend"), ("argv", "argv"),
                                       ("scratch_root", "scratch_root"),
                                       ("output_root", "output_root"),
                                       ("evidence_root", "evidence_root"),
                                       ("environment", "environment")):
        if document[prepared_name] != row[source_name]:
            raise ValueError(f"provenance PREPARED {prepared_name} challenge mismatch")
    argv = document["argv"]
    try:
        nonce_bound = (argv[argv.index("--run-id") + 1] == document["run_id"] and
                       argv[argv.index("--nonce") + 1] == document["nonce"])
    except (ValueError, IndexError):
        nonce_bound = False
    if not nonce_bound:
        raise ValueError("provenance nonce is absent from actual child argv")
    environment = document["environment"]
    if (not isinstance(environment, dict) or
            set(environment) != set(context["document"]["environment_allowlist"]) or
            any(not isinstance(value, str) for value in environment.values())):
        raise ValueError("provenance PREPARED environment allowlist mismatch")
    if document["prelaunch_assertions"] != {"scratch_root_absent": True,
                                              "output_root_absent": True,
                                              "evidence_root_absent": True}:
        raise ValueError("provenance PREPARED prelaunch absent assertion rejects pre-existing output")
    if document["scientific_output_names"] != context["document"]["scientific_output_names"]:
        raise ValueError("provenance PREPARED scientific output contract mismatch")
    if (document["controller"] != _controller_fields(context["trust"]) or
            document["bounded_trust_assumption"] != BOUNDED_TRUST_ASSUMPTION):
        raise ValueError("provenance PREPARED pinned controller trust mismatch")
    _timestamp(document["prepared_signed_utc"], "PREPARED signed")


def load_signed_prepared(prepared_path, signature_path, *, repo_root, source_commit,
                         challenge_path):
    context = _challenge(repo_root, source_commit, challenge_path)
    prepared = _canonical_document(prepared_path, "PREPARED")
    rows = [row for row in context["document"]["runs"] if row["id"] == prepared.get("run_id")]
    if len(rows) != 1:
        raise ValueError("provenance PREPARED run ID is not challenged")
    _validate_prepared(prepared, context, rows[0])
    _verify_signature(prepared_path, signature_path, public_key_path=context["public_key"],
                      fingerprint=context["trust"]["public_key_fingerprint"],
                      identity=context["trust"]["identity"],
                      namespace=context["trust"]["namespace"], label="PREPARED")
    return {"document": prepared, "row": rows[0], "context": context}


def validate_prepared_for_launch(prepared_path, signature_path, *, repo_root,
                                 source_commit, challenge_path, observed_start_utc=None):
    loaded = load_signed_prepared(prepared_path, signature_path, repo_root=repo_root,
                                  source_commit=source_commit, challenge_path=challenge_path)
    prepared, context = loaded["document"], loaded["context"]
    if _git(context["repo"], "rev-parse", "HEAD") != source_commit:
        raise ValueError("provenance launch source HEAD mismatch")
    if _git(context["repo"], "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("provenance launch worktree is not clean")
    for label in ("scratch_root", "output_root", "evidence_root"):
        if Path(prepared[label]).exists():
            raise ValueError(f"provenance pre-existing {label} rejects launch")
    if (observed_start_utc is not None and
            _timestamp(prepared["prepared_signed_utc"], "PREPARED signed") >
            _timestamp(observed_start_utc, "observed process start")):
        raise ValueError("provenance late PREPARED signature rejects launch")
    return loaded


def verify_witnessed_runs(evidence_roots, *, repo_root, source_commit, challenge_path):
    context = _challenge(repo_root, source_commit, challenge_path)
    expected = {row["id"]: row for row in context["document"]["runs"]}
    roots = [Path(root).resolve(strict=True) for root in evidence_roots]
    if len(roots) != len(expected) or len(roots) != len(set(roots)):
        raise ValueError("provenance witnessed evidence roots are incomplete or duplicated")
    verified = []
    for root in roots:
        prepared_path, prepared_sig = root / "prepared.json", root / "prepared.json.sig"
        completed_path, completed_sig = root / "completed.json", root / "completed.json.sig"
        prepared = _canonical_document(prepared_path, "PREPARED")
        row = expected.get(prepared.get("run_id"))
        if row is None:
            raise ValueError("provenance PREPARED run ID is not challenged")
        _validate_prepared(prepared, context, row)
        _verify_signature(prepared_path, prepared_sig, public_key_path=context["public_key"],
                          fingerprint=context["trust"]["public_key_fingerprint"],
                          identity=context["trust"]["identity"],
                          namespace=context["trust"]["namespace"], label="PREPARED")
        completed = _canonical_document(completed_path, "COMPLETED")
        _require_keys(completed, COMPLETED_KEYS, "COMPLETED")
        if (completed["schema"] != "gnss-doppler-lab.gcspo-stage0.a5-controller-completed.v2" or
                completed["status"] != "COMPLETED" or completed["observed_exit_code"] != 0):
            raise ValueError("provenance COMPLETED success exit is absent")
        _verify_signature(completed_path, completed_sig, public_key_path=context["public_key"],
                          fingerprint=context["trust"]["public_key_fingerprint"],
                          identity=context["trust"]["identity"],
                          namespace=context["trust"]["namespace"], label="COMPLETED")
        if (completed["run_id"] != prepared["run_id"] or
                completed["nonce"] != prepared["nonce"] or
                completed["source_commit"] != source_commit or
                completed["challenge_path"] != context["relative"] or
                completed["challenge"] != context["identity"] or
                completed["argv"] != prepared["argv"] or
                completed["environment"] != prepared["environment"] or
                completed["backend"] != prepared["backend"]):
            raise ValueError("provenance signed COMPLETED/PREPARED chain argv/source/backend mismatch")
        if (completed["prepared_envelope"] != _identity(prepared_path, allow_empty=False) or
                completed["prepared_signature"] != _identity(prepared_sig, allow_empty=False)):
            raise ValueError("provenance signed PREPARED envelope/signature chain identity mismatch")
        if (completed["controller"] != _controller_fields(context["trust"]) or
                completed["bounded_trust_assumption"] != BOUNDED_TRUST_ASSUMPTION):
            raise ValueError("provenance COMPLETED pinned controller trust mismatch")
        prepared_time = _timestamp(prepared["prepared_signed_utc"], "PREPARED signed")
        observed_start = _timestamp(completed["observed_process_start_utc"], "observed process start")
        observed_end = _timestamp(completed["observed_process_end_utc"], "observed process end")
        completed_time = _timestamp(completed["completed_signed_utc"], "COMPLETED signed")
        if not (prepared_time <= observed_start < observed_end <= completed_time):
            raise ValueError("provenance PREPARED/process/COMPLETED timestamp ordering mismatch")
        if (isinstance(completed["observed_elapsed_ns"], bool) or
                not isinstance(completed["observed_elapsed_ns"], int) or
                completed["observed_elapsed_ns"] <= 0):
            raise ValueError("provenance observed process duration is not positive")
        process_identity = completed["process_identity"]
        if (not isinstance(process_identity, dict) or
                set(process_identity) != {"pid", "proc_start_ticks"} or
                any(isinstance(process_identity[name], bool) or
                    not isinstance(process_identity[name], int) or process_identity[name] <= 0
                    for name in process_identity)):
            raise ValueError("provenance observed process identity is invalid")
        _check_identity(root / "stdout.txt", completed["stdout"], "stdout transcript")
        _check_identity(root / "stderr.txt", completed["stderr"], "stderr transcript")
        receipt_path = root / "execution_receipt.json"
        _check_identity(receipt_path, completed["execution_receipt"], "execution receipt",
                        allow_empty=False)
        receipt = _canonical_document(receipt_path, "execution receipt")
        _require_keys(receipt, RECEIPT_KEYS, "execution receipt")
        if (receipt["schema"] != "gnss-doppler-lab.gcspo-stage0.a5-execution-receipt.v1" or
                receipt["run_id"] != prepared["run_id"] or receipt["nonce"] != prepared["nonce"] or
                receipt["argv"] != prepared["argv"] or receipt["process_identity"] != process_identity or
                receipt["source_commit"] != source_commit or receipt["challenge"] != context["identity"]):
            raise ValueError("provenance execution receipt nonce/argv/process/source chain mismatch")
        child_start = _timestamp(receipt["child_started_utc"], "child start")
        child_end = _timestamp(receipt["child_finished_utc"], "child finish")
        if not (observed_start <= child_start < child_end <= observed_end):
            raise ValueError("provenance execution receipt process interval mismatch")
        output_dir = root / "bundle"
        outputs = completed["scientific_outputs"]
        if set(outputs) != set(prepared["scientific_output_names"]):
            raise ValueError("provenance signed scientific output set mismatch")
        for name, identity in outputs.items():
            _check_identity(output_dir / name, identity, f"scientific output:{name}",
                            allow_empty=False)
        if receipt["scientific_outputs"] != outputs:
            raise ValueError("provenance execution receipt output binding mismatch")
        _check_identity(output_dir / "a5_backend_truth.json", completed["backend_truth"],
                        "backend truth", allow_empty=False)
        truth = json.loads((output_dir / "a5_backend_truth.json").read_text())
        if (receipt["backend_truth"] != truth or truth.get("requested") != prepared["backend"] or
                truth.get("resolved") != prepared["backend"]):
            raise ValueError("provenance execution receipt backend truth mismatch")
        transcript_state = receipt["transcript_state"]
        if (not isinstance(transcript_state, dict) or
                set(transcript_state) != {"event_count", "sha256"} or
                isinstance(transcript_state["event_count"], bool) or
                not isinstance(transcript_state["event_count"], int) or
                transcript_state["event_count"] <= 0 or
                not isinstance(transcript_state["sha256"], str) or
                len(transcript_state["sha256"]) != 64):
            raise ValueError("provenance execution receipt transcript state is invalid")
        post = completed["post_run_source"]
        if (not isinstance(post, dict) or
                set(post) != {"head_commit", "worktree_status", "source_identities"} or
                post["head_commit"] != source_commit or post["source_identities"] != context["source"] or
                post["worktree_status"] != {"sha256": hashlib.sha256(b"").hexdigest(),
                                                "size_bytes": 0}):
            raise ValueError("provenance signed post-run source/worktree identity mismatch")
        verified.append({"root": root, "prepared": prepared, "completed": completed,
                         "receipt": receipt, "output_dir": output_dir,
                         "prepared_signature": _identity(prepared_sig),
                         "completed_signature": _identity(completed_sig)})
    uniqueness = {
        "nonce": [item["prepared"]["nonce"] for item in verified],
        "PREPARED envelope": [item["completed"]["prepared_envelope"]["sha256"] for item in verified],
        "PREPARED signature": [item["prepared_signature"]["sha256"] for item in verified],
        "COMPLETED signature": [item["completed_signature"]["sha256"] for item in verified],
        "process identity": [json.dumps(item["completed"]["process_identity"], sort_keys=True) for item in verified],
        "process interval": [item["completed"]["observed_process_start_utc"] + "/" +
                             item["completed"]["observed_process_end_utc"] for item in verified],
        "execution receipt": [item["completed"]["execution_receipt"]["sha256"] for item in verified],
    }
    for label, values in uniqueness.items():
        if len(values) != len(set(values)):
            raise ValueError(f"provenance duplicate/replayed {label} rejects independence")
    if sorted(item["prepared"]["backend"] for item in verified) != ["cpu", "cuda", "cuda"]:
        raise ValueError("provenance witnessed backend run count mismatch")
    return {"source_commit": source_commit, "challenge": context["document"],
            "trust": context["trust"], "runs": verified,
            "independence": {"status": "EXTERNALLY_WITNESSED", "distinct": list(uniqueness)}}
