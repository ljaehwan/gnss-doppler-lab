"""Round-5 controller-witness regressions using synthetic /tmp material only.

No test reads a real controller private key, protected payload, access ledger, or
attempt marker.  Every signature is made with an ephemeral Ed25519 key created
below pytest's temporary directory.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


IDENTITY = "gcspo-controller-witness"
NAMESPACE = "gnss-gcspo-controller-witness-v1"
TRUST = (
    "Independence is attested only under the bounded assumption that the external "
    "Hermes controller private key remains outside the VM and that the controller "
    "directly launches and observes each exact command; this is not hardware attestation."
)
OUTPUT_NAMES = (
    "clean_a5_report.json", "thresholds.json", "a5_numeric_trace.json",
    "a5_backend_truth.json",
)


def _canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode() + b"\n"


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(document))


def _identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repo, check=True, text=True,
                          capture_output=True).stdout.strip()


def _key(tmp_path: Path, name: str = "controller") -> tuple[Path, Path, str]:
    assert str(tmp_path).startswith("/tmp/")
    private = tmp_path / name
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name,
                    "-f", str(private)], check=True)
    public = private.with_suffix(".pub")
    fingerprint = subprocess.run(["ssh-keygen", "-lf", str(public)], check=True,
                                 text=True, capture_output=True).stdout.split()[1]
    return private, public, fingerprint


def _sign(path: Path, private: Path, namespace: str = NAMESPACE) -> Path:
    generated = Path(str(path) + ".sig")
    generated.unlink(missing_ok=True)
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(private), "-n", namespace,
                    str(path)], check=True, capture_output=True, text=True)
    return generated


def _resign(path: Path, private: Path, namespace: str = NAMESPACE) -> None:
    signature = Path(str(path) + ".sig")
    signature.unlink(missing_ok=True)
    _sign(path, private, namespace)


def _source_identities(repo: Path) -> dict:
    return {"runner.py": _identity(repo / "runner.py")}


def _make_fixture(tmp_path: Path):
    private, public, fingerprint = _key(tmp_path)
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "round5@test.invalid")
    _git(repo, "config", "user.name", "Round Five")
    (repo / "runner.py").write_text("print('synthetic witnessed run')\n")
    shutil.copy2(public, repo / "controller.pub")
    runs = []
    for run_id, nonce, backend in (
        ("round5-cuda-1", "1" * 64, "cuda"),
        ("round5-cuda-2", "2" * 64, "cuda"),
        ("round5-cpu-1", "3" * 64, "cpu"),
    ):
        scratch = tmp_path / f"scratch-{run_id}"
        evidence = tmp_path / f"evidence-{run_id}"
        argv = ["/synthetic/python", str(repo / "runner.py"), "--run-id", run_id,
                "--nonce", nonce, "--backend", backend,
                "--execution-receipt", str(evidence / "execution_receipt.json")]
        environment = {
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(repo), "TMPDIR": str(scratch / "tmp"),
        }
        runs.append({
            "id": run_id, "nonce": nonce, "backend": backend, "argv": argv, "environment": environment,
            "scratch_root": str(scratch), "output_root": str(scratch / "output"),
            "evidence_root": str(evidence),
            "controller_argv": ["/synthetic/python", "/synthetic/wrapper",
                                "--prepared-envelope", str(tmp_path / f"{run_id}.prepared.json"),
                                "--prepared-signature", str(tmp_path / f"{run_id}.prepared.json.sig"),
                                "--challenge-file", str(repo / "challenge.json")],
        })
    challenge = {
        "schema": "gnss-doppler-lab.gcspo-stage0.a5-controller-challenges.v2",
        "protected_data_access": False,
        "trust": {"public_key_path": "controller.pub", "public_key_fingerprint": fingerprint,
                  "identity": IDENTITY, "namespace": NAMESPACE,
                  "bounded_assumption": TRUST},
        "tolerance": {"absolute": 1e-5, "relative": 1e-8},
        "source_files": ["runner.py"],
        "environment_allowlist": ["PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
                                  "PYTHONPATH", "TMPDIR"],
        "scientific_output_names": list(OUTPUT_NAMES),
        "runs": runs,
    }
    _write(repo / "challenge.json", challenge)
    _git(repo, "add", "runner.py", "controller.pub", "challenge.json")
    _git(repo, "commit", "-qm", "synthetic round-5 challenge")
    source_commit = _git(repo, "rev-parse", "HEAD")
    challenge_identity = _identity(repo / "challenge.json")
    source_identities = _source_identities(repo)
    roots = []
    base = datetime(2026, 8, 14, tzinfo=timezone.utc)
    for index, row in enumerate(runs, 1):
        root = Path(row["evidence_root"])
        scratch = Path(row["scratch_root"])
        output = Path(row["output_root"])
        root.mkdir()
        output.mkdir(parents=True)
        environment = {
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(repo), "TMPDIR": str(scratch / "tmp"),
        }
        prepared_time = base + timedelta(seconds=index)
        start = prepared_time + timedelta(seconds=1)
        end = start + timedelta(seconds=2)
        prepared = {
            "schema": "gnss-doppler-lab.gcspo-stage0.a5-controller-prepared.v2",
            "status": "PREPARED", "run_id": row["id"], "nonce": row["nonce"],
            "source_commit": source_commit, "challenge_path": "challenge.json",
            "challenge": challenge_identity, "source_identities": source_identities,
            "argv": row["argv"], "environment": environment, "backend": row["backend"],
            "scratch_root": row["scratch_root"], "output_root": row["output_root"],
            "evidence_root": row["evidence_root"],
            "prelaunch_assertions": {"scratch_root_absent": True,
                                     "output_root_absent": True,
                                     "evidence_root_absent": True},
            "scientific_output_names": list(OUTPUT_NAMES),
            "controller": {"public_key_fingerprint": fingerprint, "identity": IDENTITY,
                           "namespace": NAMESPACE},
            "prepared_signed_utc": prepared_time.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "bounded_trust_assumption": TRUST,
        }
        prepared_path = root / "prepared.json"
        _write(prepared_path, prepared)
        _sign(prepared_path, private)

        backend_truth = {"schema": "synthetic.backend.v1", "requested": row["backend"],
                         "resolved": row["backend"], "cuda_available": True,
                         "device": "Synthetic GPU" if row["backend"] == "cuda" else "cpu"}
        scientific = {"schema": "synthetic.report.v1", "lambda": 100.0,
                      "thresholds": {"q99": 2.0, "q995": 3.0}, "values": [1.0, 2.0]}
        _write(output / "clean_a5_report.json", scientific)
        _write(output / "thresholds.json", {"A5": scientific["thresholds"]})
        _write(output / "a5_numeric_trace.json", {"backend": row["backend"],
                                                   "values": [1.0, 2.0]})
        _write(output / "a5_backend_truth.json", backend_truth)
        for name in OUTPUT_NAMES:
            shutil.copy2(output / name, root / "bundle" / name) if (root / "bundle").exists() else None
        if not (root / "bundle").exists():
            (root / "bundle").mkdir()
            for name in OUTPUT_NAMES:
                shutil.copy2(output / name, root / "bundle" / name)
        stdout = root / "stdout.txt"
        stderr = root / "stderr.txt"
        stdout.write_text(f"SYNTHETIC_RUN {row['id']}\n")
        stderr.write_text("")
        process_identity = {"pid": 4100 + index, "proc_start_ticks": 90000 + index}
        outputs = {name: _identity(root / "bundle" / name) for name in OUTPUT_NAMES}
        receipt = {
            "schema": "gnss-doppler-lab.gcspo-stage0.a5-execution-receipt.v1",
            "run_id": row["id"], "nonce": row["nonce"],
            "process_identity": process_identity,
            "child_started_utc": (start + timedelta(microseconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "child_finished_utc": (end - timedelta(microseconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "backend_truth": backend_truth, "source_commit": source_commit,
            "challenge": challenge_identity, "argv": row["argv"],
            "scientific_outputs": outputs,
            "transcript_state": {"event_count": 1,
                                 "sha256": hashlib.sha256(b"synthetic-state").hexdigest()},
        }
        receipt_path = root / "execution_receipt.json"
        _write(receipt_path, receipt)
        completed = {
            "schema": "gnss-doppler-lab.gcspo-stage0.a5-controller-completed.v2",
            "status": "COMPLETED", "run_id": row["id"], "nonce": row["nonce"],
            "source_commit": source_commit, "challenge_path": "challenge.json",
            "challenge": challenge_identity,
            "prepared_envelope": _identity(prepared_path),
            "prepared_signature": _identity(Path(str(prepared_path) + ".sig")),
            "argv": row["argv"], "environment": environment, "backend": row["backend"],
            "observed_exit_code": 0, "process_identity": process_identity,
            "observed_process_start_utc": start.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "observed_process_end_utc": end.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "observed_elapsed_ns": 2_000_000_000,
            "stdout": _identity(stdout), "stderr": _identity(stderr),
            "execution_receipt": _identity(receipt_path),
            "backend_truth": _identity(root / "bundle/a5_backend_truth.json"),
            "scientific_outputs": outputs,
            "post_run_source": {"head_commit": source_commit,
                                "worktree_status": {"sha256": hashlib.sha256(b"").hexdigest(),
                                                    "size_bytes": 0},
                                "source_identities": source_identities},
            "controller": {"public_key_fingerprint": fingerprint, "identity": IDENTITY,
                           "namespace": NAMESPACE},
            "completed_signed_utc": (end + timedelta(seconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "bounded_trust_assumption": TRUST,
        }
        completed_path = root / "completed.json"
        _write(completed_path, completed)
        _sign(completed_path, private)
        roots.append(root)
    return {"private": private, "repo": repo, "source_commit": source_commit,
            "roots": roots, "challenge": challenge}


def _verify(fixture):
    from gnss_doppler_lab.gcspo_provenance import verify_witnessed_runs
    return verify_witnessed_runs(fixture["roots"], repo_root=fixture["repo"],
                                 source_commit=fixture["source_commit"],
                                 challenge_path="challenge.json")


def _mutate_signed(path: Path, private: Path, change) -> None:
    document = json.loads(path.read_text())
    change(document)
    _write(path, document)
    _resign(path, private)


def test_valid_external_controller_chains_pass(tmp_path):
    fixture = _make_fixture(tmp_path)
    verified = _verify(fixture)
    assert len(verified["runs"]) == 3
    assert verified["trust"]["bounded_assumption"] == TRUST


def test_prepared_signed_after_observed_process_start_is_rejected(tmp_path):
    fixture = _make_fixture(tmp_path)
    root = fixture["roots"][0]
    completed = json.loads((root / "completed.json").read_text())
    late = (datetime.fromisoformat(completed["observed_process_start_utc"].replace("Z", "+00:00")) +
            timedelta(microseconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z")
    _mutate_signed(root / "prepared.json", fixture["private"],
                   lambda doc: doc.update(prepared_signed_utc=late))
    _mutate_signed(root / "completed.json", fixture["private"], lambda doc: doc.update(
        prepared_envelope=_identity(root / "prepared.json"),
        prepared_signature=_identity(root / "prepared.json.sig")))
    with pytest.raises(ValueError, match="PREPARED|ordering|late"):
        _verify(fixture)


@pytest.mark.parametrize("mode", ["unsigned", "wrong_key", "wrong_namespace", "self_signed"])
def test_untrusted_prepared_envelopes_are_rejected(tmp_path, mode):
    fixture = _make_fixture(tmp_path)
    root = fixture["roots"][0]
    signature = root / "prepared.json.sig"
    if mode == "unsigned":
        signature.unlink()
    else:
        wrong_private, wrong_public, _ = _key(tmp_path, f"wrong-{mode}")
        signature.unlink()
        _sign(root / "prepared.json", wrong_private,
              "wrong-namespace" if mode == "wrong_namespace" else NAMESPACE)
        if mode == "self_signed":
            shutil.copy2(wrong_public, root / "self_authored_controller.pub")
    with pytest.raises(ValueError, match="signature|controller|namespace|pinned"):
        _verify(fixture)


def test_noncanonical_or_modified_json_after_signing_is_rejected(tmp_path):
    fixture = _make_fixture(tmp_path)
    path = fixture["roots"][0] / "prepared.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="canonical|signature"):
        _verify(fixture)


def test_pretty_round5_execution_receipt_remains_rejected_even_if_resigned(tmp_path):
    fixture = _make_fixture(tmp_path)
    root = fixture["roots"][0]
    receipt = root / "execution_receipt.json"
    document = json.loads(receipt.read_text())
    receipt.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    _mutate_signed(
        root / "completed.json",
        fixture["private"],
        lambda completed: completed.update(execution_receipt=_identity(receipt)),
    )
    with pytest.raises(ValueError, match="execution receipt.*canonical"):
        _verify(fixture)


@pytest.mark.parametrize("location", ["argv", "receipt"])
def test_nonce_must_enter_actual_child_invocation_and_receipt(tmp_path, location):
    fixture = _make_fixture(tmp_path)
    root = fixture["roots"][0]
    if location == "argv":
        _mutate_signed(root / "prepared.json", fixture["private"],
                       lambda doc: doc["argv"].remove(doc["nonce"]))
        _mutate_signed(root / "completed.json", fixture["private"], lambda doc: (
            doc.update(argv=json.loads((root / "prepared.json").read_text())["argv"],
                       prepared_envelope=_identity(root / "prepared.json"),
                       prepared_signature=_identity(root / "prepared.json.sig"))))
    else:
        receipt = root / "execution_receipt.json"
        document = json.loads(receipt.read_text())
        document.pop("nonce")
        _write(receipt, document)
        _mutate_signed(root / "completed.json", fixture["private"],
                       lambda doc: doc.update(execution_receipt=_identity(receipt)))
    with pytest.raises(ValueError, match="nonce|argv|receipt"):
        _verify(fixture)


def test_copied_cuda_bundle_with_invented_signed_metadata_is_rejected(tmp_path):
    fixture = _make_fixture(tmp_path)
    first, second = fixture["roots"][:2]
    shutil.rmtree(second / "bundle")
    shutil.copytree(first / "bundle", second / "bundle")
    shutil.copy2(first / "execution_receipt.json", second / "execution_receipt.json")
    completed_path = second / "completed.json"
    _mutate_signed(completed_path, fixture["private"], lambda doc: doc.update(
        execution_receipt=_identity(second / "execution_receipt.json"),
        backend_truth=_identity(second / "bundle/a5_backend_truth.json"),
        scientific_outputs={name: _identity(second / "bundle" / name) for name in OUTPUT_NAMES}))
    with pytest.raises(ValueError, match="receipt|output|replay|independ"):
        _verify(fixture)


@pytest.mark.parametrize("artifact", ["execution_receipt.json", "prepared.json.sig", "completed.json.sig"])
def test_replayed_receipt_or_signature_is_rejected(tmp_path, artifact):
    fixture = _make_fixture(tmp_path)
    first, second = fixture["roots"][:2]
    shutil.copy2(first / artifact, second / artifact)
    with pytest.raises(ValueError, match="signature|receipt|replay|duplicate"):
        _verify(fixture)


def test_completed_signature_predating_process_end_is_rejected(tmp_path):
    fixture = _make_fixture(tmp_path)
    path = fixture["roots"][0] / "completed.json"
    def predate(doc):
        early = (datetime.fromisoformat(doc["observed_process_end_utc"].replace("Z", "+00:00")) -
                 timedelta(microseconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z")
        doc["completed_signed_utc"] = early
    _mutate_signed(path, fixture["private"], predate)
    with pytest.raises(ValueError, match="COMPLETED|ordering|early"):
        _verify(fixture)


@pytest.mark.parametrize("field", ["argv", "source", "backend", "output", "transcript"])
def test_signed_chain_mismatches_are_rejected(tmp_path, field):
    fixture = _make_fixture(tmp_path)
    root = fixture["roots"][0]
    path = root / "completed.json"
    def change(doc):
        if field == "argv":
            doc["argv"] = [*doc["argv"], "--invented"]
        elif field == "source":
            doc["source_commit"] = "f" * 40
        elif field == "backend":
            doc["backend"] = "cpu"
        elif field == "output":
            doc["scientific_outputs"]["clean_a5_report.json"]["sha256"] = "0" * 64
        else:
            doc["stdout"]["sha256"] = "0" * 64
    _mutate_signed(path, fixture["private"], change)
    with pytest.raises(ValueError, match="argv|source|backend|output|transcript|identity|chain"):
        _verify(fixture)


def test_nonzero_exit_is_rejected(tmp_path):
    fixture = _make_fixture(tmp_path)
    path = fixture["roots"][0] / "completed.json"
    _mutate_signed(path, fixture["private"],
                   lambda doc: doc.update(status="FAILED", observed_exit_code=7))
    with pytest.raises(ValueError, match="exit|COMPLETED|success"):
        _verify(fixture)


def test_preexisting_output_assertion_is_rejected(tmp_path):
    fixture = _make_fixture(tmp_path)
    root = fixture["roots"][0]
    _mutate_signed(root / "prepared.json", fixture["private"], lambda doc:
                   doc["prelaunch_assertions"].update(output_root_absent=False))
    _mutate_signed(root / "completed.json", fixture["private"], lambda doc: doc.update(
        prepared_envelope=_identity(root / "prepared.json"),
        prepared_signature=_identity(root / "prepared.json.sig")))
    with pytest.raises(ValueError, match="pre-existing|prelaunch|absent"):
        _verify(fixture)
