"""Round-6 provenance-only successor invariants; no dataset is opened."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts/gcspo_stage0_static_rerun"
ROUND5_SOURCE = "a9a6f03a8fe984ee75c15fbcf81f7c04c5ab2e46"
ROUND5_NONCES = {
    "a55e32c665f3d7cdb574adf742a6c5c4e7a9d4761c0c3ea2d9a73f46b9509a31",
    "8046efcc78ac1bd5d2879ad60c5df9df1a7ca154e3d55ede513e529bf8c9a0b9",
    "2044f7ce6885d42461850f8a478f97a40bfed9c923c70679538d8d9d771ad572",
}
FAILED_ROOTS = {
    "/tmp/gnss-gcspo-round5/round5-cuda-1-scratch",
    "/tmp/gnss-gcspo-round5/round5-cuda-1-evidence",
    "/tmp/gnss-gcspo-round5-controller",
}
OUTPUT_NAMES = [
    "clean_a5_report.json",
    "thresholds.json",
    "a5_numeric_trace.json",
    "a5_backend_truth.json",
]


def _load(name):
    return json.loads((ARTIFACT / name).read_text())


def test_round6_challenge_has_fresh_identities_roots_and_pinned_controller():
    challenge_path = ARTIFACT / "a5_round6_challenges.json"
    challenge = _load(challenge_path.name)
    runs = challenge["runs"]
    assert [row["id"] for row in runs] == [
        "round6-cuda-1", "round6-cuda-2", "round6-cpu-1",
    ]
    nonces = {row["nonce"] for row in runs}
    assert len(nonces) == 3
    assert all(len(nonce) == 64 for nonce in nonces)
    assert nonces.isdisjoint(ROUND5_NONCES)
    assert challenge["scientific_output_names"] == OUTPUT_NAMES
    assert "execution_receipt.json" not in challenge["scientific_output_names"]
    for row in runs:
        assert row["scratch_root"].startswith("/tmp/gnss-gcspo-round6/")
        assert row["output_root"] == f"{row['scratch_root']}/output"
        assert row["evidence_root"].startswith("/tmp/gnss-gcspo-round6/")
        assert row["controller_argv"][3].startswith("/tmp/gnss-gcspo-round6-controller/")
        assert row["controller_argv"][5].startswith("/tmp/gnss-gcspo-round6-controller/")
    encoded = challenge_path.read_text()
    assert not any(root in encoded for root in FAILED_ROOTS)
    assert not any(nonce in encoded for nonce in ROUND5_NONCES)
    assert challenge["trust"] == {
        "bounded_assumption": (
            "Independence is attested only under the bounded assumption that the "
            "external Hermes controller private key remains outside the VM and that "
            "the controller directly launches and observes each exact command; this "
            "is not hardware attestation."
        ),
        "identity": "gcspo-controller-witness",
        "namespace": "gnss-gcspo-controller-witness-v1",
        "public_key_fingerprint": "SHA256:L+STBb5P7+DDfilvAZUxV2eHGYZGfMQf4aDbXSsNi0c",
        "public_key_path": "config/gnss_gcspo_witness_ed25519.pub",
    }


def test_failed_round5_runtime_is_permanently_unsigned_and_excluded():
    failure = _load("round5_runtime_failure.json")
    assert failure["status"] == "REJECTED_UNSIGNED_FAIL_CLOSED"
    assert failure["challenge_source_commit"] == ROUND5_SOURCE
    assert set(failure["immutable_failed_vm_roots"]) == FAILED_ROOTS
    assert failure["completion"] == {
        "completed_envelope_exists": False,
        "completed_signature_exists": False,
        "retrospective_signature_allowed": False,
        "retrospective_acceptance_allowed": False,
    }
    assert failure["execution_receipt"]["actual"] == {
        "size_bytes": 2430,
        "sha256": "cf7801d904c760d86fa642740811039af1e1f2d92fdb7d5d6bff4670a55ede55",
    }
    assert failure["execution_receipt"]["required_compact_canonical_lf"] == {
        "size_bytes": 2082,
        "sha256": "e853071a2737ea17634745a545ae54f2976bfe6716e15e0026fb50c54b8d5721",
    }


def test_scientific_artifact_writer_is_byte_identical_to_round5_source():
    current = (ROOT / "src/gnss_doppler_lab/gcspo_artifacts.py").read_bytes()
    baseline = subprocess.run(
        ["git", "show", f"{ROUND5_SOURCE}:src/gnss_doppler_lab/gcspo_artifacts.py"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    assert current == baseline
