from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_texbat_first_spoofing_model_design_audit.py"
ARTIFACT = REPO_ROOT / "artifacts" / "texbat_first_spoofing_model_design_audit"

SPEC = importlib.util.spec_from_file_location("texbat_design_audit_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class TexbatFirstDesignAuditVerifierTest(unittest.TestCase):
    def copy_artifact(self) -> Path:
        temp_root = Path(tempfile.mkdtemp(prefix="texbat-design-audit-test-"))
        self.addCleanup(shutil.rmtree, temp_root, True)
        copy = temp_root / "artifact"
        shutil.copytree(ARTIFACT, copy)
        return copy

    @staticmethod
    def mutate_json(root: Path, name: str, callback) -> None:
        path = root / name
        document = json.loads(path.read_text(encoding="utf-8"))
        callback(document)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    def assert_semantic_failure(self, root: Path, fragment: str) -> None:
        with self.assertRaisesRegex(VERIFIER.VerificationError, fragment):
            VERIFIER.verify(root, check_manifest=False)

    def test_pristine_semantics_pass(self) -> None:
        summary = VERIFIER.verify(ARTIFACT, check_manifest=False)
        self.assertEqual(summary["models"], 33)
        self.assertEqual(summary["evidence"], 88)
        self.assertEqual(summary["remote_evidence"], 79)
        self.assertEqual(summary["local_only_snapshot_evidence"], 9)
        self.assertEqual(summary["eligible_candidates"], 0)

    def test_prior_final_verdict_misread_is_detected(self) -> None:
        root = self.copy_artifact()

        def mutate(document):
            target = next(row for row in document["models"] if row["model"] == "BITPROBE")
            target["terminal_result"] = "NO_GO_BITPROBE_INITIAL"

        self.mutate_json(root, "prior_experiment_inventory.json", mutate)
        self.assert_semantic_failure(root, "terminal verdict semantics changed")

    def test_implementation_failure_cannot_be_relabelled_physical(self) -> None:
        root = self.copy_artifact()

        def mutate(document):
            target = next(row for row in document["models"] if row["model"] == "CMTE-A2")
            target["failure_type"] = "PHYSICAL_NO_GO"

        self.mutate_json(root, "prior_experiment_inventory.json", mutate)
        self.assert_semantic_failure(root, "CMTE invalid grouping")

    def test_candidate_score_arithmetic_tamper_is_detected(self) -> None:
        root = self.copy_artifact()

        def mutate(document):
            document["candidates"][0]["weighted_total"] = 76.0

        self.mutate_json(root, "candidate_scorecard.json", mutate)
        self.assert_semantic_failure(root, "weighted score arithmetic mismatch")

    def test_selected_model_contradiction_is_detected(self) -> None:
        root = self.copy_artifact()

        def mutate(document):
            document["selected_model"] = "candidate_1"

        self.mutate_json(root, "final_verdict.json", mutate)
        self.assert_semantic_failure(root, "selects a model")

    def test_tuni_attack_access_tamper_is_detected(self) -> None:
        root = self.copy_artifact()

        def mutate(document):
            document["current_audit_operations"]["bytes_read"] = 1

        self.mutate_json(root, "tuni_unopened_preservation.json", mutate)
        self.assert_semantic_failure(root, "nonzero Tuni attack access")

    def test_causal_contract_tamper_is_detected(self) -> None:
        root = self.copy_artifact()

        def mutate(document):
            document["task_contract_preserved"]["post_decision_future_data"] = True

        self.mutate_json(root, "preregistration_draft.json", mutate)
        self.assert_semantic_failure(root, "post_decision_future_data")

    def test_manifest_tamper_is_detected(self) -> None:
        root = self.copy_artifact()
        VERIFIER.write_derived(root)
        with (root / "README.md").open("a", encoding="utf-8") as handle:
            handle.write("\ntamper\n")
        with self.assertRaisesRegex(VERIFIER.VerificationError, "mismatch: README.md"):
            VERIFIER.verify(root, check_manifest=True)


if __name__ == "__main__":
    unittest.main()
