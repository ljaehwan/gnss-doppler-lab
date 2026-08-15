#!/usr/bin/env python3
"""Add deterministic availability and provenance records to the frozen B0-CS bundle.

This reporting-only step never reads attack arrays, changes configuration, or
recomputes scientific scores. It records inputs rejected by the frozen schema,
promotes already-computed block streams into the required block table, and
documents validity limits discovered during verification.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path


METHODS = ["H0", "A0", "A1", "A2", "A3", "A4", "Full", "Linear-AR", "SimpleConsecutive"]
RUN_IDS = [
    "20260815T162404Z-b0-cs-clean-freeze",
    "20260815T162742Z-b0-cs-ds123-evaluation",
    "20260815T162835Z-b0-cs-ds123-evaluation-safe-load",
    "20260815T163122Z-b0-cs-physical-controls-bootstrap-finalize",
    "20260815T163542Z-b0-cs-availability-provenance-report",
]
DS7_PATH = "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/exports/ds7.npz"
DS8_PATH = "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/exports/ds8.npz"
RC9_PATH = "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/rc9-real-clean-domain-poc1/exports/cleanStatic.npz"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_unavailable_metrics(path: Path) -> None:
    fields, rows = read_csv(path)
    if "reason" not in fields:
        fields.append("reason")
    unavailable = {
        "DS4": "No compatible complex nine-tap NPZ with C/N0 and sample/byte lineage was found; no score was computed.",
        "DS7_DS8_family": "The located DS7 and DS8 complex nine-tap NPZs omit cn0_db_hz required by the pre-attack frozen Full pipeline; no post-freeze adapter was permitted.",
    }
    rows = [row for row in rows if row.get("scenario") not in unavailable]
    for scenario, reason in unavailable.items():
        for method in METHODS:
            row = {field: "" for field in fields}
            row.update({
                "scenario": scenario,
                "method": method,
                "status": "UNAVAILABLE",
                "source_kind": "attack",
                "common_epoch_prn_support": "UNAVAILABLE",
                "reason": reason,
            })
            rows.append(row)
    write_csv(path, fields, rows)


def add_unavailable_bootstrap(path: Path) -> None:
    fields, rows = read_csv(path)
    if "reason" not in fields:
        fields.append("reason")
    missing = {
        "DS4": "Scenario scores unavailable because no compatible frozen-schema input exists.",
        "DS7_DS8_family": "Family scores unavailable because both located inputs omit frozen nuisance field cn0_db_hz.",
    }
    rows = [row for row in rows if row.get("scenario") not in missing]
    for scenario, reason in missing.items():
        row = {field: "" for field in fields}
        row.update({
            "scenario": scenario,
            "comparison": "Full-A0 normalized pAUC at FPR<=5%",
            "status": "UNAVAILABLE",
            "repetitions": "0",
            "block_seconds": "10",
            "reason": reason,
        })
        rows.append(row)
    write_csv(path, fields, rows)


def write_block_table(root: Path) -> None:
    source = root / "per_epoch_scores.csv.gz"
    with gzip.open(source, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("method") in {"Full", "Linear-AR"}]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (root / "per_block_scores.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode("utf-8"), mtime=0))


def update_inventory(root: Path) -> None:
    path = root / "data_inventory.json"
    inventory = read_json(path)
    attacks = inventory.setdefault("attack_sources", {})
    attacks["DS4"] = {
        "status": "UNAVAILABLE",
        "kind": "attack",
        "reason": "No compatible complex nine-tap NPZ with cn0_db_hz and raw sample/byte lineage was found in the available artifact inventory.",
        "details": {"required_fields": ["complex_iq", "cn0_db_hz", "sample_count", "prn", "channel", "segment_index", "time_s"]},
    }
    attacks["DS7"] = {
        "status": "UNAVAILABLE",
        "kind": "attack",
        "reason": "Located complex nine-tap input omits cn0_db_hz required by the pre-attack frozen nuisance-conditioned pipeline.",
        "npz": DS7_PATH,
        "npz_sha256": "d0e6da4e27d51e3e96abf2ef7786501124072f28667671e4e40da756eb35f3c8",
        "manifest_sha256": "8dec0643850c9d545b3980ea45bf2c3b9081cd2e314e208faab6bc51fa6e8959",
        "observed_fields": ["channel", "complex_iq", "prn", "sample_count", "segment_index", "time_s"],
        "missing_fields": ["cn0_db_hz"],
    }
    attacks["DS8"] = {
        "status": "UNAVAILABLE",
        "kind": "attack",
        "reason": "Located complex nine-tap input omits cn0_db_hz required by the pre-attack frozen nuisance-conditioned pipeline.",
        "npz": DS8_PATH,
        "npz_sha256": "d1973fa150b7b4e7359df4827f36ce60289f206e9db11c1ac2bc1fd33a0df533",
        "manifest_sha256": "53b240d1c5ecd79c0802840575de5f00ca02131ca449c1cdd9cae30ea6c6f01d",
        "observed_fields": ["channel", "complex_iq", "prn", "sample_count", "segment_index", "time_s"],
        "missing_fields": ["cn0_db_hz"],
    }
    inventory["DS7_DS8_pre110_overlap_audit"] = {
        "status": "UNAVAILABLE",
        "reason": "Both family members failed frozen end-to-end input eligibility before scoring; no pre-110 overlap result is inferred.",
        "details": {"family_grouping": "DS7 and DS8 are one family, never independent confirmations"},
    }
    inventory["external_static_normal"] = {
        "status": "UNAVAILABLE",
        "reason": "No compatible source-distinct static normal sequence was available; deployment-level FPR claims are forbidden.",
        "details": {
            "source_dependent_evidence": "DS3 pre-onset only; DS4 unavailable",
            "rejected_candidate": RC9_PATH,
            "candidate_npz_sha256": "bc0540cb0464bdbeafb13d096829b2f72f85a70cf15453484d19be22320cbe7d",
            "candidate_manifest_sha256": "0c66fa3d9ea1bb6ad41745ec3f712b08fb40d8ed69942905459e97ca339d5d10",
            "rejection_reason": "Receiver replay uses the same TEXBAT cleanStatic raw IQ source as development, SHA-256 dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9; it is not source-distinct.",
        },
    }
    write_json(path, inventory)


def write_external_fpr(root: Path) -> None:
    _, metrics = read_csv(root / "scenario_metrics.csv")
    ds3 = next(row for row in metrics if row.get("scenario") == "DS3" and row.get("method") == "Full")
    fields = [
        "scenario", "method", "status", "reason", "false_positive_rate",
        "worst_run_external_fpr", "evidence_class", "independent_normal_evidence",
    ]
    rows = [
        {
            "scenario": "DS3_pre_onset", "method": "Full", "status": "AVAILABLE",
            "reason": "Frozen Full score evaluated before official 118.9 s signal onset.",
            "false_positive_rate": ds3.get("pre_onset_fpr", ""),
            "worst_run_external_fpr": ds3.get("pre_onset_fpr", ""),
            "evidence_class": "source-dependent external-run pre-onset", "independent_normal_evidence": "false",
        },
        {
            "scenario": "DS4_pre_onset", "method": "Full", "status": "UNAVAILABLE",
            "reason": "No compatible frozen-schema DS4 input was available.",
            "false_positive_rate": "", "worst_run_external_fpr": "",
            "evidence_class": "source-dependent external-run pre-onset", "independent_normal_evidence": "false",
        },
        {
            "scenario": "DS7_DS8_pre_onset", "method": "Full", "status": "UNAVAILABLE",
            "reason": "Located family inputs omit cn0_db_hz; raw-overlap eligibility could not be completed.",
            "false_positive_rate": "", "worst_run_external_fpr": "",
            "evidence_class": "conditional candidate only", "independent_normal_evidence": "false",
        },
        {
            "scenario": "source_distinct_static_normal", "method": "Full", "status": "UNAVAILABLE",
            "reason": "No compatible source-distinct 20-30 minute static sequence was supplied; same-source TEXBAT replay excluded.",
            "false_positive_rate": "", "worst_run_external_fpr": "",
            "evidence_class": "independent normal", "independent_normal_evidence": "false",
        },
    ]
    write_csv(root / "external_static_fpr.csv", fields, rows)


def update_timeline(root: Path) -> None:
    path = root / "timeline_inventory.json"
    timeline = read_json(path)
    timeline["availability"] = {
        "DS1": "AVAILABLE", "DS2": "AVAILABLE", "DS3": "AVAILABLE",
        "DS4": "UNAVAILABLE: no compatible frozen-schema input",
        "DS7_DS8_family": "UNAVAILABLE: located inputs omit cn0_db_hz",
    }
    write_json(path, timeline)


def update_verdict(root: Path) -> None:
    path = root / "final_verdict.json"
    verdict = read_json(path)
    blockers = verdict.setdefault("criteria", {}).setdefault("go_blockers", [])
    additions = [
        "DS4 and DS7/DS8-family Full results unavailable under the frozen input contract",
        "Full did not beat the simple consecutive threshold in two core families",
        "GRU-versus-Linear-AR significance in two core families not established",
    ]
    verdict["criteria"]["go_blockers"] = list(dict.fromkeys([*blockers, *additions]))
    write_json(path, verdict)


def write_limitations(root: Path) -> None:
    write_json(root / "validity_limitations.json", {
        "schema": "gnss-doppler-lab.b0-cs-validity-limitations.v1",
        "full_method": {
            "status": "EMPIRICALLY_BLOCK_CALIBRATED",
            "forbidden_claims": [
                "arbitrary-dependence distribution-free validity", "anytime validity",
                "deployment-level false-positive stability", "independent DS7/DS8 confirmation",
            ],
        },
        "A3_no_nuisance_ablation": {
            "status": "LIMITED",
            "reason": "The frozen reporting implementation applies np.unique after concatenating merged global calibration entries. This avoids repeated merged pools but also collapses genuine tied residual multiplicities, so A3 metrics are descriptive and are not used for the Full verdict.",
            "attack_outcome_retuning": False,
        },
        "unavailable_inputs": ["DS4", "DS7", "DS8", "source-distinct static normal"],
    })


def update_readme(root: Path) -> None:
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    old = "See `scenario_metrics.csv`. DS4 is marked `LIMITED` if it lacks the official 225 s pull-off. DS7 and DS8 are one family."
    new = ("DS3 was available; Full normalized pAUC was 1.0 with zero pre-onset alarms, but its first alarm was 62.1 s after signal onset and much later than Paper-B0/simple consecutive. "
           "DS4 is `UNAVAILABLE` because no compatible nine-tap/C/N0/lineage export was found. DS7/DS8 remain one family and are `UNAVAILABLE` because both located exports omit the frozen C/N0 nuisance input. No configuration was adapted after attack access.")
    if old in text:
        text = text.replace(old, new)
    old_ids = next((line for line in text.splitlines() if line.startswith("Runner run IDs:")), None)
    new_ids = "Runner run IDs: " + ", ".join(RUN_IDS) + ". The failed DS1-3 attempt is retained; it produced no metrics and was retried only with a Torch safe-global loader allowance."
    if old_ids:
        text = text.replace(old_ids, new_ids)
    addendum = ("\n## Structured availability and verification addendum\n\n"
                "`data_inventory.json`, `scenario_metrics.csv`, `ablation_metrics.csv`, `bootstrap_intervals.csv`, and `external_static_fpr.csv` carry explicit DS4 and DS7/DS8-family `UNAVAILABLE` records. The only additional compatible normal candidate was a receiver replay of the same TEXBAT cleanStatic raw IQ and was excluded as non-independent. `validity_limitations.json` also marks the A3 no-nuisance ablation `LIMITED` because tied calibration multiplicities are collapsed in that frozen reporting path; Full is unaffected. The artifact manifest is generated only after this addendum and all verification inputs are final.\n")
    marker = "\nRunner run IDs:"
    if "## Structured availability and verification addendum" not in text and marker in text:
        text = text.replace(marker, addendum + marker)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts/b0_cs_stage0_static")
    args = parser.parse_args()
    root = Path(args.artifact_root).resolve()
    for name in ("scenario_metrics.csv", "ablation_metrics.csv"):
        add_unavailable_metrics(root / name)
    add_unavailable_bootstrap(root / "bootstrap_intervals.csv")
    write_block_table(root)
    update_inventory(root)
    write_external_fpr(root)
    update_timeline(root)
    update_verdict(root)
    write_limitations(root)
    update_readme(root)
    print(json.dumps({"status": "COMPLETE", "artifact_root": str(root), "runner_ids": RUN_IDS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
