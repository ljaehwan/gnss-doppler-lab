from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


EXPECTED = 238


def _score_rows(offset):
    rows = []
    for index in range(EXPECTED):
        epochs = (17525 + index * 25, 17526 + index * 25)
        prns = (3, 7, 11, 19)
        rows.append({
            "window_start_s": 350.5 + index * .5,
            "availability_s": 351.5 + index * .5,
            "score": float(offset + index / 1000),
            "effective_dof": float(offset + 1),
            "prns": list(prns),
            "epoch_ids": list(epochs),
            "epoch_prn_support": [[epoch, list(prns)] for epoch in epochs],
        })
    return rows


def _identity(path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _fixture(root, *, a2_count=EXPECTED, include_a5=True):
    clean = {
        "schema": "gnss-doppler-lab.gcspo-stage0.clean-only-report.v1",
        "run_status": "CLEAN_ONLY_PASS", "attack_access_count": 0,
        "protected_attack_rows_read": False,
        "scores": {"Full_holdout": _score_rows(4), "A1_holdout": _score_rows(1)},
    }
    ablation = {
        "schema": "gnss-doppler-lab.gcspo-stage0.clean-ablation-report.v1",
        "run_status": "CLEAN_ABLATIONS_PASS", "attack_access_count": 0,
        "protected_attack_rows_read": False,
        "methods": {"A2": {"holdout": _score_rows(2)[:a2_count]}},
    }
    a5 = {
        "schema": "gnss-doppler-lab.gcspo-stage0.clean-a5-report.v1",
        "run_status": "CLEAN_A5_PASS", "attack_access_count": 0,
        "protected_attack_rows_read": False,
        "holdout": _score_rows(3),
    }
    documents = {"clean_only_report.json": clean, "clean_ablation_report.json": ablation}
    if include_a5:
        documents["clean_a5_report.json"] = a5
    for name, document in documents.items():
        (root / name).write_text(json.dumps(document, sort_keys=True) + "\n")
    return [_identity(root / name) for name in sorted(documents)]


def test_clean_holdout_cell_is_authenticated_and_present_in_both_reconstruction_paths(tmp_path):
    from gnss_doppler_lab.gcspo_evaluate import load_clean_contrast_rows
    from gnss_doppler_lab.gcspo_verify_artifacts import independently_load_clean_contrast_rows

    identities = _fixture(tmp_path)
    evaluated, source = load_clean_contrast_rows(tmp_path, identities)
    verified, verified_source = independently_load_clean_contrast_rows(tmp_path, identities)
    assert evaluated == verified
    assert source == verified_source
    assert len(evaluated) == 4 * EXPECTED
    assert {row["method"] for row in evaluated} == {"Full", "A1", "A2", "A5"}
    assert {row["scenario"] for row in evaluated} == {"cleanStatic"}
    assert all(row["phase"] == "holdout" and row["label"] is False for row in evaluated)


def test_clean_holdout_cell_rejects_tamper_missing_identity_and_wrong_count(tmp_path):
    from gnss_doppler_lab.gcspo_evaluate import load_clean_contrast_rows
    from gnss_doppler_lab.gcspo_verify_artifacts import independently_load_clean_contrast_rows

    identities = _fixture(tmp_path)
    with (tmp_path / "clean_a5_report.json").open("a") as handle:
        handle.write(" ")
    for loader in (load_clean_contrast_rows, independently_load_clean_contrast_rows):
        with pytest.raises(ValueError, match="identity"):
            loader(tmp_path, identities)

    missing_root = tmp_path / "missing"; missing_root.mkdir()
    missing = _fixture(missing_root, include_a5=False)
    for loader in (load_clean_contrast_rows, independently_load_clean_contrast_rows):
        with pytest.raises(ValueError, match="identity|cell"):
            loader(missing_root, missing)

    count_root = tmp_path / "count"; count_root.mkdir()
    wrong_count = _fixture(count_root, a2_count=EXPECTED - 1)
    for loader in (load_clean_contrast_rows, independently_load_clean_contrast_rows):
        with pytest.raises(ValueError, match="count"):
            loader(count_root, wrong_count)

    wrong_path = [dict(row) for row in wrong_count]
    wrong_path[0]["path"] = str((count_root / "not-the-source.json").resolve())
    for loader in (load_clean_contrast_rows, independently_load_clean_contrast_rows):
        with pytest.raises(ValueError, match="identity"):
            loader(count_root, wrong_path)
