from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "gnss_normal_tracking_workflow.ipynb"


def test_single_research_notebook_has_linear_normal_only_structure() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    cells = notebook["cells"]
    assert cells[0]["cell_type"] == "markdown"
    assert all(cell.get("id") for cell in cells)
    assert len({cell["id"] for cell in cells}) == len(cells)
    text = "\n".join("".join(cell["source"]) for cell in cells)
    required = [
        "연구 목적",
        "전체 연구 순서",
        "진행 상황",
        "generate_iq.py",
        "RUN_GENERATION",
        "RUN_RECEIVER",
        "GNSS_SDR_PATH",
        "SCENARIO_NAME",
        "SCENARIO_UTC",
        "POSITION_MODE",
        "LATITUDE_DEG",
        "LONGITUDE_DEG",
        "ALTITUDE_M",
        "TRAJECTORY_FILE",
        "DURATION_SECONDS",
        "RF_SAMPLE_RATE_HZ",
        "RINEX_NAV_PATH",
        "정상 IQ 기초 검증",
        "Doppler truth",
        "GNSS-SDR",
        "tracking correlator peak",
        "PEAK_RECEIVER_RUN",
        "PEAK_PRN",
        "정상-only 특징 데이터셋",
        "공개 외부 raw 데이터",
        "최종 판정",
    ]
    for phrase in required:
        assert phrase in text, f"missing: {phrase}"


def test_only_one_source_research_notebook_is_kept() -> None:
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks == [NOTEBOOK]


def test_notebook_does_not_embed_machine_specific_paths_or_outputs() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "/home/ubuntu/projects" not in text
    assert "/opt/data/gnss-doppler-lab" not in text
    notebook = json.loads(text)
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
