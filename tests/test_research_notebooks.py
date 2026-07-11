from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    "00_research_sequence.ipynb",
    "01_normal_iq_generation_and_inspection.ipynb",
    "02_gnss_sdr_receiver_analysis.ipynb",
    "03_normal_vs_spoofing_comparison.ipynb",
    "04_detection_dataset_and_baselines.ipynb",
]


def test_research_notebooks_have_ordered_executable_structure() -> None:
    for filename in EXPECTED:
        path = REPO_ROOT / "notebooks" / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        cells = notebook["cells"]
        assert cells[0]["cell_type"] == "markdown"
        assert all(cell.get("id") for cell in cells), f"{filename} has cells without stable IDs"
        assert len({cell["id"] for cell in cells}) == len(cells)
        assert any(cell["cell_type"] == "code" for cell in cells)
        text = "\n".join("".join(cell["source"]) for cell in cells)
        for section in ("연구 목적", "입력", "중간 확인", "판정", "다음 단계"):
            assert section in text, f"{filename} is missing {section}"


def test_notebooks_do_not_embed_machine_specific_project_path() -> None:
    for filename in EXPECTED:
        text = (REPO_ROOT / "notebooks" / filename).read_text(encoding="utf-8")
        assert "/home/ubuntu/projects" not in text
        assert "/opt/data/gnss-doppler-lab" not in text
