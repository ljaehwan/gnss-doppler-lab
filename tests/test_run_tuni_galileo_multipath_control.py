from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_tuni_galileo_multipath_control",
    ROOT / "scripts" / "run_tuni_galileo_multipath_control.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_longest_consecutive() -> None:
    assert MOD.longest_consecutive([]) == 0
    assert MOD.longest_consecutive([4, 5, 6, 9, 10]) == 3


def test_auc_ties_and_order() -> None:
    assert MOD.auc([1, 1, 0, 0], [2.0, 1.0, 0.0, 0.0]) == 1.0
    assert MOD.auc([1, 0], [1.0, 1.0]) == 0.5


def test_frozen_roster_and_primary_sets() -> None:
    import json

    config = json.loads(MOD.CONFIG_PATH.read_text(encoding="utf-8"))
    roster = {row["id"]: row["spoofed_prns"] for row in config["scenarios"]}
    assert roster == {
        "SS-1": [9], "SS-3": [6, 9, 23], "SS-5": [4, 6, 9, 23, 31],
        "SS-11": [31], "SS-12": [9, 31], "SS-13": [5, 9, 23, 31],
    }
    assert config["analysis"]["primary_sensitivity_scenarios"] == ["SS-12", "SS-13"]
    assert config["analysis"]["primary_specificity_scenarios"] == ["SS-11", "SS-12", "SS-13"]


def test_aggregate_decision_states() -> None:
    import json

    config = json.loads(MOD.CONFIG_PATH.read_text(encoding="utf-8"))
    rows = []
    labels = {
        "SS-11": [(31, "spoof"), (2, "authentic"), (3, "authentic")],
        "SS-12": [(9, "spoof"), (31, "spoof"), (2, "authentic"), (3, "authentic")],
        "SS-13": [(5, "spoof"), (9, "spoof"), (23, "spoof"), (2, "authentic"), (3, "authentic")],
    }
    for scenario, members in labels.items():
        for prn, label in members:
            rows.append({
                "scenario": scenario, "prn": prn, "label": label,
                "persistent_flag": label == "spoof", "coherent_score_q95": 2.0 if label == "spoof" else 0.2,
            })
    result = MOD.aggregate(config, rows)
    assert result["decision"] == "SUPPORTED"
    assert result["pooled_prn_auc_ss12_ss13"] == 1.0

