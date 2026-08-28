from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_tuni_galileo_prn_support",
    ROOT / "scripts" / "audit_tuni_galileo_prn_support.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_fixed_prn_config_assigns_every_galileo_prn(tmp_path: Path) -> None:
    text = MOD.render_fixed_prn_config(
        iq_path=tmp_path / "input.bin", output_dir=tmp_path / "out", duration_s=10.0
    )
    assert "Channels_1B.count=36" in text
    assert "Channel0.satellite=1" in text
    assert "Channel35.satellite=36" in text
    assert text.count(".satellite=") == 36
    assert "Acquisition_1B.pfa=0.000001" in text
    assert "Tracking_1B.tap_count=9" in text


def test_scenario_roster_matches_readmes() -> None:
    assert {key: sorted(value[1]) for key, value in MOD.SCENARIOS.items()} == {
        "SS-11": [31], "SS-12": [9, 31], "SS-13": [5, 9, 23, 31]
    }
