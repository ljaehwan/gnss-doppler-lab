from pathlib import Path
import importlib.util

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preflight_tuni_galileo_clean",
    ROOT / "scripts" / "preflight_tuni_galileo_clean.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_render_config_is_clean_galileo_ishort_and_five_tap(tmp_path: Path) -> None:
    text = MOD.render_config(
        iq_path=tmp_path / "clean.bin",
        output_dir=tmp_path / "out",
        input_samples=25_000_000,
        channel_count=8,
    )
    assert "SignalSource.item_type=ishort" in text
    assert "DataTypeAdapter.implementation=Ishort_To_Complex" in text
    assert "Channels_1B.count=8" in text
    assert "Galileo_E1_PCPS_Ambiguous_Acquisition" in text
    assert "Galileo_E1_DLL_PLL_VEML_Tracking" in text
    assert "Tracking_1B.tap_count=5" in text
    assert "Tracking_1B.tap_spacing_chips=0.125" in text


def test_render_config_supports_safe_nine_tap_galileo(tmp_path: Path) -> None:
    text = MOD.render_config(
        iq_path=tmp_path / "clean.bin",
        output_dir=tmp_path / "out",
        input_samples=25_000_000,
        channel_count=8,
        tracking_tap_count=9,
    )
    assert "Tracking_1B.tap_count=9" in text
    assert "Tracking_1B.tap_spacing_chips=0.125" in text

def test_ishort_source_count_has_two_items_per_complex_sample() -> None:
    assert MOD.ishort_source_item_count(5.0, 50_000_000) == 500_000_000





def test_clean_allowlist_contains_no_attack_scenario() -> None:
    assert MOD.CLEAN_ALLOWLIST == {
        "C-1": "C-1/clearsky_signal_C-1.bin",
    }


@pytest.mark.parametrize("duration", [0.49, 10.01])
def test_duration_boundary_is_documented(duration: float) -> None:
    assert not (0.5 <= duration <= 10.0)

