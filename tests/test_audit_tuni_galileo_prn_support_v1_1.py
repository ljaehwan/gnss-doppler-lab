from pathlib import Path
import importlib.util

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_tuni_galileo_prn_support_v1_1",
    ROOT / "scripts" / "audit_tuni_galileo_prn_support_v1_1.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_two_banks_exhaust_all_prns_without_duplicates() -> None:
    assert [prn for bank in MOD.PRN_BANKS for prn in bank] == list(range(1, 37))


def test_fixed_prn_config_assigns_bank_and_preserves_receiver(tmp_path: Path) -> None:
    text = MOD.render_fixed_prn_config(
        iq_path=tmp_path / "input.bin", output_dir=tmp_path / "out",
        duration_s=10.0, prns=MOD.PRN_BANKS[1],
    )
    assert "Channels_1B.count=18" in text
    assert "Channel0.satellite=19" in text
    assert "Channel17.satellite=36" in text
    assert text.count(".satellite=") == 18
    assert "Acquisition_1B.pfa=0.000001" in text
    assert "Tracking_1B.tap_count=9" in text


def test_empty_mat_sentinel_is_not_prn_one(tmp_path: Path) -> None:
    path = tmp_path / "epl_tracking_ch_0.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("PRN", data=np.array([1, 0]))
    assert MOD.channel_support([path]) == []
