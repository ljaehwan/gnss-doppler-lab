import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "preflight_cgc_code_carrier_fresh_static",
    ROOT / "scripts" / "preflight_cgc_code_carrier_fresh_static.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
np_norm = module.np_norm


def test_frozen_offsets_have_100_m_norm() -> None:
    assert np_norm([100, 0, 0]) == 100
    assert np_norm([-80, 60, 0]) == 100
    assert np_norm([60, 0, 80]) == 100
