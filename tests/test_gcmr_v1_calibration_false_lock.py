import importlib.util
from pathlib import Path
import numpy as np, pytest
S=Path(__file__).parents[1]/"scripts/diagnose_gcmr_v1_calibration_false_lock.py"
spec=importlib.util.spec_from_file_location("diag",S);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_qc_conjunction_and_no_identity_constants():
 assert m.invalid_qc(501,.49,34.9)
 assert not m.invalid_qc(500,.49,34.9)
 assert not m.invalid_qc(501,.5,34.9)
 assert not m.invalid_qc(501,.49,35)
 assert "PRN8" not in S.read_text().replace(" ","").upper()
def test_overlap_half_open():
 assert m.overlaps(339.5,340.5,340,340.5)
 assert not m.overlaps(339,340,340,340.5)
def test_remove_incident_pairs_and_support_fail_closed():
 p=np.array([[3,6],[3,8],[6,8],[6,10],[8,10],[10,11],[11,12]])
 keep=m.incident_pair_mask(p,{8}); assert p[keep].tolist()==[[3,6],[6,10],[10,11],[11,12]]
 m.require_support(p[keep],4,4)
 with pytest.raises(ValueError,match="support"):m.require_support(p[keep],4,6)


def test_topology_control_uniqueness_contract():
 assert m.root_cause_uniqueness(1.7693133951565987, 100.0, min_ratio=10.0, min_difference=10.0)
 assert not m.root_cause_uniqueness(1.7693133951565987, 2.0, min_ratio=10.0, min_difference=10.0)

def test_manifest_and_provenance_contract():
 assert m.QC_RULE == {"residual_abs_hz": {"operator": ">", "threshold": 500.0}, "median_carrier_lock_test": {"operator": "<", "threshold": 0.5}, "median_cn0_db_hz": {"operator": "<", "threshold": 35.0}, "conjunction": "all", "strict": True}
 assert m.QC_BINNING["width_s"] == 0.5
 assert m.write_sha256_manifest
