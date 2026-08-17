import numpy as np

from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.mosaic_alignment import navigation_bit_provenance, sample_bounds_status


def test_canonical_prn_known_vector_and_1023_chips():
    code = gps_l1ca_code(1)
    assert code.shape == (1023,)
    assert code[:16].tolist() == [-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
    assert set(np.unique(code)) == {-1.0, 1.0}


def test_sample_bounds_status():
    rows = [{"raw_sample_start": 0, "raw_sample_end": 25000}, {"raw_sample_start": 25000, "raw_sample_end": 50000}]
    assert sample_bounds_status(rows, raw_size_bytes=100000)["status"] == "PASS"
    bad = rows + [{"raw_sample_start": 60000, "raw_sample_end": 60001}]
    assert sample_bounds_status(bad, raw_size_bytes=100000)["status"] == "FAIL"


def test_navigation_bit_provenance_fails_closed():
    result = navigation_bit_provenance([{"navigation_bit_wipeoff_applied": True}])
    assert result["status"] == "UNAVAILABLE"
    assert "+1 fallback" in result["reason"]
