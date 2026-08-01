import numpy as np
from gnss_doppler_lab.gcmr_pi_r4_reproduction import component_agreement


def test_component_agreement_records_error_statistics_alarm_disagreements_and_correlations():
    ref = np.array([0.0, 4.0, 4.0, 9.0])
    actual = np.array([0.0, 2.0, 4.0, 8.0])
    result = component_agreement(ref, actual, threshold=3.0, times=np.array([1.0, 2.0, 3.0, 4.0]))
    assert result["max_abs_error"] == 2.0
    assert result["max_abs_error_time"] == 2.0
    assert result["alarm_agreement_rate"] == 0.75
    assert result["alarm_disagreement_events"] == [2.0]
    assert result["pearson"] is not None and result["spearman"] is not None
