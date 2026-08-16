from gnss_doppler_lab.trace_native_cadence import trace_r1_verdict


def test_needs_dump_when_retained_mapping_is_invalid():
    assert trace_r1_verdict(mapping_verified=False, native_dump_available=False, real_scores_exist=False) == "NEEDS_TRACE_SPECIFIC_RECEIVER_DUMP"


def test_go_fixture_is_reachable():
    assert trace_r1_verdict(mapping_verified=True, native_dump_available=False, real_scores_exist=True, go_conditions=[True] * 9) == "GO_FOR_TRACE_STAGE1"


def test_no_go_fixture_is_reachable():
    assert trace_r1_verdict(mapping_verified=True, native_dump_available=False, real_scores_exist=True, go_conditions=[True, False]) == "NO_GO_ACTION_EQUIVARIANCE"


def test_incomplete_provenance_overrides_performance_verdict():
    assert trace_r1_verdict(mapping_verified=True, native_dump_available=False, real_scores_exist=True, provenance_complete=False, go_conditions=[True] * 9) == "INCONCLUSIVE_BASELINE_OR_PROVENANCE"
