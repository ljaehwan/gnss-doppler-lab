from __future__ import annotations

import numpy as np
import pytest


def _context():
    from gnss_doppler_lab.gcspo_controls import ControlContext

    prns = np.asarray([3, 7, 11, 19, 23])
    times = 350.0 + np.arange(500) * .02
    raw = np.ones((5, 500, 6), float)
    raw *= np.linspace(.8, 1.2, 6)[None, None, :]
    raw += np.arange(5)[:, None, None] * .01
    other = np.zeros((5, 500, 4), float)
    other[:, :, 0] = np.sin(np.arange(500) * .01)
    residual = np.arange(5 * 500 * 10, dtype=float).reshape(5, 500, 10) / 10000
    source = residual[::-1].copy()
    cn0 = np.full((5, 500), 42.0)
    return ControlContext(prns=prns, times_s=times, raw_complex=raw, other_q=other,
                          epsilon_by_prn=np.linspace(.01, .05, 5), residual=residual,
                          source_residual=source, cn0=cn0)


@pytest.mark.parametrize("control,levels,stage", [
    ("COMMON_GAIN", [.5, .8, 1.2, 2.], "raw_q"),
    ("PROMPT_AMPLITUDE", [.5, .8, 1.2, 2.], "raw_q"),
    ("CN0_METADATA_EXCLUSION_INVARIANCE", [-3., -6., -10.], "metadata_only"),
    ("PRN_DROP_ONLY", [1, 2, 4], "mask"),
    ("EMPIRICAL_NOISE", [.5, 1., 2.], "pre_whitening_residual"),
    ("ONE_PRN_DISTURBANCE", [1., 2., 4.], "pre_whitening_residual"),
    ("INDEPENDENT_MULTIPATH_LIKE", [.5, 1., 2.], "pre_whitening_residual"),
    ("CLOCK_DRIFT", [.1, 1., 5.], "physical_state"),
])
def test_every_control_level_has_frozen_stage_seed_and_reset(control, levels, stage):
    from gnss_doppler_lab.gcspo_controls import apply_control

    for level in levels:
        result = apply_control(_context(), control_id=control, level=level, scenario="cleanStatic",
                               phase="holdout", block_id=0, var_coefficients=np.zeros((10, 10, 10)))
        assert result.stage == stage
        assert result.seed_material.startswith("23|")
        assert 0 <= result.seed < 2**128
        assert result.history_reset is True and result.first_eligible_epoch == 10
        assert result.numpy_version == np.__version__


def test_common_gain_and_prompt_amplitude_are_actual_raw_renormalizations():
    from gnss_doppler_lab.gcspo_controls import apply_control

    context = _context()
    gain = apply_control(context, control_id="COMMON_GAIN", level=.5, scenario="cleanStatic", phase="holdout", block_id=0,
                         var_coefficients=np.zeros((10, 10, 10)))
    prompt = apply_control(context, control_id="PROMPT_AMPLITUDE", level=.5, scenario="cleanStatic", phase="holdout", block_id=0,
                           var_coefficients=np.zeros((10, 10, 10)))
    assert not np.array_equal(gain.q, prompt.q)
    assert np.all(prompt.raw_complex[:, :, :2] == context.raw_complex[:, :, :2])
    assert np.all(prompt.raw_complex[:, :, 4:] == context.raw_complex[:, :, 4:])
    assert np.all(prompt.raw_complex[:, :, 2:4] == context.raw_complex[:, :, 2:4] * .5)


def test_cn0_exclusion_is_bitwise_score_input_invariant():
    from gnss_doppler_lab.gcspo_controls import apply_control

    context = _context()
    result = apply_control(context, control_id="CN0_METADATA_EXCLUSION_INVARIANCE", level=-10,
                           scenario="cleanStatic", phase="holdout", block_id=0,
                           var_coefficients=np.zeros((10, 10, 10)))
    assert result.q.tobytes() == result.baseline_q.tobytes()
    assert result.residual.tobytes() == context.residual.tobytes()
    assert np.all(result.cn0 == context.cn0 - 10)


def test_prn_drop_and_one_prn_disturbance_preserve_identity_contract():
    from gnss_doppler_lab.gcspo_controls import apply_control

    context = _context()
    dropped = apply_control(context, control_id="PRN_DROP_ONLY", level=2, scenario="cleanStatic", phase="holdout", block_id=1,
                            var_coefficients=np.zeros((10, 10, 10)))
    assert len(dropped.prns) == 3 and set(dropped.prns) < set(context.prns)
    one = apply_control(context, control_id="ONE_PRN_DISTURBANCE", level=2., scenario="cleanStatic", phase="holdout", block_id=1,
                        var_coefficients=np.zeros((10, 10, 10)))
    changed = np.any(one.residual != context.residual, axis=(1, 2))
    assert changed.sum() == 1


def test_empirical_noise_mapping_is_centered_and_deterministic():
    from gnss_doppler_lab.gcspo_controls import apply_control

    context = _context(); kwargs = dict(control_id="EMPIRICAL_NOISE", level=1., scenario="cleanStatic", phase="holdout", block_id=2,
                                        var_coefficients=np.zeros((10, 10, 10)))
    first, second = apply_control(context, **kwargs), apply_control(context, **kwargs)
    assert first.residual.tobytes() == second.residual.tobytes()
    injected = first.residual - context.residual
    assert np.max(np.abs(injected.mean(axis=1))) < 1e-12
    assert first.source_mapping


def test_independent_multipath_preserves_unlisted_rows_and_has_independent_phase():
    from gnss_doppler_lab.gcspo_controls import apply_control

    context = _context()
    result = apply_control(context, control_id="INDEPENDENT_MULTIPATH_LIKE", level=1., scenario="cleanStatic", phase="holdout", block_id=3,
                           var_coefficients=np.zeros((10, 10, 10)))
    delta = result.residual - context.residual
    assert np.all(delta[:, :, 7:] == 0)
    assert len(set(result.prn_phases_rad.values())) == len(context.prns)


def test_clock_drift_uses_physical_loading_var_once_and_reports_specificity_inputs():
    from gnss_doppler_lab.gcspo_controls import apply_control

    coefficients = np.zeros((10, 10, 10)); coefficients[0] = np.eye(10) * .2
    result = apply_control(_context(), control_id="CLOCK_DRIFT", level=1., scenario="cleanStatic", phase="holdout", block_id=4,
                           var_coefficients=coefficients)
    assert result.var_transfer_application_count == 1
    assert result.state_energy["Eclock"] > 0
    assert result.state_energy["Epos"] == 0
    assert result.state_energy["specificity_ratio"] == 0


def test_control_block_contract_is_nonoverlap_10s_and_warmup_reset():
    from gnss_doppler_lab.gcspo_controls import holdout_blocks

    blocks = holdout_blocks(350., 470.)
    assert blocks == [(index, 350. + index * 10, 360. + index * 10) for index in range(12)]


def test_control_grid_executes_every_block_level_and_records_alarm_fields():
    from gnss_doppler_lab.gcspo_controls import generate_control_grid

    contexts = [(index, 350. + index * 10, _context()) for index in range(12)]
    report = generate_control_grid(contexts, scenario="cleanStatic", phase="holdout",
                                   var_coefficients=np.zeros((10, 10, 10)), threshold=0.,
                                   scorer=lambda result: [float(np.mean(result.residual))])
    assert report["overall_status"] == "PASS"
    assert len(report["results"]) == 12 * (4 + 4 + 3 + 3 + 3 + 3 + 3 + 3)
    assert all(row["seed_material"].startswith("23|") and row["score_count"] == 1 for row in report["results"])
    assert all("persistent_alarm_ratio" in row and "max_consecutive_alarms" in row for row in report["results"])
