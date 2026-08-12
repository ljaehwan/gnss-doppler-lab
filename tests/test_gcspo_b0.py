from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


def test_gcspo_b0_dependency_is_explicit_without_changing_frozen_project_bytes():
    root = Path(__file__).parents[1]
    requirements = (root / "requirements-gcspo.txt").read_text().splitlines()
    assert "pandas>=2.1" in requirements
    module = (root / "src/gnss_doppler_lab/gcspo_b0.py").read_text()
    assert "import pandas" not in module.split("def _pandas", 1)[0]
    fresh_clone = (root / "scripts/verify_gcspo_fresh_clone.py").read_text()
    assert "requirements-gcspo.txt" in fresh_clone


def test_b0_scorer_intermediates_stay_outside_canonical_artifact_root():
    script = (Path(__file__).parents[1] / "scripts/run_gcspo_clean_b0.py").read_text()
    assert 'args.artifact_dir / "b0_clean_recomputed"' not in script
    assert "args.artifact_dir.parent" in script


def test_b0_runner_full_frame_has_explicit_lazy_pandas_dependency():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts/run_gcspo_clean_b0.py"
    spec = importlib.util.spec_from_file_location("gcspo_clean_b0_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = [{"window_start_s": 1., "score": 2.}]
    masks = [{"window_start_s": 1., "prns": [3, 7, 11, 19]}]
    frame = module._full_frame(rows, masks)
    assert frame.to_dict("records") == [{"window_start_s": 1., "prns": [3, 7, 11, 19], "score": 2.}]
    source = path.read_text()
    assert "def _pandas():" in source and "pd = _pandas()" in source


def test_b0_role_filter_requires_wholly_contained_windows():
    from gnss_doppler_lab.gcspo_b0 import role_filter

    frame = pd.DataFrame({
        "window_start_s": [219.5, 220.0, 339.0, 339.5],
        "window_end_s": [220.5, 221.0, 340.0, 340.5],
        "window_bin_s": [220.0, 220.5, 339.5, 340.0],
        "prn": [1, 1, 1, 1],
    })
    got = role_filter(frame, 220.0, 340.0)
    assert got.window_start_s.tolist() == [220.0, 339.0]




def test_b0_scheduled_windows_are_exactly_half_open_with_boundary_epsilon_and_reuse(tmp_path):
    from gnss_doppler_lab.gcspo_b0 import SAMPLE_RATE_HZ, TAP_FIELDS, build_scheduled_node_table

    raw = tmp_path / "raw"; raw.mkdir()
    below_zero = np.nextafter(0., -np.inf)
    below_one = np.nextafter(1., -np.inf)
    above_one = np.nextafter(1., np.inf)
    times = np.asarray([below_zero, 0., .25, .5, .75, below_one, 1., above_one, 1.25, 1.5, 1.75, 2.])
    samples = np.rint(times * SAMPLE_RATE_HZ).astype(np.int64)
    # Preserve exact endpoint/epsilon distinctions after the loader's sample-rate conversion.
    samples = np.asarray([-1, 0, 1, SAMPLE_RATE_HZ // 2, SAMPLE_RATE_HZ - 1,
                          SAMPLE_RATE_HZ - 2, SAMPLE_RATE_HZ, SAMPLE_RATE_HZ + 1,
                          SAMPLE_RATE_HZ + SAMPLE_RATE_HZ // 4, SAMPLE_RATE_HZ + SAMPLE_RATE_HZ // 2,
                          2 * SAMPLE_RATE_HZ - 1, 2 * SAMPLE_RATE_HZ], dtype=np.int64)
    loaded_times = samples.astype(float) / SAMPLE_RATE_HZ
    with h5py.File(raw / "epl_tracking_ch_0.mat", "w") as handle:
        handle["PRN"] = np.full(len(samples), 3)
        handle["PRN_start_sample_count"] = samples
        for field in TAP_FIELDS:
            handle[field] = np.ones(len(samples)) if field == "abs_P" else 1. + loaded_times
    frame = build_scheduled_node_table(tmp_path, {"role": (0., 2.)})
    by_start = frame.set_index("window_start_s")
    first_expected = np.mean(1. + loaded_times[(loaded_times >= 0.) & (loaded_times < 1.)])
    final_expected = np.mean(1. + loaded_times[(loaded_times >= 1.) & (loaded_times < 2.)])
    assert by_start.loc[0., "tap_E4_rel_prompt_mean"] == pytest.approx(first_expected)
    assert by_start.loc[1., "tap_E4_rel_prompt_mean"] == pytest.approx(final_expected)
    # A sample at 0.5 is legitimately reused by the overlapping [0,1) and [0.5,1.5) schedule.
    middle_expected = np.mean(1. + loaded_times[(loaded_times >= .5) & (loaded_times < 1.5)])
    assert by_start.loc[.5, "tap_E4_rel_prompt_mean"] == pytest.approx(middle_expected)
    assert 2. not in loaded_times[(loaded_times >= 1.) & (loaded_times < 2.)]


def test_b0_common_support_keeps_identical_windows_and_prns():
    from gnss_doppler_lab.gcspo_b0 import exact_common_support

    b0 = pd.DataFrame({
        "window_start_s": [1.0, 1.0, 1.5, 1.5],
        "prn": [1, 2, 1, 2], "prn_node_rmse": [.1, .2, .3, .4],
    })
    full = pd.DataFrame({
        "window_start_s": [1.0, 1.5], "prns": [[1, 2], [1, 3]], "score": [4., 5.],
    })
    got_b0, got_full = exact_common_support(b0, full)
    assert got_full.window_start_s.tolist() == [1.0]
    assert got_b0.prn.tolist() == [1, 2]


def test_b0_common_support_rejects_duplicate_prn_rows():
    from gnss_doppler_lab.gcspo_b0 import exact_common_support

    b0 = pd.DataFrame({"window_start_s": [1., 1.], "prn": [1, 1], "prn_node_rmse": [.1, .2]})
    full = pd.DataFrame({"window_start_s": [1.], "prns": [[1]], "score": [2.]})
    with pytest.raises(ValueError, match="duplicate"):
        exact_common_support(b0, full)
