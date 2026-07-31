import sys
import unittest
import importlib.util
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gnss_doppler_lab.gcmr_peak_innovation_adapter import (CausalEventBuildError, PeakWindowRecord,
    aggregate_peak_windows, build_event_record)


@dataclass
class RelationEvent:
    window_start_s: float
    window_end_s: float
    pair_prns: np.ndarray
    conditions: np.ndarray


@dataclass
class CapturedEventRecord:
    time: float
    prns: tuple[str, ...]
    epl: np.ndarray
    histories: dict[str, np.ndarray]
    cn0: np.ndarray
    elevation: np.ndarray
    pair_conditions: dict[tuple[str, str], np.ndarray]

    def validate(self, window):
        assert self.epl.shape == (len(self.prns), 3)
        assert set(self.histories) == set(self.prns)
        assert all(value.shape == (window, 3) for value in self.histories.values())


class Series:
    def __init__(self, times, magnitudes, *, taps=("E", "P", "L")):
        self.time_s = np.asarray(times, float)
        self.magnitudes = np.asarray(magnitudes, float)
        self.cn0_db_hz = np.full(len(times), 42.0)
        self.tap_names = taps


def relation(prns=(1, 17, 32)):
    pairs = []
    cond = []
    for i, a in enumerate(prns):
        for b in prns[i + 1:]:
            pairs.append((a, b))
            # First three fields are [los_dot, min elev sin, max elev sin].
            cond.append((0.1 * (a + b), 0.2 + .01 * i, 0.6 + .01 * i, 999.0))
    return RelationEvent(3.0, 4.0, np.asarray(pairs, np.int64), np.asarray(cond, float))


def windows_for(prn, *, target_bump=0.0):
    values = [np.array([prn + i, 10.0 + i, 2.0 + i]) for i in range(5)]
    values[3] = values[3] + target_bump
    return [PeakWindowRecord(float(i), float(i + 1), values[i], 40.0 + i) for i in range(5)]


class AdapterTests(unittest.TestCase):
    def test_variable_prns_and_symmetric_pair_condition_mapping(self):
        event = relation()
        record = build_event_record(event, {"1": windows_for(1), "G17": windows_for(17), 32: windows_for(32)},
                                    history_window=3, event_record_type=CapturedEventRecord)
        self.assertEqual(record.prns, ("G01", "G17", "G32"))
        self.assertEqual(record.time, 4.0)  # score availability is target-window end.
        self.assertEqual(set(record.pair_conditions), {("G01", "G17"), ("G01", "G32"), ("G17", "G32")})
        np.testing.assert_array_equal(record.pair_conditions[("G01", "G17")], np.array([1.8, .2, .6]))
        # Symmetric min/max elevation conditions give both endpoints their pair
        # midpoint; each node averages the midpoints of its incident pairs.
        np.testing.assert_allclose(record.elevation, [.4, .405, .405])
        self.assertFalse(any("G" in str(x) for x in record.epl.flat))

    def test_history_is_exactly_causal_under_target_and_future_changes(self):
        times = np.arange(.2, 5.0, .2)
        values = np.column_stack((times, 10.0 + times, 2.0 + times))
        base = Series(times, values)
        changed_values = values.copy()
        changed_values[times >= 3.0] += 10_000.0
        changed = Series(times, changed_values)
        requested = [(float(i), float(i + 1)) for i in range(5)]
        base_records = aggregate_peak_windows(base, requested)
        changed_records = aggregate_peak_windows(changed, requested)
        e = relation()
        a = build_event_record(e, {p: base_records for p in (1, 17, 32)}, history_window=3,
                               event_record_type=CapturedEventRecord)
        b = build_event_record(e, {p: changed_records for p in (1, 17, 32)}, history_window=3,
                               event_record_type=CapturedEventRecord)
        for prn in a.prns:
            np.testing.assert_array_equal(a.histories[prn], b.histories[prn])
        self.assertFalse(np.array_equal(a.epl, b.epl))

    def test_default_factory_resolves_package_event_record_after_copy(self):
        # Model the destination-package import used after this helper is copied
        # into src/gnss_doppler_lab, without importing fitting/scoring code.
        package = types.ModuleType("adapter_test_package")
        package.__path__ = []
        pipeline = types.ModuleType("adapter_test_package.gcmr_peak_innovation_pipeline")
        pipeline.EventRecord = CapturedEventRecord
        sys.modules[package.__name__] = package
        sys.modules[pipeline.__name__] = pipeline
        try:
            spec = importlib.util.spec_from_file_location(
                "adapter_test_package.gcmr_peak_innovation_adapter",
                Path(__file__).parents[1] / "src" / "gnss_doppler_lab" / "gcmr_peak_innovation_adapter.py",
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            records = {p: [module.PeakWindowRecord(float(i), float(i + 1), np.array([p + i, 10. + i, 2. + i]), 40. + i) for i in range(5)] for p in (1, 17, 32)}
            record = module.build_event_record(relation(), records, history_window=3)
            self.assertIsInstance(record, CapturedEventRecord)
        finally:
            for name in ("adapter_test_package.gcmr_peak_innovation_adapter", pipeline.__name__, package.__name__):
                sys.modules.pop(name, None)

    def test_only_real_three_taps_are_accepted(self):
        bad = Series([.2, .4], [[1, 2, 3], [2, 3, 4]], taps=("VE", "P", "VL"))
        with self.assertRaisesRegex(ValueError, "exactly"):
            aggregate_peak_windows(bad, [(0., 1.)])
        event = relation()
        windows = {p: windows_for(p) for p in (1, 17, 32)}
        windows[1][0] = PeakWindowRecord(0., 1., np.ones(3), 40., ("E", "P", "VL"))
        with self.assertRaisesRegex(ValueError, "exactly"):
            build_event_record(event, windows, history_window=3, event_record_type=CapturedEventRecord)

    def test_missing_history_fails_clearly_and_geometry_never_filters_prns(self):
        event = relation()
        windows = {p: windows_for(p) for p in (1, 17, 32)}
        windows[17] = windows[17][2:]  # target remains, but only one earlier window.
        with self.assertRaisesRegex(CausalEventBuildError, "G17 has 1 valid prior"):
            build_event_record(event, windows, history_window=3, event_record_type=CapturedEventRecord)
        # Low elevation does not remove any PRN: it remains a condition, not a filter.
        event.conditions[:, 1:3] = -0.99
        event.conditions[:, 2] = -0.98
        record = build_event_record(event, {p: windows_for(p) for p in (1, 17, 32)}, history_window=3,
                                    event_record_type=CapturedEventRecord)
        self.assertEqual(record.prns, ("G01", "G17", "G32"))


if __name__ == "__main__":
    unittest.main()
