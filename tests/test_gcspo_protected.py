import numpy as np
import pytest


def test_phase_rows_require_window_and_availability_inside_half_open_phase():
    from gnss_doppler_lab.gcspo_protected import phase_rows

    rows = [
        {"window_start_s": 9.0, "availability_s": 10.0},
        {"window_start_s": 9.5, "availability_s": 10.5},
        {"window_start_s": 10.0, "availability_s": 11.0},
    ]
    assert phase_rows(rows, 9.0, 10.5) == [rows[0]]


def test_reconstruct_normal_model_preserves_frozen_arrays():
    from gnss_doppler_lab.gcspo_protected import reconstruct_normal_model

    doc = {
        "intercept": [1.0, -1.0] + [0.0] * 8,
        "coefficients": np.zeros((2, 10, 10)).tolist(),
        "whitener_location": [0.1, 0.2] + [0.0] * 8,
        "whitener_covariance": np.eye(10).tolist(),
        "whitener_inverse_sqrt": np.eye(10).tolist(),
        "gamma": (np.eye(10) * .1).tolist(),
    }
    model, whitener, gamma = reconstruct_normal_model(doc)
    assert model.lags == 2
    assert model.intercept.tolist() == [1.0, -1.0] + [0.0] * 8
    assert whitener.transform([[0.1, 0.2] + [0.0] * 8]).tolist() == [[0.0] * 10]
    assert gamma == pytest.approx(np.eye(10) * .1)


def test_science_verdict_is_no_go_when_any_mandatory_gate_fails():
    from gnss_doppler_lab.gcspo_protected import scientific_verdict

    gates = [{"id": "G1_FALSE_ALARM", "status": "FAIL"}] + [
        {"id": name, "status": "PASS"} for name in ("G2_INCREMENTAL", "G3_GEOMETRY", "G4_PERSISTENCE", "G5_CONTROLS", "G6_SHARED")
    ]
    assert scientific_verdict(gates) == "NO_GO_PHYSICAL_HYPOTHESIS"
    assert scientific_verdict([{**row, "status": "PASS"} for row in gates]) == "GO_FOR_NEURAL_STAGE1"


def test_protected_tracking_loader_uses_manifest_capabilities_and_keeps_channel(tmp_path):
    import hashlib, json
    import h5py
    import numpy as np
    from gnss_doppler_lab.gcspo_core import AccessGate
    from gnss_doppler_lab.gcspo_clean import Q_FIELDS
    from gnss_doppler_lab.gcspo_protected import load_receiver_tracking

    files = []
    for channel in (0, 1):
        path = tmp_path / f"epl_tracking_ch_{channel}.mat"
        with h5py.File(path, "w") as handle:
            handle["PRN_start_sample_count"] = [500_000]
            handle["PRN"] = [7]
            for index, name in enumerate(Q_FIELDS): handle[name] = [1. + index + channel]
        data = path.read_bytes(); files.append({"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
    manifest = tmp_path / "receiver_manifest.json"
    manifest.write_text(json.dumps({"files": files}) + "\n")
    content = manifest.read_bytes()
    gate = AccessGate(tmp_path / "ledger.jsonl")
    gate.set_preflight(clean_only_pass=True, reviews_pass=True, freeze_sha="a" * 40, frozen_hashes={"config": "b" * 64})
    gate.set_remote_sync(local_sha="a" * 40, remote_sha="a" * 40, ahead=0, behind=0, clean=True)
    gate.register_pinned(manifest, expected_sha256=hashlib.sha256(content).hexdigest(), expected_size=len(content), kind="RECEIVER_MANIFEST")
    gate.authenticate_manifest(manifest, scenario="DS7", phase="all", purpose="receiver identities")
    loaded = load_receiver_tracking([tmp_path / row["path"] for row in files], epsilons={7: .01}, gate=gate, scenario="DS7")
    assert loaded.channel.tolist() == [0, 1] and loaded.segment.tolist() == [0, 0]
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [row["operation"] for row in records if row["record_type"] == "PRE"] == ["READ_JSON", "READ_HDF5", "READ_HDF5"]


class _SyntheticTrackingGate:
    def __init__(self, values):
        self.values = values

    def read_h5(self, path, **classification):
        return self.values


def _synthetic_tracking(samples):
    from gnss_doppler_lab.gcspo_clean import Q_FIELDS

    count = len(samples)
    values = {
        "PRN_start_sample_count": np.asarray(samples, np.int64),
        "PRN": np.full(count, 7, np.int64),
    }
    for index, name in enumerate(Q_FIELDS):
        values[name] = np.arange(count, dtype=float) + index + 1.0
    return values


def test_protected_loader_integer_bins_multiple_rows_and_boundary_samples():
    from gnss_doppler_lab.gcspo_clean import Q_FIELDS, signed_q
    from gnss_doppler_lab.gcspo_protected import load_receiver_tracking

    samples = [index * 25_000 for index in range(20)] + [500_000, 525_000]
    values = _synthetic_tracking(samples)
    loaded = load_receiver_tracking(["synthetic.mat"], epsilons={7: .01},
                                    gate=_SyntheticTrackingGate(values), scenario="DS7")
    expected_q = signed_q({name: values[name] for name in Q_FIELDS}, epsilon=np.full(len(samples), .01))
    assert loaded.epoch.tolist() == [0, 1]
    assert loaded.sample_min.tolist() == [0, 500_000]
    assert loaded.sample_max.tolist() == [475_000, 525_000]
    assert loaded.q[0] == pytest.approx(np.median(expected_q[:20], axis=0))
    assert loaded.q[1] == pytest.approx(np.median(expected_q[20:], axis=0))


def test_protected_loader_rejects_exact_duplicate_identity_before_segmenting():
    from gnss_doppler_lab.gcspo_protected import load_receiver_tracking

    values = _synthetic_tracking([499_999, 499_999])
    for name in tuple(values)[2:]:
        values[name][1] = values[name][0]
    with pytest.raises(ValueError, match="duplicate protected scientific row"):
        load_receiver_tracking(["synthetic.mat"], epsilons={7: .01},
                               gate=_SyntheticTrackingGate(values), scenario="DS7")
