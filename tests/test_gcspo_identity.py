import numpy as np

from gnss_doppler_lab.gcspo_core import aggregate_20ms
from gnss_doppler_lab.gcspo_clean import causal_histories


def test_channel_handoff_is_not_merged_at_same_epoch_prn():
    rows = [
        {"time_s": .001, "sample_count": 1, "prn": 7, "channel": 0, "segment_index": 0, "q": [1., 2.]},
        {"time_s": .002, "sample_count": 2, "prn": 7, "channel": 1, "segment_index": 0, "q": [9., 8.]},
    ]
    got = aggregate_20ms(rows)
    assert len(got) == 2
    assert [(row["channel"], row["segment_index"], row["q"].tolist()) for row in got] == [
        (0, 0, [1., 2.]), (1, 0, [9., 8.]),
    ]


def test_reacquisition_identity_breaks_causal_history_even_without_epoch_gap():
    epochs = np.arange(8)
    values = np.column_stack([epochs, -epochs]).astype(float)
    identities = np.asarray([(0, 7, 0)] * 4 + [(0, 7, 1)] * 4, dtype=object)
    histories, targets, target_epochs = causal_histories(epochs, values, lags=2, identities=identities)
    assert target_epochs.tolist() == [2, 3, 6, 7]
    assert histories[:, :, 0].tolist() == [[0, 1], [1, 2], [4, 5], [5, 6]]


def test_channel_change_breaks_history_and_never_bridges_to_same_prn():
    epochs = np.arange(7)
    values = epochs[:, None].astype(float)
    identities = np.asarray([(0, 3, 0)] * 3 + [(1, 3, 0)] * 4, dtype=object)
    _, _, target_epochs = causal_histories(epochs, values, lags=2, identities=identities)
    assert target_epochs.tolist() == [2, 5, 6]


def test_real_role_history_groups_channel_segment_and_rejects_simultaneous_prn_alias():
    from gnss_doppler_lab.gcspo_clean import AggregatedClean, role_histories

    data = AggregatedClean(
        epoch=np.asarray([0, 1, 2, 3, 4, 5]), prn=np.asarray([7] * 6),
        channel=np.asarray([0, 0, 0, 1, 1, 1]), segment=np.asarray([0, 0, 0, 1, 1, 1]),
        q=np.arange(12, dtype=float).reshape(6, 2), sample_min=np.arange(6),
        sample_max=np.arange(6), epsilons={7: .1}, source_files=("synthetic",),
    )
    histories, _, epochs, _ = role_histories(data, 0, .12, lags=2)
    assert epochs.tolist() == [2, 5]
    assert histories[:, :, 0].tolist() == [[0, 2], [6, 8]]

    ambiguous = AggregatedClean(
        epoch=np.asarray([0, 0]), prn=np.asarray([7, 7]), channel=np.asarray([0, 1]),
        segment=np.asarray([0, 0]), q=np.zeros((2, 2)), sample_min=np.asarray([0, 1]),
        sample_max=np.asarray([0, 1]), epsilons={7: .1}, source_files=("synthetic",),
    )
    with np.testing.assert_raises_regex(ValueError, "simultaneous channel identity"):
        role_histories(ambiguous, 0, .02, lags=1)
