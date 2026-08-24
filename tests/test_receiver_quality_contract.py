from __future__ import annotations

import pandas as pd
import pytest

from gnss_doppler_lab import receiver_quality_contract as contract


def make_rows(
    *,
    run_id: str = "run-a",
    prn: str = "G01",
    channel: int = 4,
    segment_index: int = 3,
    window_indexes=range(5),
    start_s: float = 0.0,
):
    rows = []
    first_index = min(window_indexes)
    for window_index in window_indexes:
        elapsed = (window_index - first_index) * 0.5
        rows.append({
            "run_id": run_id,
            "prn": prn,
            "channel": channel,
            "segment_index": segment_index,
            "window_index": window_index,
            "epoch_count": 50,
            "window_bin_s": start_s + elapsed,
            "window_start_s": start_s + elapsed,
            "window_mid_s": start_s + elapsed + 0.5,
            "window_end_s": start_s + elapsed + 1.0,
            "feature": float(window_index),
        })
    return rows


def test_reacquisition_uses_prn_segment_ordinal_not_raw_segment_index():
    frame = pd.DataFrame(
        make_rows(segment_index=3, start_s=0.0)
        + make_rows(channel=1, segment_index=0, start_s=10.0)
    )

    blocks = contract.segment_safe_blocks(frame, ["feature"])

    assert [(block.segment_index, block.prn_segment_ordinal) for block in blocks] == [
        (3, 0),
        (0, 1),
    ]
    assert [block.reacquisition_flag for block in blocks] == [0, 1]
    assert [block.sequence_restart_flag for block in blocks] == [0, 1]


def test_gap_inside_receiver_segment_starts_new_continuity_block():
    rows = make_rows(window_indexes=[0, 1, 2])
    rows += make_rows(window_indexes=[5, 6], start_s=2.5)
    frame = pd.DataFrame(rows)

    blocks = contract.segment_safe_blocks(frame, ["feature"])

    assert [block.frame.window_index.tolist() for block in blocks] == [[0, 1, 2], [5, 6]]
    assert [block.continuity_block_index for block in blocks] == [0, 1]
    assert [block.sequence_restart_flag for block in blocks] == [0, 1]


def test_score_metadata_retains_absolute_tracking_age_for_midsegment_partition():
    frame = pd.DataFrame(make_rows(window_indexes=range(20, 35), start_s=100.0))
    block = contract.segment_safe_blocks(frame, ["feature"])[0]

    metadata = contract.score_quality_metadata(block, 12, 12)

    assert metadata["target_window_index"] == 32
    assert metadata["target_sequence_position"] == 12
    assert metadata["tracking_age_s"] == pytest.approx(16.0)
    assert metadata["continuity_age_s"] == pytest.approx(6.0)
    assert metadata["segment_start_s"] == pytest.approx(90.0)
    assert metadata["history_start_window_index"] == 20
    assert metadata["history_end_window_index"] == 31
    assert metadata["history_same_segment_flag"] == 1
    assert metadata["history_length"] == 12


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda frame: frame.drop(columns=["epoch_count"]), "receiver-quality"),
        (lambda frame: frame.assign(epoch_count=0), "positive"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate receiver window",
        ),
        (
            lambda frame: pd.concat(
                [
                    frame,
                    frame.iloc[[0]].assign(
                        channel=8, segment_index=9, window_index=9
                    ),
                ],
                ignore_index=True,
            ),
            "duplicate PRN event",
        ),
    ],
)
def test_invalid_receiver_quality_contract_is_rejected(mutation, match):
    frame = pd.DataFrame(make_rows())

    with pytest.raises(ValueError, match=match):
        contract.validate_quality_node_frame(mutation(frame), ["feature"])


def test_score_contract_documents_causal_boundary():
    document = contract.score_contract_document(
        expected_stride_s=0.5, history_length=12
    )

    assert document["schema"] == contract.SCORE_CONTRACT_SCHEMA
    assert document["history_length"] == 12
    assert "residual-derived" in document["causality"]
    assert "prn_segment_ordinal > 0" in document["reacquisition_definition"]
