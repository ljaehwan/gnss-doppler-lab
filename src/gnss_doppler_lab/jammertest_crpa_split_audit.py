"""Blocked-split feasibility audit for the CRPA Stage-0B experiment."""

from __future__ import annotations


def assess_blocked_split_feasibility(rows: list[dict], block_size: int) -> dict:
    cells = []
    all_feasible = True
    for power in (15.0, 25.0, 30.0, 35.0, 40.0):
        for target_name, target_value in (("positive", True), ("negative", False)):
            sample_indices = [
                row["sample_index"]
                for row in rows
                if row["area"] == 1
                and row["transmit_power_dbm"] == power
                and (("Spoof" in row["class_name"] or "Meac" in row["class_name"]) == target_value)
            ]
            block_ids = sorted({index // block_size for index in sample_indices})
            feasible = len(block_ids) >= 2
            all_feasible &= feasible
            cells.append(
                {
                    "power_dbm": int(power),
                    "binary_class": target_name,
                    "sample_count": len(sample_indices),
                    "unique_block_count": len(block_ids),
                    "block_ids": block_ids,
                    "necessary_train_test_condition_passed": feasible,
                }
            )
    return {
        "block_size": block_size,
        "necessary_condition": "at least two distinct floor(sample_index/block_size) groups in every power/class cell",
        "guard_requirement_not_relaxed": True,
        "cells": cells,
        "balanced_block_disjoint_split_feasible": bool(all_feasible),
    }
