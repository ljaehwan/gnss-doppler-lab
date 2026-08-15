"""Execution helpers for the preregistered B0-CS Stage-0 experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gnss_doppler_lab.b0_dependence_calibrated import (
    FEATURE_COLUMNS, aggregate_receiver_scores, attach_tracked_count,
    conformal_pvalues, consecutive_alarm, higher_quantile, power_evalues,
    receiver_blocks, residual_frame, robust_pool, score_block_evidence,
    score_prn_evidence,
)


@dataclass(frozen=True)
class B0TrainingConfig:
    seq_len: int = 12
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    embedding_dim: int = 128
    dropout: float = .05
    seed: int = 11


class PaperB0GRU(nn.Module):
    """Exact historical B0 architecture; shared weights and no PRN input."""
    def __init__(self, feature_dim: int = 9, config: B0TrainingConfig = B0TrainingConfig()):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, feature_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps, features = values.shape
        encoded = self.encoder(values.reshape(batch * steps, features)).reshape(batch, steps, -1)
        output, _ = self.gru(encoded)
        return self.head(output[:, -1])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _epoch(model, loader, device, optimizer=None) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    loss_function = nn.MSELoss()
    for sequence, target in loader:
        sequence = sequence.to(device)
        target = target.to(device)
        prediction = model(sequence)
        loss = loss_function(prediction, target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach().cpu()) * len(sequence)
        count += len(sequence)
    average = total / max(1, count)
    return {"loss": average, "rmse": math.sqrt(average), "examples": count}


def train_paper_b0(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    config: B0TrainingConfig = B0TrainingConfig(),
    device: str | None = None,
) -> tuple[PaperB0GRU, list[dict[str, object]], dict[str, object]]:
    if not len(train_x) or not len(validation_x):
        raise ValueError("Paper-B0 requires nonempty train and validation examples")
    seed_everything(config.seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.batch_size, shuffle=True, generator=train_generator, num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y)),
        batch_size=config.batch_size, shuffle=False, num_workers=0,
    )
    model = PaperB0GRU(train_x.shape[-1], config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    started = time.perf_counter()
    for epoch_index in range(1, config.epochs + 1):
        train_metrics = _epoch(model, train_loader, selected_device, optimizer)
        with torch.no_grad():
            validation_metrics = _epoch(model, validation_loader, selected_device)
        record = {
            "epoch": epoch_index,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(__import__("json").dumps(record, sort_keys=True), flush=True)
        if float(validation_metrics["loss"]) < best_loss:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch_index
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Paper-B0 checkpoint selection failed")
    model.load_state_dict(best_state)
    model.to(selected_device).eval()
    peak_memory = (
        int(torch.cuda.max_memory_allocated(selected_device))
        if selected_device.type == "cuda" else None
    )
    summary = {
        "architecture": "PaperB0GRU",
        "config": asdict(config),
        "device": str(selected_device),
        "torch_version": torch.__version__,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_runtime_seconds": float(elapsed),
        "peak_cuda_memory_bytes": peak_memory,
        "uses_prn_identity": False,
        "feature_columns": list(FEATURE_COLUMNS),
        "checkpoint_selection": "strict minimum chronological validation loss; earliest exact tie",
    }
    return model, history, summary


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, *, device: str | None = None, batch_size: int = 1024) -> tuple[np.ndarray, dict[str, object]]:
    if not len(values):
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32), {
            "examples": 0, "runtime_seconds": 0.0, "runtime_per_epoch_seconds": None,
        }
    selected_device = torch.device(device or next(model.parameters()).device)
    model.to(selected_device).eval()
    outputs = []
    started = time.perf_counter()
    for begin in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[begin:begin + batch_size]).to(selected_device)
        outputs.append(model(batch).detach().cpu().numpy())
    elapsed = time.perf_counter() - started
    prediction = np.concatenate(outputs).astype(np.float32)
    return prediction, {
        "examples": int(len(values)), "runtime_seconds": float(elapsed),
        "runtime_per_epoch_seconds": float(elapsed / len(values)),
    }


def fit_linear_ar(train_x: np.ndarray, train_y: np.ndarray, *, alpha: float = .001) -> Ridge:
    if not len(train_x):
        raise ValueError("Linear-AR requires clean train examples")
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(train_x.reshape(len(train_x), -1), train_y)
    return model


def predict_linear_ar(model: Ridge, values: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(values.reshape(len(values), -1)), dtype=np.float32)


def linear_state(model: Ridge) -> dict[str, object]:
    return {
        "alpha": float(model.alpha), "fit_intercept": bool(model.fit_intercept),
        "coef": np.asarray(model.coef_).tolist(), "intercept": np.asarray(model.intercept_).tolist(),
        "feature_contract": "flattened standardized 12x9 PRN-local history to next standardized 9 taps",
    }


def linear_from_state(state: Mapping[str, object]) -> Ridge:
    model = Ridge(alpha=float(state["alpha"]), fit_intercept=bool(state["fit_intercept"]))
    model.coef_ = np.asarray(state["coef"], dtype=float)
    model.intercept_ = np.asarray(state["intercept"], dtype=float)
    model.n_features_in_ = model.coef_.shape[1]
    return model


def _binomial_tail(k: int, n: int, probability: float) -> float:
    if k <= 0 or n <= 0:
        return 0.0
    tail = sum(
        math.comb(n, index) * probability ** index * (1 - probability) ** (n - index)
        for index in range(k, n + 1)
    )
    return -math.log(max(tail, 1e-300))


def binomial_gate(
    residuals: pd.DataFrame,
    node_thresholds: Mapping[str, float],
    *,
    previous_weight: float = .75,
) -> pd.DataFrame:
    quantiles = {"q50": .50, "q70": .70, "q80": .80}
    if set(node_thresholds) != set(quantiles):
        raise ValueError("q50/q70/q80 thresholds required")
    rows = []
    for (recording, epoch), group in residuals.groupby(
        ["physical_recording_id", "window_bin_s"], sort=True
    ):
        values = group.b0_residual_rmse.to_numpy(float)
        surprises = []
        record = {
            "physical_recording_id": str(recording), "window_bin_s": float(epoch),
            "availability_time_s": float(group.window_end_s.max()),
            "tracked_prn_count": int(group.prn.nunique()),
        }
        for name, quantile in quantiles.items():
            k = int(np.sum(values > float(node_thresholds[name])))
            surprise = _binomial_tail(k, len(values), 1 - quantile)
            record[f"k_{name}"] = k
            record[f"btail_{name}"] = surprise
            surprises.append(surprise)
        record["btail_raw"] = max(surprises)
        rows.append(record)
    result = pd.DataFrame(rows)
    result["btail_ewma075"] = np.nan
    for _, indices in result.groupby("physical_recording_id", sort=False).groups.items():
        previous = 0.0
        for index in indices:
            previous = previous_weight * previous + (1 - previous_weight) * float(result.at[index, "btail_raw"])
            result.at[index, "btail_ewma075"] = previous
    return result


def calibrate_binomial_gate(residuals: pd.DataFrame) -> dict[str, object]:
    values = residuals.b0_residual_rmse.to_numpy(float)
    thresholds = {
        "q50": higher_quantile(values, .50),
        "q70": higher_quantile(values, .70),
        "q80": higher_quantile(values, .80),
    }
    events = binomial_gate(residuals, thresholds)
    return {
        "node_thresholds": thresholds,
        "event_q99_threshold": higher_quantile(events.btail_ewma075, .99),
        "source": "Paper-B0 cleanStatic calibration only",
        "independent_prn_assumption": True,
    }


def higher_quantile_thresholds(
    paper_receiver: pd.DataFrame,
    nuisance_receiver: pd.DataFrame,
    block_scores: pd.DataFrame,
) -> dict[str, object]:
    return {
        "a0_q99": higher_quantile(paper_receiver.a0_robust_pool, .99),
        "a0_q995": higher_quantile(paper_receiver.a0_robust_pool, .995),
        "set_q99": higher_quantile(nuisance_receiver.set_score.dropna(), .99),
        "set_q995": higher_quantile(nuisance_receiver.set_score.dropna(), .995),
        "block_q99": higher_quantile(block_scores.block_score, .99),
        "block_q995": higher_quantile(block_scores.block_score, .995),
        "sequential_alarm_threshold": 100.0,
        "simple_consecutive_epochs": 3,
    }


def normalized_partial_auc(labels: Sequence[int], scores: Sequence[float], max_fpr: float = .05) -> float:
    truth = np.asarray(labels, dtype=int)
    value = np.asarray(scores, dtype=float)
    if len(np.unique(truth)) < 2:
        raise ValueError("pAUC requires both classes")
    fpr, tpr, _ = roc_curve(truth, value)
    if fpr[-1] < max_fpr:
        return float(np.trapezoid(tpr, fpr) / max_fpr)
    index = int(np.searchsorted(fpr, max_fpr, side="right"))
    clipped_fpr = np.r_[fpr[:index], max_fpr]
    clipped_tpr = np.r_[tpr[:index], np.interp(max_fpr, fpr, tpr)]
    return float(np.trapezoid(clipped_tpr, clipped_fpr) / max_fpr)


def longest_true_run(flags: Sequence[bool]) -> int:
    longest = current = 0
    for flag in np.asarray(flags, dtype=bool):
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def score_metrics(
    *,
    scenario: str,
    method: str,
    scores: Sequence[float],
    times: Sequence[float],
    threshold: float,
    onset_s: float | None,
    pull_off_s: float | None = None,
) -> dict[str, object]:
    score = np.asarray(scores, dtype=float)
    time_values = np.asarray(times, dtype=float)
    finite = np.isfinite(score) & np.isfinite(time_values)
    score = score[finite]
    time_values = time_values[finite]
    alarm = score >= threshold
    base = {
        "scenario": scenario, "method": method, "threshold": float(threshold),
        "epochs_or_blocks": int(len(score)), "alarm_fraction": float(alarm.mean()) if len(alarm) else None,
        "persistent_alarm_ratio": float(np.mean([longest_true_run(alarm) >= 3])) if len(alarm) else None,
        "max_consecutive_alarms": longest_true_run(alarm),
    }
    if onset_s is None:
        base.update({
            "status": "AVAILABLE", "false_positive_rate": float(alarm.mean()) if len(alarm) else None,
            "average_run_length": float(len(alarm) / max(1, int(alarm.sum()))) if len(alarm) else None,
        })
        return base
    pre = time_values < onset_s
    post = time_values >= onset_s
    transition = (time_values >= onset_s) & (time_values < onset_s + 10)
    established = time_values >= onset_s + 10
    if not pre.any() or not post.any():
        base.update({"status": "LIMITED", "reason": "missing pre-onset or post-onset support"})
        return base
    labels = post.astype(int)
    hits = np.flatnonzero(alarm & post)
    first_alarm = None if not len(hits) else float(time_values[hits[0]])
    base.update({
        "status": "AVAILABLE",
        "roc_auc": float(roc_auc_score(labels, score)),
        "normalized_pauc_fpr_le_0_05": normalized_partial_auc(labels, score),
        "pr_auc": float(average_precision_score(labels, score)),
        "pre_onset_fpr": float(alarm[pre].mean()),
        "attack_detection_rate": float(alarm[post].mean()),
        "transition_detection_rate": float(alarm[transition].mean()) if transition.any() else None,
        "established_detection_rate": float(alarm[established].mean()) if established.any() else None,
        "first_alarm_time_s": first_alarm,
        "first_alarm_delay_from_signal_s": None if first_alarm is None else first_alarm - onset_s,
        "first_alarm_delay_from_pull_off_s": (
            None if first_alarm is None or pull_off_s is None else first_alarm - pull_off_s
        ),
    })
    return base


def method_streams(
    paper_residuals: pd.DataFrame,
    linear_residuals: pd.DataFrame,
    historical_residuals: pd.DataFrame,
    *,
    calibrator,
    calibration_block_scores: Sequence[float],
    block_seconds: float,
    thresholds: Mapping[str, object],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    paper_evidence = score_prn_evidence(paper_residuals, calibrator, nuisance_conditioned=True)
    paper_global = score_prn_evidence(paper_residuals, calibrator, nuisance_conditioned=False)
    linear_evidence = score_prn_evidence(linear_residuals, calibrator, nuisance_conditioned=True)
    paper_receiver = aggregate_receiver_scores(paper_evidence)
    global_receiver = aggregate_receiver_scores(paper_global)
    linear_receiver = aggregate_receiver_scores(linear_evidence)
    full_blocks = score_block_evidence(
        receiver_blocks(paper_receiver, block_seconds=block_seconds), calibration_block_scores
    )
    linear_blocks = score_block_evidence(
        receiver_blocks(linear_receiver, block_seconds=block_seconds), calibration_block_scores
    )
    paper_gate = binomial_gate(paper_residuals, thresholds["paper_binomial"]["node_thresholds"])
    historical_gate = binomial_gate(
        historical_residuals, thresholds["historical_binomial"]["node_thresholds"]
    )

    def epoch(name: str, frame: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
        result = frame[["physical_recording_id", "window_bin_s", "availability_time_s", column]].copy()
        result = result.rename(columns={column: "score"})
        result["method"] = name
        result["threshold"] = float(threshold)
        result["alarm"] = result.score >= float(threshold)
        return result

    streams = {
        "H0": epoch("H0", historical_gate, "btail_ewma075", float(thresholds["historical_binomial"]["event_q99_threshold"])),
        "A0": epoch("A0", paper_receiver, "a0_robust_pool", float(thresholds["a0_q99"])),
        "A1": epoch("A1", paper_gate, "btail_ewma075", float(thresholds["paper_binomial"]["event_q99_threshold"])),
        "A2": epoch("A2", paper_receiver, "a0_robust_pool", float(thresholds["a0_q99"])),
        "A3": epoch("A3", global_receiver, "set_score", float(thresholds["set_q99"])),
        "A4": epoch("A4", paper_receiver, "set_score", float(thresholds["set_q99"])),
        "Full": epoch("Full", full_blocks.rename(columns={"block_end_s": "availability_time_s", "block_start_s": "window_bin_s"}), "e_cusum", 100.0),
        "Linear-AR": epoch("Linear-AR", linear_blocks.rename(columns={"block_end_s": "availability_time_s", "block_start_s": "window_bin_s"}), "e_cusum", 100.0),
    }
    simple = streams["A0"].copy()
    simple["method"] = "SimpleConsecutive"
    simple["alarm"] = consecutive_alarm(simple.score >= float(thresholds["a0_q99"]), consecutive_epochs=3)
    streams["SimpleConsecutive"] = simple
    return streams, paper_evidence, linear_evidence
