#!/usr/bin/env python3
"""Train a conditional integrated normal-only PRN relation predictor.

Design goal:
- Learn from per-PRN node features with a shared encoder; no PRN ID is used as model input.
- Aggregate the currently visible PRN set with permutation-invariant pooling.
- Use receiver graph/relation features as context, but weight relation surprise at scoring time
  only when local PRN/node surprise is high.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

META_NODE = {
    "run_id", "source_fingerprint", "split", "label", "prn", "channel", "sample_rate_hz",
    "segment_index", "window_index", "window_start_s", "window_end_s", "window_mid_s",
    "window_bin_s", "epoch_count",
}
META_GRAPH = {
    "run_id", "label", "split", "window_bin_s", "window_mid_s_min", "window_mid_s_max",
    "window_start_s_min", "window_end_s_max", "tracked_prn_count", "node_count",
    "tracked_prns", "prns",
}


@dataclass
class TrainConfig:
    node_csv: str
    graph_csv: str
    output_dir: str
    seq_len: int = 10
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    node_emb_dim: int = 64
    graph_emb_dim: int = 32
    max_prns: int = 32
    seed: int = 7
    graph_loss_weight: float = 0.35
    gate_center_quantile: float = 0.90


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def numeric_feature_columns(df: pd.DataFrame, meta: set[str]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in meta:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (values.astype(np.float32) - mean) / std
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def aggregate_node_targets(x: np.ndarray) -> np.ndarray:
    # x: [n_prn, node_dim], already standardized. No PRN identity is used.
    # Keep this robust to object-array containers: np.array(list_of_equal_shape_arrays,
    # dtype=object) can yield per-window object dtype arrays, and numpy std then fails.
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] == 0:
        return np.zeros((x.shape[1] * 3,), dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    maxabs = np.max(np.abs(x), axis=0)
    return np.concatenate([mean, std, maxabs], axis=0).astype(np.float32)


class PrnSetSequenceDataset(Dataset):
    def __init__(self, groups: list[tuple[np.ndarray, np.ndarray]], seq_len: int, max_prns: int):
        self.groups = groups
        self.seq_len = seq_len
        self.max_prns = max_prns
        self.index: list[tuple[int, int]] = []
        for gi, (nodes, _graphs) in enumerate(groups):
            if len(nodes) >= seq_len + 1:
                for start in range(0, len(nodes) - seq_len):
                    self.index.append((gi, start))
        if not self.index:
            raise ValueError("no training windows; lower seq_len or provide longer runs")

    def __len__(self) -> int:
        return len(self.index)

    def _pad_nodes(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # x is object array/list of [n_prn, node_dim] arrays for seq_len timesteps.
        node_dim = x[0].shape[1]
        out = np.zeros((len(x), self.max_prns, node_dim), dtype=np.float32)
        mask = np.zeros((len(x), self.max_prns), dtype=np.float32)
        for ti, arr in enumerate(x):
            # Deterministic ordering is allowed for batching, but PRN ID is not an input and pooling is invariant.
            take = min(len(arr), self.max_prns)
            if take:
                out[ti, :take, :] = arr[:take]
                mask[ti, :take] = 1.0
        return out, mask

    def __getitem__(self, idx: int):
        gi, start = self.index[idx]
        nodes, graphs = self.groups[gi]
        node_seq, mask_seq = self._pad_nodes(nodes[start : start + self.seq_len])
        graph_seq = graphs[start : start + self.seq_len]
        target_nodes = aggregate_node_targets(nodes[start + self.seq_len])
        target_graph = graphs[start + self.seq_len]
        return (
            torch.from_numpy(node_seq),
            torch.from_numpy(mask_seq),
            torch.from_numpy(graph_seq.astype(np.float32)),
            torch.from_numpy(target_nodes),
            torch.from_numpy(target_graph.astype(np.float32)),
        )


class ConditionalIntegratedGRU(nn.Module):
    def __init__(self, node_dim: int, graph_dim: int, node_target_dim: int, cfg: TrainConfig):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, cfg.node_emb_dim), nn.LayerNorm(cfg.node_emb_dim), nn.GELU(),
            nn.Linear(cfg.node_emb_dim, cfg.node_emb_dim), nn.GELU(),
        )
        # mean + max pooling over current visible PRN set
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_dim, cfg.graph_emb_dim), nn.LayerNorm(cfg.graph_emb_dim), nn.GELU(),
        )
        fusion_dim = cfg.node_emb_dim * 2 + cfg.graph_emb_dim
        self.gru = nn.GRU(input_size=fusion_dim, hidden_size=cfg.hidden_dim, batch_first=True)
        self.node_head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, node_target_dim))
        self.graph_head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, graph_dim))

    def forward(self, node_seq: torch.Tensor, mask_seq: torch.Tensor, graph_seq: torch.Tensor):
        # node_seq [B,T,P,F], mask_seq [B,T,P], graph_seq [B,T,G]
        b, t, p, f = node_seq.shape
        emb = self.node_encoder(node_seq.reshape(b * t * p, f)).reshape(b, t, p, -1)
        mask = mask_seq.unsqueeze(-1)
        count = mask.sum(dim=2).clamp_min(1.0)
        mean_pool = (emb * mask).sum(dim=2) / count
        emb_for_max = emb.masked_fill(mask <= 0, -1e9)
        max_pool = emb_for_max.max(dim=2).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        gemb = self.graph_encoder(graph_seq)
        fused = torch.cat([mean_pool, max_pool, gemb], dim=-1)
        out, _ = self.gru(fused)
        h = out[:, -1, :]
        return self.node_head(h), self.graph_head(h)


def build_groups(node_df: pd.DataFrame, graph_df: pd.DataFrame, node_cols: list[str], graph_cols: list[str], splits: Iterable[str] | None, cfg: TrainConfig):
    if splits is not None and "split" in node_df.columns and "split" in graph_df.columns:
        node_df = node_df[node_df["split"].isin(splits)].copy()
        graph_df = graph_df[graph_df["split"].isin(splits)].copy()
    train_node_df = node_df
    train_graph_df = graph_df
    node_mean, node_std = fit_standardizer(train_node_df[node_cols].to_numpy(np.float32))
    graph_mean, graph_std = fit_standardizer(train_graph_df[graph_cols].to_numpy(np.float32))

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for run_id, g_run in graph_df.groupby("run_id", sort=True):
        n_run = node_df[node_df["run_id"] == run_id]
        if n_run.empty:
            continue
        graph_rows = []
        node_sets = []
        for _, grow in g_run.sort_values("window_bin_s").iterrows():
            bin_s = grow["window_bin_s"]
            nodes = n_run[n_run["window_bin_s"] == bin_s].sort_values("prn")
            if nodes.empty:
                continue
            nx = standardize(nodes[node_cols].to_numpy(np.float32), node_mean, node_std)
            gx = standardize(grow[graph_cols].to_numpy(np.float32)[None, :], graph_mean, graph_std)[0]
            node_sets.append(nx)
            graph_rows.append(gx)
        if len(node_sets) >= cfg.seq_len + 1:
            # Store a 1-D object array of per-window float32 PRN sets. Using
            # np.array(node_sets, dtype=object) can create a higher-rank object
            # array when all windows have the same PRN count, causing dtype=object
            # arrays to leak into aggregate_node_targets.
            node_obj = np.empty(len(node_sets), dtype=object)
            node_obj[:] = node_sets
            groups.append((node_obj, np.stack(graph_rows).astype(np.float32)))
    return groups, {"node_mean": node_mean, "node_std": node_std, "graph_mean": graph_mean, "graph_std": graph_std}


def run_epoch(model, loader, opt, device, graph_loss_weight: float):
    train = opt is not None
    model.train(train)
    total = node_total = graph_total = 0.0
    n = 0
    mse = nn.MSELoss(reduction="mean")
    for node_seq, mask_seq, graph_seq, target_nodes, target_graph in loader:
        node_seq = node_seq.to(device)
        mask_seq = mask_seq.to(device)
        graph_seq = graph_seq.to(device)
        target_nodes = target_nodes.to(device)
        target_graph = target_graph.to(device)
        pred_node, pred_graph = model(node_seq, mask_seq, graph_seq)
        node_loss = mse(pred_node, target_nodes)
        graph_loss = mse(pred_graph, target_graph)
        loss = node_loss + graph_loss_weight * graph_loss
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = node_seq.shape[0]
        total += float(loss.detach().cpu()) * bs
        node_total += float(node_loss.detach().cpu()) * bs
        graph_total += float(graph_loss.detach().cpu()) * bs
        n += bs
    return {"loss": total / n, "node_loss": node_total / n, "graph_loss": graph_total / n, "count": n}


@torch.no_grad()
def score_model(model, loader, device, gate_center: float):
    model.eval()
    rows = []
    for node_seq, mask_seq, graph_seq, target_nodes, target_graph in loader:
        node_seq = node_seq.to(device)
        mask_seq = mask_seq.to(device)
        graph_seq = graph_seq.to(device)
        target_nodes = target_nodes.to(device)
        target_graph = target_graph.to(device)
        pred_node, pred_graph = model(node_seq, mask_seq, graph_seq)
        node_err = ((pred_node - target_nodes) ** 2).mean(dim=1).sqrt()
        graph_err = ((pred_graph - target_graph) ** 2).mean(dim=1).sqrt()
        # Conditional relation: graph surprise becomes important only when local/node surprise is high.
        gate = torch.sigmoid((node_err - gate_center) * 8.0)
        joint = node_err + gate * graph_err
        for a, b, c, d in zip(node_err.cpu(), graph_err.cpu(), gate.cpu(), joint.cpu()):
            rows.append({"node_rmse": float(a), "graph_rmse": float(b), "relation_gate": float(c), "joint_score": float(d)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-csv", default="artifacts/model_datasets/normal_multi_prn_morphology_dynamics_v2/normal_prn_node_windows.csv")
    ap.add_argument("--graph-csv", default="artifacts/model_datasets/normal_multi_prn_morphology_dynamics_v2/normal_receiver_graph_windows.csv")
    ap.add_argument("--output-dir", default="artifacts/conditional_integrated_gru_poc")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=10)
    args = ap.parse_args()
    cfg = TrainConfig(node_csv=args.node_csv, graph_csv=args.graph_csv, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size, seq_len=args.seq_len)
    seed_all(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    node_df = pd.read_csv(cfg.node_csv)
    graph_df = pd.read_csv(cfg.graph_csv)
    node_cols = numeric_feature_columns(node_df, META_NODE)
    graph_cols = numeric_feature_columns(graph_df, META_GRAPH)
    if not node_cols or not graph_cols:
        raise RuntimeError("missing numeric node/graph features")
    if "split" in node_df.columns:
        train_splits = ["train"] if "train" in set(node_df["split"].astype(str)) else None
        val_splits = ["val", "validation"] if any(s in set(node_df["split"].astype(str)) for s in ["val", "validation"]) else None
    else:
        train_splits = val_splits = None

    if train_splits:
        train_groups, stats = build_groups(node_df, graph_df, node_cols, graph_cols, train_splits, cfg)
        val_groups, _ = build_groups(node_df, graph_df, node_cols, graph_cols, val_splits, cfg) if val_splits else ([], None)
    else:
        # Run-level split, not random row split, to avoid leakage across adjacent windows.
        run_ids = sorted(graph_df["run_id"].astype(str).unique())
        cut = max(1, int(len(run_ids) * 0.8))
        train_runs = set(run_ids[:cut])
        val_runs = set(run_ids[cut:]) or set(run_ids[-1:])
        train_groups, stats = build_groups(node_df[node_df.run_id.astype(str).isin(train_runs)], graph_df[graph_df.run_id.astype(str).isin(train_runs)], node_cols, graph_cols, None, cfg)
        val_groups, _ = build_groups(node_df[node_df.run_id.astype(str).isin(val_runs)], graph_df[graph_df.run_id.astype(str).isin(val_runs)], node_cols, graph_cols, None, cfg)

    if not val_groups:
        # Last-resort small validation from training groups, sufficient for PoC smoke training only.
        val_groups = train_groups[-1:]
        train_groups = train_groups[:-1] or train_groups

    train_ds = PrnSetSequenceDataset(train_groups, cfg.seq_len, cfg.max_prns)
    val_ds = PrnSetSequenceDataset(val_groups, cfg.seq_len, cfg.max_prns)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalIntegratedGRU(len(node_cols), len(graph_cols), len(node_cols) * 3, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_val = math.inf
    best_path = out / "conditional_integrated_gru_predictor.pt"
    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, opt, device, cfg.graph_loss_weight)
        va = run_epoch(model, val_loader, None, device, cfg.graph_loss_weight)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "node_feature_columns": node_cols,
                "graph_feature_columns": graph_cols,
                "standardizer": {k: v.tolist() for k, v in stats.items()},
                "architecture_note": "shared PRN node encoder + permutation-invariant set pooling + graph context encoder + GRU; no PRN ID input; relation graph contributes through conditional gate during scoring",
            }, best_path)

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    # Load best before scoring
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    raw_val_scores = score_model(model, val_loader, device, gate_center=0.0)
    gate_center = float(raw_val_scores["node_rmse"].quantile(cfg.gate_center_quantile))
    val_scores = score_model(model, val_loader, device, gate_center=gate_center)
    val_scores.to_csv(out / "validation_conditional_scores.csv", index=False)

    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": torch.cuda.get_device_name(0), "total_memory_gb": props.total_memory / 1024**3}
    summary = {
        "purpose": "normal-only spoofing anomaly PoC: learn PRN-local behavior first; use inter-PRN relation as conditional context when node/local surprise is high",
        "architecture": "ConditionalIntegratedGRU",
        "single_model": True,
        "uses_prn_id_as_input": False,
        "relation_policy": "conditional gate: joint_score = node_rmse + sigmoid((node_rmse - q90_node_rmse)*8) * graph_rmse",
        "device": str(device),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": gpu},
        "data": {"node_csv": cfg.node_csv, "graph_csv": cfg.graph_csv, "node_rows": len(node_df), "graph_rows": len(graph_df), "train_windows": len(train_ds), "val_windows": len(val_ds)},
        "features": {"node_feature_count": len(node_cols), "graph_feature_count": len(graph_cols), "node_feature_columns": node_cols, "graph_feature_columns": graph_cols},
        "best_val_loss": best_val,
        "gate_center_node_rmse_q90": gate_center,
        "validation_score_summary": val_scores.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "artifacts": {"model": str(best_path), "history": str(out / "training_history.csv"), "scores": str(out / "validation_conditional_scores.csv")},
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
