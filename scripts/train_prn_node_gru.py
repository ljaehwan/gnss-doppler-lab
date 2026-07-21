#!/usr/bin/env python3
"""Train a PRN-local normal-only GRU predictor from Doppler/tap features.

No PRN relation, no receiver graph features, and no PRN ID input are used. Each
sample is one PRN's own feature history predicting that same PRN's next feature
vector. Spoofing score is next-window prediction RMSE on standardized features.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

META_COLS = {
    "run_id", "source_fingerprint", "split", "label", "prn", "channel", "sample_rate_hz",
    "segment_index", "window_index", "window_start_s", "window_end_s", "window_mid_s",
    "window_bin_s", "epoch_count", "tap_count", "tap_layout",
}

@dataclass
class TrainConfig:
    node_csv: str
    output_dir: str
    seq_len: int = 12
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    emb_dim: int = 128
    dropout: float = 0.05
    seed: int = 11
    feature_subset: str = "all_numeric"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def select_feature_columns(df: pd.DataFrame, subset: str) -> list[str]:
    cols = numeric_feature_columns(df)
    if subset == "all_numeric":
        return cols
    if subset == "tap_rel_prompt_mean":
        return [c for c in cols if c.startswith("tap_") and "_rel_prompt_mean" in c]
    raise ValueError(f"unknown --feature-subset: {subset}")


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


class PrnSequenceDataset(Dataset):
    def __init__(self, series: list[np.ndarray], seq_len: int):
        self.series = series
        self.seq_len = seq_len
        self.index: list[tuple[int, int]] = []
        for gi, arr in enumerate(series):
            if len(arr) >= seq_len + 1:
                for start in range(len(arr) - seq_len):
                    self.index.append((gi, start))
        if not self.index:
            raise ValueError("no PRN-local training windows; lower --seq-len or provide longer data")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        gi, start = self.index[idx]
        arr = self.series[gi]
        return torch.from_numpy(arr[start:start+self.seq_len]), torch.from_numpy(arr[start+self.seq_len])


class PrnLocalGRU(nn.Module):
    def __init__(self, feature_dim: int, cfg: TrainConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, cfg.emb_dim), nn.LayerNorm(cfg.emb_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.GELU(),
        )
        self.gru = nn.GRU(input_size=cfg.emb_dim, hidden_size=cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        z = self.encoder(x.reshape(b*t, f)).reshape(b, t, -1)
        out, _ = self.gru(z)
        return self.head(out[:, -1])


def build_series(df: pd.DataFrame, feature_cols: list[str], mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    series: list[np.ndarray] = []
    for (_run, _prn), g in df.groupby(["run_id", "prn"], sort=True):
        g = g.sort_values("window_bin_s")
        x = standardize(g[feature_cols].to_numpy(np.float32), mean, std)
        if len(x):
            series.append(x)
    return series


def run_epoch(model, loader, opt, device):
    train = opt is not None
    model.train(train)
    mse = nn.MSELoss()
    total = 0.0
    n = 0
    for seq, target in loader:
        seq = seq.to(device)
        target = target.to(device)
        pred = model(seq)
        loss = mse(pred, target)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = seq.shape[0]
        total += float(loss.detach().cpu()) * bs
        n += bs
    return {"loss": total / max(1, n), "rmse": math.sqrt(total / max(1, n)), "count": n}


@torch.no_grad()
def score_dataset(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for seq, target in loader:
        seq = seq.to(device)
        target = target.to(device)
        pred = model(seq)
        rmse = torch.sqrt(((pred - target) ** 2).mean(dim=1))
        mae = torch.mean(torch.abs(pred - target), dim=1)
        for r, a in zip(rmse.cpu(), mae.cpu()):
            rows.append({"prn_node_rmse": float(r), "prn_node_mae": float(a)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--feature-subset", choices=["all_numeric", "tap_rel_prompt_mean"], default="all_numeric",
                    help="Feature family for shared PRN encoder; tap_rel_prompt_mean matches q70 morphology framing.")
    args = ap.parse_args()
    cfg = TrainConfig(node_csv=args.node_csv, output_dir=args.output_dir, seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size, feature_subset=args.feature_subset)
    seed_all(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.node_csv)
    feature_cols = select_feature_columns(df, cfg.feature_subset)
    if not feature_cols:
        raise RuntimeError("no numeric Doppler/tap feature columns found")
    run_ids = sorted(df["run_id"].astype(str).unique())
    if len(run_ids) > 1:
        cut = max(1, int(len(run_ids) * 0.8))
        train_runs = set(run_ids[:cut])
        val_runs = set(run_ids[cut:]) or set(run_ids[-1:])
        train_df = df[df["run_id"].astype(str).isin(train_runs)].copy()
        val_df = df[df["run_id"].astype(str).isin(val_runs)].copy()
        split_doc = {"mode": "run_holdout", "train_runs": sorted(train_runs), "val_runs": sorted(val_runs)}
    else:
        prns = sorted(df["prn"].astype(str).unique())
        val_prns = set(prns[-max(1, len(prns)//5):])
        train_df = df[~df["prn"].astype(str).isin(val_prns)].copy()
        val_df = df[df["prn"].astype(str).isin(val_prns)].copy()
        split_doc = {"mode": "prn_holdout", "val_prns": sorted(val_prns)}
    mean, std = fit_standardizer(train_df[feature_cols].to_numpy(np.float32))
    train_series = build_series(train_df, feature_cols, mean, std)
    val_series = build_series(val_df, feature_cols, mean, std)
    train_ds = PrnSequenceDataset(train_series, cfg.seq_len)
    val_ds = PrnSequenceDataset(val_series, cfg.seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrnLocalGRU(len(feature_cols), cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = math.inf
    history = []
    best_path = out / "prn_local_gru_predictor.pt"
    for epoch in range(1, cfg.epochs+1):
        tr = run_epoch(model, train_loader, opt, device)
        va = run_epoch(model, val_loader, None, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "node_feature_columns": feature_cols,
                "standardizer": {"node_mean": mean.tolist(), "node_std": std.tolist()},
                "architecture_note": "PRN-local GRU next-window predictor; no receiver graph, no PRN relation, no PRN ID input",
            }, best_path)
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_scores = score_dataset(model, val_loader, device)
    val_scores.to_csv(out / "validation_prn_node_scores.csv", index=False)
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": torch.cuda.get_device_name(0), "total_memory_gb": props.total_memory / 1024**3}
    summary = {
        "purpose": "normal-only spoofing anomaly baseline using only per-PRN Doppler/tap features",
        "architecture": "PrnLocalGRU",
        "uses_prn_id_as_input": False,
        "uses_receiver_graph_or_prn_relation": False,
        "score": "per-PRN next-window standardized feature prediction RMSE",
        "device": str(device),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": gpu},
        "data": {"node_csv": cfg.node_csv, "node_rows": len(df), "train_windows": len(train_ds), "val_windows": len(val_ds), "split": split_doc},
        "features": {"feature_subset": cfg.feature_subset, "node_feature_count": len(feature_cols), "node_feature_columns": feature_cols},
        "best_val_loss": best_val,
        "validation_score_summary": val_scores.describe(percentiles=[.5, .9, .95, .99, .995, .999]).to_dict(),
        "artifacts": {"model": str(best_path), "history": str(out / "training_history.csv"), "scores": str(out / "validation_prn_node_scores.csv")},
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
