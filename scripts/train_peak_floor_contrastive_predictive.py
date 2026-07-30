#!/usr/bin/env python3
"""Train normal-only Peak–Floor Contrastive Predictive Coding (PF-CPC).

This is not late fusion of B0/M1 scores. It jointly encodes per-PRN nine-tap
correlation morphology and receiver-wide raw-IQ floor features at each epoch,
then predicts the next joint embedding from a causal history. Symmetric InfoNCE
learns clean temporal continuity; cosine surprisal is the anomaly score.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

_BASE_PATH = Path(__file__).with_name("train_peak_floor_temporal_autoencoder.py")
_spec = importlib.util.spec_from_file_location("_peak_floor_data_contract", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
assert _spec.loader is not None
_spec.loader.exec_module(_base)

MAX_PRNS = _base.MAX_PRNS
DEFAULT_MORPH_FEATURES = _base.DEFAULT_MORPH_FEATURES
DEFAULT_FLOOR_FEATURES = _base.DEFAULT_FLOOR_FEATURES
DEFAULT_SPLIT_RULES = _base.DEFAULT_SPLIT_RULES
AlignedData = _base.AlignedData
align_modalities = _base.align_modalities
partition_aligned = _base.partition_aligned
fit_robust_scalers = _base.fit_robust_scalers
apply_scalers = _base.apply_scalers
validate_normal_only_inputs = _base.validate_normal_only_inputs
sha256 = _base.sha256
SCHEMA = "gnss-doppler-lab.peak-floor-cpc.v1"


@dataclass
class PredictivePairs:
    context_times: np.ndarray
    target_times: np.ndarray
    available_times: np.ndarray
    context_morph: np.ndarray
    context_floor: np.ndarray
    context_mask: np.ndarray
    target_morph: np.ndarray
    target_floor: np.ndarray
    target_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.target_times.shape[0])


@dataclass
class ModelConfig:
    morph_dim: int
    floor_dim: int
    context_len: int = 12
    hidden_dim: int = 64
    embedding_dim: int = 32
    token_layers: int = 2
    token_heads: int = 4
    dropout: float = 0.1
    temperature: float = 0.1


class PairDataset(Dataset):
    def __init__(self, pairs: PredictivePairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs
        return (
            torch.from_numpy(p.context_morph[i]), torch.from_numpy(p.context_floor[i]),
            torch.from_numpy(p.context_mask[i]), torch.from_numpy(p.target_morph[i]),
            torch.from_numpy(p.target_floor[i]), torch.from_numpy(p.target_mask[i]),
        )


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_predictive_pairs(data: AlignedData, context_len: int, horizon: int = 1, stride: int = 1,
                          expected_cadence_s: float = 0.5, tolerance_s: float = 1e-4) -> PredictivePairs:
    if context_len < 1 or horizon < 1 or stride < 1:
        raise ValueError("context_len, horizon, and stride must be positive")
    context_times=[]; target_times=[]; available=[]; cm=[]; cf=[]; ck=[]; tm=[]; tf=[]; tk=[]
    stop = len(data.times) - horizon
    for end in range(context_len - 1, stop, stride):
        start = end - context_len + 1
        target = end + horizon
        support_times = data.times[start:target + 1]
        if not np.allclose(np.diff(support_times), expected_cadence_s, atol=tolerance_s, rtol=0):
            continue
        context_times.append(data.times[start:end + 1]); target_times.append(data.times[target])
        available.append(max(float(data.available_times[target]), float(data.times[target])))
        cm.append(data.morph[start:end + 1]); cf.append(data.floor[start:end + 1]); ck.append(data.prn_mask[start:end + 1])
        tm.append(data.morph[target]); tf.append(data.floor[target]); tk.append(data.prn_mask[target])
    if not target_times:
        raise ValueError("no contiguous predictive pairs were built")
    return PredictivePairs(
        np.asarray(context_times, np.float64), np.asarray(target_times, np.float64), np.asarray(available, np.float64),
        np.asarray(cm, np.float32), np.asarray(cf, np.float32), np.asarray(ck, bool),
        np.asarray(tm, np.float32), np.asarray(tf, np.float32), np.asarray(tk, bool),
    )


class PeakFloorCPC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__(); self.cfg = cfg
        h = cfg.hidden_dim
        self.morph_encoder = nn.Sequential(nn.Linear(cfg.morph_dim, h), nn.LayerNorm(h), nn.SiLU())
        self.floor_encoder = nn.Sequential(nn.Linear(cfg.floor_dim, h), nn.LayerNorm(h), nn.SiLU())
        self.type_embedding = nn.Parameter(torch.zeros(2, h))
        layer = nn.TransformerEncoderLayer(h, cfg.token_heads, dim_feedforward=2*h,
                                           dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=False)
        self.relation_encoder = nn.TransformerEncoder(layer, cfg.token_layers)
        self.epoch_projector = nn.Sequential(nn.Linear(2*h, h), nn.SiLU(), nn.Linear(h, cfg.embedding_dim))
        self.context_gru = nn.GRU(cfg.embedding_dim, h, batch_first=True)
        self.future_predictor = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cfg.embedding_dim))

    def encode_epoch(self, morph: torch.Tensor, floor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        prefix = morph.shape[:-2]
        n = math.prod(prefix) if prefix else 1
        m = morph.reshape(n, MAX_PRNS, self.cfg.morph_dim)
        f = floor.reshape(n, self.cfg.floor_dim)
        valid = mask.reshape(n, MAX_PRNS)
        morph_tokens = self.morph_encoder(m) + self.type_embedding[0]
        floor_token = self.floor_encoder(f).unsqueeze(1) + self.type_embedding[1]
        tokens = torch.cat([morph_tokens, floor_token], dim=1)
        padding = torch.cat([~valid, torch.zeros(n, 1, dtype=torch.bool, device=valid.device)], dim=1)
        related = self.relation_encoder(tokens, src_key_padding_mask=padding)
        denom = valid.sum(1, keepdim=True).clamp_min(1).to(related.dtype)
        morph_pool = (related[:, :MAX_PRNS] * valid.unsqueeze(-1)).sum(1) / denom
        floor_pool = related[:, -1]
        z = self.epoch_projector(torch.cat([morph_pool, floor_pool], dim=-1))
        return F.normalize(z, dim=-1).reshape(*prefix, self.cfg.embedding_dim)

    def forward(self, context_morph, context_floor, context_mask, target_morph, target_floor, target_mask):
        context_z = self.encode_epoch(context_morph, context_floor, context_mask)
        _, state = self.context_gru(context_z)
        predicted = F.normalize(self.future_predictor(state[-1]), dim=-1)
        actual = self.encode_epoch(target_morph, target_floor, target_mask)
        return {"predicted": predicted, "actual": actual, "context": context_z}


def symmetric_info_nce(predicted: torch.Tensor, actual: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    predicted = F.normalize(predicted, dim=-1); actual = F.normalize(actual, dim=-1)
    logits = predicted @ actual.T / temperature
    labels = torch.arange(len(predicted), device=predicted.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def train_model(train: PredictivePairs, validation: PredictivePairs, cfg: ModelConfig, epochs: int,
                batch_size: int, lr: float, weight_decay: float, device: torch.device):
    model = PeakFloorCPC(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(PairDataset(train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(PairDataset(validation), batch_size=batch_size, shuffle=False)
    best = math.inf; best_state = None; history=[]
    for epoch in range(1, epochs + 1):
        model.train(); losses=[]
        for batch in train_loader:
            if batch[0].shape[0] < 2: continue
            batch=[x.to(device) for x in batch]; opt.zero_grad(set_to_none=True)
            out=model(*batch); loss=symmetric_info_nce(out["predicted"], out["actual"], cfg.temperature)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        if not losses: raise ValueError("training requires at least one batch with two predictive pairs")
        model.eval(); vals=[]
        with torch.no_grad():
            for batch in val_loader:
                if batch[0].shape[0] < 2: continue
                out=model(*[x.to(device) for x in batch])
                vals.append(float(symmetric_info_nce(out["predicted"], out["actual"], cfg.temperature).cpu()))
        val=float(np.mean(vals)) if vals else float(np.mean(losses))
        if val < best:
            best=val; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({"epoch":epoch,"train_info_nce":float(np.mean(losses)),"validation_info_nce":val})
    assert best_state is not None; model.load_state_dict(best_state)
    return model, history, best


def score_pairs(model: PeakFloorCPC, pairs: PredictivePairs, batch_size: int, device: torch.device) -> pd.DataFrame:
    model.eval(); values=[]
    with torch.no_grad():
        for batch in DataLoader(PairDataset(pairs), batch_size=batch_size, shuffle=False):
            out=model(*[x.to(device) for x in batch])
            values.extend((1.0 - F.cosine_similarity(out["predicted"], out["actual"], dim=-1)).cpu().numpy().tolist())
    return pd.DataFrame({"window_start_s":pairs.target_times, "available_time_s":pairs.available_times,
                         "pf_cpc_surprisal":np.clip(np.asarray(values, float), 0.0, 2.0)})


def _tail_pvalues(scores: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    calibration=np.asarray(calibration,float); scores=np.asarray(scores,float)
    return np.asarray([(1.0 + np.sum(calibration >= s)) / (len(calibration) + 1.0) for s in scores])


def _json(path: Path, value) -> None:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"{type(obj).__name__} is not JSON serializable")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def run_campaign(morph_csv: Path, floor_csv: Path, output_dir: Path, *, epochs: int=20, batch_size: int=128,
                 context_len: int=12, hidden_dim: int=64, embedding_dim: int=32, token_layers: int=2,
                 token_heads: int=4, temperature: float=0.1, dropout: float=0.1, lr: float=1e-3,
                 weight_decay: float=1e-4, split_rules: Mapping[str, tuple[float|None,float|None]]|None=None,
                 device: str="auto", seed: int=17):
    seed_all(seed); morph_csv=Path(morph_csv); floor_csv=Path(floor_csv); output_dir=Path(output_dir)
    morph=pd.read_csv(morph_csv); floor=pd.read_csv(floor_csv)
    contract=validate_normal_only_inputs(morph,floor)
    aligned=align_modalities(morph,floor,DEFAULT_MORPH_FEATURES,DEFAULT_FLOOR_FEATURES)
    rules=dict(split_rules or DEFAULT_SPLIT_RULES); parts=partition_aligned(aligned,rules)
    scalers=fit_robust_scalers(parts["train"])
    scaled={k:apply_scalers(v,scalers) for k,v in parts.items()}
    pairs={k:make_predictive_pairs(v,context_len=context_len) for k,v in scaled.items()}
    cfg=ModelConfig(len(DEFAULT_MORPH_FEATURES),len(DEFAULT_FLOOR_FEATURES),context_len,hidden_dim,
                    embedding_dim,token_layers,token_heads,dropout,temperature)
    dev=_device(device); model,history,best=train_model(pairs["train"],pairs["validation"],cfg,epochs,batch_size,lr,weight_decay,dev)
    calibration=score_pairs(model,pairs["calibration"],batch_size,dev)
    held=score_pairs(model,pairs["held_clean"],batch_size,dev)
    held["conformal_p_value"]=_tail_pvalues(held.pf_cpc_surprisal.to_numpy(),calibration.pf_cpc_surprisal.to_numpy())
    output_dir.mkdir(parents=True,exist_ok=False)
    torch.save({"model_state_dict":model.state_dict(),"model_config":asdict(cfg),
                "feature_contract":{"morph":DEFAULT_MORPH_FEATURES,"floor":DEFAULT_FLOOR_FEATURES}},output_dir/"model.pt")
    _json(output_dir/"model_metadata.json",{"schema":SCHEMA,"architecture":"PeakFloorCPC",
          "objective":"symmetric_info_nce_future_prediction","normal_only_training":True,"seed":seed,
          "device":str(dev),"parameters":sum(p.numel() for p in model.parameters()),"best_validation_loss":best,
          "model_config":asdict(cfg),"normal_input_contract":contract,
          "source":{"morph_csv":str(morph_csv.resolve()),"morph_sha256":sha256(morph_csv),
                    "floor_csv":str(floor_csv.resolve()),"floor_sha256":sha256(floor_csv)}})
    _json(output_dir/"scalers.json",scalers)
    _json(output_dir/"split_manifest.json",{k:{"rule":rules[k],"epochs":len(parts[k].times),"pairs":len(pairs[k])} for k in rules})
    pd.DataFrame(history).to_csv(output_dir/"training_history.csv",index=False)
    calibration.to_csv(output_dir/"calibration_scores.csv",index=False)
    _json(output_dir/"calibration.json",{"method":"empirical_upper_tail_rank","scores":len(calibration),
          "quantiles":{str(q):float(calibration.pf_cpc_surprisal.quantile(q)) for q in (0.95,0.99,0.999)}})
    held.to_csv(output_dir/"held_clean_scores.csv",index=False)
    summary={"windows":len(held),"mean_p_value":float(held.conformal_p_value.mean()),
             "median_p_value":float(held.conformal_p_value.median()),
             "fraction_p_le_0_05":float((held.conformal_p_value<=0.05).mean()),
             "fraction_p_le_0_01":float((held.conformal_p_value<=0.01).mean())}
    _json(output_dir/"held_clean_summary.json",summary)
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output_dir.iterdir() if p.is_file()}
    _json(output_dir/"campaign_manifest.json",{"schema":SCHEMA,"artifacts":artifacts})
    return {"held_clean":summary,"best_validation_loss":best,"output_dir":str(output_dir)}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--morph-csv",type=Path,required=True); p.add_argument("--floor-csv",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=128); p.add_argument("--context-len",type=int,default=12)
    p.add_argument("--hidden-dim",type=int,default=64); p.add_argument("--embedding-dim",type=int,default=32)
    p.add_argument("--token-layers",type=int,default=2); p.add_argument("--token-heads",type=int,default=4)
    p.add_argument("--temperature",type=float,default=0.1); p.add_argument("--device",default="auto")
    p.add_argument("--seed",type=int,default=17); return p.parse_args()


def main():
    a=parse_args(); result=run_campaign(a.morph_csv,a.floor_csv,a.output_dir,epochs=a.epochs,batch_size=a.batch_size,
        context_len=a.context_len,hidden_dim=a.hidden_dim,embedding_dim=a.embedding_dim,
        token_layers=a.token_layers,token_heads=a.token_heads,temperature=a.temperature,device=a.device,seed=a.seed)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__": main()
