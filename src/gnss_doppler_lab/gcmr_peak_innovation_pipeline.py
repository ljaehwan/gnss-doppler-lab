"""E/P/L-to-GCMR-PI adapter with normal-only fitting and attack-only scoring.

This module deliberately has no dataset I/O.  An EventRecord contains a single
receiver epoch plus exactly W *preceding* E/P/L samples for each visible PRN.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import torch
from torch import nn
from .gcmr_peak_innovation import (safe_prompt_normalize, ConditionalInnovationWhitener,
    NormalOnlyCalibrator, EventDiagnostics, common_drive_statistics,
    normal_loading_threshold, relation_destruction, aggregate_event_score, binomial_tail_from_exceedances)


@dataclass(frozen=True)
class EventRecord:
    """One variable-cardinality epoch; history[prn] is ordered oldest to newest."""
    time: float
    prns: tuple[str, ...]
    epl: np.ndarray                 # (N, 3), current target observation
    histories: Mapping[str, np.ndarray]  # each (W, 3), strictly before ``time``
    cn0: np.ndarray                 # (N,), used as whitener context
    elevation: np.ndarray           # (N,)
    pair_conditions: Mapping[tuple[str, str], np.ndarray]  # symmetric [dot,min_el,max_el]

    def validate(self, window: int, require_pairs: bool = True) -> None:
        n = len(self.prns)
        if n < 2 or len(set(self.prns)) != n:
            raise ValueError("event requires at least two unique PRNs")
        if np.asarray(self.epl).shape != (n, 3) or np.asarray(self.cn0).shape != (n,) or np.asarray(self.elevation).shape != (n,):
            raise ValueError("epl, cn0, and elevation shapes must match PRNs")
        if not all(np.isfinite(np.asarray(x, float)).all() for x in (self.epl, self.cn0, self.elevation)):
            raise ValueError("event values must be finite")
        if set(self.histories) != set(self.prns): raise ValueError("histories must cover exactly the event PRNs")
        for prn in self.prns:
            if np.asarray(self.histories[prn]).shape != (window, 3):
                raise ValueError("every history must have shape (W, 3)")
        if require_pairs:
            for i, a in enumerate(self.prns):
                for b in self.prns[i + 1:]:
                    value = self.pair_conditions.get((a,b), self.pair_conditions.get((b,a)))
                    if value is None or np.asarray(value).shape != (3,) or not np.isfinite(value).all():
                        raise ValueError("every PRN pair needs finite symmetric 3-vector condition")


class _GRU(nn.Module):
    def __init__(self, hidden: int):
        super().__init__(); self.gru = nn.GRU(3, hidden, batch_first=True); self.out = nn.Linear(hidden, 3)
    def forward(self, x): return self.out(self.gru(x)[0][:, -1])


class DirectPairRelationModel:
    """Ridge expected-cosine model, fitted directly from supplied pair conditions."""
    def __init__(self, ridge: float = 1e-3): self.ridge = ridge
    def fit(self, normal_pairs: Sequence[tuple[np.ndarray, float]]):
        if not normal_pairs: raise ValueError("normal pair fitting requires pairs")
        x = np.asarray([p[0] for p in normal_pairs], float); y = np.asarray([p[1] for p in normal_pairs], float)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError("invalid normal pairs")
        X = np.c_[np.ones(len(x)), x]
        self.coef_ = np.linalg.solve(X.T @ X + self.ridge*np.eye(4), X.T @ y); return self
    def expected(self, condition: np.ndarray) -> float:
        if not hasattr(self, "coef_"): raise RuntimeError("fit on normal events first")
        c=np.asarray(condition,float)
        if c.shape != (3,) or not np.isfinite(c).all(): raise ValueError("condition must be finite (3,)")
        return float(np.clip(np.r_[1., c] @ self.coef_, -1., 1.))


@dataclass(frozen=True)
class EventScore:
    time: float; n: int; diagnostics: EventDiagnostics
    scores: Mapping[str, float]; destroyed_pair_score: float


class GCMRPeakInnovationPipeline:
    """Shared no-identity GRU and GCMR-PI event scorer.

    ``fit_normal`` is the only mutating calibration API.  ``score_attack`` has
    no fitting arguments and raises until normal calibration is complete.
    """
    def __init__(self, window: int, hidden_size: int = 8, epochs: int = 30, learning_rate: float = .02, seed: int = 0):
        if window < 1 or hidden_size < 1 or epochs < 1 or learning_rate <= 0: raise ValueError("invalid configuration")
        self.window, self.epochs, self.learning_rate, self.seed = window, epochs, learning_rate, seed
        torch.manual_seed(seed); self.network = _GRU(hidden_size)
        self.whitener = ConditionalInnovationWhitener(min_bin_samples=2)
        self.pairs = DirectPairRelationModel(); self.calibrator = NormalOnlyCalibrator()

    def _arrays(self, event: EventRecord):
        event.validate(self.window)
        # PRN strings only select rows; they are never presented to the neural network.
        h=np.stack([event.histories[p] for p in event.prns]); raw=np.asarray(event.epl,float)
        hn, hv=safe_prompt_normalize(h); yn, yv=safe_prompt_normalize(raw)
        if not hv.all() or not yv.all(): raise ValueError("all E/P/L prompts must exceed the safe normalization threshold")
        return hn, yn
    def _predict_residual(self, event: EventRecord):
        h,y=self._arrays(event)
        with torch.no_grad(): pred=self.network(torch.tensor(h,dtype=torch.float32)).cpu().numpy()
        return y-pred
    @staticmethod
    def _cos(a,b):
        return float(a@b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
    def _pair_score(self, z, event):
        vals=[]
        for i,a in enumerate(event.prns):
            for j,b in enumerate(event.prns[i+1:], i+1):
                c=event.pair_conditions.get((a,b), event.pair_conditions.get((b,a)))
                vals.append(abs(self._cos(z[i],z[j])-self.pairs.expected(c)))
        return float(np.mean(vals))
    def fit_normal(self, normal_train: Sequence[EventRecord], normal_validation: Sequence[EventRecord]):
        """Fit predictor/calibration exclusively from supplied normal-role records."""
        if not normal_train or not normal_validation: raise ValueError("normal train and validation are required")
        train=[self._arrays(e) for e in normal_train]
        x=np.concatenate([a for a,_ in train]); y=np.concatenate([b for _,b in train])
        opt=torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.network.train()
        for _ in range(self.epochs):
            opt.zero_grad(); loss=((self.network(torch.tensor(x,dtype=torch.float32))-torch.tensor(y,dtype=torch.float32))**2).mean(); loss.backward(); opt.step()
        self.network.eval()
        raw=[self._predict_residual(e) for e in normal_validation]
        all_r=np.concatenate(raw); all_c=np.concatenate([e.cn0 for e in normal_validation])
        self.whitener.fit(all_r, all_c)
        zs=[self.whitener.transform(r,e.cn0) for r,e in zip(raw,normal_validation)]
        normal_pairs=[]
        for z,e in zip(zs,normal_validation):
            for i,a in enumerate(e.prns):
                for j,b in enumerate(e.prns[i+1:],i+1):
                    c=e.pair_conditions.get((a,b),e.pair_conditions.get((b,a))); normal_pairs.append((c,self._cos(z[i],z[j])))
        self.pairs.fit(normal_pairs); self.loading_threshold_=normal_loading_threshold(zs)
        normal_scalars=np.concatenate([np.linalg.norm(r,axis=1) for r in raw])
        self.scalar_threshold_=float(np.quantile(normal_scalars, .99))
        # Empirical normal rate is calibration-only and drives A1's binomial baseline.
        self.normal_exceedance_rate_=float(np.clip(np.mean(normal_scalars > self.scalar_threshold_), 1e-6, 1.-1e-6))
        diags=[self._diagnostics(z,e) for z,e in zip(zs,normal_validation)]
        self.calibrator.fit({k: [getattr(d,k) for d in diags] for k in ('scalar_rmse','energy','s_common','n_eff','s_pair')})
        self.fitted_=True; return self
    def _diagnostics(self,z,event):
        stats=common_drive_statistics(z,self.loading_threshold_)
        raw=self._predict_residual(event); scalar=np.linalg.norm(raw,axis=1)
        btail = 1.0 if not hasattr(self, "scalar_threshold_") else binomial_tail_from_exceedances(
            int(np.count_nonzero(scalar > self.scalar_threshold_)), len(z), self.normal_exceedance_rate_)
        return EventDiagnostics(stats.n,stats.n_eff,stats.loading_count,stats.at_least_four,stats.s_common,
          self._pair_score(z,event),float(np.mean(np.sum(z*z,axis=1))),float(np.sqrt(np.mean(scalar*scalar))), btail)
    def score_attack(self, event: EventRecord, destruction_seed: int = 0) -> EventScore:
        """Read-only scoring: no attack record can alter model or calibration state."""
        if not getattr(self,'fitted_',False): raise RuntimeError("call fit_normal with normal data before attack scoring")
        residual=self._predict_residual(event); z=self.whitener.transform(residual,event.cn0); d=self._diagnostics(z,event)
        scores={name: aggregate_event_score(d,self.calibrator,name) for name in ('A0','A1','A2','A3','A4','Full')}
        return EventScore(event.time,len(event.prns),d,scores,self._pair_score(relation_destruction(z,destruction_seed),event))
