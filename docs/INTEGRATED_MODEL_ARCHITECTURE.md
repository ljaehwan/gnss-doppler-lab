# Integrated PRN-relation normal-only model design

## Key decisions

1. The final detector is **one integrated model**, not three independent models.
2. The two CSV files are two input views:
   - `normal_prn_node_windows.csv`: variable visible-PRN node features.
   - `normal_receiver_graph_windows.csv`: receiver-level relation/context features computed from whichever PRNs are currently visible/tracked.
3. The model must **not memorize fixed PRN IDs** as the core relationship definition. Different times and locations see different GPS PRNs, so relation reasoning must be permutation-aware over the currently available PRN set.
4. Inter-PRN relation is not an always-on equal-weight detector. Its purpose is conditional context: when PRN-local evidence becomes abnormal, relation context helps decide whether the abnormality is isolated to one PRN or jointly expressed across other currently visible PRNs.
5. Training should use GPU automatically when available. On the current GNSS VM check, CUDA is **not visible yet**: `nvidia-smi` is missing, `/dev/nvidia*` is absent, and PyTorch reports `cuda_available=False`. The code should still use `cuda` when the VM gets A6000 passthrough/driver support.

## Data interpretation

At each time window `t`:

```text
Visible/tracked PRN set: P_t = {currently observed PRNs}
Node input: X_node[t, p, :] for p in P_t
Graph/context input: X_graph[t, :] computed from P_t as a set/relation summary
Mask: M[t, p] indicates which PRNs are present
```

The model should treat PRNs as a dynamic set, not a fixed ordered list such as G01..G32 with learned identity shortcuts.

Acceptable use of PRN ID:
- optional metadata or weak embedding for diagnostics/ablation.

Preferred core relation features:
- statistics across currently tracked PRNs;
- common-mode removed Doppler/code residual spread;
- morphology spread across current PRNs;
- tracked count and masks;
- optionally geometry/elevation/LOS context later.

## Recommended architecture

```text
PRN-node sequence [B, T, N, F_node] + mask [B, T, N]
        │
        ▼
Shared node encoder applied to every PRN
        │
        ▼
Permutation-aware pooling/attention over current PRNs
        │
        ├──────────────┐
        │              ▼
        │       Conditional relation gate
        │              ▲
        ▼              │
Graph/context encoder from receiver graph features [B, T, F_graph]
        │              │
        └──────► Fusion over local evidence + relation context
                       │
                       ▼
                  GRU temporal backbone
                       │
                       ▼
              Next normal-state prediction
                       │
                       ▼
              One joint anomaly score
```

## Loss and score

Training loss:

```text
L_total = L_node + lambda_graph * g(local_error) * L_graph
```

For the first implementation, `g` can be a stable learned/sigmoid gate or a simple monotonic function of normalized local prediction error. The key point is that graph/relation context is conditionally weighted, not blindly dominant at every time.

Inference score:

```text
S_total = normalized_node_error + lambda_graph * gate * normalized_graph_error
```

Final decision:

```text
S_total > threshold
```

PRN-level errors may be saved for explanation, but the detector output is one receiver-level anomaly score.

## GPU execution rule

Training code should select device as:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

and log:

```text
torch version
cuda availability
GPU name
VRAM
batch size
mixed precision on/off
```

If CUDA is not visible, do not pretend GPU training is active; report that VM GPU passthrough/driver setup is missing.
