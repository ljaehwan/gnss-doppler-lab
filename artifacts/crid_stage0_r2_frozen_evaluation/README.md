# CRID-GNSS Stage-0 R2 frozen evaluation

Final verdict: `INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT`. Phase B was not authorized and no attack payload was opened.

## 1. What CRID measures

CRID tests whether four counterfactual tracking configurations applied to the same IQ can be explained by one shared, dynamics-corrected delay/carrier state. It is not an IQ-power, C/N0, or single-residual detector.

## 2. Physical controls

The R1 all-waveform 0.15-chip/-3-dB fixed duplicate replay terminated correctly and was deterministic, but its q99 alarm ratio was only 3.71%. More importantly, it is not the frozen single/four-PRN smooth-pull-off stimulus. The frozen JSON requires 15 negative and 18 positive cases per domain, while the frozen generator has only nine conforming negative transforms and no conforming PRN-selective smooth positive transform. Phase A therefore cannot be completed without introducing new, unregistered scientific choices.

## 3. TEXBAT/OAKBAT performance

No attack performance was computed. TEXBAT DS1/DS3/DS4/DS7/DS8 and OAKBAT OS3/OS4 payloads remained unopened because Phase A did not authorize Phase B.

## 4. Difference from B0

B0 is an exact baseline only when it can be rerun on identical support. CRID instead tests counterfactual receiver-configuration invariance after causal dynamics compensation. No B0 comparison is claimed here.

## 5. Novelty boundary

Comparing multiple tracking-loop configurations or observing auxiliary peaks is not itself novel. A possible contribution would require the combined same-IQ counterfactual replay, shared-state H0 versus configuration-dependent H1 with complexity correction, and multi-PRN persistence. This R2 result does not establish that contribution.

## 6. Successful and failed scenarios

R1 termination, deterministic C0 reproduction, ten dumps per configuration, and ±1-sample support passed for the diagnostic control. The frozen Phase-A stimulus binding failed. No attack scenario was evaluated.

## 7. Claims

It is valid to claim that receiver replay engineering is repaired and that the preregistered Phase-A generator is incomplete. It is not valid to claim spoofing detection, physical-hypothesis failure, detection advantage, or Stage-1 readiness.

## 8. One next action

Freeze one audited raw-IQ generator implementing all negative controls plus PRN-selective single/four-PRN smooth pull-off, then repeat Phase A before opening attack data.
