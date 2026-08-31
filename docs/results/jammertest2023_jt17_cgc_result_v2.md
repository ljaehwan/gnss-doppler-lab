# JammerTest 2023 JT23-17.1.6 CGC result v2

Date: 2026-08-31

## Outcome

`REAL_CARRYOFF_TRANSFER_SUPPORTED`

The frozen complex-nine-tap, support-normalized CGC rule transferred without
JammerTest threshold or template fitting to one public outdoor coherent-spoof
recording. All preregistered terminal gates passed. This is external
carry-off-sensitivity evidence, not a universal multipath-versus-spoof field
confusion matrix.

## Data and release identity

- Dataset: FGI Jammer Test 2023, `JT23_17.1.6_L1_E1.iq`
- Size: 72,110,000,000 bytes
- Published and verified SHA-256:
  `4bd6e3963f0b3d6806670db5d1a653de05fd9fe602ec42b005eb6cc4d45931e3`
- RF spoof onset: 226 s (independent published label)
- Planned route-motion onset: 526 s (`226 + 300` s fixed-position hold)
- Original preregistration commit: `7f4c880`
- Score-free adapter-amendment commit: `14fe05a`

The v1 support run stopped before tap or score access. It exposed two receiver
interface assumptions: the tracking dump is 50 Hz rather than 1 kHz, and the
receiver performs one 80 ms clock-settle correction. The v2 amendment was
committed and pushed before score access. It changed only the support adapter:
40/50 epochs per PRN/bin and longest cadence-consistent timing segment. The
nine taps, signed-delay template, eight-PRN requirement, primary intervals,
partial-F threshold, and 3-of-5 persistence were unchanged.

## Score-free support

| Region | Eligible geometry bins | Required | PRNs/bin |
|---|---:|---:|---:|
| Clean `[40,200)` | 158 | 60 | 8 |
| Aligned spoof `[246,500)` | 163 | 60 | 8--9 |
| Carry-off onset `[526,556)` | 30 | 20 | 9 |

Nine healthy broadcast-ephemeris PRNs were available. The stable observables
segment began at receiver second 41.34; only full bins from second 42 onward
were admitted. The timing fit residual in the selected 25,918-row segment was
`5.82e-11` s.

## Frozen detector result

| Metric | Clean | Aligned spoof (descriptive) | Carry-off onset |
|---|---:|---:|---:|
| Geometry bins | 158 | 163 | 30 |
| Median partial-F p-value | 0.4551 | 0.2674 | 0.1476 |
| Raw alarm rate | 6.96% | 21.47% | 30.00% |
| Persistent alarm rate | 0.00% | 14.72% | 6.67% |

- First onset-reset 3-of-5 alarm: receiver second 549
- Latency from planned motion onset: 23 s (gate: at most 30 s)
- Clean persistent false-alarm rate: 0% (gate: at most 5%)
- Median-p direction gate: 0.1476 below clean 0.4551
- Secondary serial-bin AUC, clean versus carry-off onset: 0.7842
- Delay rows: 4,199; geometry rows: 423

All six recorded gates passed: the three minimum-bin gates, clean specificity,
onset alarm latency, and median-p direction.

## Interpretation and paper boundary

The result supports the narrow claim that PRN-wise signed correlator delays
become more consistent with one LOS-constrained displacement during the first
prompt-local part of this real coherent carry-off, and that the unchanged
persistent CGC rule alarms without persistent clean false alarms in this
recording. The 23 s latency and AUC 0.7842 also show that the transfer is useful
but not perfect. The paper must not describe this as universal field accuracy
or as a labelled real multipath-versus-spoof confusion matrix.

This result is sufficient to continue the WCL draft and request supervisor
review. Acceptance is not guaranteed; the strongest defensible presentation
is a compact physical-mechanism letter supported by matched receiver-RF,
TEXBAT directionality, GNSS-OpenIF negative control, and this one independently
labelled outdoor coherent-carry-off sensitivity case.

## Immutable local artifacts

- Receiver manifest SHA-256:
  `9234f2bb5e0f923ad3e4fce8662382d94df0d499590d7ff3ed54b4f6bd823eaa`
- Support summary SHA-256:
  `bf21f016ce13ad11df5ce521f0d5fe29bbee0858675512cacc94b12de988f156`
- Evaluation summary SHA-256:
  `77a466c55f410130f5b3ae01a3c5f99085f21b80bb53660b483569841c41c472`

Large I/Q and receiver artifacts remain under
`/home/ubuntu/hdd_data/jammertest2023/` and are not committed to Git.
