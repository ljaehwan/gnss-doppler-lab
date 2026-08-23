# Spatial identifiability review

A 4×4 sample covariance is algebraically possible if the inferred four-channel complex layout is confirmed. A 1024-sample snapshot gives at most 1024 temporal observations, enough to form a full-rank 4×4 covariance, but not enough by itself to make its eigenvectors stable under nonstationary wideband waveforms, clipping, multipath, or channel mismatch. Stability must be measured, not assumed.

The working physical hypothesis is narrower than satellite-AoA resolution: a strong terrestrial emitter can create cross-channel coherence or a dominant spatial mode. Authentic GNSS is normally below the pre-correlation noise floor in a 10 µs wideband snapshot, so the released representation does not justify a claim that separate satellite directions will be visible in raw covariance. Thus low rank may indicate any dominant terrestrial emitter, including a non-deceptive jammer, rather than spoofing.

MUSIC/MVDR requires element coordinates/order and steering calibration; these are absent. Calibration-free pairwise coherence/eigenvalue ratios could be computed after phase continuity is proven, but fixed RF-chain offsets, element gains, and coupling remain nuisance factors. Multipath can raise apparent rank; multiple emitters can raise it further; a distributed/multi-antenna spoofer can defeat the common-direction premise. Power matching and phase-destruction controls are mandatory to separate spatial evidence from received-power shortcuts.

Conclusion: covariance is potentially measurable, but spoof-vs-jammer identifiability and calibrated direction inference are not established from metadata.
