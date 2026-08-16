# Q-COMET Stage-0 static method

Q-COMET asks whether peak-shape innovations outside a clean normal nuisance
manifold begin at a common receiver-visible time across PRNs, while allowing a
different signed complex deformation vector for every PRN.

For PRN `i`, the causal clean predictor gives `c_hat[i,t]`. Calibration-A gives
the shrinkage covariance `Sigma`. With observed-local-peak nuisance Jacobian
`J`, the innovation is

`r = Sigma^(-1/2) (c - c_hat)`, `v = (I - J(J'J)^(-1)J') r`.

The Jacobian removes amplitude/gain, common carrier/navigation-bit phase,
small delay recentering from the measured local peak derivative, and a
tap-dependent Doppler phase tangent. It preserves all remaining signed complex
directions; Full never reduces the vector to scalar RMSE.

For fixed onset `k`, each PRN has its own coefficient matrix on frozen step,
linear-ramp, and first-order-transient bases. A Gaussian analytic marginal
likelihood supplies `B_i(k,t)`. Full evaluates

`S_t = max_k sum_i log((1-pi) + pi B_i(k,t)) - 0.5 log(L+1)`

in a finite 10 s window. `pi=0.5`, and fewer than four valid PRNs produces
`NO_SCORE`. Timestamp gaps and recording boundaries break predictor history.

All structure, candidate selection, priors, windows, covariance policy, and
threshold quantiles are fixed in `pre_evaluation_freeze.json` before attack
evaluation. Attack results cannot fit or calibrate any component.

The detection target is the first receiver-correlator-visible common change,
not guaranteed physical RF turn-on. An exactly code/Doppler/carrier-aligned
counterfeit that remains on the normal one-source manifold can be
information-theoretically undetectable. Physical controls in this Stage-0 are
correlator-domain checks and must not be described as raw-IQ physical proof.

