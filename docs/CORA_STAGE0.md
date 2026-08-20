# CORA-GNSS Stage-0

CORA asks whether nominal-signal-removed residuals from concurrently tracked
PRNs contain a fourth-order dependence compatible with one shared emitter. It
does not use PRN identity, absolute time, raw power, or absolute Doppler in its
primary score.

For each authenticated 1 ms tracker interval, the implementation reads the
exact raw-IQ sample interval, creates the receiver-faithful nominal code and
carrier replica, fits and removes its complex amplitude, and recorrelates the
raw residual on the frozen 3-by-3 delay/Doppler grid. Constant, delay, and
Doppler tangent directions are projected out and the complex token is norm
quotiented. A PRN-shared clean-only ridge conditioner and shrinkage whitener
produce innovations.

For different PRNs, the alpha-zero statistic is

`cum(x,x*,y,y*) = E|x|²|y|² - E|x|²E|y|² - |E[xy*]|² - |E[xy]|²`.

The implementation uses an unbiased fourth k-statistic, averages fixed token
projections into a symmetric matrix, and compares the independent off-diagonal
null against a rank-1 shared-emitter alternative. The score is twice the
Gaussian log-likelihood improvement minus a BIC penalty.

Configuration was frozen and pushed at
`c226b942a82dbd63c6682e76e44b2aefe1c60156` before attack payload access.
The completed experiment returned `NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS`.
Neural Stage-1, score fusion, and threshold retuning are therefore prohibited.
