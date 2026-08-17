# Navigation-bit requirement audit

For a single GPS L1 C/A code epoch, write the received samples as
`y[n] = alpha_(i,k) b_(i,k) c_i[n-tau] exp(j phase[n]) + w[n]`, where `b_(i,k)` is constant over the exact 1 ms integration. Every delay tap is therefore multiplied by the same scalar `b_(i,k)`. The least-squares epoch amplitude is `a_hat=(r^H y)/(r^H r)`; replacing `y` by `-y` replaces `a_hat` by `-a_hat`, leaving the fitted normalized complex tap shape, Prompt-normalized vector, complex cosine, and magnitude ranks unchanged.

The proof requires an exact 1 ms epoch, per-epoch complex-amplitude/global-phase normalization, and no absolute carrier-phase claim. Those conditions are checked here. Consequently Stage-0A is `NAV_BIT_NOT_REQUIRED_AFTER_PER_EPOCH_COMPLEX_NORMALIZATION`. Stage-0B must synthesize a continuous receiver input across navigation-symbol boundaries, so it remains `NAV_BIT_PROVENANCE_STILL_REQUIRED`; no +1 fallback is authorized.
