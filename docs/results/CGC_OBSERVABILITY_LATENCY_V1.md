# CGC observability-latency hypothesis v1

## Evidence that motivated the hypothesis

Two different outcome-unseen TEXBAT releases detected spoofing with no
persistent or sequential pre-onset alarm, but both missed a frozen 60 s delay
gate:

| Release | Attack | Alarm rule | First alarm end | Nominal/official onset | Delay |
|---|---|---|---:|---:|---:|
| real detection v1 | DS7 | 3 of 5 lower-5% crossings | 219 s | 100 s | 119 s |
| sequential DS8 v1 | DS8 | robust Page-CUSUM | 201 s | 110 s | 91 s |

For DS7, 20 s median residuals stayed at `0.6909`, `0.7332`, and `0.6553`
through 170 s and then fell to `0.5688`, `0.5374`, and `0.3345` over the next
three 20 s blocks. For DS8, the first four post-onset 20 s medians were
`0.7101`, `0.7738`, `0.6283`, and `0.7314`; they then fell to `0.5503` and
`0.2839`.

The alarm mechanisms differed, but both physical residual traces show that the
large spoof-consistent change appeared well after the published attack start.
The comparison is descriptive and post hoc; it is not a new validation result.

## Physical hypothesis

Clock-centered cross-satellite correlator geometry is an *observability*
statistic, not necessarily an attack-onset statistic. A carry-off attack becomes
strongly observable only after the counterfeit and authentic correlation
components have enough relative code separation and power balance to create a
stable, signed secondary-delay pattern across satellites. Before that point,
the nine-tap profile can remain prompt-dominated or fluctuate like normal
tracking distortion even though spoof transmission has begun.

Consequently, expected detection delay should be modeled against a physical
state such as effective code separation in chips, counterfeit/authentic power
ratio, and peak-overlap regime rather than against scenario start time alone.

## Next test required

Use controlled paired RF simulations, and later a controlled replay or live
test, that sweep:

- code separation or induced pseudorange displacement,
- counterfeit-to-authentic power ratio,
- carry-off rate,
- common versus satellite-specific delay structure, and
- independent multipath delay, amplitude, and phase.

For every condition, record the earliest time or physical separation at which
the clock-centered residual crosses a normal-only confidence boundary. The
primary analysis should estimate a detection boundary in separation-power
space and compare it with independent multipath. Parameters and gates must be
frozen before any new held-out condition is evaluated.

This narrower study can support a WCL contribution even if onset detection is
not immediate: it characterizes when the proposed physical discriminant is and
is not identifiable, rather than presenting it as a universally prompt
detector.
