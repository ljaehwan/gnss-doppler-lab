# Confound analysis

Stage-0B establishes simultaneous four-channel structure. Stage-0C asks only whether calibration-free spatial relationships add discrimination between spoof/meacon and non-deceptive terrestrial jammer at matched transmit-power strata; it is not clean-versus-spoof detection.

M0 audits received-power shortcuts and M1 audits single-channel amplitude/spectrum shortcuts. M2 removes pair identity and coherence phase; M3 intentionally retains phase/order sensitivity as a location-shortcut diagnostic. B/C/D retraining separates spatial structure from per-channel waveform distributions, while actual-trained cross-application measures distribution-shift fragility.

Official recording ID, timestamp/day grouping, transmitter position, receiver orientation, and array calibration remain unavailable. Frozen sample-index groups are leakage-reduction proxies, not proof of recording independence. Strong OOF performance cannot identify spoofing physics versus transmitter location.

In the realized primary OOF support, M0 and M1 both reach AUROC 1.0, so received power and single-channel waveform information fully explain label separation before spatial features are considered. The positive OOF class contains Meac but no Spoof rows; Spoof recall is therefore not estimable. M3 falls from AUROC 1.0 to 0.388876 under the fixed channel permutation, confirming phase/order sensitivity, while permutation-invariant M2 remains at 0.979172 but does not improve on either baseline.
