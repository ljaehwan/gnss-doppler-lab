# Confound analysis

This is **not** clean-versus-spoofing detection. The bounded object has no clean CRPA class. The only intended task is spoof/meacon versus non-deceptive terrestrial jammer.

The official release does not bind snapshots to recording ID, timestamp/day, transmitter identity/position, receiver orientation, antenna ordering, geometry, cable/channel calibration, or VGA. Sample-index blocks are therefore only a leakage-reduction proxy, never proof of recording independence.

Within Area 1, the nominal common transmit-power values are severely class-imbalanced. At least one negative power cell occupies only one block, so no block-disjoint train/test assignment can contain every exact power cell on both sides for all frozen block sizes. A snapshot-random substitute would violate the contract.

Even when actual tuples exceed channel-mismatched and phase-destroyed controls, this demonstrates simultaneous multi-channel structure, not spoofing-specific directionality. Transmitter location, waveform family, multipath, and unknown array calibration remain plausible causes.
