# Array phase-coherence evidence

## Directly supported

The official README/data paper calls the receiver a 2×2 patch CRPA, says it is quadrature sampled over 10 µs, and states an array/direction-finding use. The released reader takes one array item, exposes four complex streams as eight I/Q columns, and never beamforms or magnitude-reduces those columns. The LFS size is exactly consistent with 42,673 × 4 × 1024 complex64 values plus a 128-byte NPY header.

## Not supported

No released text defines element order, element coordinates/spacing, common clock/trigger, cable or RF-chain phase offsets, phase calibration, receiver orientation, or a phase-coherence acceptance measurement. The payload header was not opened. Therefore the size-consistent shape is an inference, and four-channel relative phase preservation is not direct evidence.

## Gate

The mandatory phase gate is closed. `READY_FOR_CRPA_MINIMAL_SUBSET_DOWNLOAD` is forbidden. The next bounded schema step would require the single CRPA LFS object (NPY, despite the contractual verdict name mentioning H5) and a publisher statement or calibration record; a sample alone cannot establish absolute array calibration.
