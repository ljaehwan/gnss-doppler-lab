# Public raw-signal GNSS spoofing dataset audit v2

Date: 2026-08-31

## Decision

The strongest unopened near-term candidate for the frozen clock-centered
geometry (CGC) detector is the **FGI-SpoofRepo targeted DFMC L1/E1 recording**.
It is a continuous live-sky-plus-spoofer raw-IF stream, contains eleven
simulated GPS satellites matched to the live constellation, and implements a
time-synchronous dynamic false-position trajectory.  The L1/E1 file is
9,961,930,752 bytes and the documented attack onset is approximately 138 s.

The strongest independent field candidate is **FGI Jammertest 2023
JT23-17.1.6**.  It is a coherent spoofing recording made outdoors in Andoya,
Norway, using broadcast ephemerides.  It is scientifically valuable because it
contains real propagation effects and differs from the laboratory batteries,
but its single raw-IQ file is 72,110,000,000 bytes.  The complete three-scenario
release is 151,670,000,000 bytes.  It is not executable on the current SSD,
which has only about 9.5 GB free.

No attack file is opened by this audit.  In particular, no delay estimate,
geometry statistic, alarm score, or outcome plot is computed.  Candidate
selection is based only on official metadata and publications.

## Eligibility requirements

A primary confirmatory recording must provide:

1. continuous GPS L1 raw IF/IQ, not RINEX, UBX, PVT, spectrum-only, or Android
   observables;
2. an authentic pre-attack interval and a known attack onset;
3. coherent or time-synchronous spoofing capable of affecting an already
   tracking receiver;
4. preferably at least eight simultaneously usable GPS PRNs;
5. enough bandwidth and sample continuity to form complex multi-tap
   correlations and signed per-PRN delays; and
6. an outcome that has not been used to tune the frozen detector.

## Tier A: suitable raw-signal candidates

### A1. FGI-SpoofRepo targeted DFMC -- first preflight

- Official record: <https://doi.org/10.23729/7a648509-2ca8-4a7d-8223-0b429182f857>
- Characterization paper:
  <https://doi.org/10.1007/s10291-024-01719-2>
- Candidate file: `FGISpoofRepo/TG_DFMC/TGD_L1_E1.dat`
- Exact size: 9,961,930,752 bytes
- SHA-256:
  `10aad73665db7c5e530d9ef1d3b2fdb57bab0a7b9b19177a0128867fbad2606b`
- Signal: GPS L1 C/A plus Galileo E1, 26 MSps real IF, 6.39 MHz IF,
  approximately 4.2 MHz front-end bandwidth.
- Recording: about 373 s; the first roughly 130 s are attack-free.
- Attack: targeted, time-synchronous, dual-frequency multi-constellation
  spoofing; the intended false position follows a circular trajectory.
- GPS support: eleven GPS satellites are simulated to match the live
  constellation in the targeted recordings.

This is the best immediate physics match.  A single common false trajectory
should induce a satellite-direction-dependent signed-delay pattern.  Its
limitations are that the spoofer is simulator-generated, the receiver is
static, and the live and spoof signals are combined in a controlled setup.

The publication reports a 2-bit NSL Stereo-v2 front end, while the released
file size is consistent with one stored byte per real sample.  The adapter must
verify signedness, packing, and timebase without examining detector scores.

### A2. FGI Jammertest 2023 JT23-17.1.6 -- field confirmation

- Official record:
  <https://doi.org/10.23729/fd-06d27736-45cb-3ca2-aff8-725d42c6caeb>
- Independent analysis paper:
  <https://doi.org/10.1007/s10291-026-02102-z>
- Candidate file: `JammerTest2023/JT23_17.1.6/JT23_17.1.6_L1_E1.iq`
- Exact size: 72,110,000,000 bytes
- SHA-256:
  `4bd6e3963f0b3d6806670db5d1a653de05fd9fe602ec42b005eb6cc4d45931e3`
- Signal: complex GPS L1 C/A plus Galileo E1 at 30.69 MSps, zero IF,
  30 MHz bandwidth.
- Front end: LabSat 3 Wideband; the release metadata describes 16 bits per
  complex I+Q sample, while the analysis paper reports 3-bit quantization
  stored in 8-bit bins.
- Attack: coherent stationary-spoofer transmission using broadcast true
  ephemerides; published evaluation onset 226 s.
- Environment: authority-controlled outdoor field campaign in Andoya,
  Norway.

This is the strongest independent operational-domain candidate, but coherent
spoofing does not by itself guarantee a sizeable carry-off displacement.
Therefore a no-alarm result before measurable signed delays appear would not
be a detector failure.  The receiver output must first establish whether the
recording contains a resolvable secondary-code trajectory.

Official metadata and the 2026 analysis use different duration conventions
for this scenario.  The file size and sample format imply a much longer raw
recording than the 500 s evaluation window; onset and scoring windows must be
fixed from the publication and release metadata before any detector output is
opened.

### A3. OAKBAT GPS L1 C/A -- established independent battery

- Official record: <https://doi.org/10.13139/ORNLNCCS/1664429>
- Contents: binary GPS L1 C/A RF recordings for clean and spoofing scenarios.
- Local status: already downloaded and already used by earlier detector
  experiments.

OAKBAT remains a valid benchmark and format-transfer check, but it cannot be
called a new blind confirmation.  Its spoofing outcomes are already open.

### A4. TEXBAT -- canonical benchmark

- Official record: <https://radionavlab.ae.utexas.edu/texbat/>
- Contents: continuous high-fidelity GPS L1 C/A raw IF including matched-power
  and carrier-phase-aligned carry-off scenarios.
- Local status: downloaded, processed, and outcome-opened.

TEXBAT remains important for comparison and latency interpretation, but it is
development evidence rather than a fresh confirmatory dataset.

## Tier B: raw signal exists, but does not presently satisfy CGC

### TUNI2025 SS-33

- Official collection: <https://doi.org/10.5281/zenodo.17413258>
- Scenario record: <https://zenodo.org/records/17249727>
- Raw file: 40.0 GB, 50 MSps interleaved I/Q; delayed spoofer injection with
  all documented spoofing PRNs active.

This is a useful all-spoofer positive control.  However, the TUNI descriptor
states that spoofed PRNs were intentionally selected to be distinct from the
genuine visible PRNs for RF-fingerprinting studies.  Until the SS-33 README
proves same-PRN authentic/spoof overlap and a carry-off trajectory, it is not a
clean test of the CGC common-displacement mechanism.  It also exceeds current
storage.

### SatGrid

- Official record: <https://doi.org/10.7294/SE62-7X13>
- Total release: 75,634,916,096 bytes in `.mat` and `.dat` archives.
- Scenarios: genuine live GPS, GPS-SDR-SIM spoofing, and USRP-B210 replay at
  multiple powers; the G25/S10 pair contains eight PRNs.

The accompanying Spotr paper identifies the released detector features as
GNSS-SDR early, prompt, and late correlator outputs.  This makes SatGrid useful
for a three-tap baseline or a hardware-fingerprint comparison, but it does not
provide the nine-tap continuous raw-IF contract required for the primary CGC
test.  It must not be described as an independent nine-tap evaluation.

### TUNI raw-IQ RFF fingerprinting release

- Official record: <https://doi.org/10.5281/zenodo.13846381>
- Public contents: a randomly selected subset of 1,000 samples per scenario;
  two spoofing PRNs mixed with authentic PRNs.

The public subset is not a continuous tracking stream and has insufficient
coordinated PRN support.  It is appropriate for RF-fingerprint classification,
not the present delay-geometry test.

### Fraunhofer IIS Jammertest 2025

- Official repository:
  <https://github.com/FelixOtt94/FraunhoferIIS_Jammertest2025>
- Contents: labelled 8-bit raw-IQ HDF5 items from outdoor spoofing,
  meaconing, jamming, and multi-emitter events.
- Release size: about 358.8 GB from the previously audited LFS manifest.

The Innosense items are independently indexed approximately 1.62 ms snippets,
and CRPA snapshots are 10 microseconds.  They are useful for waveform and
spectral classification, but do not preserve the multi-second continuous
tracking stream needed for per-PRN nine-tap geometry.

## Excluded despite the word "raw"

| Dataset | What is actually released | Exclusion reason |
|---|---|---|
| GNSS Dataset Under Jamming, Spoofing, and Meaconing Conditions | UBX, RINEX, RF-monitoring, and navigation outputs | no antenna raw IF/IQ |
| Yunnan GNSS interference and spoofing dataset | UBX-derived JSON including RXM-RAWX and MON-SPAN | observables/spectra, not correlator-replayable IF |
| CG-SpoofGNSS | Android/u-blox per-satellite observations, positions, and sensors | "raw" means raw measurements, not RF samples |
| GNSS-OpenIF S1/S2 | continuous authentic raw I/Q | no labelled spoofing; already used as multipath negative controls |

## Watch list

The ION GNSS+ 2026 abstract
[Toward Realistic Evaluation of GNSS Anti-Spoofing: A Spoofing Dataset from
Urban Experiments](https://www.ion.org/gnss/abstracts.cfm?paperID=16974)
describes separate clean, spoof-only, and composite raw IF, centimeter-level
truth, 3-D ray-traced multipath labels, pedestrian and vehicle motion, and a
spoof injection near 40 s.  This would be an unusually strong validation set,
but no working public data-download endpoint was located during this audit.
It remains a watch-list item, not available evidence.

## Frozen next-step protocol

1. Use FGI-SpoofRepo `TGD_L1_E1.dat` as the first unopened support preflight.
2. Before downloading signal bytes, commit the format adapter, allowed time
   prefix, PRN-support rule, onset, minimum post-onset duration, and terminal
   `INSUFFICIENT_SUPPORT` outcome.
3. Run acquisition/tracking and count eligible PRNs without loading the delay
   template or computing CGC scores.
4. Continue only if at least eight PRNs satisfy the frozen one-second support
   rule for at least 60 primary bins.
5. Reveal the unchanged score, threshold, and 3-of-5 persistence rule once.
   Report detection probability, false alarms in the pre-attack segment,
   latency from the documented onset, and the signed-delay/common-displacement
   trajectory.
6. Acquire additional storage before JT23-17.1.6.  Treat it as a separately
   preregistered field-domain validation and do not tune on FGI-SpoofRepo
   outcomes first.

## Claim boundary

Passing FGI-SpoofRepo would support transfer to an unopened live-sky composite
and a different RF front end.  Passing Jammertest 2023 would add independent
outdoor coherent-spoof evidence.  Neither alone proves universal detection of
all spoofing: asynchronous injection, acquisition-time capture, non-carry-off
attacks, and attacks with too few supported PRNs remain outside the primary
claim.
