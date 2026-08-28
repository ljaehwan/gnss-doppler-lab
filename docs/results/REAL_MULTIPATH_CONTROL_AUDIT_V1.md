# Real multipath control audit v1

## Decision

As of 2026-08-28, the locally available corpus does **not** contain a usable
GPS L1 recording that is simultaneously:

1. real RF collected in a physical multipath environment;
2. multipath-only, without spoofing;
3. paired with a sufficiently comparable low-multipath recording; and
4. labelled strongly enough to serve as the negative control for the claim
   "multipath is not mistaken for spoofing."

The repository does contain synthetic multipath controls, real clean RF,
real spoof RF without multipath, and real spoof RF with multipath. These are
useful for distinct validation questions but must not be relabelled as a real
multipath-only negative control.

## Locally available evidence

| Corpus | Local raw data | Scenario label | Valid use | Not valid for |
|---|---|---|---|---|
| Generated `independent_multipath` | yes | synthetic injection with generated epoch truth | controlled mechanism and classifier-negative tests | real-field multipath claim |
| TEXBAT `cleanStatic` / `cleanDynamic` | yes | authentic clean recording; no path-level multipath truth | real normal baseline | labelled multipath negative |
| OAKBAT `cleanStatic` / `cleanDynamic` | yes | authentic clean recording; no path-level multipath truth | independent real normal baseline | labelled multipath negative |
| TUNI GPS C-5 | yes | static, no multipath, no spoofer | labelled no-multipath baseline | multipath response |
| TUNI GPS SS-17 / SS-18 / SS-20 | yes | static, no multipath, 1/2/4 spoofers | real no-multipath spoof validation | multipath-only false-alarm control |
| TUNI Galileo SS-11 / SS-12 / SS-13 | yes | controlled-lab RF; multipath with 1/2/4 spoofed PRNs | within-recording authentic-PRN multipath controls and spoof robustness | GPS or uncontrolled field-multipath claim |
| TUNI GPS C-7 | no independent payload | conflicting historical label; official size and MD5 equal C-5 | metadata audit only | any independent experimental count |

The generated control is located at
`/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-r3-control-generator-foundation/controls/TEX/TEX.negative.independent_multipath/`.
Its `truth.json` identifies a generated output, target PRNs, an epoch-truth
file, and the modified sample interval. It is therefore a reproducible
synthetic intervention, not a field recording.

The current raw mounts are:

- TEXBAT: `/home/ubuntu/unraid_hdd/texbat/raw/`;
- OAKBAT: `/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw/`;
- TUNI2025: `/home/ubuntu/unraid_hdd/tuni2025/`.

TUNI's official metadata describes controlled-laboratory recordings made with
a USRP-2945R and a Spectracom GSG-6 where spoofing was used. In SS-11 only PRN
31 is spoofed; in SS-12 PRNs 9 and 31 are spoofed; and in SS-13 PRNs 5, 9, 23,
and 31 are spoofed. Other tracked PRNs in the same recording are authentic
signals under the same labelled multipath condition. They therefore form a
strong same-stream specificity control at the PRN level, although they are not
an uncontrolled urban field recording. The attacks transmit a true-position
solution, so they also test a different, near-zero-displacement regime from the
simulated carry-off geometry campaign.

The TUNI README calls the I/Q interleaved 32-bit floats, but clean C-1 byte
inspection and receiver preflight show interleaved signed 16-bit I and 16-bit Q
(GNU Radio `ishort`, 32 bits per complex sample). A 5-second clean-only C-1
preflight using a 50-to-12.5 MHz receiver resampler tracked PRNs 2, 30, and 36
for 1,778 valid epochs. Treat the README sample-format text as erroneous;
the provided FGI-GSRx `sampleSize=32` setting is consistent with the bytes.

No attack payload was decoded during this compatibility preflight. The local
`DO_NOT_OPEN_BEFORE_MODEL_FREEZE.txt` boundary remains in force until a Galileo
clean-only model and the evaluation contract are committed.

The shared dataset volume currently has about 16 GB free. This is insufficient
for an additional 126--132 GB SJTU environment archive without first expanding

The tracked TEXBAT links under `data/external/texbat/raw/` still point to the
old `/home/ubuntu/unraid/gnss-datasets/texbat/raw/` mount. Experiments should
resolve the current mount explicitly or repair the links in a separately
reviewed data-mount change.

## TUNI integrity exclusion

TUNI GPS C-7 cannot be used as a second clean recording or as the historically
named multipath recording. The local download manifest records:

- C-5 official size: 29,999,832,000 bytes;
- C-7 official size: 29,999,832,000 bytes;
- C-5 and C-7 official MD5:
  `a03dedd79ac4208f6d60b4c916484dba`;
- C-7 was not downloaded as an independent payload.

Thus C-7 is byte-identical to C-5 according to the official metadata and must
contribute zero independent samples. The audit source is
`/home/ubuntu/unraid_hdd/tuni2025/gps/download_manifest_20260824T172706Z.json`.

## External candidates

### Immediately actionable field proxy

The [SJTU GNSS dataset](https://bat.sjtu.edu.cn/en/gnss-dataset/) provides raw
vehicular IF recordings from labelled environments. Downtown and boulevard
segments are described as multipath-heavy, while the suburb segment is a
lower-multipath comparison. This can support an environment-stratified field
proxy, subject to format adaptation and a frozen segment-selection rule.

Limitations are important: the labels describe environments, not individual
reflected paths or per-epoch multipath truth, and each environment is roughly
126--132 GB. It is a defensible robustness proxy, not an exact causal
multipath ground truth. The current 16 GB free space also prevents an immediate
full-environment download.

### Stronger future target

The ION GNSS+ 2026 abstract
[Toward Realistic Evaluation of GNSS Anti-Spoofing: A Spoofing Dataset from
Urban Experiments](https://www.ion.org/gnss/abstracts.cfm?paperID=16974&sessionID=2068)
describes clean, spoofed, and composite raw IF with time-domain labels derived
using 3D ray tracing. This is close to the desired validation design, but the
conference is scheduled for September 2026 and no public download was located
during this audit. It is a watch-list item, not current evidence.

## Claim-safe validation ladder

The available data support the following staged tests:

1. use generated independent multipath as the controlled negative class;
2. use TEXBAT and OAKBAT clean recordings to measure real-normal false alarms;
3. use TUNI C-5 versus GPS SS-17/18/20 for an independent real no-multipath
   clean/spoof test;
4. freeze a Galileo-compatible clean model and use the authentic, non-target
   PRNs inside SS-11/12/13 as same-stream multipath specificity controls;
5. compare common spoofed PRNs across TUNI Galileo no-multipath and multipath
   scenarios to test whether multipath degrades spoof detection;
6. obtain storage and preregister an SJTU low-/high-multipath comparison;
7. reserve the operational statement "field multipath is distinguished from
   spoofing" until a real multipath-only negative control is tested.

Until step 7, the paper-safe wording is: the method separates simulated
independent multipath from coherent simulated spoofing and is additionally
checked on real normal/spoof RF; field multipath specificity remains an open
external-validity item.

