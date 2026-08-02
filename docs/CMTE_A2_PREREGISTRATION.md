# CMTE-A2 TEXBAT DS7/DS8 사전등록

- **모델:** CMTE-A2 / Multivariate Conformal Epoch Evidence detector
- **기준 commit:** `d19115e48f8859330ae05bbab59c0ffd5f3d2004`
- **브랜치:** `research/cmte-a2-texbat-ds78`
- **예정 사전등록 commit:** `SELF`
- **작성 시각:** `2026-08-02T10:42:34Z`
- **상태:** `created_before_confirmatory_scoring = true`

이 문서는 **preregistration-only** 단계에서 작성되었다. 이 시점까지 CMTE-A2로 DS7/DS8 model inference, signed residual, nonconformity, conformal p-value, epoch score, alarm 또는 confirmatory metric을 생성하지 않았다. 이후 이들 중 하나라도 노출되면 이 문서와 `configs/cmte_a2_preregistration.json`을 수정하지 않는다. 수정이 필요하면 추가 holdout 접근 전에 별도의 superseding preregistration을 만든다.

## 1. 주장 범위와 사전 노출

- DS1–DS4는 development/reproduction tier다. A2 aggregation과 q99.5 선택에 이미 노출되었으므로 confirmatory가 아니다.
- DS7/DS8은 **CMTE-A2-specific one-shot holdout**이다.
- 그러나 DS7/DS8은 project-wide pristine holdout이 아니다. TEXBAT corpus와 공격 metadata, repository 자료, prepared-input metadata, historical Method-A/node 및 B0 evaluator/artifact가 이전에 노출되었다.
- 따라서 가능한 주장은 “미리 고정한 CMTE-A2의 DS7/DS8 one-shot 결과”이며, “프로젝트 전체에서 처음 보는 corpus”라는 주장은 금지한다.
- DS7/DS8 결과를 보고 architecture, feature, residual/covariance, conformal 식, aggregation, threshold, timing, metric, bootstrap 또는 success criterion을 바꾸지 않는다.

## 2. 권위 metadata와 입력 freeze

### 권위 자료

- Scenario manifest
  - path: `/home/ubuntu/projects/gnss-doppler-cmte-a2-texbat-ds78/data/external/texbat/manifests/scenarios.json`
  - SHA-256: `4a9f011bd2bc151e8b98862199c5460427887a8a98fe13d980bd21899f3bf86e`
- TEXBAT DS7/DS8 PDF
  - path: `/home/ubuntu/projects/gnss-doppler-cmte-a2-texbat-ds78/data/external/texbat/docs/texbat_ds7_and_ds8.pdf`
  - SHA-256: `3ac9a01f79a7f099f93b91f2def613b6fa41897f0cf700cb4434795b974aa149`
- ScienceDB mirror manifest
  - path: `/home/ubuntu/unraid/gnss-datasets/texbat/manifests/ds7_ds8_scidb_manifest.tsv`
  - SHA-256: `05fe26aeae3a14726ce84d260aae9d1f16563d87cfcbd00f8e4872aa687681a5`

### DS7

- 공격: `Carrier-Aligned Matched-Power Time Push`
- 공식 phase:
  - `[0,110)`: spoofing 없음
  - `[110,130)`: carrier phase/amplitude ramp
  - `[130,150)`: takeover 및 antipodal hold
  - `>=150`: `1.2 m/s` time push
- Raw IQ
  - path: `/home/ubuntu/unraid/gnss-datasets/texbat/raw/ds7.bin`
  - SHA-256: `d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e`
  - size: `27059290112` bytes
  - format: 25 MHz `ishort` interleaved IQ
  - byte-derived actual coverage: `270.59290112 s`; PDF가 더 긴 구간을 기술하더라도 실제 availability는 이 coverage를 따른다.
- Primary prepared complex NPZ
  - path: `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/exports/ds7.npz`
  - SHA-256: `d0e6da4e27d51e3e96abf2ef7786501124072f28667671e4e40da756eb35f3c8`
  - manifest: `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/exports/ds7.manifest.json`
  - manifest SHA-256: `8dec0643850c9d545b3980ea45bf2c3b9081cd2e314e208faab6bc51fa6e8959`
  - `complex_iq` shape/order: `[351746,9,2]`, `[epoch,tap,component]`, components `[I,Q]`

### DS8

- 공격: `SCER Carrier-Aligned Time Push` / `time_push_scer`
- nominal phase boundary는 DS7과 같은 `110/130/150 s`다.
- zero-delay SCER의 약 `50 µs` bit-estimation 효과를 1초 window에서 별도 onset 또는 delay 주장으로 사용하지 않는다.
- Raw IQ
  - path: `/home/ubuntu/unraid/gnss-datasets/texbat/raw/ds8.bin`
  - SHA-256: `1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78`
  - size: `47000141168` bytes
  - format: 25 MHz `ishort` interleaved IQ
  - coverage: `470.00141168 s`
- 현재 primary complex NPZ는 없다. 모델 접근 전에 raw source만 사용하는 deterministic prep을 수행하고 생성 NPZ, manifest, rendered config, wrapper hash를 freeze한다.
- Prep 고정값:
  - patched GNSS-SDR: `/home/ubuntu/build-gnss-sdr-complex9/build-complex/src/main/gnss-sdr`
  - SHA-256: `6c4512adefcfe49ae7d964c0425b26bfffd8b988ad7f9a0cf6f4b2e30fc5cafb`
  - 11 channels, 9 taps, spacing `0.125 chips`, 25 MHz
  - exporter git object: `f8a97987a8f48bd8f2f15dc249fc181a65d97842:src/gnss_doppler_lab/multiview/complex_taps.py`
  - exporter content SHA-256: `30a45f988cec15fdce84552ff30747b472c7d76df07d93f79d6ae236166d4039`
- Prep 실패 시 DS8 primary는 이유를 명시한 `NA`다. Historical Method-A node는 sensitivity로만 보고할 수 있으며 primary를 조용히 대체할 수 없다.

### 공통 converter와 DS4 caveat

cleanStatic, DS1–DS3, DS7, fresh DS8에는 같은 converter를 적용한다.

- path: `/home/ubuntu/projects/gnss-doppler-cmte-a2-texbat-ds78/src/gnss_doppler_lab/cmte_inputs.py`
- base content SHA-256: `5e6db2ef2a07b01a5753d2ac3729df0da47c95821ce61fc4118b634efb671f5a`
- tap magnitude: `m_j = hypot(I_j,Q_j)`
- per-epoch prompt-relative normalization
- 1초 window, 0.5초 stride
- role/split/recording/segment/channel/cadence gap을 연결하거나 fill/interpolate하지 않는다.
- 새 CMTE-A2 wrapper 및 구현은 DS7/DS8 평가 전 별도 freeze commit에서 hash 고정한다.

DS4에는 complex NPZ가 없다. 기존 node source는 explicit mixed-producer development sensitivity일 뿐 confirmatory 근거가 아니다.

## 3. B0 predictor 고정

### 입력과 architecture

- tap order: `E4,E3,E2,E,P,L,L2,L3,L4`
- 9차원 shared PRN-local causal model
- history: 12 epochs
- PRN identity/embedding 없음
- encoder: `Linear(9,128) → LayerNorm(128) → GELU → Dropout(.05) → Linear(128,128) → GELU`
- recurrent: one-layer unidirectional `GRU(128,128,batch_first=True)`
- head: `Linear(128,128) → GELU → Linear(128,9)`
- 마지막 causal GRU timestep으로 다음 9-vector를 예측한다.

### Chronological training

- cleanStatic prefix `[0,240)`에 완전히 포함되는 window만 사용한다.
- prefix 안에서 기존 sorted-PRN holdout을 유지하되 held PRN은 model selection에만 쓴다. Gradient와 scaler fit에서는 제외한다.
- scaler는 prefix gradient-training PRN에서 feature별 `nanmean`, population `nanstd(ddof=0)`로 fit한다. non-finite mean은 0, non-finite 또는 `<1e-6` std는 1, scaling 후 non-finite는 0으로 한다.
- AdamW: lr `1e-3`, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, amsgrad false
- epochs `25`, batch `256`, seed `11`
- loss: batch와 9 output 전체 MSE
- global gradient clip: `1.0`
- scheduler/early stopping 없음; 25 epoch 모두 실행
- held-PRN prefix validation MSE의 **strict minimum** checkpoint를 선택하고 tie면 앞선 checkpoint를 유지한다.
- `250 s` 이후 자료는 training, scaler, checkpoint selection에서 제외한다.

History는 role, split, physical recording, segment, channel, cadence gap에서 reset한다. PRN-local sequence를 gap 너머로 만들지 않는다.

## 4. CMTE-A2 식과 normal-data 역할

### Signed residual과 full-shrinkage nonconformity

Prefix-scaled feature를 `x`, 예측을 `xhat`이라 할 때:

```text
r = x - xhat
mu = (1/n) sum_j r_j
S = sum_j (r_j-mu)(r_j-mu)^T / max(1,n-1)
epsilon = 1e-8
lambda = min(1, 10/max(10,n))
D = diag(max(diag(S), epsilon))
Sigma = (1-lambda)S + lambda D + epsilon I_9
q(r) = max(0, (r-mu)^T Sigma^{-1} (r-mu))
```

`mu/Sigma`는 cleanStatic `[0,240)`의 eligible residual만으로 fit한다. Held-PRN model-selection row와 `[240,∞)`, Qcal, threshold role, clean test, DS1–DS8은 fit에서 제외한다.

### Qcal

- source: cleanStatic `[250,290)`에 완전히 포함되는 window
- `Qcal`은 residual distribution 또는 threshold fit에 재사용하지 않는다.
- finite-sample inclusive-tie p-value:

```text
p_i = (1 + #{q in Qcal : q >= q_i}) / (n_cal + 1)
```

### Primary epoch score

같은 physical recording과 `window_end_s`에서 현재 추적되는 `N_t`개 PRN만 모은다. Duplicate PRN은 fail closed한다.

```text
CMTE-A2_t = mean_i[-ln(p_i)]
alarm_t iff CMTE-A2_t > threshold
```

**사용하지 않는 것:** mixture e, mean e, e-CUSUM, betting/restart capital, 어떤 sequential score도, online normalization/adaptation도 사용하지 않는다.

### Threshold와 clean test

- threshold role: cleanStatic `[300,330)`에 완전히 포함되는 window만 사용
- primary: NumPy-style higher q99.5
  - 오름차순 `z[0..n-1]`에서 `q_higher(p)=z[ceil(p*(n-1))]`
- alarm은 strict `score > threshold`
- higher q99와 empirical target-1%는 diagnostic-only다. empirical target-1%는 strict exceedance가 `<=1%`가 되는 가장 작은 observed threshold-role score이며 tie면 높은 threshold를 쓴다.
- clean test: cleanStatic `window_start_s >=340`; end-to-end 독립 평가다. 이 구간은 어떤 fit, selection, calibration, threshold 조정에도 쓰지 않는다.

## 5. Comparator 고정

모든 comparator threshold는 normal threshold role `[300,330)`만 사용하며 attack label을 사용하지 않는다. Primary alarm threshold는 higher q99.5와 strict `>`다.

### Chronological B0 with exact gate semantics (B0-Exact)

반드시 **`chronological B0 with exact gate semantics`**로 표기한다.

- 같은 새 prefix-only chronological checkpoint와 signed residual의 per-PRN scalar RMSE를 쓴다.
- threshold role에서 node q50/q70/q80을 재산출한다.
- 각 epoch와 q에 대해 `K = count(RMSE > node threshold)`, `N = current tracked PRNs`.
- nominal `p=(.5,.3,.2)`로 exact finite binomial tail `P[X>=K]`를 계산한다.
- surprise는 `-ln(max(tail,1e-300))`, raw는 세 surprise의 max다.
- causal retention EWMA:

```text
state_before_first = 0
state_t = 0.75*state_(t-1) + 0.25*raw_t
```

- physical recording/run에서 reset한다.
- 새 checkpoint에 맞춰 node/final threshold를 normal-only로 재보정한다. **old checkpoint에 묶인 historical threshold를 재사용하지 않는다.**
- 별도로 historical precomputed node input에 gate만 적용해 K/N, exact tail, max, EWMA, alarm을 historical evaluator/golden과 numerical-equivalence 확인한다. 이것은 새 chronological checkpoint score equivalence 주장이 아니다.

### B0-Enhanced

별도 comparator다.

- threshold-role node q50/q70/q80
- 같은 role의 empirical strict exceedance probabilities
- max exact-binomial-tail surprise
- pandas-style `ewm(alpha=.75,adjust=False)`: current weight .75, previous .25, 첫 raw에서 초기화
- previous CMTE A1 semantics이며 B0-Exact로 부르지 않는다.

### A0

현재 epoch의 per-PRN scalar RMSE의 max다.

### FPR 비교

- Primary independent-clean FPR의 “similar” 허용치는 absolute `0.005`다.
- matched-clean-FPR는 diagnostic-only다. `[300,330)` 안에서 A2 threshold-role strict-exceedance occupancy와 가장 가까운 comparator observed threshold를 고르고, tie면 높은 threshold를 고정한 뒤 clean test와 DS7/DS8에 적용한다. Primary threshold나 success decision을 대체하지 않는다.

## 6. Timing과 metric

### Timing

Metadata onset `110 s`를 우선한다.

- stable pre: `[30,onset-20) = [30,90)`
- transition: `[90,110)`
- established/post: `>=110`
- phase sensitivity: `[110,130)`, `[130,150)`, `>=150`
- primary availability time: `window_end_s`
- delay: `window_end_s - 110`
- `90/110/130/150` 경계를 가로지르는 window는 양쪽 phase metric에서 제외한다.

### Point metrics

- clean FPR: independent clean-test epoch alarm occupancy
- stable-pre FPR: `[30,90)` epoch alarm occupancy
- post detection rate: `[110,end)` alarm occupancy
- persistent detection rate: `>=150` alarm occupancy
- first alarm delay: 첫 eligible post-onset alarm의 `window_end_s-110`; 없으면 censored/NA
- persistent-3-epoch delay: 같은 recording에서 0.5초 연속인 alarm 3개 최초 run의 첫 epoch delay; 없으면 censored/NA
- ROC-AUC: stable pre를 negative, post-onset을 positive로 사용하고 transition/boundary-crossing window는 제외
- 각 `110–130`, `130–150`, `>=150` detection rate를 sensitivity로 보고한다.

PRN count diagnostic은 clean test, DS7, DS8 각각에 대해 epoch N 분포, exact-N별 score 분포, alarm occupancy, clean/stable/post count를 기록한다. Sparse N을 pool하거나 aggregation 변경 근거로 쓰지 않고 `NA` 이유를 기록한다.

## 7. Moving-block bootstrap

- seed: `20260802`
- replicates: `2000`
- 95% percentile CI
- cadence: 0.5초
- block: 20 epochs = 10초
- 대상: ROC-AUC, stable-pre FPR, persistent detection rate, post detection rate
- scenario/physical recording/phase/gap-safe chunk 안에서 phase 시작에 anchor한 non-overlapping complete 20-epoch block을 만든다.
- statistic에 필요한 phase stratum마다 whole block을 replacement sampling하며 원래 complete-block 수를 유지한다.
- phase, recording, segment/channel-derived gap 또는 cadence gap을 넘지 않는다.
- point estimate는 모든 eligible epoch를 쓰고 complete block은 CI에만 적용한다.
- 필요한 각 phase/class에 complete block이 최소 2개 없으면 CI는 이유가 포함된 `NA`다.
- IID fallback은 금지한다.

## 8. Success criteria

모든 항목이 통과해야 성공이다. Required primary DS가 `NA`면 정직하게 `NA`로 보고하되 overall success는 실패한다.

1. CMTE-A2 independent clean FPR `<=1.5%`.
2. DS7과 DS8 각각 stable-pre FPR `<=5%`.
3. DS7/DS8 어느 것도 stable-pre FPR `>=20%`가 아님. 항목 2가 더 엄격하지만 catastrophic guard를 별도 보존한다.
4. DS7/DS8 중 적어도 하나에서, independent clean FPR 차이가 `<=0.5 percentage point`인 B0-Exact보다 다음 중 하나가 개선됨:
   - first-alarm 또는 persistent-3-epoch delay가 최소 한 cadence(`0.5 s`) 빠름,
   - post 또는 persistent detection rate가 strict하게 높음(`1e-12` numerical tolerance).
   - 이미 pre-onset alarm인 detector는 해당 scenario에서 delay advantage를 주장할 수 없다.
5. 나머지 scenario에서 CMTE-A2가 B0-Exact보다 가능한 모든 attack metric(first delay, persistent-3 delay, post rate, persistent rate)에서 동시에 나쁘지 않음. B0만 detect하고 A2가 censored이면 A2가 나쁜 것, 둘 다 censored면 tie다. 항목 1–3 guard도 그대로 적용한다.
6. clean test, DS7, DS8의 PRN-count diagnostic을 만들고 sparse stratum의 명시적 `NA` 이유를 포함하며 N을 보고 aggregation을 변경하지 않는다.

## 9. 금지 사항과 future audit

금지한다:

- mixture e/e-value, e-CUSUM, capital, sequential score
- online scaler/covariance/threshold adaptation
- PRN identity/embedding, cross-PRN graph/relation
- DS7/DS8를 이용한 model/threshold/aggregation/timing/metric 변경
- old checkpoint용 B0 threshold 재사용
- DS8 prep 실패 시 historical Method-A node의 silent primary substitution
- IID bootstrap fallback
- role/split/recording/segment/channel/cadence gap bridging

향후 confirmatory report는 implementation/freeze commit SHA, 모든 source/input/wrapper/checkpoint/scaler/state/Qcal/threshold hash, role 및 reset row-count audit, historical gate-only numerical-equivalence, primary와 diagnostic operating point의 분리, point estimate와 CI/NA 이유, success-criterion audit, 그리고 사전등록 파일을 result exposure 뒤 수정하지 않았다는 문장을 포함해야 한다.
