**GERMAN NATIONAL PV GENERATION**

**ROLLING FORECAST PIPELINE**

Claude Code Implementation Guide --- v3 (Simulated Real-Time)

Aligned to: Technical Assignment --- Internship Assessment --- Solar PV
Forecasting

Version 3.0 \| March 2026

VM: 8 vCPU, 32 GB RAM, no GPU \| Runtime: \<45 min (compute only, excl.
download)

Paradigm: Simulated real-time --- at each step T, only data ≤ T is
visible to the model

  -----------------------------------------------------------------------
  **CORE PARADIGM:** This is NOT a retrospective backfill where all data
  is available. The pipeline simulates a real-time prediction stream. At
  each quarter-hour T, the model can only see data up to and including T,
  and must forecast T+1, T+2, \..., T+x. The full-year computation
  (harmonise + feature build + train + 35,040 rolling prediction steps)
  must complete within \~45 minutes. Data download time is NOT counted
  toward this budget.

  -----------------------------------------------------------------------

0\. Core Paradigm: Simulated Real-Time Rolling Forecast

  -----------------------------------------------------------------------
  **⭐ KEY CONCEPT:** At every quarter-hour T in the backfill year, the
  model stands at time T and produces predictions for T+1, T+2, ..., T+x.
  It may ONLY use weather, actuals, and derived features from timestamps
  ≤ T. This is the single most important invariant in the entire
  pipeline. Every design decision flows from it.

  -----------------------------------------------------------------------

This means:

- **Weather/irradiance at time T is known** (it's
  reanalysis/observation, already happened). Weather at T+1, T+2 etc. is
  NOT known unless you have an NWP forecast issued before T.

- **Actual PV generation has \~60 min publication lag.** ENTSO-E
  publishes actuals with roughly a 1-hour delay. At time T, the most
  recent available actual is approximately T-4 (one hour ago), not T-1.
  Therefore the model uses actual_lag_4 as the most recent lag, not
  actual_lag_1. Lags 1--3 would be unavailable in a real-time system and
  are excluded. Additional lags: actual_lag_8 (2h ago) and actual_lag_96
  (same QH yesterday).

- **Deterministic features** (clear-sky GHI, solar zenith, calendar) are
  known for ALL times including future, because they're computed from
  astronomy and the calendar. These are the only features available for
  the forecast horizon T+1..T+x.

- **Weather at time T (ERA5):** ERA5 is reanalysis, not a real-time
  feed. In reality it has a \~5-day publication delay. However, the
  assignment uses ERA5 as a proxy for "observed weather that has already
  occurred." We treat weather at time T as available at T (the event
  happened), while acknowledging in the technical note that a live
  system would use NWP forecasts instead. Weather at T+1 and beyond is
  NOT available --- the model cannot use future irradiance or
  temperature.

- **The synthetic proxy** uses observed irradiance, so it's only
  available for ≤ T (like actuals). For the forecast horizon, a
  clear-sky-only proxy can be computed.

0.1 Forecast Horizon Choice

The horizon x (how many quarter-hours ahead to predict) is a design
decision you must justify. Reasonable choices:

- **x = 1 (15 min ahead):** Simplest. One prediction per step. The model
  uses rich recent context (lags, weather at T) to predict the immediate
  next interval. Good for demonstrating the rolling framework cleanly.

- **x = 4 (1 hour ahead):** More practical for trading. Produces 4
  predictions per step. Features for T+2..T+4 degrade since observed
  weather is only available at T.

- **x = 96 (24h / day-ahead):** Most ambitious. Only deterministic
  features (clear-sky, calendar) are available for most of the horizon.
  Lag features and observed weather only help the near-term end.

  -----------------------------------------------------------------------
  **⚠ NOTE:** Recommendation for the assignment: start with x=1
  (simplest, easiest to validate, lowest risk of leakage). Mention x=4 or
  x=96 as extensions in the technical note. If time permits, implement
  x=4 as a stretch goal.

  -----------------------------------------------------------------------

0.2 What This Changes vs. a Retrospective Backfill

In a retrospective backfill, you'd build a feature matrix once with all
data, train, and predict. Here, the prediction loop must **simulate time
progression**. Concretely:

1.  **Training is still done once** on the training split (e.g.,
    Jan--Aug). The model is fitted on historical data where targets are
    known. This is normal.

2.  **Prediction is done step-by-step:** for each T in the evaluation
    period, assemble features using only data ≤ T, call model.predict(),
    store the result. Move to T+1.

3.  **Feature assembly must be fast:** the rolling loop runs 35,040
    times. If each step takes \>75ms, you blow the 45-minute budget.
    This is why vectorised/batch prediction is critical (see Section 6).
    Note: the 45-min budget covers compute only (harmonise + features +
    train + predict), not data download.

1\. Hard Constraints

  -----------------------------------------------------------------------
  **⚠ HARD CONSTRAINT:** VM: 8 vCPU, 32 GB RAM, no GPU. Peak memory
  target: \<16 GB.

  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  **⚠ HARD CONSTRAINT:** Runtime: The compute portion (harmonise +
  feature build + train + 35,040 rolling prediction steps) must complete
  within \~45 minutes. Data download/fetch time is excluded from this
  budget.

  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  **⚠ HARD CONSTRAINT:** Language: Python ≥ 3.11. No R, Julia, or
  shell-heavy logic.

  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  **⚠ HARD CONSTRAINT:** Output: \`make full-run\` must produce
  outputs/predictions.parquet and outputs/runtime.log.

  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  **⚠ HARD CONSTRAINT:** Storage: All intermediates in Parquet. Use
  float32 where safe. Partition by year/month for large datasets.

  -----------------------------------------------------------------------

  -----------------------------------------------------------------------
  **⚠ HARD CONSTRAINT:** Reproducibility: uv / Poetry / Conda with pinned
  lockfile. Optional Dockerfile.

  -----------------------------------------------------------------------

**Design implications:**

- **No database.** Parquet-only. The assignment doesn't call for
  PostgreSQL.

- **Prefer polars.** polars.scan_parquet() + lazy evaluation keeps
  memory low. pandas only where required by library APIs (pvlib,
  entsoe-py).

- **The rolling loop must be vectorised.** You CANNOT call
  model.predict() 35,040 times in a Python for-loop with per-step
  feature assembly. Instead, pre-compute the full feature matrix (all
  timestamps), then walk through it with a sliding window or mask. See
  Section 6.

- **No GPU models.** Temporal CNN must train + infer on CPU. If training
  exceeds 10 min, drop it.

2\. Repository Structure

Note the addition of src/forecaster.py --- this is the rolling
prediction loop, separate from model training.

  ------------------------------------- ---------------------------------
  **Path**                              **Purpose**

  pv-forecast/                          

  ├─ Makefile                           make full-run entry point

  ├─ pyproject.toml                     uv/Poetry deps, Python ≥ 3.11

  ├─ uv.lock                            Pinned lockfile

  ├─ Dockerfile                         (optional)

  ├─ config/                            

  │ ├─ data_sources.yaml                Endpoints, API keys, CRS

  │ ├─ features.yaml                    Feature list, dtypes, avail rules

  │ └─ models.yaml                      Hyperparams, forecast horizon

  ├─ src/                               

  │ ├─ data/                            Phase 1.1--1.3

  │ │ ├─ fetch_era5.py                  CDS API → Parquet

  │ │ ├─ fetch_openmeteo.py             Alt NWP source

  │ │ ├─ fetch_actuals.py               ENTSO-E/SMARD → Parquet

  │ │ ├─ fetch_capacity.py              MaStR → Parquet

  │ │ └─ harmonise.py                   1.4: align + clean + merge

  │ ├─ features/                        Phase 1.5 + eng

  │ │ ├─ solar.py                       pvlib clear-sky + zenith

  │ │ ├─ proxy.py                       Synthetic PV proxy

  │ │ └─ build.py                       Feature matrix builder

  │ ├─ models/                          Phase 2

  │ │ ├─ baseline.py                    2.1: Ridge / irradiance reg

  │ │ ├─ candidate.py                   2.2: LightGBM + optional TCNN

  │ │ └─ evaluate.py                    2.3: rolling eval + metrics

  │ ├─ forecaster.py                    Rolling forecast loop

  │ └─ pipeline.py                      Orchestrator: make full-run

  ├─ data/                              Parquet storage

  │ ├─ raw/                             Downloads

  │ ├─ interim/                         Cleaned intermediates

  │ └─ processed/                       Full feature table

  ├─ outputs/                           

  │ ├─ predictions.parquet              REQUIRED deliverable

  │ └─ runtime.log                      REQUIRED deliverable

  ├─ report/                            Technical note (PDF)

  └─ tests/                             Unit + integration
  ------------------------------------- ---------------------------------

3\. Makefile & Pipeline Orchestrator

> \# Makefile
>
> .PHONY: full-run fetch clean
>
> \# Data download (run once, NOT timed)
>
> fetch:
>
> \@mkdir -p data/raw
>
> python -m src.data.fetch_era5
>
> python -m src.data.fetch_actuals
>
> python -m src.data.fetch_capacity
>
> \# Full compute pipeline (THIS is timed, must be \<45 min)
>
> \# Assumes data/raw/ is already populated via \`make fetch\`
>
> full-run:
>
> \@mkdir -p outputs data/interim data/processed
>
> python -m src.pipeline 2\>&1 \| tee outputs/runtime.log
>
> clean:
>
> rm -rf data/ outputs/ \*.log

3.1 Pipeline Orchestrator (src/pipeline.py)

1.  Log start time, git commit hash, Python version.

2.  Assert data/raw/ exists and contains downloaded data (fail fast if
    fetch was not run).

3.  Phase 1 compute: harmonise → build features (including proxy). No
    downloads here.

4.  Phase 2 compute: train baseline → train candidate → run rolling
    forecast → evaluate.

5.  Log elapsed time per phase and peak memory (resource.getrusage).

6.  Assert outputs/predictions.parquet exists with expected schema.

7.  Log total wall-clock time. This is the number that must be \< 45
    min.

4\. Data Sourcing & Ingestion (Phase 1.1--1.3)

  -------------- ---------------- ---------------- ----------- ------------ -----------
  **Source**     **Access         **Resolution**   **Volume    **Format**   **Phase**
                 Method**                          (1yr)**                  

  ERA5 (ssrd,    CDS API (cdsapi) Hourly           \~200 MB    GRIB2 →      1.1
  t2m, tcc)                                        GRIB        Parquet      

  Open-Meteo     open-meteo.com   15 min native    \~50 MB     JSON →       1.1 alt
  (alt.)         REST                              JSON        Parquet      

  MaStR (PV      Bulk XML/CSV     Static snapshot  \~500 MB    CSV →        1.2
  registry)                                        raw         Parquet      

  OSM PV (proxy  Overpass API     Static           \~20 MB     GeoJSON →    1.2 alt
  alt.)                                                        Parquet      

  ENTSO-E        entsoe-py / REST 15 min           \~15 MB     XML →        1.3
  (actual gen)                                                 Parquet      

  SMARD (BNetzA, smard.de CSV     15 min           \~10 MB     CSV →        1.3 alt
  alt.)                                                        Parquet      

  pvlib          Python library   Arbitrary        Computed    In-memory    1.5
  (clear-sky)                                                               
  -------------- ---------------- ---------------- ----------- ------------ -----------

  -----------------------------------------------------------------------
  **⚠ NOTE:** Document every source you use AND every source you
  considered and rejected. The assignment explicitly evaluates this (40%
  weight on data sourcing & engineering).

  -----------------------------------------------------------------------

4.1 Phase 1.1: Weather Layer

**Primary: ERA5 via CDS API.** Variables: ssrd (GHI), t2m, tcc. Grid:
0.25° over Germany. Hourly. Download month-by-month. Convert GRIB →
Parquet. Compute capacity-weighted spatial average. Convert ssrd
cumulative J/m² → instantaneous W/m². Interpolate to 15 min (cubic for
GHI, linear for t2m/tcc).

**Alternative: Open-Meteo.** Free, no signup, 15-min native. Point
forecasts for 5--10 representative German locations. Document the
trade-off vs ERA5.

4.2 Phase 1.2: PV Spatial Layer

MaStR bulk export, filter to Solar, aggregate by Bundesland. Output:
capacity weights per region. Alternative: hard-coded Bundesland
capacities from BNetzA publication.

4.3 Phase 1.3: Target Series

ENTSO-E doc type A16, PsrType B16, zone DE_LU. 15-min resolution. Gap
handling: linear interp \< 2h, NaN \> 2h. Alternative: SMARD CSV.

5\. Harmonisation & Features (Phase 1.4--1.5)

5.1 Phase 1.4: Harmonisation (src/data/harmonise.py)

Create canonical 15-min DatetimeIndex (UTC), join all datasets. Unit
conversions (ssrd→W/m², t2m K→°C, tcc →\[0,1\]). Clip pv_gen_mw to \[0,
capacity_mw\]. Output: data/interim/harmonised.parquet, all float32.

5.2 Phase 1.5: Synthetic Proxy (src/features/proxy.py)

Formula: pv_proxy = ghi_wm2 × efficiency × temp_correction × capacity.
Calibrate a scaling constant on the training period. The proxy uses
observed GHI, so it is **only computable for past timestamps** (≤ T).
For future steps (T+1..T+x), a clear-sky-only proxy can serve as a
degraded reference.

5.3 Feature Catalog

The **Avail @ T?** column is critical. It defines what the model can see
at prediction time T for a target at T+k.

  --------------- ----------------- ----------- ---------- --------- -------------------
  **Feature**     **Source**        **dtype**   **Unit**   **Avail @ **Notes**
                                                           T?**      

  ghi_wm2         ERA5/Open-Meteo   float32     W/m²       Yes       Weather at time T
                                                           (past)    is known at T

  t2m_c           ERA5/Open-Meteo   float32     °C         Yes       Used as feature for
                                                           (past)    steps ≤ T

  tcc             ERA5/Open-Meteo   float32     \[0,1\]    Yes       Total cloud cover
                                                           (past)    fraction

  clear_sky_ghi   pvlib             float32     W/m²       Yes (all) Deterministic;
                                                                     known for future
                                                                     too

  solar_zenith    pvlib             float32     degrees    Yes (all) Deterministic;
                                                                     night filter \>85°

  clear_sky_idx   Derived           float32     ratio      Past only ghi/cs_ghi. Only
                                                                     for T, not T+k

  hour_sin/cos    Calendar          float32     \[-1,1\]   Yes (all) Known for future
                                                                     target times

  month_sin/cos   Calendar          float32     \[-1,1\]   Yes (all) Known for future
                                                                     target times

  capacity_gw     MaStR             float32     GW         Yes       Static per month

  pv_proxy_norm   Synthetic         float32     \[0,1\]    Past only Uses ghi, so only
                                                                     known for ≤ T

  actual_lag_4    ENTSO-E           float32     norm       T-4       Most recent
                                                                     available (\~1h pub
                                                                     lag)

  actual_lag_8    ENTSO-E           float32     norm       T-8       2 hours ago

  actual_lag_96   ENTSO-E           float32     norm       T-96      Same QH yesterday

  target: pv_norm ENTSO-E/cap       float32     \[0,1\]    N/A       What we predict for
                                                                     T+1..T+x
  --------------- ----------------- ----------- ---------- --------- -------------------

  -----------------------------------------------------------------------
  **⭐ KEY CONCEPT:** Features marked "Past only" (ghi, tcc,
  clear_sky_idx, proxy, lags) are available for the model's input context
  at time T but NOT for the forecast targets T+1..T+x. Features marked
  "Yes (all)" (clear_sky_ghi, solar_zenith, calendar) are deterministic
  and available for all times. This asymmetry is central to the model
  design.

  -----------------------------------------------------------------------

5.4 Feature Matrix Builder (src/features/build.py)

Build the full feature matrix for the entire year. Include all features
--- both past-only and deterministic. The rolling forecaster (Section 6)
is responsible for masking features correctly at inference time. The
builder just computes everything.

- **Lag features:** Compute actual_lag_4 (shift by 4), actual_lag_8
  (shift by 8), actual_lag_96 (shift by 96). Note: lag_1 through lag_3
  are excluded because ENTSO-E has \~60 min publication delay, making
  them unavailable in a real-time scenario. The first 96 rows will have
  NaN for lag_96; fill with 0 or forward-fill.

- **Split column:** Add \'train\'/\'val\'/\'test\' based on the temporal
  split. This is used by the training scripts, not by the forecaster.

- **Output:** data/processed/features.parquet. \~35K rows × \~20 columns
  × float32 ≈ \~3 MB. Trivial memory.

**Acceptance:** features.parquet has all catalog columns, dtypes are
float32, lag columns are properly shifted (lag_4\[T\] == actual\[T-4\]),
lag_1/lag_2/lag_3 do NOT exist (excluded due to publication lag).

6\. Rolling Forecast Loop (src/forecaster.py)

  -----------------------------------------------------------------------
  **⭐ KEY CONCEPT:** This is the heart of the pipeline. It simulates a
  real-time prediction stream by stepping through the year quarter-hour
  by quarter-hour, assembling features from data ≤ T at each step, and
  producing predictions for T+1..T+x.

  -----------------------------------------------------------------------

6.1 The Naive Approach (Too Slow)

A straightforward implementation would be:

> \# WRONG: \~35K Python iterations with per-step feature lookup
>
> \# This will take 30+ minutes for the prediction loop alone
>
> for t_idx in range(len(timestamps)):
>
> features_at_t = build_features_up_to(t_idx) \# expensive
>
> pred = model.predict(features_at_t) \# 1 row
>
> results.append(pred)

This is too slow because: (a) per-step feature assembly in Python is
expensive, (b) model.predict() on a single row has high overhead per
call.

6.2 The Vectorised Approach (Correct)

Pre-compute the entire feature matrix. Then for each target step T+k,
the model input is simply a row lookup. The key insight: for x=1, the
features at step T for predicting T+1 are just the row at index T in the
feature matrix (lagged features are already pre-computed as shifted
columns).

> \# CORRECT: Vectorised rolling prediction
>
> import polars as pl
>
> import numpy as np
>
> \# 1. Load pre-computed feature matrix (all timestamps, all features)
>
> df = pl.read_parquet(\'data/processed/features.parquet\')
>
> \# 2. Define which features the model uses as input at time T
>
> \# These are: all weather/lag features AT TIME T, plus
>
> \# deterministic features FOR TIME T+1 (the target time)
>
> input_features = \[
>
> \# Features observed at T (recent context)
>
> \'ghi_wm2\', \'t2m_c\', \'tcc\', \'clear_sky_idx\',
>
> \'actual_lag_4\', \# most recent available (\~1h pub lag)
>
> \'actual_lag_8\', \# 2 hours ago
>
> \'actual_lag_96\', \# same QH yesterday
>
> \'pv_proxy_norm\',
>
> \# Deterministic features for T+1 (known in advance)
>
> \'clear_sky_ghi_ahead_1\', \'solar_zenith_ahead_1\',
>
> \'hour_sin_ahead_1\', \'hour_cos_ahead_1\',
>
> \'month_sin_ahead_1\', \'month_cos_ahead_1\',
>
> \]
>
> \# 3. Pre-compute the \'ahead\' columns (just shift deterministic
>
> \# features by -1 so row T contains T+1\'s values)
>
> for col in \[\'clear_sky_ghi\',\'solar_zenith\',\'hour_sin\',\...\]:
>
> df = df.with_columns(
>
> pl.col(col).shift(-1).alias(f\'{col}\_ahead_1\')
>
> )
>
> \# 4. Batch predict on the eval split
>
> eval_mask = df\[\'split\'\].is_in(\[\'val\', \'test\'\])
>
> X_eval = df.filter(eval_mask).select(input_features).to_numpy()
>
> preds = model.predict(X_eval) \# ONE call, \~35ms for LightGBM
>
> \# 5. This is valid because:
>
> \# - Row T contains observed features at T (past)
>
> \# - Row T contains shifted deterministic features for T+1 (future ok)
>
> \# - Lag columns (actual_lag_4 at row T = actual at T-4) respect pub
> lag
>
> \# - NO future weather or actuals leak into the input

6.3 Multi-Step Horizon (x \> 1)

For x = 4 (predict T+1, T+2, T+3, T+4), there are two strategies:

**Strategy A --- Direct multi-output:** Train one model per horizon step
(model_h1, model_h2, model_h3, model_h4). Each model is trained with
appropriate features: model_h1 gets lag_4 (most recent available),
model_h4 may drop shorter lags since the target is further away. Simple,
no error accumulation. Recommended.

**Strategy B --- Recursive:** Predict T+1, plug prediction into features
for T+2, predict T+2, etc. More complex, error accumulates. Not
recommended for the assignment.

**Strategy A implementation:** pre-compute clear_sky_ghi_ahead_k and
solar_zenith_ahead_k for k=1..4 (shift by -k). Train 4 LightGBM models
during Phase 2.2. Batch-predict all 4 in the eval loop. Store
predictions in separate columns of predictions.parquet.

6.4 Night Shortcut

When solar_zenith at T+1 (or all T+1..T+x) exceeds 85°, skip model
inference and output 0 MW. This saves \~50% of compute (nighttime is
\~half the year). Check: if all(solar_zenith_ahead_k \> 85 for k in
range(1,x+1)): pred = 0.

6.5 Runtime Budget (Compute Only, Excl. Download)

The \~45 min budget covers everything AFTER data is downloaded.
Breakdown:

- Harmonise + feature build (Phase 1.4--1.5): \~2--5 min

- Model training (Phase 2.1--2.2): \~2--5 min (LightGBM on \~25K train
  rows is seconds; optional Optuna adds \~3 min)

- Rolling prediction over 35,040 steps (Phase 2.3): \~1--3 min
  (vectorised batch predict)

- Evaluation + Parquet write: \<1 min

Total compute: \~5--15 min typical, well within 45 min.

  -----------------------------------------------------------------------
  **⚠ HARD CONSTRAINT:** If the compute pipeline (excl. download) takes
  \>30 min, something is wrong. The rolling prediction alone should take
  \<3 min when vectorised. Data download (ERA5, ENTSO-E) can take 10--30
  min depending on API speed but is NOT part of the timed budget.

  -----------------------------------------------------------------------

**Acceptance:** The rolling prediction loop produces one prediction per
quarter-hour for the eval period. No prediction uses any feature from a
timestamp \> T. Verify by spot-checking: prediction at T=2023-07-15
12:00 uses lags from ≤12:00, not 12:15 or later.

7\. Modelling (Phase 2)

  -------------- ---------------------- ----------------- -----------------
  **Model**      **Role**               **Features at T** **Expected nMAE**

  Synthetic      Phase 1.5 floor ref    cs_ghi(T+k) × cap \~8--12%
  Proxy                                                   

  Baseline       Phase 2.1 transparent  proxy + cal +     \~5--8%
  (Ridge)                               lags              

  LightGBM       Phase 2.2 candidate    All avail         \~3--5%
                                        features          

  (Opt.)         Phase 2.2 if time      Sequence input    \~3--5%
  Temporal CNN                                            
  -------------- ---------------------- ----------------- -----------------

7.1 Phase 2.1: Baseline (src/models/baseline.py)

Ridge regression. Features: the subset of input_features defined in
Section 6.2. StandardScaler fitted on train split only. For x=1, this is
a single model. For x=4, train one Ridge per horizon step.

The proxy itself (irradiance × capacity) is NOT a model --- it's a
feature and a floor reference. The baseline is a learned regression that
takes the proxy and other features as input.

7.2 Phase 2.2: Candidate (src/models/candidate.py)

**LightGBM.** Same feature set. Hyperparameters: num_leaves=63, lr=0.05,
n_estimators=1000, early_stopping=50 on val split. Optional Optuna
tuning. For x\>1, train one LightGBM per horizon step.

- **Feature importance:** Extract and save top-10 feature importances.
  These go in the technical note.

- **Optional Temporal CNN:** Only if LightGBM is done and tested. Input:
  sequence of recent features (e.g., last 24 QH). Keep \< 200K params.
  Train on CPU.

7.3 Phase 2.3: Validation (src/models/evaluate.py)

Temporal Split

  --------------- --------------------------- ---------------------------
  **Split**       **Period (1yr example)**    **Usage**

  Train           Jan 1 -- Aug 31 (8 months)  Fit model parameters

  Validation      Sep 1 -- Oct 31 (2 months)  Early stopping, HP tuning

  Test            Nov 1 -- Dec 31 (2 months)  Final reported metrics ONLY
  --------------- --------------------------- ---------------------------

Models are trained on train split. Early stopping on val split. Final
metrics reported on test split only.

Metrics

- **Primary: nMAE.** = MAE / installed_capacity. Equivalently: MAE on
  pv_norm.

- **Secondary: Skill score vs. proxy.** skill = 1 − MAE_model /
  MAE_proxy. Must be \> 0.

- **Additional:** RMSE, R², MAE by time-of-day and by horizon step (if
  x\>1).

Leakage Controls (document in technical note)

- **No future weather:** observed ghi/t2m/tcc only from ≤ T. Only
  deterministic features (clear-sky, calendar) are used for T+k. ERA5 is
  treated as observed weather at T (event has occurred), not as a
  forecast.

- **No future actuals + publication lag respected:** lag features start
  at T-4 (not T-1) to model ENTSO-E's \~60 min publication delay. lag_1
  through lag_3 are excluded. Target at T+k is never in the feature set.

- **Scaler on train only.** StandardScaler.fit() on train, .transform()
  on val/test.

- **No HP tuning on test.** Optuna runs on val. Test is touched once at
  the end.

predictions.parquet Schema

> \# Required columns:
>
> timestamp_utc: datetime\[ns, UTC\] \# the TARGET time (T+k, what was
> predicted)
>
> issued_at: datetime\[ns, UTC\] \# when the prediction was made (T)
>
> horizon_step: int8 \# k (1 for T+1, 4 for T+4, etc.)
>
> pv_actual_mw: float32 \# realised generation at target time
>
> pv_pred_mw: float32 \# candidate model prediction (denormalised)
>
> pv_baseline_mw: float32 \# baseline prediction (denormalised)
>
> pv_proxy_mw: float32 \# synthetic proxy at target time (if avail)
>
> split: string \# \'train\'/\'val\'/\'test\'
>
> \# For x=1: one row per quarter-hour. \~35K rows.
>
> \# For x=4: four rows per quarter-hour (one per horizon step). \~140K
> rows.
>
> \# The test-split rows with horizon_step=1 are used for headline
> metrics.

**Acceptance:** predictions.parquet has expected schema. candidate
nMAE_test \< baseline nMAE_test. Skill score \> 0. No prediction at time
T uses any feature from T+1 or later.

8\. Reproducible Environment

Use **uv** (preferred) or Poetry.

> \[project\]
>
> name = \"pv-forecast\"
>
> version = \"0.1.0\"
>
> requires-python = \"\>= 3.11\"
>
> dependencies = \[
>
> \"polars\>=1.0\", \"pyarrow\>=15.0\", \"pandas\>=2.2\",
>
> \"pvlib\>=0.11\", \"lightgbm\>=4.3\", \"scikit-learn\>=1.4\",
>
> \"entsoe-py\>=0.6\", \"cdsapi\>=0.7\", \"cfgrib\>=0.9\",
>
> \"xarray\>=2024.1\", \"requests\", \"pyyaml\", \"joblib\",
>
> \]
>
> \[project.optional-dependencies\]
>
> dev = \[\"pytest\", \"ruff\", \"optuna\"\]

- **Lock:** uv lock and commit uv.lock.

- **Seeds:** numpy.random.seed(42), lightgbm random_state=42.

- **Artifacts:** Save models to outputs/ with meta.json (features, HPs,
  train dates, metrics).

- **Git hash:** Log at pipeline start.

9\. Technical Note Outline (1--6 pages, PDF)

4.  **Data Sources (1--2pp).** Considered vs. used. Rejected sources and
    why. CRS, resampling, calibration.

5.  **Harmonisation & Proxy (0.5--1pp).** Timestamp alignment, unit
    conversions, proxy formula + calibration. Discuss publication lag
    modelling: ERA5 treated as observed weather (event occurred at T,
    even though ERA5 publishes days later); ENTSO-E actuals modelled
    with \~60 min lag (lag_4 is most recent, lag_1--3 excluded).

6.  **Modelling (1--2pp).** Rolling forecast paradigm explanation.
    Feature availability at time T. Horizon choice and justification.
    Model architecture. Vectorisation strategy for runtime constraint.

7.  **Results (0.5--1pp).** nMAE, RMSE, skill score table. Error by
    time-of-day. Error by horizon step (if x\>1). Feature importance.
    Runtime log summary.

8.  **Limitations & Next Steps (0.5pp).** NWP integration for longer
    horizons (ICON-EU with \~3h publication lag replacing ERA5
    reanalysis). True real-time actuals feed with measured publication
    delay rather than assumed 60 min. Probabilistic forecasting
    (quantile regression). Multi-model ensemble. Retraining strategies.
    Satellite-derived irradiance.

  -----------------------------------------------------------------------
  **⚠ NOTE:** Evaluation weights: Data Sourcing & Engineering 40%,
  Modelling & Validation 40%, Reporting & Reproducibility 20%. The
  rolling forecast paradigm is where you demonstrate modelling judgment.
  Explain the leakage controls clearly.

  -----------------------------------------------------------------------

10\. Implementation Order

  ----------- ------------------ ---------------------------- ----------------- --------------
  **Phase**   **Deliverable**    **Output**                   **Acceptance      **Mem Note**
                                                              Test**            

  1.1         Weather layer      raw/era5/\*.parquet          1yr hourly, no    Stream
                                                              NaN in ssrd       per-month

  1.2         PV spatial layer   raw/capacity/\*.parquet      Total GW ±5% of   Filter MaStR
                                                              known             early

  1.3         Target series      raw/actuals/\*.parquet       15min, gap\<0.5%  \~15 MB
                                                                                trivial

  1.4         Harmonise          interim/harmonised.parquet   Aligned ts, units Scan lazily
                                                              OK                

  1.5         Proxy + features   processed/features.parquet   Proxy corr\>0.85  float32
                                                              vs actual         throughout

  2.1         Baseline model     baseline_model.joblib        nMAE computed, \> Sklearn fast
                                                              proxy             

  2.2         Candidate model    candidate_model.lgbm         nMAE \< baseline  LightGBM
                                                                                \~1-2GB

  2.3         Rolling predict +  predictions.parquet          35K rows, compute Vectorised
              eval                                            \<45min           loop
  ----------- ------------------ ---------------------------- ----------------- --------------

**END OF DOCUMENT**

Feed to Claude Code section by section. The single most important thing
to get right is Section 6: the rolling forecast loop must never peek
into the future. Build Phase 1 first, validate features, then implement
the simplest rolling forecast (x=1) before attempting multi-step
horizons. When in doubt, choose simplicity and document the trade-off.
