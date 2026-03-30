# Solar PV Forecasting — Germany (Quarter-Hourly)

Intraday rolling forecast pipeline for quarter-hourly national solar PV generation in Germany. Predicts 4 hours ahead (16 steps at 15-min resolution) using ERA5 reanalysis, ICON-D2 NWP forecasts, and SMARD actual generation data.

**Key results** (walk-forward evaluation over 2025):
- LightGBM nMAE: **0.121** (MAE: 254 MW)
- Skill vs Ridge baseline: **14.6%**
- Monthly retraining benefit vs frozen model: **4.0%**

## Quick Start

```bash
# Install dependencies (requires Python >= 3.11 and uv)
make install

# Run full pipeline: fetch data -> harmonise -> proxy -> tune + walk-forward -> evaluate
make full-run
```

Output: `output/predictions.parquet`, `output/evaluation_summary.json`, diagnostic plots.

## Pipeline Architecture

```
 Data Fetch (Phase 1)
 ├── ERA5 weather (hourly, 16 states)         fetch_weather.py
 ├── ICON-D2 NWP forecasts (15-min native)    fetch_nwp.py
 ├── SMARD actual solar generation (15-min)   fetch_target.py
 └── PV capacity per Bundesland (static)      fetch_pv_capacity.py
         │
         ▼
 Harmonise (Phase 1.4)                        harmonise.py
 ├── Interpolate ERA5 hourly -> 15-min (cubic spline GHI, linear others)
 ├── Capacity-weighted national averages (ERA5 + ICON-D2)
 └── Merge with target, gap-fill, cast to float32
         │
         ▼
 Synthetic Proxy (Phase 1.5)                  build_proxy.py
 └── P_syn = eta * sum(GHI_i * w_i), eta fitted via OLS on 2024
         │
         ▼
 Feature Engineering (Phase 2)                features.py
 ├── Calendar: hour_sin/cos, doy_sin/cos, month
 ├── Solar position: zenith, elevation, clearsky GHI/index (pvlib)
 └── Lags: actual_lag_1d (shift 96), actual_lag_7d (shift 672)
         │
         ▼
 Multi-Horizon Expansion                      train.py
 └── For each origin T, horizons h=1..16:
     ├── Origin features from T (weather, lags, proxy)
     ├── Target features from T+h (calendar, solar, NWP forecast)
     └── Target value: actual generation at T+h
         │
         ▼
 Optuna HP Tuning (once on 2024)              tune.py
 └── 30 trials, 3-fold rolling-origin CV, 5-min timeout
         │
         ▼
 Walk-Forward Evaluation (12 rounds)          walk_forward.py
 ├── Round 1:  train Jan-Dec 2024,        test Jan 2025
 ├── Round 2:  train Jan 2024-Jan 2025,   test Feb 2025
 ├── ...
 └── Round 12: train Jan 2024-Nov 2025,   test Dec 2025
     ├── Retrain baseline (Ridge) + LightGBM each round
     └── Frozen model (Round 1 only) for comparison
         │
         ▼
 Evaluate + Report                            evaluate.py
 ├── Overall / monthly / horizon / time-of-day metrics
 ├── Retrained vs frozen model comparison
 ├── Diagnostic plots (4 PNG files)
 └── predictions.parquet (final deliverable)
```

## Project Structure

```
solar-pv-forecast/
├── Makefile                          Pipeline orchestration
├── pyproject.toml                    Dependencies (uv/hatch)
├── CLAUDE.md                         AI assistant instructions
│
├── src/solar_pv_forecast/            Python package
│   ├── config.py                     All constants and configuration
│   ├── utils.py                      Logging (loguru), timing
│   ├── cli.py                        Click CLI entry point
│   ├── dashboard.py                  Streamlit interactive dashboard
│   │
│   ├── data/                         Phase 1: data sourcing
│   │   ├── fetch_weather.py          Open-Meteo ERA5 API client
│   │   ├── fetch_nwp.py              Open-Meteo ICON-D2 NWP forecasts
│   │   ├── fetch_target.py           SMARD solar generation download
│   │   ├── fetch_pv_capacity.py      PV capacity weights per state
│   │   └── harmonise.py              Align, interpolate, merge to 15-min
│   │
│   ├── proxy/                        Phase 1.5: synthetic proxy
│   │   └── build_proxy.py            Capacity-weighted GHI -> MW
│   │
│   └── model/                        Phase 2: modelling
│       ├── features.py               Feature engineering (calendar, solar, lags)
│       ├── train.py                  Model fitting utilities (Ridge, LightGBM)
│       ├── tune.py                   Optuna HP tuning on 2024
│       ├── walk_forward.py           12-round monthly walk-forward loop
│       └── evaluate.py               Metrics, plots, final predictions
│
├── data/
│   ├── raw/                          Downloaded source files
│   │   ├── weather_hourly.parquet        ERA5 (280K rows, 16 states)
│   │   ├── nwp_icon_d2_15min.parquet     ICON-D2 (1.1M rows, 16 states)
│   │   ├── actual_solar_generation.parquet  SMARD (70K rows)
│   │   └── pv_capacity_by_state.parquet  BNetzA (16 rows)
│   ├── interim/                      Intermediate (15-min interpolated weather)
│   └── processed/                    Modelling tables, feature tables, predictions
│       ├── modelling_table.parquet       70K rows x 26 cols (7.6 MB)
│       ├── features_table.parquet        70K rows x 38 cols (9.3 MB)
│       └── test_predictions.parquet      560K rows x 10 cols (9.7 MB)
│
├── models/                           Saved model artifacts
│   ├── tuned_lgbm_params.json        Optuna-tuned hyperparameters
│   ├── walk_forward_summary.json     Walk-forward results summary
│   ├── baseline_ridge.pkl            Ridge model + scaler
│   └── lightgbm_model.txt            LightGBM booster
│
├── output/                           Final deliverables
│   ├── predictions.parquet           560K predictions (all horizons, all months)
│   ├── evaluation_summary.json       Overall metrics
│   ├── monthly_metrics.csv           Per-round nMAE
│   ├── horizon_metrics.csv           Per-horizon nMAE
│   ├── tod_metrics.csv               Time-of-day nMAE
│   └── *.png                         Diagnostic plots
│
├── report/                           Technical note
│   └── technical_note.tex            LaTeX source (compile with tectonic)
│
└── tests/                            Unit tests
```

## Make Targets

| Command | Description |
|---------|-------------|
| `make full-run` | Full pipeline: install -> data -> proxy -> walk-forward -> evaluate -> dashboard |
| `make install` | Set up Python environment with `uv sync` |
| `make data` | Fetch and harmonise all data (ERA5 + ICON-D2 + SMARD + capacity) |
| `make data-nwp` | Fetch ICON-D2 NWP forecasts only |
| `make proxy` | Build synthetic PV proxy |
| `make tune` | Run Optuna HP tuning on 2024 (standalone) |
| `make train` | Single-split train (no walk-forward) |
| `make walk-forward` | Optuna tuning + 12-round walk-forward evaluation |
| `make evaluate` | Compute metrics, generate plots, write predictions.parquet |
| `make dashboard` | Launch Streamlit interactive dashboard |
| `make report` | Compile LaTeX technical note to PDF |
| `make clean` | Remove intermediate and output files |

## Data Sources

| Source | Variables | Resolution | Access |
|--------|----------|-----------|--------|
| Open-Meteo ERA5 | GHI, temperature, wind speed, humidity | Hourly, 0.25deg | Free, no key |
| Open-Meteo ICON-D2 | GHI, temperature, cloud cover | 15-min native, ~2 km | Free, no key |
| SMARD (BNetzA) | Actual solar generation | 15-min | Free, no key |
| BNetzA / MaStR | PV capacity per Bundesland | Static (2024) | Public statistics |
| pvlib | Clear-sky GHI, solar zenith | Computed | Python library |

## Feature Design

At prediction time T, forecasting T+h (h = 1..16):

| Type | Source time | Features |
|------|-----------|----------|
| **Origin** (observed) | T | ERA5 GHI/temp/wind, proxy, clearsky index, lag 1d/7d |
| **Target deterministic** | T+h | Clear-sky GHI, solar zenith, hour/DoY sin/cos, month |
| **Target NWP** | T+h | ICON-D2 forecast GHI, temperature, cloud cover |

**18 features total** for LightGBM; **6 features** for Ridge baseline.

Night mask: predictions zeroed when target solar zenith > 85deg.

## Constraints

- **Compute**: 8 vCPU, 32 GB RAM, no GPU
- **Runtime**: Full pipeline ~15 min (well within 45-min budget)
- **Memory**: Peak ~0.5 GB (70K rows x 16 horizons, float32)
- **Language**: Python >= 3.11

## Dashboard

```bash
make dashboard
# or: streamlit run src/solar_pv_forecast/dashboard.py
```

4 tabs:
1. **Raw Data** — GHI vs PV production, per-state weather
2. **Predictions vs Actual** — h=1 and h=4 time series, scatter plots, KPIs
3. **Horizon Analysis** — MAE/nMAE curves across 16 horizon steps, time-of-day error
4. **Walk-Forward Metrics** — Monthly nMAE trend, retrained vs frozen comparison

## Technical Note

See [`report/technical_note.tex`](report/technical_note.tex). Compile with:

```bash
tectonic report/technical_note.tex
# or: cd report && pdflatex technical_note.tex && pdflatex technical_note.tex
```

Covers: data sources, harmonisation, proxy construction, modelling paradigm, validation, results (all numbers from pipeline outputs), limitations and next steps.
