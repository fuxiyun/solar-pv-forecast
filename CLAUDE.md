# CLAUDE.md

## Project Overview
This is a solar PV forecasting pipeline that predicts quarter-hourly national solar generation for Germany. It's a technical assessment for an internship at an energy trading company.

## Architecture
The pipeline has 5 stages:
1. **Data fetch** — Weather (Open-Meteo ERA5), target generation (SMARD), PV capacity (BNetzA)
2. **Harmonise** — Align to 15-min UTC, interpolate weather, merge to Parquet
3. **Synthetic proxy** — Capacity-weighted GHI aggregation: P_syn = η × Σ(GHI_i × w_i)
4. **Train** — Baseline (ridge regression) + LightGBM with calendar/solar/weather features
5. **Evaluate** — Rolling-origin nMAE, skill score, diagnostic plots → predictions.parquet

## Running the Pipeline
```bash
# Full pipeline
export PYTHONPATH="$PWD/src:$PYTHONPATH"
make full-run

# Individual steps
python -m solar_pv_forecast.data.fetch_weather
python -m solar_pv_forecast.data.fetch_target
python -m solar_pv_forecast.data.fetch_pv_capacity
python -m solar_pv_forecast.data.harmonise
python -m solar_pv_forecast.proxy.build_proxy
python -m solar_pv_forecast.model.train
python -m solar_pv_forecast.model.evaluate

# Diagnostics (validates every stage with plots)
python -m solar_pv_forecast.diagnostics
```

## Key Files
- `src/solar_pv_forecast/config.py` — All constants, coordinates, model hyperparameters
- `src/solar_pv_forecast/data/` — Data fetching and harmonisation
- `src/solar_pv_forecast/proxy/build_proxy.py` — Synthetic proxy construction
- `src/solar_pv_forecast/model/features.py` — Feature engineering
- `src/solar_pv_forecast/model/train.py` — Model training (baseline + LightGBM)
- `src/solar_pv_forecast/model/evaluate.py` — Metrics, plots, final output
- `src/solar_pv_forecast/diagnostics.py` — 6-check validation suite
- `report/technical_note.tex` — LaTeX technical report

## Tech Stack
- Python ≥3.11, polars/pandas, pyarrow (Parquet)
- pvlib (solar position, clear-sky), LightGBM, scikit-learn
- Open-Meteo API (weather), SMARD API (actual generation)
- LaTeX for report, matplotlib for plots

## Constraints
- 8 vCPU, 32 GB RAM, no GPU
- Full-year backfill in ~45 min
- Float32 where safe, Parquet partitioned by year/month

## Data Locations
- `data/raw/` — Downloaded source files
- `data/interim/` — Intermediate processing (15-min interpolated weather)
- `data/processed/` — Clean modelling table, feature table, test predictions
- `output/` — Final predictions.parquet, plots, diagnostics, runtime log
- `models/` — Saved model artifacts (ridge pkl, LightGBM txt)

## Current Status
- Pipeline scaffolding complete, validated end-to-end with synthetic data
- Synthetic data generator in `data/generate_synthetic.py` (for offline testing)
- Need to run with real API data (Open-Meteo + SMARD) on a machine with internet
- Report drafted but needs real results filled in

## Code Style
- Use loguru for logging, not print()
- Type hints on function signatures
- Docstrings on all modules and public functions
- Keep functions focused and under ~50 lines
