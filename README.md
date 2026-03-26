# Solar PV Forecasting — Germany (Quarter-Hourly)

End-to-end pipeline for predicting quarter-hourly national solar PV generation
in Germany, combining ERA5 weather data, installed capacity distribution, and
realised generation from SMARD.

## Quick Start

```bash
# Install dependencies (requires Python ≥3.11 and uv)
make install

# Run full pipeline: data → proxy → train → evaluate
make full-run
```

Output: `output/predictions.parquet` + `output/runtime.log`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources (Phase 1)                      │
│  Open-Meteo (ERA5)  ·  PV Capacity (BNetzA)  ·  SMARD (target) │
└────────────┬────────────────────┬────────────────────┬──────────┘
             │                    │                    │
             ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│            Harmonise & Align (15-min UTC, Parquet)               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│     Synthetic Proxy: P_syn = η × Σ(GHI_i × capacity_weight_i)  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
     ┌──────────────────┐       ┌──────────────────────┐
     │ Baseline (Ridge) │       │ LightGBM + features  │
     └────────┬─────────┘       └──────────┬───────────┘
              │                            │
              ▼                            ▼
     ┌──────────────────────────────────────────────┐
     │ Rolling-origin validation: nMAE, skill score │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
               predictions.parquet + report
```

## Data Sources

| Layer | Source | Resolution | Access |
|-------|--------|-----------|--------|
| Weather | Open-Meteo Historical API (ERA5) | Hourly → 15-min interp. | Free, no API key |
| PV spatial | Bundesnetzagentur (aggregated) | Per federal state | Public statistics |
| Target | SMARD (Bundesnetzagentur) | 15-min native | Free, no registration |

## Project Structure

```
solar-pv-forecast/
├── Makefile                    # Single-command entry points
├── pyproject.toml              # Dependencies and project metadata
├── README.md
│
├── src/solar_pv_forecast/      # Python package
│   ├── config.py               # All constants and configuration
│   ├── utils.py                # Logging, timing utilities
│   ├── data/
│   │   ├── fetch_weather.py    # Open-Meteo API client
│   │   ├── fetch_target.py     # SMARD solar generation download
│   │   ├── fetch_pv_capacity.py # PV capacity weights
│   │   └── harmonise.py        # Align and merge all layers
│   ├── proxy/
│   │   └── build_proxy.py      # Synthetic national PV proxy
│   └── model/
│       ├── features.py         # Feature engineering
│       ├── train.py            # Baseline + LightGBM training
│       └── evaluate.py         # Metrics, plots, predictions.parquet
│
├── data/
│   ├── raw/                    # Downloaded source data
│   ├── interim/                # Intermediate processing
│   └── processed/              # Clean modelling tables
│
├── models/                     # Saved model artefacts
├── output/                     # Final predictions + plots + logs
├── report/                     # LaTeX technical note
│   └── technical_note.tex
├── notebooks/                  # Exploratory analysis (optional)
└── tests/                      # Unit tests
```

## Make Targets

| Command | Description |
|---------|-------------|
| `make full-run` | Run entire pipeline end-to-end |
| `make install` | Set up Python environment |
| `make data` | Phase 1: fetch and harmonise all data |
| `make proxy` | Build synthetic PV proxy |
| `make train` | Train baseline + LightGBM |
| `make evaluate` | Evaluate and produce predictions.parquet |
| `make report` | Compile LaTeX report to PDF |
| `make clean` | Remove intermediate files |

## Constraints

- **Compute**: 8 vCPU, 32 GB RAM, no GPU
- **Language**: Python ≥ 3.11
- **Runtime**: Full-year backfill target ≈ 45 min
- **Memory**: Peak well below 32 GB (uses Polars lazy eval + float32)

## Technical Note

See [`report/technical_note.tex`](report/technical_note.tex) for the full
writeup covering data sourcing rationale, proxy construction, modelling
choices, validation setup, results, and limitations.
