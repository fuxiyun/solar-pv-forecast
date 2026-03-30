"""Pipeline configuration: paths, constants, and data source definitions."""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = ROOT_DIR / "models"

# Create directories on import
for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Temporal scope ──────────────────────────────────────────────
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"  # will be clipped to available data

# ── Weather data: representative coordinates per Bundesland ─────
# Each tuple: (state_name, latitude, longitude)
# Chosen near PV-dense population/industry centres, avoiding
# extreme alpine or coastal locations.
GERMAN_STATES = {
    "Schleswig-Holstein":       (54.22, 9.98),
    "Hamburg":                  (53.55, 10.00),
    "Niedersachsen":            (52.45, 9.75),
    "Bremen":                   (53.08, 8.80),
    "Nordrhein-Westfalen":      (51.43, 7.66),
    "Hessen":                   (50.65, 8.77),
    "Rheinland-Pfalz":          (49.75, 7.44),
    "Baden-Württemberg":        (48.78, 9.18),
    "Bayern":                   (48.79, 11.50),
    "Saarland":                 (49.24, 6.99),
    "Berlin":                   (52.52, 13.40),
    "Brandenburg":              (52.41, 13.07),
    "Mecklenburg-Vorpommern":   (53.63, 11.40),
    "Sachsen":                  (51.05, 13.74),
    "Sachsen-Anhalt":           (51.95, 11.69),
    "Thüringen":                (50.98, 11.03),
}

# ── Open-Meteo API ──────────────────────────────────────────────
OPENMETEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARIABLES = [
    "shortwave_radiation",
    "temperature_2m",
    "wind_speed_10m",
    "relative_humidity_2m",
]

# ── PV capacity by Bundesland (MWp, approximate 2024 values) ───
# Source: Bundesnetzagentur / OPSD. Will be updated from live data
# if available; these serve as fallback.
PV_CAPACITY_MWP = {
    "Bayern":                   21_800,
    "Baden-Württemberg":        10_200,
    "Nordrhein-Westfalen":       8_900,
    "Niedersachsen":             7_600,
    "Brandenburg":               6_100,
    "Sachsen-Anhalt":            4_200,
    "Schleswig-Holstein":        3_200,
    "Sachsen":                   3_100,
    "Rheinland-Pfalz":           3_800,
    "Hessen":                    4_400,
    "Thüringen":                 2_800,
    "Mecklenburg-Vorpommern":    3_000,
    "Saarland":                    700,
    "Berlin":                      300,
    "Hamburg":                     200,
    "Bremen":                      100,
}

# ── Forecast configuration ──────────────────────────────────────
FORECAST_HORIZON_STEPS = 16   # 4 hours ahead at 15-min resolution
FORECAST_STEP_MINUTES = 15

# ── Walk-forward configuration ─────────────────────────────────
WALK_FORWARD_TEST_YEAR = 2025
WALK_FORWARD_N_ROUNDS = 12
EARLY_STOP_VAL_MONTHS = 1     # last N months of train window for early stopping

# ── Model configuration ─────────────────────────────────────────
TRAIN_END_DATE = "2024-12-31"  # train on 2024, test on 2025
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "verbose": -1,
    "random_state": 42,
}
EARLY_STOPPING_ROUNDS = 50
