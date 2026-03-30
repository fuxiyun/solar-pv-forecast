"""Model fitting and prediction utilities.

Provides reusable functions for fitting baseline (Ridge) and candidate
(LightGBM) models, and for generating predictions.  These are called
by the walk-forward loop (walk_forward.py).

Run standalone (single-split, no walk-forward):
    python -m solar_pv_forecast.model.train
"""

import json
import pickle

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from solar_pv_forecast.config import (
    EARLY_STOPPING_ROUNDS,
    FORECAST_HORIZON_STEPS,
    LIGHTGBM_PARAMS,
    MODEL_DIR,
    PROCESSED_DIR,
    TRAIN_END_DATE,
)
from solar_pv_forecast.model.features import engineer_features
from solar_pv_forecast.utils import log_step, setup_logger

# ── Feature sets ────────────────────────────────────────────────
# Origin features: observed at time T (past only — NOT available for future)
ORIGIN_FEATURES = [
    "proxy_solar_mw",
    "ghi_wm2_national",
    "temperature_2m_national",
    "wind_speed_10m_national",
    "clearsky_index",
    "actual_lag_1d",
    "actual_lag_7d",
]

# Target features: deterministic, known for future T+h
# These get prefixed with "target_" in the multi-horizon table
TARGET_DETERMINISTIC_FEATURES = [
    "clearsky_ghi",
    "solar_zenith",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month",
]

TARGET = "actual_solar_mw"

# Combined feature list used for model training (built by build_multihorizon_data)
BASELINE_FEATURES = [
    "proxy_solar_mw", "clearsky_index", "forecast_horizon",
    "target_clearsky_ghi", "target_solar_zenith",
]
CANDIDATE_FEATURES = (
    ORIGIN_FEATURES
    + [f"target_{f}" for f in TARGET_DETERMINISTIC_FEATURES]
    + ["forecast_horizon"]
)


# ── Multi-horizon expansion ────────────────────────────────────
def build_multihorizon_data(df: pd.DataFrame) -> pd.DataFrame:
    """Expand feature table into multi-horizon rolling forecast rows.

    For each origin index i and horizon h (1..H):
      - Origin features (weather, lags, proxy) come from row i (time T)
      - Target deterministic features (calendar, solar) come from row i+h (time T+h)
      - Target value comes from row i+h
      - forecast_horizon = h

    This respects the real-time invariant: at time T, only data <= T
    is used for weather/lags, while deterministic features for T+h
    are known in advance.
    """
    n = len(df)
    origin_cols = [c for c in ORIGIN_FEATURES if c in df.columns]
    target_det_cols = [c for c in TARGET_DETERMINISTIC_FEATURES if c in df.columns]

    chunks = []
    for h in range(1, FORECAST_HORIZON_STEPS + 1):
        n_rows = n - h

        # Origin features from row T (index 0..n-h-1)
        origin = df[origin_cols].iloc[:n_rows].reset_index(drop=True)

        # Target deterministic features from row T+h (index h..n-1)
        target_det = df[target_det_cols].iloc[h:].reset_index(drop=True)
        target_det.columns = [f"target_{c}" for c in target_det.columns]

        # Target value and timestamp from row T+h
        target_val = df[[TARGET, "timestamp"]].iloc[h:].reset_index(drop=True)

        chunk = pd.concat([origin, target_det, target_val], axis=1)
        chunk["forecast_horizon"] = np.int8(h)
        chunk["origin_timestamp"] = df["timestamp"].values[:n_rows]

        chunks.append(chunk)

    result = pd.concat(chunks, ignore_index=True)
    logger.info(
        f"  Multi-horizon expansion: {len(df):,} → {len(result):,} rows "
        f"({FORECAST_HORIZON_STEPS} horizons)"
    )
    return result


# ── Baseline (Ridge) ───────────────────────────────────────────
def fit_baseline(
    train: pd.DataFrame,
) -> dict:
    """Fit ridge regression baseline.  Returns dict with model artifacts."""
    feats = [f for f in BASELINE_FEATURES if f in train.columns]

    X = train[feats].values
    y = train[TARGET].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_s, y)

    mae = np.abs(y - model.predict(X_s)).mean()
    logger.info(f"    Baseline fit — train MAE: {mae:.0f} MW ({len(train):,} rows)")

    return {"model": model, "scaler": scaler, "features": feats}


def _apply_night_mask(preds: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Zero out predictions where target solar zenith > 85° (nighttime)."""
    if "target_solar_zenith" in df.columns:
        preds = preds.copy()
        preds[df["target_solar_zenith"].values > 85] = 0.0
    return preds


def predict_baseline(model_info: dict, df: pd.DataFrame) -> np.ndarray:
    """Generate clipped predictions from a fitted baseline."""
    X = df[model_info["features"]].values
    X_s = model_info["scaler"].transform(X)
    preds = np.clip(model_info["model"].predict(X_s), 0, None)
    return _apply_night_mask(preds, df)


# ── LightGBM (candidate) ──────────────────────────────────────
def fit_lightgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    lgbm_params: dict | None = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> dict:
    """Fit LightGBM candidate model with early stopping on val set.

    Returns dict with model artifacts.
    """
    import lightgbm as lgb

    params = (lgbm_params or LIGHTGBM_PARAMS).copy()
    n_estimators = params.pop("n_estimators", 500)
    params.pop("random_state", None)

    feats = [f for f in CANDIDATE_FEATURES if f in train.columns]

    X_train, y_train = train[feats], train[TARGET]
    X_val, y_val = val[feats], val[TARGET]

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    callbacks = [
        lgb.early_stopping(early_stopping_rounds),
        lgb.log_evaluation(0),  # silent during walk-forward
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    mae_train = np.abs(y_train.values - np.clip(model.predict(X_train), 0, None)).mean()
    mae_val = np.abs(y_val.values - np.clip(model.predict(X_val), 0, None)).mean()
    logger.info(
        f"    LightGBM fit — train MAE: {mae_train:.0f} MW, "
        f"val MAE: {mae_val:.0f} MW "
        f"(best iter: {model.best_iteration})"
    )

    importance = dict(zip(feats, model.feature_importance(importance_type="gain")))
    return {"model": model, "features": feats, "importance": importance}


def predict_lightgbm(model_info: dict, df: pd.DataFrame) -> np.ndarray:
    """Generate clipped predictions from a fitted LightGBM."""
    X = df[model_info["features"]]
    preds = np.clip(model_info["model"].predict(X), 0, None)
    return _apply_night_mask(preds, df)


# ── Standalone entry point (single split, no walk-forward) ─────
def main():
    """Train once on 2024, test on 2025 (no walk-forward)."""
    setup_logger()

    with log_step("Load and engineer features"):
        df = pd.read_parquet(PROCESSED_DIR / "modelling_table.parquet")
        df = engineer_features(df)
        df.to_parquet(PROCESSED_DIR / "features_table.parquet", index=False)

    with log_step("Build multi-horizon data"):
        mh = build_multihorizon_data(df)

    with log_step("Train/test split"):
        cutoff = pd.Timestamp(TRAIN_END_DATE)
        all_train = mh[mh["origin_timestamp"] <= cutoff].copy()
        all_train = all_train[all_train["timestamp"] <= cutoff]
        test = mh[mh["origin_timestamp"] > cutoff].copy()

        # Val: last month of training
        val_start = cutoff - pd.DateOffset(months=1)
        val = all_train[all_train["origin_timestamp"] >= val_start].copy()
        train = all_train[all_train["origin_timestamp"] < val_start].copy()

        feat_cols = [f for f in CANDIDATE_FEATURES if f in train.columns]
        train = train.dropna(subset=feat_cols)
        val = val.dropna(subset=feat_cols)
        test = test.dropna(subset=feat_cols)

        logger.info(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    with log_step("Train baseline"):
        bl = fit_baseline(train)
        bl_preds = predict_baseline(bl, test)

    with log_step("Train LightGBM"):
        lgb_info = fit_lightgbm(train, val)
        lgb_preds = predict_lightgbm(lgb_info, test)

    # Save predictions
    actual = test[TARGET].values
    mae_bl = np.abs(actual - bl_preds).mean()
    mae_lgb = np.abs(actual - lgb_preds).mean()
    skill = 1 - mae_lgb / mae_bl
    logger.info(f"  Baseline MAE: {mae_bl:.0f} | LightGBM MAE: {mae_lgb:.0f} | Skill: {skill:.4f}")

    test_preds = test[["origin_timestamp", "timestamp", "forecast_horizon", TARGET]].copy()
    test_preds["pred_baseline"] = bl_preds
    test_preds["pred_lightgbm"] = lgb_preds
    test_preds.to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)

    # Save models
    with open(MODEL_DIR / "baseline_ridge.pkl", "wb") as f:
        pickle.dump(bl, f)
    lgb_info["model"].save_model(str(MODEL_DIR / "lightgbm_model.txt"))
    with open(MODEL_DIR / "lightgbm_features.json", "w") as f:
        json.dump(lgb_info["features"], f)

    summary = {
        "baseline": {"mae_test": float(mae_bl)},
        "lightgbm": {"mae_test": float(mae_lgb)},
        "skill_score": float(skill),
        "forecast_horizon_steps": FORECAST_HORIZON_STEPS,
    }
    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
