"""Train baseline (ridge) and candidate (LightGBM) models.

Run:  python -m solar_pv_forecast.model.train
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
    LIGHTGBM_PARAMS,
    MODEL_DIR,
    PROCESSED_DIR,
    TRAIN_END_MONTH,
)
from solar_pv_forecast.model.features import engineer_features
from solar_pv_forecast.utils import log_step, setup_logger

# Features used by each model
BASELINE_FEATURES = ["proxy_solar_mw", "clearsky_index"]
CANDIDATE_FEATURES = [
    "proxy_solar_mw",
    "ghi_wm2_national",
    "temperature_2m_national",
    "wind_speed_10m_national",
    "clearsky_index",
    "clearsky_ghi",
    "solar_zenith",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month",
    "actual_lag_1d",
    "actual_lag_7d",
]
TARGET = "actual_solar_mw"


def split_train_test(df: pd.DataFrame):
    """Time-aware train/test split by month."""
    train = df[df["timestamp"].dt.month <= TRAIN_END_MONTH].copy()
    test = df[df["timestamp"].dt.month > TRAIN_END_MONTH].copy()

    # Drop rows with NaN in lag features (first days of train set)
    train = train.dropna(subset=CANDIDATE_FEATURES)
    test = test.dropna(subset=CANDIDATE_FEATURES)

    logger.info(
        f"  Train: {len(train):,} rows "
        f"({train['timestamp'].min().date()} → {train['timestamp'].max().date()})"
    )
    logger.info(
        f"  Test:  {len(test):,} rows "
        f"({test['timestamp'].min().date()} → {test['timestamp'].max().date()})"
    )
    return train, test


def train_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train ridge regression baseline on proxy + clear-sky index."""
    feats = [f for f in BASELINE_FEATURES if f in train.columns]

    X_train = train[feats].values
    y_train = train[TARGET].values
    X_test = test[feats].values
    y_test = test[TARGET].values

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    # Clip predictions to non-negative
    y_pred_test = np.clip(y_pred_test, 0, None)

    mae_train = np.abs(y_train - y_pred_train).mean()
    mae_test = np.abs(y_test - y_pred_test).mean()
    nmae_test = mae_test / y_test[y_test > 0].mean()

    logger.info(f"  Baseline — Train MAE: {mae_train:.0f} MW")
    logger.info(f"  Baseline — Test  MAE: {mae_test:.0f} MW | nMAE: {nmae_test:.4f}")

    # Save model
    with open(MODEL_DIR / "baseline_ridge.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": feats}, f)

    return {
        "model_name": "baseline_ridge",
        "mae_train": float(mae_train),
        "mae_test": float(mae_test),
        "nmae_test": float(nmae_test),
        "predictions": y_pred_test,
    }


def train_lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train LightGBM candidate model."""
    import lightgbm as lgb

    feats = [f for f in CANDIDATE_FEATURES if f in train.columns]

    X_train = train[feats]
    y_train = train[TARGET]
    X_test = test[feats]
    y_test = test[TARGET]

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    params = LIGHTGBM_PARAMS.copy()
    n_estimators = params.pop("n_estimators", 500)
    random_state = params.pop("random_state", 42)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    y_pred_train = np.clip(model.predict(X_train), 0, None)
    y_pred_test = np.clip(model.predict(X_test), 0, None)

    mae_train = np.abs(y_train.values - y_pred_train).mean()
    mae_test = np.abs(y_test.values - y_pred_test).mean()
    nmae_test = mae_test / y_test[y_test > 0].mean()

    logger.info(f"  LightGBM — Train MAE: {mae_train:.0f} MW")
    logger.info(f"  LightGBM — Test  MAE: {mae_test:.0f} MW | nMAE: {nmae_test:.4f}")

    # Feature importance
    importance = dict(zip(feats, model.feature_importance(importance_type="gain")))
    top_feats = sorted(importance.items(), key=lambda x: -x[1])[:5]
    logger.info(f"  Top features: {top_feats}")

    # Save model
    model.save_model(str(MODEL_DIR / "lightgbm_model.txt"))
    with open(MODEL_DIR / "lightgbm_features.json", "w") as f:
        json.dump(feats, f)

    return {
        "model_name": "lightgbm",
        "mae_train": float(mae_train),
        "mae_test": float(mae_test),
        "nmae_test": float(nmae_test),
        "predictions": y_pred_test,
        "feature_importance": importance,
    }


def main():
    setup_logger()

    with log_step("Load and engineer features"):
        df = pd.read_parquet(PROCESSED_DIR / "modelling_table.parquet")
        df = engineer_features(df)

        # Save feature-engineered table
        df.to_parquet(PROCESSED_DIR / "features_table.parquet", index=False)

    with log_step("Train/test split"):
        train, test = split_train_test(df)

    with log_step("Train baseline model"):
        baseline_results = train_baseline(train, test)

    with log_step("Train LightGBM model"):
        lgbm_results = train_lightgbm(train, test)

    # Compute skill score
    skill_score = 1 - lgbm_results["mae_test"] / baseline_results["mae_test"]
    logger.info(f"  Skill score (LightGBM vs baseline): {skill_score:.4f}")

    # Save summary
    summary = {
        "baseline": {
            k: v for k, v in baseline_results.items() if k != "predictions"
        },
        "lightgbm": {
            k: v
            for k, v in lgbm_results.items()
            if k not in ("predictions", "feature_importance")
        },
        "skill_score": float(skill_score),
    }
    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary → {MODEL_DIR / 'training_summary.json'}")

    # Save test predictions for evaluation
    test_preds = test[["timestamp", TARGET]].copy()
    test_preds["pred_baseline"] = baseline_results["predictions"]
    test_preds["pred_lightgbm"] = lgbm_results["predictions"]
    test_preds.to_parquet(
        PROCESSED_DIR / "test_predictions.parquet", index=False
    )


if __name__ == "__main__":
    main()
