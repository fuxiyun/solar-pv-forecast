"""Optuna hyperparameter tuning for LightGBM on 2024 data.

Uses rolling-origin cross-validation within 2024 to select HPs.
The best HPs are saved and used by the walk-forward loop for all
12 monthly rounds (only model weights are retrained, not HPs).

Rolling-origin CV with 3 folds on 2024:
  Fold 1: train Jan-Jun,  val Jul-Aug
  Fold 2: train Jan-Aug,  val Sep-Oct
  Fold 3: train Jan-Oct,  val Nov-Dec

Run:  python -m solar_pv_forecast.model.tune
"""

import json

import numpy as np
import pandas as pd
from loguru import logger

from solar_pv_forecast.config import (
    EARLY_STOPPING_ROUNDS,
    LIGHTGBM_PARAMS,
    MODEL_DIR,
    OPTUNA_CV_FOLDS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SEC,
    PROCESSED_DIR,
    TRAIN_END_DATE,
)
from solar_pv_forecast.model.features import engineer_features
from solar_pv_forecast.model.train import (
    CANDIDATE_FEATURES,
    TARGET,
    build_multihorizon_data,
)
from solar_pv_forecast.utils import log_step, setup_logger


def _build_cv_folds(
    mh: pd.DataFrame, n_folds: int = OPTUNA_CV_FOLDS
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Build rolling-origin CV folds within 2024 data.

    Each fold expands the training window by 2 months and validates
    on the next 2 months.
    """
    cutoff = pd.Timestamp(TRAIN_END_DATE)
    data_2024 = mh[mh["origin_timestamp"] <= cutoff].copy()
    data_2024 = data_2024[data_2024["timestamp"] <= cutoff]

    feat_cols = [f for f in CANDIDATE_FEATURES if f in data_2024.columns]
    data_2024 = data_2024.dropna(subset=feat_cols)

    # Determine fold boundaries: split 12 months into n_folds+1 chunks
    # of 2 months each for validation, with expanding train
    months_per_val = 2
    # Fold i: train months 1..(6 + 2*i), val months (7+2*i)..(8+2*i)
    # For 3 folds: train 1-6/val 7-8, train 1-8/val 9-10, train 1-10/val 11-12
    folds = []
    base_train_months = 12 - months_per_val * (n_folds + 1)
    if base_train_months < 2:
        base_train_months = 4  # minimum 4 months train

    for i in range(n_folds):
        train_end_month = base_train_months + months_per_val * (i + 1)
        val_start_month = train_end_month + 1
        val_end_month = val_start_month + months_per_val - 1

        if val_end_month > 12:
            break

        train_end_ts = pd.Timestamp(2024, train_end_month, 1) + pd.offsets.MonthEnd(0)
        val_start_ts = pd.Timestamp(2024, val_start_month, 1)
        val_end_ts = pd.Timestamp(2024, val_end_month, 1) + pd.offsets.MonthEnd(0)

        train = data_2024[data_2024["origin_timestamp"] <= train_end_ts].copy()
        val = data_2024[
            (data_2024["origin_timestamp"] >= val_start_ts)
            & (data_2024["origin_timestamp"] <= val_end_ts)
        ].copy()

        if len(train) > 0 and len(val) > 0:
            folds.append((train, val))
            logger.info(
                f"    Fold {len(folds)}: train → {train_end_ts.date()} "
                f"({len(train):,}), val {val_start_ts.date()} → "
                f"{val_end_ts.date()} ({len(val):,})"
            )

    return folds


def _evaluate_params(
    params: dict, folds: list[tuple[pd.DataFrame, pd.DataFrame]]
) -> float:
    """Evaluate a set of LightGBM params via rolling-origin CV.

    Returns mean validation MAE across folds.
    """
    import lightgbm as lgb

    p = params.copy()
    n_estimators = p.pop("n_estimators", 500)
    p.pop("random_state", None)

    feats = [f for f in CANDIDATE_FEATURES if f in folds[0][0].columns]
    fold_maes = []

    for train, val in folds:
        X_train, y_train = train[feats], train[TARGET]
        X_val, y_val = val[feats], val[TARGET]

        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(0),
        ]

        model = lgb.train(
            p,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        preds = np.clip(model.predict(X_val), 0, None)
        mae = float(np.abs(y_val.values - preds).mean())
        fold_maes.append(mae)

    return float(np.mean(fold_maes))


def run_optuna(
    mh: pd.DataFrame,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT_SEC,
) -> dict:
    """Run Optuna HP search on 2024 data with rolling-origin CV.

    Returns the best LightGBM params dict (same format as LIGHTGBM_PARAMS).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"  Building CV folds within 2024...")
    folds = _build_cv_folds(mh)
    if not folds:
        logger.warning("  No valid CV folds, using default HPs")
        return LIGHTGBM_PARAMS.copy()

    # Evaluate default params first for reference
    default_mae = _evaluate_params(LIGHTGBM_PARAMS, folds)
    logger.info(f"  Default HPs CV MAE: {default_mae:.1f} MW")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": 800,
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return _evaluate_params(params, folds)

    study = optuna.create_study(direction="minimize")

    # Enqueue default params as first trial
    study.enqueue_trial({
        "num_leaves": LIGHTGBM_PARAMS["num_leaves"],
        "learning_rate": LIGHTGBM_PARAMS["learning_rate"],
        "min_child_samples": LIGHTGBM_PARAMS["min_child_samples"],
        "subsample": LIGHTGBM_PARAMS["subsample"],
        "colsample_bytree": 1.0,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
    })

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    logger.info(
        f"  Optuna complete: {len(study.trials)} trials, "
        f"best MAE: {best.value:.1f} MW (vs default {default_mae:.1f})"
    )

    best_params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "num_leaves": best.params["num_leaves"],
        "learning_rate": best.params["learning_rate"],
        "n_estimators": 800,
        "min_child_samples": best.params["min_child_samples"],
        "subsample": best.params["subsample"],
        "subsample_freq": 1,
        "colsample_bytree": best.params["colsample_bytree"],
        "reg_alpha": best.params["reg_alpha"],
        "reg_lambda": best.params["reg_lambda"],
        "random_state": 42,
    }

    # Only use tuned params if they actually beat the default
    if best.value < default_mae:
        logger.info(f"  Tuned HPs beat default by {default_mae - best.value:.1f} MW")
        return best_params
    else:
        logger.info(f"  Default HPs are best, keeping them")
        return LIGHTGBM_PARAMS.copy()


def main():
    setup_logger()

    with log_step("Load and engineer features"):
        df = pd.read_parquet(PROCESSED_DIR / "modelling_table.parquet")
        df = engineer_features(df)

    with log_step("Build multi-horizon data"):
        mh = build_multihorizon_data(df)

    with log_step("Optuna HP tuning on 2024"):
        best_params = run_optuna(mh)

    # Save best params
    out_path = MODEL_DIR / "tuned_lgbm_params.json"
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"  Best params → {out_path}")
    logger.info(f"  {best_params}")


if __name__ == "__main__":
    main()
