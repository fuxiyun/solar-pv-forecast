"""Monthly walk-forward evaluation with periodic retraining.

Simulates how the system would operate in production: each month,
retrain on all available history, then forecast the next month.

12 evaluation rounds for 2025:
  Round 1:  train Jan-Dec 2024,         test Jan 2025
  Round 2:  train Jan 2024 - Jan 2025,  test Feb 2025
  ...
  Round 12: train Jan 2024 - Nov 2025,  test Dec 2025

Run:  python -m solar_pv_forecast.model.walk_forward
"""

import json
import pickle
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from solar_pv_forecast.config import (
    EARLY_STOP_VAL_MONTHS,
    FORECAST_HORIZON_STEPS,
    LIGHTGBM_PARAMS,
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    WALK_FORWARD_N_ROUNDS,
    WALK_FORWARD_TEST_YEAR,
)
from solar_pv_forecast.model.features import engineer_features
from solar_pv_forecast.model.train import (
    CANDIDATE_FEATURES,
    TARGET,
    build_multihorizon_data,
    fit_baseline,
    fit_lightgbm,
    predict_baseline,
    predict_lightgbm,
)
from solar_pv_forecast.model.tune import run_optuna
from solar_pv_forecast.utils import log_step, setup_logger


def _month_start(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year, month, 1)


def _month_end(year: int, month: int) -> pd.Timestamp:
    """Last moment of the month (inclusive for <=)."""
    if month == 12:
        return pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(seconds=1)
    return pd.Timestamp(year, month + 1, 1) - pd.Timedelta(seconds=1)


def walk_forward_train_predict(
    mh: pd.DataFrame,
    lgbm_params: dict | None = None,
) -> pd.DataFrame:
    """Run 12 monthly walk-forward rounds.

    For each test month in 2025:
      1. Slice train data: everything before test month start
      2. Split early-stop validation: last EARLY_STOP_VAL_MONTHS of train
      3. Fit baseline + LightGBM with fixed HPs, early stopping on val
      4. Predict on test month
      5. Collect predictions

    Also produces a "frozen" baseline: Round 1 model applied to all months,
    quantifying the benefit of monthly retraining.

    Returns concatenated predictions for all 12 test months.
    """
    params = lgbm_params or LIGHTGBM_PARAMS
    feat_cols = [f for f in CANDIDATE_FEATURES if f in mh.columns]
    test_year = WALK_FORWARD_TEST_YEAR

    all_preds = []
    frozen_bl_info = None
    frozen_lgb_info = None

    for round_idx in range(1, WALK_FORWARD_N_ROUNDS + 1):
        test_month = round_idx
        test_start = _month_start(test_year, test_month)
        test_end = _month_end(test_year, test_month)
        train_cutoff = test_start - pd.Timedelta(seconds=1)

        logger.info(
            f"  Round {round_idx:2d}/12: "
            f"train → {train_cutoff.date()}, "
            f"test {test_start.date()} – {test_end.date()}"
        )

        # ── Slice train / val / test ───────────────────────────
        train_all = mh[mh["origin_timestamp"] <= train_cutoff].copy()
        # Ensure no target timestamps leak into future
        train_all = train_all[train_all["timestamp"] <= train_cutoff]

        test_df = mh[
            (mh["origin_timestamp"] >= test_start)
            & (mh["origin_timestamp"] <= test_end)
        ].copy()

        # Early-stop validation: last N months of the train window
        val_start = train_cutoff - pd.DateOffset(months=EARLY_STOP_VAL_MONTHS)
        val = train_all[train_all["origin_timestamp"] >= val_start].copy()
        train = train_all[train_all["origin_timestamp"] < val_start].copy()

        # Drop NaN rows
        train = train.dropna(subset=feat_cols)
        val = val.dropna(subset=feat_cols)
        test_df = test_df.dropna(subset=feat_cols)

        if len(train) == 0 or len(val) == 0 or len(test_df) == 0:
            logger.warning(f"    Skipping round {round_idx}: insufficient data")
            continue

        logger.info(
            f"    train: {len(train):,}  val: {len(val):,}  test: {len(test_df):,}"
        )

        # ── Fit models ─────────────────────────────────────────
        bl_info = fit_baseline(train)
        lgb_info = fit_lightgbm(train, val, lgbm_params=params)

        # Save Round 1 model as the "frozen" reference
        if round_idx == 1:
            frozen_bl_info = bl_info
            frozen_lgb_info = lgb_info

        # ── Predict ────────────────────────────────────────────
        pred_bl = predict_baseline(bl_info, test_df)
        pred_lgb = predict_lightgbm(lgb_info, test_df)
        pred_frozen_bl = predict_baseline(frozen_bl_info, test_df)
        pred_frozen_lgb = predict_lightgbm(frozen_lgb_info, test_df)

        chunk = test_df[[
            "origin_timestamp", "timestamp", "forecast_horizon", TARGET,
        ]].copy()
        chunk["pred_baseline"] = pred_bl
        chunk["pred_lightgbm"] = pred_lgb
        chunk["pred_frozen_baseline"] = pred_frozen_bl
        chunk["pred_frozen_lightgbm"] = pred_frozen_lgb
        chunk["train_end_date"] = train_cutoff.date()
        chunk["round"] = round_idx

        all_preds.append(chunk)

        # Log round metrics
        actual = test_df[TARGET].values
        daytime = actual > 0
        if daytime.any():
            denom = actual[daytime].mean()
            nmae_lgb = np.abs(actual - pred_lgb).mean() / denom
            nmae_bl = np.abs(actual - pred_bl).mean() / denom
            logger.info(
                f"    nMAE — baseline: {nmae_bl:.4f}, "
                f"LightGBM: {nmae_lgb:.4f}"
            )

    result = pd.concat(all_preds, ignore_index=True)
    logger.info(f"  Walk-forward complete: {len(result):,} total predictions")
    return result


def main():
    setup_logger()

    with log_step("Load and engineer features"):
        df = pd.read_parquet(PROCESSED_DIR / "modelling_table.parquet")
        df = engineer_features(df)
        df.to_parquet(PROCESSED_DIR / "features_table.parquet", index=False)

    with log_step("Build multi-horizon data"):
        mh = build_multihorizon_data(df)

    with log_step("Optuna HP tuning on 2024"):
        tuned_params = run_optuna(mh)
        # Save tuned params
        with open(MODEL_DIR / "tuned_lgbm_params.json", "w") as f:
            json.dump(tuned_params, f, indent=2)
        logger.info(f"  Tuned params saved → {MODEL_DIR / 'tuned_lgbm_params.json'}")

    with log_step("Walk-forward evaluation (12 rounds)"):
        preds = walk_forward_train_predict(mh, lgbm_params=tuned_params)

    with log_step("Save predictions"):
        preds.to_parquet(
            PROCESSED_DIR / "test_predictions.parquet", index=False
        )
        logger.info(
            f"  Saved {len(preds):,} predictions → "
            f"{PROCESSED_DIR / 'test_predictions.parquet'}"
        )

    # ── Print summary table ────────────────────────────────────
    logger.info("")
    logger.info("=" * 72)
    logger.info("  WALK-FORWARD SUMMARY")
    logger.info("=" * 72)
    logger.info(
        f"  {'Round':>5} {'Test Month':>12} {'Train End':>12} "
        f"{'nMAE-BL':>9} {'nMAE-LGB':>9} {'nMAE-Frozen':>12}"
    )
    logger.info("-" * 72)

    actual = preds[TARGET].values
    overall_daytime = actual[actual > 0]

    for rnd, grp in preds.groupby("round"):
        a = grp[TARGET].values
        daytime = a > 0
        if not daytime.any():
            continue
        denom = a[daytime].mean()
        nmae_bl = np.abs(a - grp["pred_baseline"].values).mean() / denom
        nmae_lgb = np.abs(a - grp["pred_lightgbm"].values).mean() / denom
        nmae_frozen = np.abs(a - grp["pred_frozen_lightgbm"].values).mean() / denom
        test_month = grp["timestamp"].iloc[0].strftime("%Y-%m")
        train_end = str(grp["train_end_date"].iloc[0])
        logger.info(
            f"  {rnd:5d} {test_month:>12} {train_end:>12} "
            f"{nmae_bl:9.4f} {nmae_lgb:9.4f} {nmae_frozen:12.4f}"
        )

    # Overall
    denom = overall_daytime.mean() if len(overall_daytime) > 0 else 1.0
    nmae_bl_all = np.abs(actual - preds["pred_baseline"].values).mean() / denom
    nmae_lgb_all = np.abs(actual - preds["pred_lightgbm"].values).mean() / denom
    nmae_frozen_all = np.abs(actual - preds["pred_frozen_lightgbm"].values).mean() / denom
    skill_vs_bl = 1 - nmae_lgb_all / nmae_bl_all
    retrain_benefit = 1 - nmae_lgb_all / nmae_frozen_all

    logger.info("-" * 72)
    logger.info(f"  Overall nMAE (baseline):         {nmae_bl_all:.4f}")
    logger.info(f"  Overall nMAE (LightGBM):         {nmae_lgb_all:.4f}")
    logger.info(f"  Overall nMAE (frozen LightGBM):  {nmae_frozen_all:.4f}")
    logger.info(f"  Skill score vs baseline:         {skill_vs_bl:.4f}")
    logger.info(f"  Retrain benefit vs frozen:       {retrain_benefit:+.4f}")
    logger.info("=" * 72)

    # Save summary JSON
    summary = {
        "overall_nmae_baseline": float(nmae_bl_all),
        "overall_nmae_lightgbm": float(nmae_lgb_all),
        "overall_nmae_frozen_lightgbm": float(nmae_frozen_all),
        "skill_score_vs_baseline": float(skill_vs_bl),
        "retrain_benefit_vs_frozen": float(retrain_benefit),
        "n_rounds": WALK_FORWARD_N_ROUNDS,
        "forecast_horizon_steps": FORECAST_HORIZON_STEPS,
        "tuned_lgbm_params": tuned_params,
    }
    with open(MODEL_DIR / "walk_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
