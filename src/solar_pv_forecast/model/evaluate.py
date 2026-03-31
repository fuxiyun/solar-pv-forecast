"""Evaluate walk-forward predictions and produce final outputs.

Reports metrics stratified by:
  - Walk-forward round / test month (12 values)
  - Forecast horizon (15 min to 4 h)
  - Time of day
  - Retrained vs frozen model comparison

Run:  python -m solar_pv_forecast.model.evaluate
"""

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance

from solar_pv_forecast.config import (
    FORECAST_STEP_MINUTES,
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    TRAIN_END_DATE,
)
from solar_pv_forecast.model.train import (
    CANDIDATE_FEATURES,
    TARGET,
    build_multihorizon_data,
)
from solar_pv_forecast.model.features import engineer_features
from solar_pv_forecast.utils import log_step, setup_logger


# ── Metrics ─────────────────────────────────────────────────────
def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, nMAE (daytime-only denominator)."""
    daytime = actual > 0
    mae = float(np.abs(actual - predicted).mean())
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mean_daytime = float(actual[daytime].mean()) if daytime.any() else np.nan
    nmae = mae / mean_daytime if mean_daytime > 0 else np.nan

    return {
        "mae_mw": mae,
        "rmse_mw": rmse,
        "nmae": float(nmae),
        "mean_actual_daytime_mw": mean_daytime,
        "n_samples": int(len(actual)),
    }


def _metrics_by_group(df: pd.DataFrame, group_col: str, model_cols: list[str]) -> pd.DataFrame:
    """Compute metrics grouped by a column for multiple model columns."""
    records = []
    for key, grp in df.groupby(group_col):
        actual = grp["actual_solar_mw"].values
        for col in model_cols:
            m = compute_metrics(actual, grp[col].values)
            m[group_col] = key
            m["model"] = col.replace("pred_", "")
            records.append(m)
    return pd.DataFrame(records)


# ── Per-round monthly metrics ──────────────────────────────────
def compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-test-month metrics (one value per round)."""
    model_cols = ["pred_baseline", "pred_lightgbm", "pred_frozen_lightgbm"]
    model_cols = [c for c in model_cols if c in df.columns]
    return _metrics_by_group(df, "round", model_cols)


# ── Horizon metrics ────────────────────────────────────────────
def compute_horizon_metrics(df: pd.DataFrame) -> pd.DataFrame:
    model_cols = ["pred_baseline", "pred_lightgbm"]
    model_cols = [c for c in model_cols if c in df.columns]
    result = _metrics_by_group(df, "forecast_horizon", model_cols)
    result["horizon_minutes"] = result["forecast_horizon"] * FORECAST_STEP_MINUTES
    return result


# ── Time-of-day metrics ───────────────────────────────────────
def compute_tod_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute nMAE by hour of day (aggregated across all months)."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    model_cols = ["pred_baseline", "pred_lightgbm"]
    model_cols = [c for c in model_cols if c in df.columns]
    return _metrics_by_group(df, "hour", model_cols)


# ── Plots ──────────────────────────────────────────────────────
def plot_monthly_nmae_trend(monthly: pd.DataFrame, out_dir):
    """Monthly nMAE trend: retrained vs frozen, showing improvement with data growth."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    months = sorted(monthly["round"].unique())

    for model, color, label in [
        ("baseline", "#3B8BD4", "Baseline (ridge)"),
        ("lightgbm", "#D85A30", "LightGBM (retrained)"),
        ("frozen_lightgbm", "#888888", "LightGBM (frozen, Round 1)"),
    ]:
        sub = monthly[monthly["model"] == model].sort_values("round")
        if sub.empty:
            continue
        ax.plot(
            sub["round"], sub["nmae"],
            marker="o", markersize=5, color=color, label=label, linewidth=1.5,
        )

    ax.set_xlabel("Walk-forward round (test month)")
    ax.set_ylabel("nMAE")
    ax.set_title("Monthly nMAE: retrained vs frozen model")
    ax.set_xticks(months)
    ax.set_xticklabels([f"M{m}" for m in months])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "monthly_nmae_walkforward.png", dpi=150)
    plt.close()
    logger.info(f"  Saved monthly nMAE trend → {out_dir / 'monthly_nmae_walkforward.png'}")


def plot_horizon_metrics(horizon_df: pd.DataFrame, out_dir):
    """MAE and nMAE as a function of forecast horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for model, color, label in [
        ("baseline", "#3B8BD4", "Baseline (ridge)"),
        ("lightgbm", "#D85A30", "LightGBM"),
    ]:
        sub = horizon_df[horizon_df["model"] == model].sort_values("horizon_minutes")
        if sub.empty:
            continue
        axes[0].plot(
            sub["horizon_minutes"], sub["mae_mw"],
            marker="o", markersize=4, color=color, label=label,
        )
        axes[1].plot(
            sub["horizon_minutes"], sub["nmae"],
            marker="o", markersize=4, color=color, label=label,
        )

    axes[0].set_xlabel("Forecast horizon (min)")
    axes[0].set_ylabel("MAE (MW)")
    axes[0].set_title("MAE by forecast horizon")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Forecast horizon (min)")
    axes[1].set_ylabel("nMAE")
    axes[1].set_title("Normalised MAE by forecast horizon")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "horizon_metrics.png", dpi=150)
    plt.close()
    logger.info(f"  Saved horizon metrics plot → {out_dir / 'horizon_metrics.png'}")


def plot_tod_metrics(tod: pd.DataFrame, out_dir):
    """nMAE by hour of day."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for model, color, label in [
        ("baseline", "#3B8BD4", "Baseline"),
        ("lightgbm", "#D85A30", "LightGBM"),
    ]:
        sub = tod[tod["model"] == model].sort_values("hour")
        if sub.empty:
            continue
        daytime = sub[sub["nmae"].notna() & (sub["nmae"] < 10)]
        ax.plot(
            daytime["hour"], daytime["nmae"],
            marker="o", markersize=4, color=color, label=label,
        )

    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("nMAE")
    ax.set_title("nMAE by time of day (all months, all horizons)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    fig.savefig(out_dir / "tod_nmae.png", dpi=150)
    plt.close()
    logger.info(f"  Saved time-of-day nMAE → {out_dir / 'tod_nmae.png'}")


def plot_predictions_sample(df: pd.DataFrame, out_dir):
    """Plot a sample day showing rolling 4-hour forecasts vs actuals."""
    mid_date = df["timestamp"].iloc[len(df) // 2].normalize()
    day_df = df[
        (df["timestamp"] >= mid_date)
        & (df["timestamp"] < mid_date + pd.Timedelta(days=1))
    ]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    h1 = day_df[day_df["forecast_horizon"] == 1].sort_values("timestamp")
    ax.plot(
        h1["timestamp"], h1["actual_solar_mw"],
        color="#333", linewidth=1.5, label="Actual", alpha=0.9,
    )

    origins = sorted(day_df["origin_timestamp"].unique())
    colours = plt.cm.tab10(np.linspace(0, 1, min(len(origins), 10)))
    for idx, origin in enumerate(origins[::16]):
        window = day_df[day_df["origin_timestamp"] == origin].sort_values("timestamp")
        if len(window) == 0:
            continue
        c = colours[idx % len(colours)]
        label = f"Fcst {pd.Timestamp(origin).strftime('%H:%M')}" if idx < 6 else None
        ax.plot(
            window["timestamp"], window["pred_lightgbm"],
            color=c, linewidth=1, alpha=0.7, linestyle="--", label=label,
        )

    ax.set_ylabel("Solar generation (MW)")
    ax.set_title(f"Intraday rolling forecasts — {mid_date.strftime('%Y-%m-%d')}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "predictions_sample_day.png", dpi=150)
    plt.close()
    logger.info(f"  Saved sample day plot → {out_dir / 'predictions_sample_day.png'}")


# ── Feature importance ────────────────────────────────────────
def _load_model_and_features():
    """Load saved LightGBM model and its feature list.

    Uses the model's embedded feature names (authoritative) rather than
    the separately saved JSON, which may be out of sync with the
    multi-horizon column naming (target_ prefix).
    """
    import lightgbm as lgb

    model = lgb.Booster(model_file=str(MODEL_DIR / "lightgbm_model.txt"))
    features = model.feature_name()
    return model, features


def _build_validation_set(features: list[str]) -> pd.DataFrame:
    """Reconstruct the validation set used during single-split training.

    Renames multi-horizon columns to match the model's feature names
    when a target_ prefix was added during build_multihorizon_data but
    the model was trained with unprefixed names (or vice-versa).
    """
    df = pd.read_parquet(PROCESSED_DIR / "modelling_table.parquet")
    df = engineer_features(df)
    mh = build_multihorizon_data(df)

    cutoff = pd.Timestamp(TRAIN_END_DATE)
    all_train = mh[mh["origin_timestamp"] <= cutoff].copy()
    all_train = all_train[all_train["timestamp"] <= cutoff]

    val_start = cutoff - pd.DateOffset(months=1)
    val = all_train[all_train["origin_timestamp"] >= val_start].copy()

    # Align column names: if the model expects "clearsky_ghi" but the
    # DataFrame has "target_clearsky_ghi", rename to match.
    missing = [f for f in features if f not in val.columns]
    rename_map = {}
    for feat in missing:
        prefixed = f"target_{feat}"
        if prefixed in val.columns:
            rename_map[prefixed] = feat
    if rename_map:
        val = val.rename(columns=rename_map)
        logger.info(f"  Renamed {len(rename_map)} columns to match model feature names")

    val = val.dropna(subset=[f for f in features if f in val.columns])
    return val


def analyse_native_importance(model, features: list[str]) -> list[dict]:
    """Extract split-based feature importance from LightGBM model.

    Returns list of {feature, importance} dicts sorted descending.
    """
    raw = model.feature_importance(importance_type="split")
    ranked = sorted(
        zip(features, raw.tolist()), key=lambda x: x[1], reverse=True
    )
    results = [{"feature": f, "importance": v} for f, v in ranked]

    logger.info("  Native (split-based) feature importance — top 10:")
    for i, entry in enumerate(results[:10], 1):
        logger.info(f"    {i:2d}. {entry['feature']:35s} {entry['importance']:>8d}")

    out_path = OUTPUT_DIR / "feature_importance_native.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved → {out_path}")
    return results


class _BoosterWrapper(BaseEstimator, RegressorMixin):
    """Wrap lgb.Booster so sklearn's permutation_importance accepts it."""

    def __init__(self, booster=None):
        self.booster = booster

    def fit(self, X, y):
        return self  # already trained

    def predict(self, X):
        return self.booster.predict(X)


def analyse_permutation_importance(
    model, features: list[str], val: pd.DataFrame,
) -> list[dict]:
    """Compute permutation importance on the validation set.

    Uses neg_mean_absolute_error with 10 repeats.
    Returns list of {feature, mean, std, ci95_low, ci95_high} dicts.
    """
    X_val = val[features]
    y_val = val[TARGET]

    logger.info(
        f"  Running permutation importance on {len(val):,} validation rows "
        f"(n_repeats=10)…"
    )
    wrapper = _BoosterWrapper(model)
    perm_result = permutation_importance(
        wrapper, X_val, y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    entries = []
    for i, feat in enumerate(features):
        mean_imp = float(perm_result["importances_mean"][i])
        std_imp = float(perm_result["importances_std"][i])
        entries.append({
            "feature": feat,
            "mean": mean_imp,
            "std": std_imp,
            "ci95_low": float(mean_imp - 1.96 * std_imp),
            "ci95_high": float(mean_imp + 1.96 * std_imp),
        })
    entries.sort(key=lambda x: x["mean"], reverse=True)

    logger.info("  Permutation importance (validation set) — top 10:")
    for i, e in enumerate(entries[:10], 1):
        ci_note = " *" if e["ci95_low"] <= 0 else ""
        logger.info(
            f"    {i:2d}. {e['feature']:35s} "
            f"mean={e['mean']:>10.1f}  ±{e['std']:.1f}{ci_note}"
        )

    # Flag features with 95% CI including zero
    zero_ci = [e["feature"] for e in entries if e["ci95_low"] <= 0]
    if zero_ci:
        logger.info(
            f"  Features with 95% CI including 0 (candidates for removal): "
            f"{', '.join(zero_ci)}"
        )

    out_path = OUTPUT_DIR / "feature_importance_permutation.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info(f"  Saved → {out_path}")
    return entries


def plot_importance_comparison(
    native: list[dict], perm: list[dict], out_dir,
) -> None:
    """Side-by-side horizontal bar chart of native vs permutation importance."""
    n_features = min(15, len(native))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Native
    feats_n = [e["feature"] for e in native[:n_features]]
    vals_n = [e["importance"] for e in native[:n_features]]
    ax1.barh(range(len(feats_n)), vals_n, color="#3B8BD4", alpha=0.8)
    ax1.set_yticks(range(len(feats_n)))
    ax1.set_yticklabels(feats_n, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Split count")
    ax1.set_title("Native (split-based)")

    # Permutation
    feats_p = [e["feature"] for e in perm[:n_features]]
    vals_p = [e["mean"] for e in perm[:n_features]]
    stds_p = [e["std"] for e in perm[:n_features]]
    ax2.barh(range(len(feats_p)), vals_p, xerr=stds_p,
             color="#D85A30", alpha=0.8, capsize=3)
    ax2.set_yticks(range(len(feats_p)))
    ax2.set_yticklabels(feats_p, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean MAE increase")
    ax2.set_title("Permutation (validation set)")

    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance_comparison.png", dpi=150)
    plt.close()
    logger.info(f"  Saved comparison plot → {out_dir / 'feature_importance_comparison.png'}")


# ── Main ───────────────────────────────────────────────────────
def main():
    setup_logger()

    with log_step("Load test predictions"):
        df = pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["origin_timestamp"] = pd.to_datetime(df["origin_timestamp"])
        has_frozen = "pred_frozen_lightgbm" in df.columns
        logger.info(
            f"  Loaded {len(df):,} predictions, "
            f"{df['round'].nunique() if 'round' in df.columns else 1} rounds, "
            f"{df['forecast_horizon'].nunique()} horizons"
        )

    with log_step("Compute overall metrics"):
        actual = df["actual_solar_mw"].values
        results = {}
        model_cols = ["pred_baseline", "pred_lightgbm"]
        if has_frozen:
            model_cols.append("pred_frozen_lightgbm")

        for col in model_cols:
            name = col.replace("pred_", "")
            m = compute_metrics(actual, df[col].values)
            results[name] = m
            logger.info(
                f"  {name:22s} — MAE: {m['mae_mw']:.0f} MW | "
                f"nMAE: {m['nmae']:.4f} | RMSE: {m['rmse_mw']:.0f} MW"
            )

        skill = 1 - results["lightgbm"]["mae_mw"] / results["baseline"]["mae_mw"]
        results["skill_score_vs_baseline"] = float(skill)
        logger.info(f"  Skill score vs baseline: {skill:.4f}")

        if has_frozen:
            retrain_benefit = (
                1 - results["lightgbm"]["mae_mw"]
                / results["frozen_lightgbm"]["mae_mw"]
            )
            results["retrain_benefit_vs_frozen"] = float(retrain_benefit)
            logger.info(f"  Retrain benefit vs frozen: {retrain_benefit:+.4f}")

    with log_step("Compute monthly metrics (per round)"):
        monthly = compute_monthly_metrics(df)
        monthly.to_csv(OUTPUT_DIR / "monthly_metrics.csv", index=False)

        # Log monthly nMAE table
        lgb_monthly = monthly[monthly["model"] == "lightgbm"].sort_values("round")
        nmae_values = lgb_monthly["nmae"].values
        logger.info(f"  Monthly nMAE (LightGBM): mean={nmae_values.mean():.4f} ± {nmae_values.std():.4f}")
        for _, row in lgb_monthly.iterrows():
            logger.info(f"    Round {int(row['round']):2d}: nMAE={row['nmae']:.4f}")

    with log_step("Compute horizon metrics"):
        horizon = compute_horizon_metrics(df)
        horizon.to_csv(OUTPUT_DIR / "horizon_metrics.csv", index=False)

        lgb_h = horizon[horizon["model"] == "lightgbm"]
        for _, row in lgb_h.iterrows():
            logger.info(
                f"    h={int(row['forecast_horizon']):2d} "
                f"({int(row['horizon_minutes']):3d} min) — "
                f"MAE: {row['mae_mw']:.0f} MW | nMAE: {row['nmae']:.4f}"
            )

    with log_step("Compute time-of-day metrics"):
        tod = compute_tod_metrics(df)
        tod.to_csv(OUTPUT_DIR / "tod_metrics.csv", index=False)

    with log_step("Feature importance analysis"):
        model, features = _load_model_and_features()
        native = analyse_native_importance(model, features)
        val = _build_validation_set(features)
        perm = analyse_permutation_importance(model, features, val)
        plot_importance_comparison(native, perm, OUTPUT_DIR)

    with log_step("Generate plots"):
        plot_monthly_nmae_trend(monthly, OUTPUT_DIR)
        plot_horizon_metrics(horizon, OUTPUT_DIR)
        plot_tod_metrics(tod, OUTPUT_DIR)
        plot_predictions_sample(df, OUTPUT_DIR)

    with log_step("Produce final predictions.parquet"):
        out_cols = [
            "origin_timestamp", "timestamp", "forecast_horizon",
            "actual_solar_mw", "pred_lightgbm", "pred_baseline",
            "train_end_date", "round",
        ]
        if has_frozen:
            out_cols.append("pred_frozen_lightgbm")
        output = df[out_cols].rename(columns={
            "pred_lightgbm": "predicted_solar_mw",
            "pred_baseline": "pred_baseline_mw",
        })
        out_path = OUTPUT_DIR / "predictions.parquet"
        output.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"  Final output: {len(output):,} rows → {out_path}")

    # Save evaluation summary
    with open(OUTPUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Evaluation summary → {OUTPUT_DIR / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
