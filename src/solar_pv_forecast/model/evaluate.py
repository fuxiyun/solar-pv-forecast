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

from solar_pv_forecast.config import (
    FORECAST_STEP_MINUTES,
    OUTPUT_DIR,
    PROCESSED_DIR,
)
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
