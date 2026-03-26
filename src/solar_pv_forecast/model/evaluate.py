"""Evaluate models and produce final predictions.parquet.

Computes rolling-origin metrics, generates diagnostic plots,
and outputs the final prediction file.

Run:  python -m solar_pv_forecast.model.evaluate
"""

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from loguru import logger

from solar_pv_forecast.config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR
from solar_pv_forecast.utils import log_step, setup_logger


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, nMAE (daytime-only denominator)."""
    daytime = actual > 0
    mae = np.abs(actual - predicted).mean()
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    mean_actual_daytime = actual[daytime].mean() if daytime.any() else np.nan
    nmae = mae / mean_actual_daytime if mean_actual_daytime > 0 else np.nan

    return {
        "mae_mw": float(mae),
        "rmse_mw": float(rmse),
        "nmae": float(nmae),
        "mean_actual_daytime_mw": float(mean_actual_daytime),
        "n_samples": int(len(actual)),
    }


def compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-month rolling-origin metrics."""
    records = []
    for month, group in df.groupby(df["timestamp"].dt.month):
        actual = group["actual_solar_mw"].values
        for model_col in ["pred_baseline", "pred_lightgbm"]:
            pred = group[model_col].values
            m = compute_metrics(actual, pred)
            m["month"] = int(month)
            m["model"] = model_col.replace("pred_", "")
            records.append(m)

    return pd.DataFrame(records)


def plot_predictions_sample(df: pd.DataFrame, out_dir):
    """Plot a representative week of predictions vs actuals."""
    # Pick a week in the middle of the test set
    mid = df["timestamp"].iloc[len(df) // 2]
    start = mid - pd.Timedelta(days=3)
    end = mid + pd.Timedelta(days=4)
    week = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        week["timestamp"], week["actual_solar_mw"],
        color="#333", linewidth=1.2, label="Actual", alpha=0.9,
    )
    ax.plot(
        week["timestamp"], week["pred_baseline"],
        color="#3B8BD4", linewidth=1, label="Baseline (ridge)",
        linestyle="--", alpha=0.8,
    )
    ax.plot(
        week["timestamp"], week["pred_lightgbm"],
        color="#D85A30", linewidth=1, label="LightGBM", alpha=0.8,
    )
    ax.set_ylabel("Solar generation (MW)")
    ax.set_xlabel("")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_title("Sample week: predictions vs actual solar generation")
    plt.tight_layout()
    fig.savefig(out_dir / "predictions_sample_week.png", dpi=150)
    plt.close()
    logger.info(f"  Saved sample week plot → {out_dir / 'predictions_sample_week.png'}")


def plot_monthly_metrics(monthly: pd.DataFrame, out_dir):
    """Bar chart of nMAE per month for both models."""
    fig, ax = plt.subplots(figsize=(10, 4))
    months = sorted(monthly["month"].unique())
    x = np.arange(len(months))
    width = 0.35

    baseline = monthly[monthly["model"] == "baseline"].set_index("month")
    lgbm = monthly[monthly["model"] == "lightgbm"].set_index("month")

    ax.bar(
        x - width / 2,
        [baseline.loc[m, "nmae"] if m in baseline.index else 0 for m in months],
        width, label="Baseline", color="#3B8BD4", alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        [lgbm.loc[m, "nmae"] if m in lgbm.index else 0 for m in months],
        width, label="LightGBM", color="#D85A30", alpha=0.8,
    )

    ax.set_ylabel("nMAE")
    ax.set_xlabel("Month")
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{m}" for m in months])
    ax.legend()
    ax.set_title("Normalised MAE by month (test set)")
    plt.tight_layout()
    fig.savefig(out_dir / "monthly_nmae.png", dpi=150)
    plt.close()
    logger.info(f"  Saved monthly nMAE plot → {out_dir / 'monthly_nmae.png'}")


def main():
    setup_logger()

    with log_step("Load test predictions"):
        df = pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logger.info(f"  Loaded {len(df):,} test predictions")

    with log_step("Compute overall metrics"):
        actual = df["actual_solar_mw"].values
        results = {}
        for model_col in ["pred_baseline", "pred_lightgbm"]:
            name = model_col.replace("pred_", "")
            m = compute_metrics(actual, df[model_col].values)
            results[name] = m
            logger.info(
                f"  {name:12s} — MAE: {m['mae_mw']:.0f} MW | "
                f"nMAE: {m['nmae']:.4f} | RMSE: {m['rmse_mw']:.0f} MW"
            )

        skill = 1 - results["lightgbm"]["mae_mw"] / results["baseline"]["mae_mw"]
        results["skill_score"] = float(skill)
        logger.info(f"  Skill score: {skill:.4f}")

    with log_step("Compute monthly metrics"):
        monthly = compute_monthly_metrics(df)
        monthly.to_csv(OUTPUT_DIR / "monthly_metrics.csv", index=False)
        logger.info(f"  Monthly metrics → {OUTPUT_DIR / 'monthly_metrics.csv'}")

    with log_step("Generate plots"):
        plot_predictions_sample(df, OUTPUT_DIR)
        plot_monthly_metrics(monthly, OUTPUT_DIR)

    with log_step("Produce final predictions.parquet"):
        # Final output: timestamp + best model prediction
        output = df[["timestamp", "actual_solar_mw", "pred_lightgbm"]].rename(
            columns={"pred_lightgbm": "predicted_solar_mw"}
        )
        output["pred_baseline_mw"] = df["pred_baseline"]
        out_path = OUTPUT_DIR / "predictions.parquet"
        output.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"  Final output: {len(output):,} rows → {out_path}")

    # Save evaluation summary
    with open(OUTPUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Evaluation summary → {OUTPUT_DIR / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
