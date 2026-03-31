"""Diagnostic analysis: why is the model systematically underpredicting?

Runs 7 steps to pinpoint the source of bias before attempting any fix.
Uses the actual column names from this pipeline (pandas-based).

Run:  python src/analysis/diagnose_bias.py
"""

import numpy as np
import pandas as pd

from solar_pv_forecast.config import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TRAIN_END_DATE,
)

SEP = "=" * 70


def main():
    # ── Load data ─────────────────────────────────────────────────
    pred = pd.read_parquet(OUTPUT_DIR / "predictions.parquet")
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])

    features = pd.read_parquet(PROCESSED_DIR / "features_table.parquet")
    features["timestamp"] = pd.to_datetime(features["timestamp"])

    test_preds = pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")
    test_preds["timestamp"] = pd.to_datetime(test_preds["timestamp"])
    test_preds["origin_timestamp"] = pd.to_datetime(test_preds["origin_timestamp"])

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Quantify the bias precisely
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 1: Quantify the bias")
    print(SEP)

    residuals = pred["predicted_solar_mw"] - pred["actual_solar_mw"]
    mean_actual = pred["actual_solar_mw"].mean()

    print(f"Mean error (bias):       {residuals.mean():>10.1f} MW")
    print(f"Median error:            {residuals.median():>10.1f} MW")
    print(f"MAE:                     {residuals.abs().mean():>10.1f} MW")
    print(f"Bias as % of mean actual:{residuals.mean() / mean_actual * 100:>9.1f}%")
    print(f"Mean actual:             {mean_actual:>10.1f} MW")
    print(f"Mean predicted:          {pred['predicted_solar_mw'].mean():>10.1f} MW")

    # Add helper columns
    df = pred.copy()
    df["error"] = df["predicted_solar_mw"] - df["actual_solar_mw"]
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Bias by generation level (deciles)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 2: Bias by generation level")
    print(SEP)
    print("KEY QUESTION: is underprediction concentrated at high generation?")
    print("  If yes → model clips peaks (common with MAE objective)")
    print("  If uniform → level shift, possibly data/capacity issue\n")

    daytime = df[df["actual_solar_mw"] > 10].copy()
    daytime["gen_decile"] = pd.qcut(
        daytime["actual_solar_mw"], 10, labels=False, duplicates="drop"
    )

    bias_by_level = (
        daytime.groupby("gen_decile")
        .agg(
            mean_error=("error", "mean"),
            mean_actual=("actual_solar_mw", "mean"),
            mean_pred=("predicted_solar_mw", "mean"),
            n=("error", "size"),
        )
        .reset_index()
    )
    bias_by_level["bias_pct"] = (
        bias_by_level["mean_error"] / bias_by_level["mean_actual"] * 100
    )

    print(f"{'Decile':>7} {'MeanActual':>11} {'MeanPred':>10} {'MeanErr':>9} "
          f"{'Bias%':>7} {'N':>8}")
    print("-" * 60)
    for _, row in bias_by_level.iterrows():
        print(
            f"  q{int(row['gen_decile']):>4d} {row['mean_actual']:>11.0f} "
            f"{row['mean_pred']:>10.0f} {row['mean_error']:>9.0f} "
            f"{row['bias_pct']:>6.1f}% {int(row['n']):>8d}"
        )

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Bias by hour
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 3: Bias by hour of day (daytime only)")
    print(SEP)

    bias_by_hour = (
        daytime.groupby("hour")
        .agg(
            mean_error=("error", "mean"),
            mean_actual=("actual_solar_mw", "mean"),
            n=("error", "size"),
        )
        .reset_index()
    )
    bias_by_hour["bias_pct"] = (
        bias_by_hour["mean_error"] / bias_by_hour["mean_actual"] * 100
    )

    print(f"{'Hour':>5} {'MeanActual':>11} {'MeanErr':>9} {'Bias%':>7} {'N':>8}")
    print("-" * 45)
    for _, row in bias_by_hour.iterrows():
        print(
            f"  {int(row['hour']):>3d} {row['mean_actual']:>11.0f} "
            f"{row['mean_error']:>9.0f} {row['bias_pct']:>6.1f}% "
            f"{int(row['n']):>8d}"
        )

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Does the baseline have the same bias?
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 4: Bias comparison across models")
    print(SEP)
    print("If ALL models underpredict → data/capacity issue")
    print("If only candidate → model issue\n")

    cand_bias = (
        test_preds["pred_lightgbm"] - test_preds["actual_solar_mw"]
    ).mean()
    base_bias = (
        test_preds["pred_baseline"] - test_preds["actual_solar_mw"]
    ).mean()

    print(f"Baseline (Ridge) bias:   {base_bias:>10.1f} MW")
    print(f"LightGBM (retrained):    {cand_bias:>10.1f} MW")

    if "pred_frozen_lightgbm" in test_preds.columns:
        frozen_bias = (
            test_preds["pred_frozen_lightgbm"] - test_preds["actual_solar_mw"]
        ).mean()
        print(f"LightGBM (frozen):       {frozen_bias:>10.1f} MW")

    # Proxy bias from modelling table
    cutoff = pd.Timestamp(TRAIN_END_DATE)
    test_feat = features[features["timestamp"] > cutoff].copy()
    if "proxy_solar_mw" in test_feat.columns:
        proxy_bias = (
            test_feat["proxy_solar_mw"] - test_feat["actual_solar_mw"]
        ).mean()
        print(f"Synthetic proxy bias:    {proxy_bias:>10.1f} MW")

    # ══════════════════════════════════════════════════════════════
    # STEP 5: Capacity mismatch between train and test
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 5: Capacity and generation shift (train vs test)")
    print(SEP)

    # Check if PV capacity data has time variation
    cap_path = RAW_DIR / "pv_capacity_by_state.parquet"
    if cap_path.exists():
        cap_df = pd.read_parquet(cap_path)
        print(f"PV capacity file columns: {list(cap_df.columns)}")
        total_cap = cap_df["capacity_mwp"].sum()
        print(f"Total installed capacity: {total_cap:,.0f} MWp")
        print("(This is a STATIC snapshot — same for train and test)")
        print("If real capacity grew in 2025, the model won't know.")

    # Check actual generation levels by year-month
    feat_ym = features.copy()
    feat_ym["ym"] = feat_ym["timestamp"].dt.to_period("M")
    gen_by_month = (
        feat_ym.groupby("ym")
        .agg(
            mean_actual=("actual_solar_mw", "mean"),
            max_actual=("actual_solar_mw", "max"),
            p95_actual=("actual_solar_mw", lambda x: np.percentile(x, 95)),
        )
        .reset_index()
    )
    print(f"\n{'Month':>8} {'MeanActual':>11} {'MaxActual':>10} {'P95':>8}")
    print("-" * 42)
    for _, row in gen_by_month.iterrows():
        print(
            f"  {row['ym']} {row['mean_actual']:>11.0f} "
            f"{row['max_actual']:>10.0f} {row['p95_actual']:>8.0f}"
        )

    # ══════════════════════════════════════════════════════════════
    # STEP 6: Distribution shift between train and test
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 6: Target distribution shift (train vs test)")
    print(SEP)

    train_feat = features[features["timestamp"] <= cutoff]
    test_feat = features[features["timestamp"] > cutoff]

    print(f"Train ({train_feat['timestamp'].min().date()} to "
          f"{train_feat['timestamp'].max().date()}):")
    print(f"  mean actual:  {train_feat['actual_solar_mw'].mean():>10.1f} MW")
    print(f"  std actual:   {train_feat['actual_solar_mw'].std():>10.1f} MW")
    print(f"  max actual:   {train_feat['actual_solar_mw'].max():>10.1f} MW")
    print(f"  p95 actual:   {np.percentile(train_feat['actual_solar_mw'], 95):>10.1f} MW")

    print(f"Test ({test_feat['timestamp'].min().date()} to "
          f"{test_feat['timestamp'].max().date()}):")
    print(f"  mean actual:  {test_feat['actual_solar_mw'].mean():>10.1f} MW")
    print(f"  std actual:   {test_feat['actual_solar_mw'].std():>10.1f} MW")
    print(f"  max actual:   {test_feat['actual_solar_mw'].max():>10.1f} MW")
    print(f"  p95 actual:   {np.percentile(test_feat['actual_solar_mw'], 95):>10.1f} MW")

    shift_pct = (
        (test_feat["actual_solar_mw"].mean() - train_feat["actual_solar_mw"].mean())
        / train_feat["actual_solar_mw"].mean()
        * 100
    )
    print(f"\n  Distribution shift: {shift_pct:+.1f}% (test vs train mean)")
    if shift_pct > 3:
        print("  ⚠ Test generation is HIGHER than train — capacity likely grew.")
    elif shift_pct < -3:
        print("  ⚠ Test generation is LOWER than train — unusual.")
    else:
        print("  ✓ No significant shift.")

    # ══════════════════════════════════════════════════════════════
    # STEP 7: Prediction distribution — peak clipping?
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("STEP 7: Prediction distribution (peak clipping?)")
    print(SEP)

    test_pred_vals = pred["predicted_solar_mw"]
    test_act_vals = pred["actual_solar_mw"]

    print(f"{'':>20} {'Predicted':>12} {'Actual':>12} {'Gap':>10}")
    print("-" * 56)
    for label, fn in [
        ("Mean", lambda s: s.mean()),
        ("Std", lambda s: s.std()),
        ("P75", lambda s: np.percentile(s, 75)),
        ("P90", lambda s: np.percentile(s, 90)),
        ("P95", lambda s: np.percentile(s, 95)),
        ("P99", lambda s: np.percentile(s, 99)),
        ("Max", lambda s: s.max()),
    ]:
        p = fn(test_pred_vals)
        a = fn(test_act_vals)
        print(f"  {label:>18} {p:>12.0f} {a:>12.0f} {p - a:>+10.0f}")

    # Check if model compresses the upper tail
    act_above_p90 = np.percentile(test_act_vals, 90)
    high_gen = pred[pred["actual_solar_mw"] > act_above_p90]
    if len(high_gen) > 0:
        high_bias = (
            high_gen["predicted_solar_mw"] - high_gen["actual_solar_mw"]
        ).mean()
        print(f"\n  Bias for actual > P90 ({act_above_p90:.0f} MW): "
              f"{high_bias:+.0f} MW")
        print(f"  (This shows if the model clips high-generation peaks)")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("SUMMARY — possible causes of underprediction")
    print(SEP)
    print("""
Check the output above for these patterns:

1. PEAK CLIPPING (Step 2 + 7)
   Upper deciles have large negative bias, prediction max << actual max.
   → MAE objective encourages conservative predictions.
   → Fix: add asymmetric loss, or post-hoc bias correction.

2. CAPACITY GROWTH (Step 5 + 6)
   Test mean > train mean, static capacity snapshot used.
   → 2025 has more PV installed than 2024; model was trained on lower levels.
   → Fix: use time-varying capacity, or add capacity as a feature.

3. SYSTEMATIC LEVEL SHIFT (Step 2)
   Uniform negative bias across all deciles.
   → Possible data issue (e.g., proxy scaling factor η fitted on 2024 only).
   → Fix: refit proxy on expanding window, or add bias correction term.

4. ALL MODELS BIASED (Step 4)
   Baseline and LightGBM both underpredict.
   → Not a model-specific issue; likely data or capacity.

5. ONLY CANDIDATE BIASED (Step 4)
   Baseline is unbiased but LightGBM underpredicts.
   → Model-specific: MAE loss, regularization, or feature issue.
""")


if __name__ == "__main__":
    main()
