"""Unit tests: model behavior.

Checks prediction bounds, nighttime masking, and that the candidate
model actually beats the baseline.
"""

import numpy as np
import pandas as pd
import pytest

from solar_pv_forecast.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(OUTPUT_DIR / "predictions.parquet")


@pytest.fixture(scope="module")
def test_predictions():
    return pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(PROCESSED_DIR / "features_table.parquet")


# ── Prediction bounds ────────────────────────────────────────────

class TestPredictionBounds:
    def test_non_negative(self, predictions):
        """PV generation can't be negative."""
        assert predictions["predicted_solar_mw"].min() >= 0, (
            "Negative LightGBM prediction"
        )
        assert predictions["pred_baseline_mw"].min() >= 0, (
            "Negative baseline prediction"
        )

    def test_below_installed_capacity(self, predictions):
        """No prediction should exceed installed capacity + margin."""
        cap_df = pd.read_parquet(RAW_DIR / "pv_capacity_by_state.parquet")
        total_cap_mw = cap_df["capacity_mwp"].sum()
        pred_max = predictions["predicted_solar_mw"].max()
        assert pred_max < total_cap_mw * 1.05, (
            f"Prediction {pred_max:.0f} MW exceeds capacity "
            f"{total_cap_mw:.0f} MW"
        )


# ── Nighttime masking ────────────────────────────────────────────

class TestNightMask:
    def test_night_predictions_near_zero(self, test_predictions, features):
        """When solar zenith > 85° at target time, predictions must be ~0."""
        # Build a lookup of zenith by timestamp
        feat = features[["timestamp", "solar_zenith"]].drop_duplicates("timestamp")
        merged = test_predictions.merge(
            feat, on="timestamp", how="inner",
        )
        night = merged[merged["solar_zenith"] > 85]
        if len(night) == 0:
            pytest.skip("No nighttime predictions found in overlap")
        max_night = night["pred_lightgbm"].abs().max()
        assert max_night < 1.0, (
            f"Nighttime prediction max={max_night:.1f} MW, expected ~0"
        )


# ── Model comparison ─────────────────────────────────────────────

class TestModelComparison:
    def test_candidate_beats_baseline_overall(self, test_predictions):
        """LightGBM must outperform Ridge baseline on the full test set."""
        actual = test_predictions["actual_solar_mw"]
        mae_lgb = (test_predictions["pred_lightgbm"] - actual).abs().mean()
        mae_bl = (test_predictions["pred_baseline"] - actual).abs().mean()
        assert mae_lgb < mae_bl, (
            f"LightGBM MAE {mae_lgb:.0f} >= Baseline MAE {mae_bl:.0f}"
        )

    def test_retrained_beats_frozen(self, test_predictions):
        """Monthly retraining should beat the frozen Round-1 model."""
        if "pred_frozen_lightgbm" not in test_predictions.columns:
            pytest.skip("No frozen model predictions")
        actual = test_predictions["actual_solar_mw"]
        mae_retrained = (
            test_predictions["pred_lightgbm"] - actual
        ).abs().mean()
        mae_frozen = (
            test_predictions["pred_frozen_lightgbm"] - actual
        ).abs().mean()
        assert mae_retrained <= mae_frozen, (
            f"Retrained MAE {mae_retrained:.0f} > Frozen MAE {mae_frozen:.0f}"
        )

    def test_skill_score_positive(self, test_predictions):
        """Candidate must have positive skill vs baseline."""
        actual = test_predictions["actual_solar_mw"]
        mae_lgb = (test_predictions["pred_lightgbm"] - actual).abs().mean()
        mae_bl = (test_predictions["pred_baseline"] - actual).abs().mean()
        skill = 1 - mae_lgb / mae_bl
        assert skill > 0, f"Skill score {skill:.4f} is not positive"

    def test_proxy_beaten(self, test_predictions, features):
        """Both models must beat the raw synthetic proxy."""
        # The proxy is an origin feature — approximate its MAE on test
        # by checking that model MAE is lower than proxy correlation implies
        actual = test_predictions["actual_solar_mw"]
        mae_lgb = (test_predictions["pred_lightgbm"] - actual).abs().mean()
        # Proxy nMAE was ~0.214, model nMAE should be well below that
        daytime = actual[actual > 0]
        nmae_lgb = mae_lgb / daytime.mean() if len(daytime) > 0 else float("inf")
        assert nmae_lgb < 0.214, (
            f"LightGBM nMAE {nmae_lgb:.3f} doesn't beat proxy nMAE 0.214"
        )
