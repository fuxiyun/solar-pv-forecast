"""Integration tests: end-to-end pipeline validation.

Checks that all pipeline artifacts exist, are consistent with each
other, and form a coherent whole.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from solar_pv_forecast.config import (
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    INTERIM_DIR,
)


# ── Artifact existence ───────────────────────────────────────────

class TestArtifactsExist:
    """All pipeline outputs must exist after a full run."""

    @pytest.mark.parametrize("path", [
        RAW_DIR / "actual_solar_generation.parquet",
        RAW_DIR / "weather_hourly.parquet",
        RAW_DIR / "nwp_icon_d2_15min.parquet",
        RAW_DIR / "pv_capacity_by_state.parquet",
    ])
    def test_raw_data_exists(self, path):
        assert path.exists(), f"Missing raw data: {path.name}"

    @pytest.mark.parametrize("path", [
        INTERIM_DIR / "weather_15min.parquet",
    ])
    def test_interim_data_exists(self, path):
        assert path.exists(), f"Missing interim data: {path.name}"

    @pytest.mark.parametrize("path", [
        PROCESSED_DIR / "modelling_table.parquet",
        PROCESSED_DIR / "features_table.parquet",
        PROCESSED_DIR / "test_predictions.parquet",
    ])
    def test_processed_data_exists(self, path):
        assert path.exists(), f"Missing processed data: {path.name}"

    @pytest.mark.parametrize("path", [
        MODEL_DIR / "lightgbm_model.txt",
        MODEL_DIR / "lightgbm_features.json",
        MODEL_DIR / "baseline_ridge.pkl",
    ])
    def test_model_artifacts_exist(self, path):
        assert path.exists(), f"Missing model artifact: {path.name}"

    @pytest.mark.parametrize("path", [
        OUTPUT_DIR / "predictions.parquet",
        OUTPUT_DIR / "evaluation_summary.json",
    ])
    def test_output_exists(self, path):
        assert path.exists(), f"Missing output: {path.name}"


# ── Schema consistency ───────────────────────────────────────────

class TestSchemaConsistency:
    def test_predictions_has_required_columns(self):
        """Final predictions must have all expected columns."""
        df = pd.read_parquet(OUTPUT_DIR / "predictions.parquet")
        required = {
            "origin_timestamp",
            "timestamp",
            "forecast_horizon",
            "actual_solar_mw",
            "predicted_solar_mw",
            "pred_baseline_mw",
            "round",
            "train_end_date",
        }
        missing = required - set(df.columns)
        assert not missing, f"Missing columns in predictions: {missing}"

    def test_features_table_has_engineered_cols(self):
        """Features table must include all engineered features."""
        df = pd.read_parquet(PROCESSED_DIR / "features_table.parquet")
        expected = [
            "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
            "solar_zenith", "clearsky_ghi", "clearsky_index",
            "actual_lag_1d", "actual_lag_7d",
        ]
        missing = [c for c in expected if c not in df.columns]
        assert not missing, f"Missing feature columns: {missing}"


# ── Cross-file consistency ───────────────────────────────────────

class TestCrossFileConsistency:
    def test_predictions_cover_12_rounds(self):
        """Walk-forward must produce exactly 12 monthly rounds."""
        df = pd.read_parquet(OUTPUT_DIR / "predictions.parquet")
        rounds = sorted(df["round"].unique())
        assert rounds == list(range(1, 13)), (
            f"Expected rounds 1-12, got {rounds}"
        )

    def test_predictions_cover_16_horizons(self):
        """Each round must have all 16 forecast horizons."""
        df = pd.read_parquet(OUTPUT_DIR / "predictions.parquet")
        horizons = sorted(df["forecast_horizon"].unique())
        assert horizons == list(range(1, 17)), (
            f"Expected horizons 1-16, got {horizons}"
        )

    def test_modelling_table_feeds_features(self):
        """Features table must have at least as many rows as modelling table."""
        mt = pd.read_parquet(PROCESSED_DIR / "modelling_table.parquet")
        ft = pd.read_parquet(PROCESSED_DIR / "features_table.parquet")
        assert len(ft) == len(mt), (
            f"Features table ({len(ft)}) != modelling table ({len(mt)})"
        )

    def test_evaluation_summary_valid_json(self):
        """Evaluation summary must be parseable and contain key metrics."""
        with open(OUTPUT_DIR / "evaluation_summary.json") as f:
            summary = json.load(f)
        assert "lightgbm" in summary, "Missing lightgbm in summary"
        assert "baseline" in summary, "Missing baseline in summary"
        assert "skill_score_vs_baseline" in summary

    def test_model_features_match_training(self):
        """Saved feature list must match what the model expects."""
        import lightgbm as lgb

        model = lgb.Booster(
            model_file=str(MODEL_DIR / "lightgbm_model.txt")
        )
        model_feats = model.feature_name()
        with open(MODEL_DIR / "lightgbm_features.json") as f:
            saved_feats = json.load(f)
        # Feature count must match
        assert len(model_feats) == len(saved_feats), (
            f"Model has {len(model_feats)} features, "
            f"JSON has {len(saved_feats)}"
        )
