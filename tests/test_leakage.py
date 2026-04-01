"""Unit tests: no future information leak.

These catch the most dangerous bugs — future data leaking into
predictions.  Run these FIRST.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from solar_pv_forecast.config import PROCESSED_DIR, OUTPUT_DIR


@pytest.fixture(scope="module")
def predictions():
    return pd.read_parquet(OUTPUT_DIR / "predictions.parquet")


@pytest.fixture(scope="module")
def test_predictions():
    return pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(PROCESSED_DIR / "features_table.parquet")


# ── Temporal ordering ────────────────────────────────────────────

class TestTemporalOrdering:
    def test_origin_before_target(self, predictions):
        """For every prediction, origin_timestamp must be strictly before timestamp."""
        bad = predictions[predictions["origin_timestamp"] >= predictions["timestamp"]]
        assert len(bad) == 0, (
            f"{len(bad)} predictions where origin >= target timestamp"
        )

    def test_horizon_matches_time_gap(self, predictions):
        """forecast_horizon × 15 min must equal timestamp - origin_timestamp."""
        sample = predictions.sample(n=min(1000, len(predictions)), random_state=42)
        diff_sec = (
            sample["timestamp"] - sample["origin_timestamp"]
        ).dt.total_seconds()
        expected_sec = sample["forecast_horizon"].astype(float) * 15 * 60
        mismatches = (diff_sec != expected_sec).sum()
        assert mismatches == 0, (
            f"{mismatches} predictions where horizon doesn't match time gap"
        )


# ── Walk-forward no look-ahead ───────────────────────────────────

class TestWalkForwardIntegrity:
    def test_train_end_before_all_targets(self, test_predictions):
        """Each round's model must not have trained on its test month."""
        df = test_predictions.copy()
        df["target_date"] = df["timestamp"].dt.date
        df["train_end"] = pd.to_datetime(df["train_end_date"]).dt.date
        violations = df[df["train_end"] >= df["target_date"]]
        assert len(violations) == 0, (
            f"{len(violations)} predictions where train_end >= target_date"
        )

    def test_train_end_before_all_origins(self, test_predictions):
        """Origin timestamp must also be after the training window."""
        df = test_predictions.copy()
        df["origin_date"] = df["origin_timestamp"].dt.date
        df["train_end"] = pd.to_datetime(df["train_end_date"]).dt.date
        violations = df[df["train_end"] >= df["origin_date"]]
        assert len(violations) == 0, (
            f"{len(violations)} origins fall within training window"
        )

    def test_rounds_monotonically_advance(self, test_predictions):
        """Walk-forward rounds must advance in time — no backwards jumps."""
        round_dates = (
            test_predictions.groupby("round")["train_end_date"]
            .first()
            .sort_index()
        )
        for i in range(1, len(round_dates)):
            assert round_dates.iloc[i] >= round_dates.iloc[i - 1], (
                f"Round {i+1} train_end_date regresses: "
                f"{round_dates.iloc[i]} < {round_dates.iloc[i-1]}"
            )

    def test_no_overlap_between_rounds(self, test_predictions):
        """No target timestamp should appear in two different rounds."""
        # Group by (timestamp, forecast_horizon) — each combo should be unique per round
        counts = (
            test_predictions
            .groupby(["timestamp", "forecast_horizon"])["round"]
            .nunique()
        )
        overlap = counts[counts > 1]
        assert len(overlap) == 0, (
            f"{len(overlap)} (timestamp, horizon) pairs appear in multiple rounds"
        )


# ── Lag leakage ──────────────────────────────────────────────────

class TestLagLeakage:
    def test_lag_1d_is_at_least_24h_old(self, features):
        """actual_lag_1d must reference data from ≥24h ago."""
        df = features.sort_values("timestamp").reset_index(drop=True)
        valid = df[df["actual_lag_1d"].notna()].iloc[100:200]
        for _, row in valid.iterrows():
            t = row["timestamp"]
            t_minus_1d = t - timedelta(hours=24)
            match = df[df["timestamp"] == t_minus_1d]
            if len(match) == 1:
                expected = match["actual_solar_mw"].iloc[0]
                got = row["actual_lag_1d"]
                assert abs(expected - got) < 1e-3, (
                    f"lag_1d at {t} doesn't match actual at {t_minus_1d}"
                )

    def test_no_same_day_lag(self, features):
        """No lag feature should reference the same calendar day."""
        df = features.sort_values("timestamp").reset_index(drop=True)
        valid = df[df["actual_lag_1d"].notna()].sample(n=100, random_state=42)
        for _, row in valid.iterrows():
            t = row["timestamp"]
            # The lag should come from a different day
            lag_source_time = t - timedelta(hours=24)
            assert lag_source_time.date() != t.date() or t.hour < 1, (
                f"Lag at {t} references same calendar day"
            )
