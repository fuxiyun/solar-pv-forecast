"""Unit tests: feature correctness.

Verifies lag shifts, clear-sky determinism, forbidden features, and
calendar encoding.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from solar_pv_forecast.config import PROCESSED_DIR


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(PROCESSED_DIR / "features_table.parquet")


# ── Lag correctness ──────────────────────────────────────────────

class TestLags:
    def test_lag_1d_shift_correctness(self, features):
        """Verify actual_lag_1d is actual_solar_mw shifted by exactly 96 rows (24h)."""
        df = features.sort_values("timestamp").reset_index(drop=True)
        # Check 100 non-null rows in the middle of the dataset
        valid = df[df["actual_lag_1d"].notna()].iloc[100:200]
        for idx, row in valid.iterrows():
            t = row["timestamp"]
            t_minus_1d = t - timedelta(hours=24)
            match = df[df["timestamp"] == t_minus_1d]
            if len(match) == 1:
                expected = match["actual_solar_mw"].iloc[0]
                got = row["actual_lag_1d"]
                assert abs(expected - got) < 1e-3, (
                    f"lag_1d at {t}: expected {expected}, got {got}"
                )

    def test_lag_7d_shift_correctness(self, features):
        """Verify actual_lag_7d is actual_solar_mw shifted by 672 rows (7d)."""
        df = features.sort_values("timestamp").reset_index(drop=True)
        valid = df[df["actual_lag_7d"].notna()].iloc[100:150]
        for idx, row in valid.iterrows():
            t = row["timestamp"]
            t_minus_7d = t - timedelta(days=7)
            match = df[df["timestamp"] == t_minus_7d]
            if len(match) == 1:
                expected = match["actual_solar_mw"].iloc[0]
                got = row["actual_lag_7d"]
                assert abs(expected - got) < 1e-3, (
                    f"lag_7d at {t}: expected {expected}, got {got}"
                )

    def test_no_forbidden_lags(self, features):
        """Lags < 24h must not exist (ENTSO-E ~60-min publication delay)."""
        for forbidden in [
            "actual_lag_1", "actual_lag_2", "actual_lag_3",
            "actual_lag_1h", "actual_lag_2h", "actual_lag_3h",
        ]:
            assert forbidden not in features.columns, (
                f"FORBIDDEN: {forbidden} exists — leaks future actuals!"
            )


# ── Clear-sky features ───────────────────────────────────────────

class TestClearSky:
    def test_deterministic_across_years(self, features):
        """Clear-sky GHI is astronomical — nearly identical for same day across years."""
        jan15_noon = features[
            (features["timestamp"].dt.month == 1)
            & (features["timestamp"].dt.day == 15)
            & (features["timestamp"].dt.hour == 12)
            & (features["timestamp"].dt.minute == 0)
        ]
        if len(jan15_noon) > 1:
            vals = jan15_noon["clearsky_ghi"].values
            # Should differ by <2% year-to-year (tiny orbital variation)
            assert vals.max() - vals.min() < vals.max() * 0.02, (
                f"Clear-sky GHI varies too much across years: {vals}"
            )

    def test_clearsky_index_range(self, features):
        """Clearsky index should be in [0, 1.5] (clamped in feature eng)."""
        assert features["clearsky_index"].min() >= 0
        assert features["clearsky_index"].max() <= 1.5

    def test_solar_zenith_range(self, features):
        """Solar zenith must be in [0°, ~180°] for any location on Earth."""
        assert features["solar_zenith"].min() >= 0
        assert features["solar_zenith"].max() <= 180


# ── Calendar features ────────────────────────────────────────────

class TestCalendar:
    def test_hour_sin_cos_unit_circle(self, features):
        """sin² + cos² must equal 1 for hour encoding."""
        norm = features["hour_sin"] ** 2 + features["hour_cos"] ** 2
        assert np.allclose(norm, 1.0, atol=1e-5), (
            f"hour sin²+cos² range: [{norm.min():.6f}, {norm.max():.6f}]"
        )

    def test_doy_sin_cos_unit_circle(self, features):
        """sin² + cos² must equal 1 for day-of-year encoding."""
        norm = features["doy_sin"] ** 2 + features["doy_cos"] ** 2
        assert np.allclose(norm, 1.0, atol=1e-5)

    def test_month_range(self, features):
        """Month must be 1-12."""
        assert features["month"].min() >= 1
        assert features["month"].max() <= 12
