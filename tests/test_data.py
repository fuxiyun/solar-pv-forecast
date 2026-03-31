"""Unit tests: data quality checks on raw and harmonised data.

Catches corruption, gaps, physical impossibilities, and dtype violations.
"""

import pandas as pd
import pytest

from solar_pv_forecast.config import PROCESSED_DIR, RAW_DIR, INTERIM_DIR


# ── Raw actuals ──────────────────────────────────────────────────

class TestRawActuals:
    @pytest.fixture(autouse=True)
    def load(self):
        self.df = pd.read_parquet(RAW_DIR / "actual_solar_generation.parquet")

    def test_no_duplicates(self):
        """Duplicate timestamps cause silent double-counting."""
        dupes = self.df["timestamp"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate timestamps found"

    def test_coverage(self):
        """Gaps in actuals mean missing targets and broken lags."""
        df = self.df.sort_values("timestamp")
        diffs = df["timestamp"].diff().dropna()
        expected = pd.Timedelta(minutes=15)
        gaps = diffs[diffs > expected]
        gap_pct = len(gaps) / len(df) * 100
        assert gap_pct < 0.5, f"Too many gaps: {gap_pct:.3f}%"

    def test_physical_bounds(self):
        """Generation can't be negative or exceed installed capacity."""
        assert self.df["actual_solar_mw"].min() >= 0, "Negative generation"
        # German PV capacity is ~80-90 GW in 2024-2025
        assert self.df["actual_solar_mw"].max() < 100_000, (
            "Generation exceeds plausible capacity"
        )


# ── Raw weather ──────────────────────────────────────────────────

class TestRawWeather:
    @pytest.fixture(autouse=True)
    def load(self):
        self.df = pd.read_parquet(RAW_DIR / "weather_hourly.parquet")

    def test_ghi_bounds(self):
        """GHI can't be negative; >1500 W/m² is physically impossible."""
        assert self.df["ghi_wm2"].min() >= 0, "Negative irradiance"
        assert self.df["ghi_wm2"].max() < 1500, "GHI > 1500 W/m²"

    def test_temperature_bounds(self):
        """Temperature should be plausible for Germany."""
        assert self.df["temperature_2m"].min() > -40, "Temp below -40°C"
        assert self.df["temperature_2m"].max() < 50, "Temp above 50°C"

    def test_all_16_states_present(self):
        """Weather must cover all 16 Bundesländer."""
        n_states = self.df["state"].nunique()
        assert n_states == 16, f"Only {n_states} states, expected 16"


# ── Modelling table (harmonised output) ─────────────────────────

class TestModellingTable:
    @pytest.fixture(autouse=True)
    def load(self):
        self.df = pd.read_parquet(PROCESSED_DIR / "modelling_table.parquet")

    def test_float32_dtypes(self):
        """All numeric columns must be float32 per spec."""
        for col in self.df.columns:
            if col == "timestamp":
                continue
            assert self.df[col].dtype.name == "float32", (
                f"{col} is {self.df[col].dtype}, expected float32"
            )

    def test_row_count(self):
        """Should have ~96 quarter-hours × 365/366 days per year."""
        years = self.df["timestamp"].dt.year.unique()
        expected = sum(366 * 96 if y % 4 == 0 else 365 * 96 for y in years)
        # Allow up to 96 rows tolerance (1 day) for edge effects
        assert abs(len(self.df) - expected) <= 96, (
            f"Row count {len(self.df)} vs expected ~{expected}"
        )

    def test_no_nulls_in_key_columns(self):
        """Key columns must not have NaNs after harmonisation."""
        key_cols = [
            "ghi_wm2_national",
            "temperature_2m_national",
            "actual_solar_mw",
        ]
        for col in key_cols:
            if col in self.df.columns:
                n_null = self.df[col].isna().sum()
                assert n_null == 0, f"{col} has {n_null} nulls"

    def test_timestamp_sorted(self):
        """Timestamps must be monotonically increasing."""
        ts = self.df["timestamp"]
        assert ts.is_monotonic_increasing, "Timestamps are not sorted"


# ── Interpolated weather (interim) ───────────────────────────────

class TestInterimWeather:
    @pytest.fixture(autouse=True)
    def load(self):
        self.df = pd.read_parquet(INTERIM_DIR / "weather_15min.parquet")

    def test_15min_resolution(self):
        """Interpolated weather must be at 15-min resolution."""
        sample = self.df[self.df["state"] == self.df["state"].iloc[0]]
        sample = sample.sort_values("timestamp")
        diffs = sample["timestamp"].diff().dropna()
        modal = diffs.mode().iloc[0]
        assert modal == pd.Timedelta(minutes=15), (
            f"Modal interval is {modal}, expected 15 min"
        )

    def test_ghi_non_negative_after_interpolation(self):
        """Cubic interpolation can produce negative GHI — must be clamped."""
        assert self.df["ghi_wm2"].min() >= 0, (
            f"Negative GHI after interpolation: {self.df['ghi_wm2'].min()}"
        )
