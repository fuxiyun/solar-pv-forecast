"""Feature engineering for solar PV forecasting models.

Creates calendar, solar position, clear-sky, and lag features
from the modelling table.
"""

import numpy as np
import pandas as pd
from loguru import logger


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclically-encoded calendar features."""
    ts = df["timestamp"]

    # Hour of day (sin/cos for cyclic encoding)
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24).astype("float32")

    # Day of year (sin/cos)
    doy = ts.dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")

    # Month (for seasonal grouping)
    df["month"] = ts.dt.month.astype("int8")

    # Weekend indicator
    df["is_weekend"] = (ts.dt.weekday >= 5).astype("int8")

    return df


def add_solar_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add solar zenith angle and clear-sky GHI via pvlib.

    Uses Germany's approximate centroid (51.0°N, 10.5°E) for
    national-level solar position. This is a simplification;
    per-state positions would be more precise but add complexity
    for marginal gain at the national aggregation level.
    """
    try:
        from pvlib.solarposition import get_solarposition
        from pvlib.irradiance import get_extra_radiation
        from pvlib.clearsky import simplified_solis

        lat, lon = 51.0, 10.5  # Germany centroid
        times = pd.DatetimeIndex(df["timestamp"], tz="UTC")

        solpos = get_solarposition(times, lat, lon)
        df["solar_zenith"] = solpos["zenith"].values.astype("float32")
        df["solar_elevation"] = solpos["elevation"].values.astype("float32")

        # Simple clear-sky model
        apparent_zenith = solpos["apparent_zenith"].values
        aoi = np.clip(apparent_zenith, 0, 90)
        # Approximate clear-sky GHI using cosine of zenith
        extra = get_extra_radiation(times.dayofyear)
        clearsky_ghi = extra * np.cos(np.radians(aoi))
        clearsky_ghi = np.clip(clearsky_ghi, 0, None)
        df["clearsky_ghi"] = clearsky_ghi.astype("float32")

        # Clear-sky index
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = df["ghi_wm2_national"] / df["clearsky_ghi"]
        kt = np.where(df["clearsky_ghi"] > 10, kt, 0)  # avoid division issues
        df["clearsky_index"] = np.clip(kt, 0, 1.5).astype("float32")

        logger.info("  Added solar position + clear-sky features (pvlib)")

    except ImportError:
        logger.warning("  pvlib not available; skipping solar position features")
        # Fallback: simple zenith approximation
        hour_frac = (
            df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        )
        # Very rough solar elevation proxy
        df["solar_zenith"] = np.nan
        df["solar_elevation"] = np.nan
        df["clearsky_ghi"] = np.nan
        df["clearsky_index"] = np.nan

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged actual generation features.

    Uses lags of >=24h to avoid operational leakage. These features
    assume access to yesterday's realised values, which is realistic
    for day-ahead or intraday forecasting contexts.
    """
    # 1-day lag (96 intervals of 15 min)
    df["actual_lag_1d"] = df["actual_solar_mw"].shift(96).astype("float32")

    # 7-day lag (672 intervals)
    df["actual_lag_7d"] = df["actual_solar_mw"].shift(672).astype("float32")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_solar_position_features(df)
    df = add_lag_features(df)

    logger.info(f"  Feature matrix: {df.shape[1]} columns")
    return df
