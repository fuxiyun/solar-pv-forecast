"""Harmonise weather, PV capacity, and target data into a modelling table.

Aligns timestamps to 15-min UTC, interpolates hourly weather to 15-min,
joins capacity weights, and persists as Parquet.

Run:  python -m solar_pv_forecast.data.harmonise
"""

import pandas as pd
from loguru import logger

from solar_pv_forecast.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)
from solar_pv_forecast.utils import log_step, setup_logger


def interpolate_weather_to_15min(weather: pd.DataFrame) -> pd.DataFrame:
    """Interpolate hourly weather data to 15-minute resolution per state."""
    frames = []
    for state in weather["state"].unique():
        df_state = (
            weather[weather["state"] == state]
            .set_index("timestamp")
            .sort_index()
        )
        # Create 15-min index spanning the data range
        idx_15m = pd.date_range(
            start=df_state.index.min(),
            end=df_state.index.max(),
            freq="15min",
        )
        df_15m = df_state.reindex(idx_15m)

        # Interpolate: cubic for GHI (smoother), linear for others
        df_15m["ghi_wm2"] = (
            df_15m["ghi_wm2"]
            .interpolate(method="cubicspline")
            .clip(lower=0)  # GHI cannot be negative
        )
        for col in ["temperature_2m", "wind_speed_10m", "relative_humidity_2m"]:
            if col in df_15m.columns:
                df_15m[col] = df_15m[col].interpolate(method="linear")

        # Forward-fill metadata columns
        df_15m["state"] = state
        df_15m["latitude"] = df_15m["latitude"].ffill().bfill()
        df_15m["longitude"] = df_15m["longitude"].ffill().bfill()

        df_15m.index.name = "timestamp"
        frames.append(df_15m.reset_index())

    return pd.concat(frames, ignore_index=True)


def build_weighted_national_weather(
    weather_15m: pd.DataFrame,
    capacity: pd.DataFrame,
) -> pd.DataFrame:
    """Compute capacity-weighted national averages of weather variables."""
    # Merge capacity weights
    merged = weather_15m.merge(
        capacity[["state", "weight"]],
        on="state",
        how="left",
    )

    # Weighted average per timestamp
    weather_vars = ["ghi_wm2", "temperature_2m", "wind_speed_10m"]
    for var in weather_vars:
        merged[f"{var}_weighted"] = merged[var] * merged["weight"]

    national = (
        merged.groupby("timestamp")
        .agg({f"{v}_weighted": "sum" for v in weather_vars})
        .rename(columns={f"{v}_weighted": f"{v}_national" for v in weather_vars})
        .reset_index()
    )

    # Also keep per-state GHI for the proxy (wide format)
    ghi_wide = (
        weather_15m.pivot_table(
            index="timestamp",
            columns="state",
            values="ghi_wm2",
        )
        .add_prefix("ghi_")
        .reset_index()
    )

    result = national.merge(ghi_wide, on="timestamp", how="left")
    return result


def build_weighted_national_nwp(
    nwp_15m: pd.DataFrame,
    capacity: pd.DataFrame,
) -> pd.DataFrame:
    """Compute capacity-weighted national averages of NWP forecast variables.

    NWP data is already at 15-min resolution (ICON-D2 native), so no
    interpolation is needed.
    """
    merged = nwp_15m.merge(
        capacity[["state", "weight"]],
        on="state",
        how="left",
    )

    nwp_vars = ["nwp_ghi_wm2", "nwp_temperature_2m", "nwp_cloud_cover"]
    nwp_vars = [v for v in nwp_vars if v in merged.columns]

    for var in nwp_vars:
        merged[f"{var}_weighted"] = merged[var] * merged["weight"]

    national = (
        merged.groupby("timestamp")
        .agg({f"{v}_weighted": "sum" for v in nwp_vars})
        .rename(columns={f"{v}_weighted": f"{v}_national" for v in nwp_vars})
        .reset_index()
    )
    return national


def main():
    setup_logger()

    # ── Load raw data ───────────────────────────────────────────
    with log_step("Load raw data"):
        weather = pd.read_parquet(RAW_DIR / "weather_hourly.parquet")
        target = pd.read_parquet(RAW_DIR / "actual_solar_generation.parquet")
        capacity = pd.read_parquet(RAW_DIR / "pv_capacity_by_state.parquet")
        logger.info(
            f"  Weather: {len(weather):,} rows | "
            f"Target: {len(target):,} rows | "
            f"States: {len(capacity)}"
        )

    # ── Interpolate weather to 15-min ───────────────────────────
    with log_step("Interpolate weather → 15-min"):
        weather_15m = interpolate_weather_to_15min(weather)
        logger.info(f"  Interpolated: {len(weather_15m):,} rows")
        weather_15m.to_parquet(
            INTERIM_DIR / "weather_15min.parquet", index=False
        )

    # ── Build national weighted weather ─────────────────────────
    with log_step("Build national weighted weather"):
        national = build_weighted_national_weather(weather_15m, capacity)
        logger.info(f"  National weather: {len(national):,} timestamps")

    # ── Load and merge NWP forecast data (if available) ──────────
    nwp_path = RAW_DIR / "nwp_icon_d2_15min.parquet"
    nwp_national = None
    if nwp_path.exists():
        with log_step("Build national weighted NWP forecasts"):
            nwp = pd.read_parquet(nwp_path)
            nwp["timestamp"] = pd.to_datetime(nwp["timestamp"])
            logger.info(f"  NWP raw: {len(nwp):,} rows")
            nwp_national = build_weighted_national_nwp(nwp, capacity)
            logger.info(f"  NWP national: {len(nwp_national):,} timestamps")
    else:
        logger.warning(f"  NWP data not found at {nwp_path}, skipping NWP features")

    # ── Join with target ────────────────────────────────────────
    with log_step("Join weather + target → modelling table"):
        modelling = national.merge(target, on="timestamp", how="inner")

        # Merge NWP if available
        if nwp_national is not None:
            modelling = modelling.merge(nwp_national, on="timestamp", how="left")

        # Handle missing values: forward-fill gaps < 4 hours
        gap_limit = 4 * 4  # 4 hours × 4 intervals/hour = 16 intervals
        for col in modelling.columns:
            if col != "timestamp":
                modelling[col] = modelling[col].ffill(limit=gap_limit)

        # Drop rows still missing
        n_before = len(modelling)
        modelling = modelling.dropna(how="any") 
        n_dropped = n_before - len(modelling)
        if n_dropped > 0:
            logger.warning(f"  Dropped {n_dropped} rows with remaining NaN")

        # Convert to float32 where safe (not timestamps)
        for col in modelling.columns:
            if modelling[col].dtype == "float64" and col != "timestamp":
                modelling[col] = modelling[col].astype("float32")

        out_path = PROCESSED_DIR / "modelling_table.parquet"
        modelling.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            f"  Modelling table: {len(modelling):,} rows × "
            f"{len(modelling.columns)} cols → {out_path}"
        )
        logger.info(
            f"  Time range: {modelling['timestamp'].min()} → "
            f"{modelling['timestamp'].max()}"
        )


if __name__ == "__main__":
    main()
