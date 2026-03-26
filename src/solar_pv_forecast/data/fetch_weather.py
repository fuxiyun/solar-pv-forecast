"""Fetch historical weather data from Open-Meteo for German federal states.

Uses the Open-Meteo Historical Weather API (ERA5 backend) to retrieve
hourly GHI, temperature, and wind speed for representative coordinates.

Run:  python -m solar_pv_forecast.data.fetch_weather
"""

import json
import time

import pandas as pd
import requests
from loguru import logger

from solar_pv_forecast.config import (
    GERMAN_STATES,
    OPENMETEO_HISTORICAL_URL,
    RAW_DIR,
    START_DATE,
    END_DATE,
    WEATHER_VARIABLES,
)
from solar_pv_forecast.utils import log_step, setup_logger


def fetch_state_weather(
    state: str, lat: float, lon: float, start: str, end: str
) -> pd.DataFrame:
    """Fetch hourly weather for a single coordinate from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
    }

    resp = requests.get(OPENMETEO_HISTORICAL_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "ghi_wm2": hourly["shortwave_radiation"],
        "temperature_2m": hourly["temperature_2m"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
    })
    df["state"] = state
    df["latitude"] = lat
    df["longitude"] = lon

    return df


def fetch_all_weather() -> pd.DataFrame:
    """Fetch weather data for all German states and concatenate."""
    frames = []
    for state, (lat, lon) in GERMAN_STATES.items():
        logger.info(f"  Fetching weather: {state} ({lat:.2f}, {lon:.2f})")
        try:
            df = fetch_state_weather(state, lat, lon, START_DATE, END_DATE)
            frames.append(df)
            time.sleep(0.5)  # rate-limit courtesy
        except Exception as e:
            logger.warning(f"  Failed for {state}: {e}")
            continue

    if not frames:
        raise RuntimeError("No weather data fetched for any state.")

    return pd.concat(frames, ignore_index=True)


def main():
    setup_logger()

    with log_step("Fetch weather data"):
        df = fetch_all_weather()
        out_path = RAW_DIR / "weather_hourly.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            f"  Saved {len(df):,} rows × {len(df.columns)} cols → {out_path}"
        )

    # Also save metadata
    meta = {
        "source": "Open-Meteo Historical Weather API (ERA5)",
        "url": OPENMETEO_HISTORICAL_URL,
        "variables": WEATHER_VARIABLES,
        "temporal_resolution": "hourly",
        "spatial_resolution": "0.25° (~25 km, ERA5 native)",
        "period": f"{START_DATE} to {END_DATE}",
        "n_states": len(GERMAN_STATES),
        "coordinates": {k: list(v) for k, v in GERMAN_STATES.items()},
    }
    meta_path = RAW_DIR / "weather_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata → {meta_path}")


if __name__ == "__main__":
    main()
