"""Fetch archived ICON-D2 NWP forecasts from Open-Meteo Historical Forecast API.

ICON-D2 is DWD's high-resolution regional model for Central Europe:
  - 2 km spatial resolution
  - Native 15-min temporal resolution
  - Updated every 3 hours (00, 03, 06, 09, 12, 15, 18, 21 UTC)
  - ~1.5–2 h publication delay

The Historical Forecast API serves the best-available archived NWP output
at each timestamp, which is what would have been operationally available.

Run:  python -m solar_pv_forecast.data.fetch_nwp
"""

import json
import time

import pandas as pd
import requests
from loguru import logger

from solar_pv_forecast.config import (
    END_DATE,
    GERMAN_STATES,
    NWP_MODEL,
    NWP_VARIABLES_15MIN,
    OPENMETEO_FORECAST_URL,
    RAW_DIR,
    START_DATE,
)
from solar_pv_forecast.utils import log_step, setup_logger


def fetch_state_nwp(
    state: str, lat: float, lon: float, start: str, end: str
) -> pd.DataFrame:
    """Fetch 15-min ICON-D2 forecast data for a single coordinate.

    Chunks by month to stay within API response limits.
    """
    frames = []
    date_range = pd.date_range(start, end, freq="MS")  # month starts

    for month_start in date_range:
        month_end = (month_start + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
        ms = month_start.strftime("%Y-%m-%d")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": ms,
            "end_date": month_end,
            "minutely_15": ",".join(NWP_VARIABLES_15MIN),
            "models": NWP_MODEL,
            "timezone": "UTC",
        }

        for attempt in range(3):
            try:
                resp = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=120)
                resp.raise_for_status()
                break
            except (requests.RequestException, requests.HTTPError) as e:
                if attempt < 2:
                    logger.warning(f"    Retry {attempt+1} for {state} {ms}: {e}")
                    time.sleep(2 ** (attempt + 1))
                else:
                    logger.error(f"    Failed {state} {ms} after 3 attempts: {e}")
                    continue

        data = resp.json()

        minutely = data.get("minutely_15", {})
        if not minutely or "time" not in minutely:
            logger.warning(f"    No minutely_15 data for {state} {ms}")
            continue

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(minutely["time"]),
            "nwp_ghi_wm2": minutely.get("shortwave_radiation"),
            "nwp_temperature_2m": minutely.get("temperature_2m"),
            "nwp_cloud_cover": minutely.get("cloud_cover"),
        })
        df["state"] = state
        df["latitude"] = lat
        df["longitude"] = lon
        frames.append(df)

        time.sleep(0.3)  # rate-limit courtesy

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_all_nwp() -> pd.DataFrame:
    """Fetch ICON-D2 forecast data for all German states."""
    frames = []
    for state, (lat, lon) in GERMAN_STATES.items():
        logger.info(f"  Fetching NWP: {state} ({lat:.2f}, {lon:.2f})")
        try:
            df = fetch_state_nwp(state, lat, lon, START_DATE, END_DATE)
            if len(df) > 0:
                frames.append(df)
                logger.info(f"    {len(df):,} rows")
            else:
                logger.warning(f"    No data for {state}")
        except Exception as e:
            logger.warning(f"  Failed for {state}: {e}")
            continue

    if not frames:
        raise RuntimeError("No NWP data fetched for any state.")

    return pd.concat(frames, ignore_index=True)


def main():
    setup_logger()

    with log_step("Fetch ICON-D2 NWP forecast data"):
        df = fetch_all_nwp()
        out_path = RAW_DIR / "nwp_icon_d2_15min.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            f"  Saved {len(df):,} rows x {len(df.columns)} cols -> {out_path}"
        )

    meta = {
        "source": "Open-Meteo Historical Forecast API (ICON-D2)",
        "url": OPENMETEO_FORECAST_URL,
        "model": NWP_MODEL,
        "variables": NWP_VARIABLES_15MIN,
        "temporal_resolution": "15-min (native)",
        "spatial_resolution": "~2 km (ICON-D2)",
        "period": f"{START_DATE} to {END_DATE}",
        "n_states": len(GERMAN_STATES),
        "coordinates": {k: list(v) for k, v in GERMAN_STATES.items()},
    }
    meta_path = RAW_DIR / "nwp_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
