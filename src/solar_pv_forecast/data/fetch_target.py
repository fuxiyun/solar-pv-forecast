"""Fetch realised German solar generation from SMARD (Bundesnetzagentur).

SMARD provides 15-minute resolution generation data via a public API.
No registration or API key required.

Run:  python -m solar_pv_forecast.data.fetch_target
"""

import io
from datetime import datetime, timezone

import pandas as pd
import requests
from loguru import logger

from solar_pv_forecast.config import RAW_DIR, START_DATE, END_DATE
from solar_pv_forecast.utils import log_step, setup_logger

# SMARD API configuration
# Filter 4069 = Solar generation
# Region DE = Germany total
SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
SMARD_FILTER = 4069        # Realised solar generation
SMARD_REGION = "DE"
SMARD_RESOLUTION = "quarterhour"


def _ts_to_smard_ms(date_str: str) -> int:
    """Convert date string to SMARD millisecond timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_smard_index() -> list[int]:
    """Get available timestamp indices from SMARD for solar generation."""
    url = (
        f"{SMARD_BASE_URL}/{SMARD_FILTER}/{SMARD_REGION}"
        f"/index_{SMARD_RESOLUTION}.json"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["timestamps"]


def fetch_smard_chunk(timestamp_ms: int) -> pd.DataFrame:
    """Fetch one chunk of SMARD data starting at the given timestamp."""
    url = (
        f"{SMARD_BASE_URL}/{SMARD_FILTER}/{SMARD_REGION}"
        f"/{SMARD_FILTER}_{SMARD_REGION}_{SMARD_RESOLUTION}_{timestamp_ms}.json"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    series = data["series"]
    rows = []
    for ts_ms, value in series:
        if value is not None:
            rows.append({
                "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                "actual_solar_mw": value,
            })
    return pd.DataFrame(rows)


def fetch_solar_generation(start: str, end: str) -> pd.DataFrame:
    """Fetch all available SMARD solar generation data in date range."""
    start_ms = _ts_to_smard_ms(start)
    end_ms = _ts_to_smard_ms(end)

    logger.info("  Fetching SMARD index...")
    timestamps = fetch_smard_index()

    # Filter to relevant timestamps (each chunk covers ~1 week)
    relevant = [ts for ts in timestamps if ts >= start_ms - 7 * 86400_000
                and ts <= end_ms + 7 * 86400_000]
    logger.info(f"  Found {len(relevant)} relevant chunks to download.")

    frames = []
    for i, ts in enumerate(relevant):
        try:
            df = fetch_smard_chunk(ts)
            frames.append(df)
            if (i + 1) % 10 == 0:
                logger.info(f"    Downloaded {i + 1}/{len(relevant)} chunks")
        except Exception as e:
            logger.warning(f"    Chunk {ts} failed: {e}")
            continue

    if not frames:
        raise RuntimeError("No SMARD data fetched.")

    result = pd.concat(frames, ignore_index=True)

    # Remove timezone info for consistency, filter to date range
    result["timestamp"] = result["timestamp"].dt.tz_localize(None)
    mask = (
        (result["timestamp"] >= start)
        & (result["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    )
    result = result.loc[mask].drop_duplicates(subset="timestamp").sort_values("timestamp")

    return result.reset_index(drop=True)


def main():
    setup_logger()

    with log_step("Fetch SMARD solar generation"):
        df = fetch_solar_generation(START_DATE, END_DATE)
        out_path = RAW_DIR / "actual_solar_generation.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            f"  Saved {len(df):,} rows, "
            f"range: {df['timestamp'].min()} → {df['timestamp'].max()}"
        )
        logger.info(
            f"  Generation stats: "
            f"mean={df['actual_solar_mw'].mean():.0f} MW, "
            f"max={df['actual_solar_mw'].max():.0f} MW"
        )


if __name__ == "__main__":
    main()
