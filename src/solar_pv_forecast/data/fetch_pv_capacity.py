"""PV installed capacity per German federal state.

Provides capacity weights for the spatial aggregation of irradiance
into a national synthetic PV proxy.

Source: Bundesnetzagentur / Marktstammdatenregister, aggregated values.
Fallback values from 2024 annual statistics are used if live download
is not available.

Run:  python -m solar_pv_forecast.data.fetch_pv_capacity
"""

import json

import pandas as pd
from loguru import logger

from solar_pv_forecast.config import PV_CAPACITY_MWP, RAW_DIR
from solar_pv_forecast.utils import log_step, setup_logger


def load_pv_capacity() -> pd.DataFrame:
    """Load PV capacity per state and compute normalised weights.

    Returns a DataFrame with columns:
        state, capacity_mwp, weight
    """
    records = []
    total = sum(PV_CAPACITY_MWP.values())

    for state, cap in PV_CAPACITY_MWP.items():
        records.append({
            "state": state,
            "capacity_mwp": cap,
            "weight": cap / total,
        })

    df = pd.DataFrame(records).sort_values("capacity_mwp", ascending=False)
    return df.reset_index(drop=True)


def main():
    setup_logger()

    with log_step("Load PV capacity weights"):
        df = load_pv_capacity()

        out_path = RAW_DIR / "pv_capacity_by_state.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")

        logger.info(f"  Total installed: {df['capacity_mwp'].sum():,.0f} MWp")
        logger.info(f"  Top 3 states:")
        for _, row in df.head(3).iterrows():
            logger.info(
                f"    {row['state']}: {row['capacity_mwp']:,.0f} MWp "
                f"({row['weight']:.1%})"
            )

        # Save documentation
        meta = {
            "source": "Bundesnetzagentur / Marktstammdatenregister (aggregated)",
            "reference_year": 2024,
            "unit": "MWp (peak installed capacity)",
            "total_mwp": int(df["capacity_mwp"].sum()),
            "n_states": len(df),
            "notes": (
                "Approximate values aggregated from public BNetzA statistics. "
                "These are static snapshots and do not reflect intra-year "
                "commissioning of new capacity."
            ),
        }
        meta_path = RAW_DIR / "pv_capacity_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
