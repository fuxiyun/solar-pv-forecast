"""Fetch monthly PV installed capacity per German federal state.

Provides time-varying capacity weights for the spatial aggregation of
irradiance into a national synthetic PV proxy.  Accounts for capacity
growth between 2024 and 2025 (~83 GWp → ~117 GWp).

Strategy:
  1. Try Fraunhofer Energy-Charts API for monthly national totals.
  2. Fall back to hardcoded monthly values from BNetzA statistics.
  3. Distribute national totals across states using the static weight
     template (state-level distribution is stable over time).

Outputs:
  data/raw/pv_capacity_monthly.parquet   — (month, state, capacity_mwp, weight)
  data/raw/pv_capacity_by_state.parquet  — static snapshot (backward compat)

Run:  python -m solar_pv_forecast.data.fetch_pv_capacity
"""

import json

import pandas as pd
import requests
from loguru import logger

from solar_pv_forecast.config import (
    ENERGY_CHARTS_API_URL,
    PV_CAPACITY_MWP,
    PV_MONTHLY_CAPACITY_MWP_FALLBACK,
    PV_STATE_WEIGHTS,
    RAW_DIR,
    START_DATE,
    END_DATE,
)
from solar_pv_forecast.utils import log_step, setup_logger


def fetch_monthly_national_capacity() -> dict[str, float]:
    """Fetch monthly installed PV capacity (MW) from Energy-Charts API.

    Returns dict mapping 'YYYY-MM' → total national capacity in MWp.
    Falls back to hardcoded values if the API is unreachable.
    """
    try:
        resp = requests.get(
            ENERGY_CHARTS_API_URL,
            params={
                "country": "de",
                "time_step": "monthly",
                "installation_decommission": "false",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Energy-Charts returns {production_type: [...], time: [...], ...}
        # Find the solar/PV entry
        types = data.get("production_types", [])
        times = data.get("unix_seconds", [])

        solar_data = None
        for entry in types:
            name = entry.get("name", "").lower()
            if "solar" in name or "photovoltaic" in name:
                solar_data = entry.get("data", [])
                break

        if solar_data is None or not times:
            logger.warning("  Energy-Charts response missing solar data, using fallback")
            return {k: float(v) for k, v in PV_MONTHLY_CAPACITY_MWP_FALLBACK.items()}

        monthly = {}
        for ts, val in zip(times, solar_data):
            if val is None:
                continue
            dt = pd.Timestamp(ts, unit="s")
            ym = dt.strftime("%Y-%m")
            # API returns MW — keep as MWp
            monthly[ym] = float(val)

        # Filter to our date range
        start_ym = START_DATE[:7]
        end_ym = END_DATE[:7]
        filtered = {
            k: v for k, v in monthly.items()
            if start_ym <= k <= end_ym
        }

        if len(filtered) < 12:
            logger.warning(
                f"  Energy-Charts returned only {len(filtered)} months "
                f"in range, supplementing with fallback"
            )
            for k, v in PV_MONTHLY_CAPACITY_MWP_FALLBACK.items():
                if k not in filtered:
                    filtered[k] = v

        logger.info(f"  Fetched {len(filtered)} months from Energy-Charts API")
        return filtered

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning(f"  Energy-Charts API failed ({e}), using fallback values")
        return {k: float(v) for k, v in PV_MONTHLY_CAPACITY_MWP_FALLBACK.items()}


def build_monthly_state_capacity(
    national_monthly: dict[str, float],
) -> pd.DataFrame:
    """Distribute national monthly totals across states.

    Uses the static state-level weight template.  Returns a DataFrame
    with columns: month, state, capacity_mwp, weight.
    """
    records = []
    for ym, national_total in sorted(national_monthly.items()):
        for state, w in PV_STATE_WEIGHTS.items():
            records.append({
                "month": ym,
                "state": state,
                "capacity_mwp": national_total * w,
                "weight": w,
                "national_total_mwp": national_total,
            })
    return pd.DataFrame(records)


def load_pv_capacity() -> pd.DataFrame:
    """Load static PV capacity per state (backward compatibility).

    Returns a DataFrame with columns: state, capacity_mwp, weight.
    """
    total = sum(PV_CAPACITY_MWP.values())
    records = [
        {"state": state, "capacity_mwp": cap, "weight": cap / total}
        for state, cap in PV_CAPACITY_MWP.items()
    ]
    return pd.DataFrame(records).sort_values(
        "capacity_mwp", ascending=False
    ).reset_index(drop=True)


def main():
    setup_logger()

    # ── Fetch monthly national capacity ───────────────────────────
    with log_step("Fetch monthly PV capacity"):
        national_monthly = fetch_monthly_national_capacity()

        # Log growth
        months = sorted(national_monthly.keys())
        if len(months) >= 2:
            first = national_monthly[months[0]]
            last = national_monthly[months[-1]]
            logger.info(
                f"  Capacity range: {first/1000:.1f} GWp ({months[0]}) → "
                f"{last/1000:.1f} GWp ({months[-1]}) "
                f"[+{(last-first)/1000:.1f} GWp]"
            )

    # ── Build per-state monthly table ─────────────────────────────
    with log_step("Build monthly state-level capacity"):
        monthly_df = build_monthly_state_capacity(national_monthly)

        out_path = RAW_DIR / "pv_capacity_monthly.parquet"
        monthly_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            f"  Monthly capacity: {len(monthly_df):,} rows "
            f"({monthly_df['month'].nunique()} months × "
            f"{monthly_df['state'].nunique()} states) → {out_path}"
        )

    # ── Save static snapshot (backward compatibility) ─────────────
    with log_step("Save static capacity snapshot"):
        static_df = load_pv_capacity()
        out_path = RAW_DIR / "pv_capacity_by_state.parquet"
        static_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"  Static snapshot: {static_df['capacity_mwp'].sum():,.0f} MWp")

    # ── Save metadata ─────────────────────────────────────────────
    meta = {
        "source": "Fraunhofer Energy-Charts API + BNetzA fallback",
        "period": f"{months[0]} to {months[-1]}",
        "unit": "MWp (peak installed capacity)",
        "national_total_start": national_monthly[months[0]],
        "national_total_end": national_monthly[months[-1]],
        "n_states": 16,
        "n_months": len(months),
        "state_distribution": "proportional to mid-2024 BNetzA snapshot",
        "notes": (
            "Monthly national totals from Energy-Charts API (or fallback). "
            "State-level distribution uses fixed weight ratios from BNetzA 2024. "
            "This is a simplification — new capacity is not perfectly proportional, "
            "but the approximation is sufficient for national-level proxy weighting."
        ),
    }
    with open(RAW_DIR / "pv_capacity_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
