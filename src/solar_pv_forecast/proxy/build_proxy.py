"""Build the synthetic national PV proxy from weather + capacity weights.

The proxy is: P_syn_t = η_m × C_m × Σ_i (GHI_i,t × w_i)
where w_i is the normalised capacity weight for state i,
C_m is the national capacity for month m (time-varying),
and η_m is fitted via OLS on the training set.

When monthly capacity data is available, the proxy accounts for
Germany's rapid PV expansion (~83 GWp Jan 2024 → ~117 GWp Dec 2025).

Run:  python -m solar_pv_forecast.proxy.build_proxy
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

from solar_pv_forecast.config import (
    PV_CAPACITY_MWP,
    PV_MONTHLY_CAPACITY_MWP_FALLBACK,
    PV_STATE_WEIGHTS,
    PROCESSED_DIR,
    RAW_DIR,
    TRAIN_END_DATE,
)
from solar_pv_forecast.utils import log_step, setup_logger


def _load_monthly_national_capacity(df: pd.DataFrame) -> pd.Series:
    """Return a Series indexed like df with monthly national capacity (MWp).

    Uses pv_capacity_monthly.parquet if available, else fallback constants.
    """
    path = RAW_DIR / "pv_capacity_monthly.parquet"
    if path.exists():
        monthly = pd.read_parquet(path)
        # Get national total per month (one row per month)
        nat = (
            monthly.groupby("month")["national_total_mwp"]
            .first()
            .to_dict()
        )
        logger.info(f"  Loaded time-varying capacity ({len(nat)} months)")
    else:
        nat = {k: float(v) for k, v in PV_MONTHLY_CAPACITY_MWP_FALLBACK.items()}
        logger.info("  Using fallback monthly capacity values")

    # Map each row's month to its national capacity
    ym = df["timestamp"].dt.to_period("M").astype(str)
    cap_series = ym.map(nat)

    # Fill any unmapped months with the nearest available value
    if cap_series.isna().any():
        fallback_val = float(np.median(list(nat.values())))
        n_missing = cap_series.isna().sum()
        logger.warning(
            f"  {n_missing} rows have no monthly capacity, "
            f"using median ({fallback_val:.0f} MWp)"
        )
        cap_series = cap_series.fillna(fallback_val)

    return cap_series.astype("float64")


def compute_raw_proxy(df: pd.DataFrame) -> pd.Series:
    """Compute the raw capacity-weighted GHI sum (before η scaling).

    Uses static state-level weights (distribution is stable) but
    scales by time-varying national capacity so the proxy tracks
    capacity growth.

    proxy_raw_t = C_national_m(t) × Σ_i (GHI_i,t × w_i)

    This has units of MWp × W/m², which η then converts to MW.
    """
    # Weighted-average GHI (using static state weights)
    ghi_weighted = pd.Series(0.0, index=df.index, dtype="float64")
    for state, w in PV_STATE_WEIGHTS.items():
        col = f"ghi_{state}"
        if col in df.columns:
            ghi_weighted += df[col].astype("float64") * w
        else:
            logger.warning(f"  Missing column {col}, skipping state.")

    # Scale by monthly national capacity
    monthly_cap = _load_monthly_national_capacity(df)
    proxy_raw = ghi_weighted * monthly_cap

    return proxy_raw


def fit_scaling_factor(
    proxy_raw: pd.Series,
    actual: pd.Series,
    train_mask: pd.Series,
) -> float:
    """Fit η via OLS: actual ≈ η × proxy_raw (daytime only)."""
    # Only fit on daytime observations (proxy > 0 and actual > 0)
    mask = train_mask & (proxy_raw > 0) & (actual > 0)
    X = proxy_raw[mask].values.reshape(-1, 1)
    y = actual[mask].values

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    eta = reg.coef_[0]
    r2 = reg.score(X, y)
    logger.info(f"  Fitted η = {eta:.2f} (R² = {r2:.4f} on training daytime)")
    return eta


def main():
    setup_logger()

    with log_step("Build synthetic PV proxy"):
        df = pd.read_parquet(PROCESSED_DIR / "modelling_table.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Compute raw proxy
        proxy_raw = compute_raw_proxy(df)
        df["proxy_raw"] = proxy_raw.astype("float32")

        # Define train mask
        train_mask = df["timestamp"] <= TRAIN_END_DATE

        # Fit scaling factor
        eta = fit_scaling_factor(
            proxy_raw, df["actual_solar_mw"].astype("float64"), train_mask
        )

        # Apply scaling
        df["proxy_solar_mw"] = (proxy_raw * eta).astype("float32")

        # Add national capacity as a feature (for the model to learn
        # the capacity → generation scaling directly)
        monthly_cap = _load_monthly_national_capacity(df)
        df["national_capacity_mwp"] = monthly_cap.astype("float32")

        # Diagnostics
        daytime = df["actual_solar_mw"] > 0
        corr = df.loc[daytime, "proxy_solar_mw"].corr(
            df.loc[daytime, "actual_solar_mw"]
        )
        mae = (
            (df.loc[daytime, "proxy_solar_mw"] - df.loc[daytime, "actual_solar_mw"])
            .abs()
            .mean()
        )
        logger.info(f"  Proxy correlation (daytime): {corr:.4f}")
        logger.info(f"  Proxy MAE (daytime): {mae:.0f} MW")

        # Save
        out_path = PROCESSED_DIR / "modelling_table.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"  Updated modelling table with proxy → {out_path}")


if __name__ == "__main__":
    main()
