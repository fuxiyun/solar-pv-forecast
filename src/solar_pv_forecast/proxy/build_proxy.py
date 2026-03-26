"""Build the synthetic national PV proxy from weather + capacity weights.

The proxy is: P_syn_t = η × Σ_i (GHI_i,t × w_i)
where w_i is the normalised capacity weight for state i,
and η is fitted via OLS on the training set.

Run:  python -m solar_pv_forecast.proxy.build_proxy
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

from solar_pv_forecast.config import (
    PV_CAPACITY_MWP,
    PROCESSED_DIR,
    TRAIN_END_MONTH,
)
from solar_pv_forecast.utils import log_step, setup_logger


def compute_raw_proxy(df: pd.DataFrame) -> pd.Series:
    """Compute the raw capacity-weighted GHI sum (before scaling).

    Looks for columns named ghi_<state> in the dataframe and applies
    capacity weights to produce a single national proxy series.
    """
    total_cap = sum(PV_CAPACITY_MWP.values())
    proxy = pd.Series(0.0, index=df.index, dtype="float64")

    for state, cap in PV_CAPACITY_MWP.items():
        col = f"ghi_{state}"
        if col in df.columns:
            proxy += df[col].astype("float64") * (cap / total_cap)
        else:
            logger.warning(f"  Missing column {col}, skipping state.")

    return proxy


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
        train_mask = df["timestamp"].dt.month <= TRAIN_END_MONTH

        # Fit scaling factor
        eta = fit_scaling_factor(
            proxy_raw, df["actual_solar_mw"].astype("float64"), train_mask
        )

        # Apply scaling
        df["proxy_solar_mw"] = (proxy_raw * eta).astype("float32")

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
