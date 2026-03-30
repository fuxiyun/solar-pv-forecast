"""Interactive Streamlit dashboard for solar PV forecast visualisation.

Shows raw weather data, the SMARD target curve, walk-forward predictions,
horizon-stratified metrics, and monthly retrained vs frozen comparison.

Run with::

    streamlit run src/solar_pv_forecast/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_WEATHER     = ROOT / "data" / "raw" / "weather_hourly.parquet"
RAW_TARGET      = ROOT / "data" / "raw" / "actual_solar_generation.parquet"
MODELLING_TABLE = ROOT / "data" / "processed" / "modelling_table.parquet"
PREDICTIONS     = ROOT / "output" / "predictions.parquet"
MONTHLY_METRICS = ROOT / "output" / "monthly_metrics.csv"
HORIZON_METRICS = ROOT / "output" / "horizon_metrics.csv"
TOD_METRICS     = ROOT / "output" / "tod_metrics.csv"
EVAL_SUMMARY    = ROOT / "output" / "evaluation_summary.json"

# ── Colours ────────────────────────────────────────────────────────────────
C_ACTUAL   = "#2C2C2C"
C_BASELINE = "#3B8BD4"
C_LGBM     = "#D85A30"
C_FROZEN   = "#888888"
C_PROXY    = "#8B5CF6"
C_GHI      = "#F59E0B"
C_TEMP     = "#EF4444"
C_WIND     = "#10B981"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar PV Forecast — Germany 2025",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar PV Forecast — Germany 2025")
st.caption(
    "Quarter-hourly national solar generation · Open-Meteo ERA5 weather · "
    "SMARD actual generation · Walk-forward evaluation (12 rounds) · "
    "Ridge baseline + LightGBM · 16-step (4 h) intraday horizon"
)

# ── Data loaders ───────────────────────────────────────────────────────────

@st.cache_data
def load_raw_weather() -> pd.DataFrame:
    df = pd.read_parquet(RAW_WEATHER)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def load_raw_target() -> pd.DataFrame:
    df = pd.read_parquet(RAW_TARGET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def load_modelling_table() -> pd.DataFrame:
    df = pd.read_parquet(MODELLING_TABLE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PREDICTIONS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "origin_timestamp" in df.columns:
        df["origin_timestamp"] = pd.to_datetime(df["origin_timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def load_monthly_metrics() -> pd.DataFrame:
    return pd.read_csv(MONTHLY_METRICS)


@st.cache_data
def load_horizon_metrics() -> pd.DataFrame:
    return pd.read_csv(HORIZON_METRICS)


@st.cache_data
def load_tod_metrics() -> pd.DataFrame:
    return pd.read_csv(TOD_METRICS)


@st.cache_data
def load_eval_summary() -> dict:
    with open(EVAL_SUMMARY) as f:
        return json.load(f)


def resample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    numeric = df.select_dtypes("number").columns.tolist()
    return (
        df.set_index("timestamp")[numeric]
        .resample(freq)
        .mean()
        .reset_index()
    )


# ── Check raw data availability ──────────────────────────────────────────

raw_missing = [p for p in (RAW_WEATHER, RAW_TARGET) if not p.exists()]
if raw_missing:
    st.error(
        "Raw data files are missing. Run the fetch steps first:\n\n"
        + "\n".join(f"- `{p.relative_to(ROOT)}`" for p in raw_missing)
    )
    st.stop()

weather_raw = load_raw_weather()
target_raw  = load_raw_target()

pipeline_ready = all(
    p.exists() for p in (MODELLING_TABLE, PREDICTIONS, MONTHLY_METRICS, EVAL_SUMMARY)
)
horizon_ready = HORIZON_METRICS.exists()
tod_ready = TOD_METRICS.exists()

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")

    freq_label = st.radio("Aggregation", ["15-min", "Hourly", "Daily"], index=1)
    freq_map   = {"15-min": "15min", "Hourly": "h", "Daily": "D"}
    freq       = freq_map[freq_label]

    date_min = target_raw["timestamp"].dt.tz_localize(None).min().date()
    date_max = target_raw["timestamp"].dt.tz_localize(None).max().date()
    date_range = st.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if len(date_range) == 2:
        d0 = pd.Timestamp(date_range[0], tz="UTC")
        d1 = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        d0 = pd.Timestamp(date_min, tz="UTC")
        d1 = pd.Timestamp(date_max, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    st.divider()
    states = sorted(weather_raw["state"].unique())
    selected_states = st.multiselect("States (GHI plot)", states, default=states)

    if pipeline_ready:
        st.divider()
        st.subheader("Model series")
        show_proxy    = st.checkbox("Synthetic proxy", value=False)
        show_baseline = st.checkbox("Baseline (Ridge)", value=True)
        show_lgbm     = st.checkbox("LightGBM", value=True)

# ── Filter raw data ────────────────────────────────────────────────────────

tgt_filt = target_raw[(target_raw["timestamp"] >= d0) & (target_raw["timestamp"] <= d1)].copy()
wx_filt  = weather_raw[
    (weather_raw["timestamp"] >= d0) &
    (weather_raw["timestamp"] <= d1) &
    (weather_raw["state"].isin(selected_states))
].copy()

ghi_national = (
    wx_filt.groupby("timestamp", as_index=False)["ghi_wm2"]
    .mean()
    .rename(columns={"ghi_wm2": "ghi_national"})
)

if freq != "15min":
    tgt_rs = resample_df(tgt_filt, freq)
else:
    tgt_rs = tgt_filt.copy()

if freq != "h":
    ghi_rs_src = ghi_national.copy()
    ghi_rs_src["timestamp"] = pd.to_datetime(ghi_rs_src["timestamp"])
    if freq == "D":
        ghi_rs = resample_df(ghi_rs_src, "D")
    else:
        ghi_rs = ghi_rs_src
else:
    ghi_rs = ghi_national.copy()

# ── Tabs ───────────────────────────────────────────────────────────────────

tabs = ["Raw Data"]
if pipeline_ready:
    tabs += ["Predictions vs Actual", "Horizon Analysis", "Walk-Forward Metrics"]

tab_objects = st.tabs(tabs)
tab_raw = tab_objects[0]

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — Raw Data
# ──────────────────────────────────────────────────────────────────────────

with tab_raw:
    st.markdown("#### Irradiance (GHI) vs Target PV Production")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=ghi_rs["timestamp"],
            y=ghi_rs["ghi_national"],
            name=f"GHI avg ({len(selected_states)} states, W/m²)",
            line=dict(color=C_GHI, width=1.2),
            mode="lines",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=tgt_rs["timestamp"],
            y=tgt_rs["actual_solar_mw"],
            name="Actual PV (MW)",
            line=dict(color=C_ACTUAL, width=1.5),
            mode="lines",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        hovermode="x unified", height=440,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="GHI (W/m²)", secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="Solar generation (MW)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Date (UTC)")
    st.plotly_chart(fig, use_container_width=True)

    if len(selected_states) > 1:
        with st.expander("Per-state GHI series", expanded=False):
            state_fig = go.Figure()
            for state in selected_states:
                s_df = wx_filt[wx_filt["state"] == state].copy()
                if freq == "D":
                    s_df = resample_df(s_df[["timestamp", "ghi_wm2"]], "D")
                state_fig.add_trace(go.Scatter(
                    x=s_df["timestamp"], y=s_df["ghi_wm2"],
                    name=state, mode="lines", line=dict(width=1),
                ))
            state_fig.update_layout(
                xaxis_title="Date (UTC)", yaxis_title="GHI (W/m²)",
                height=380, hovermode="x unified",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(state_fig, use_container_width=True)

    st.divider()
    st.markdown("#### Raw data tables")
    col_tgt, col_wx = st.columns(2)

    with col_tgt:
        st.markdown("**Target: actual solar generation**")
        st.caption(f"{len(tgt_filt):,} rows · 15-min · MW")
        st.dataframe(
            tgt_filt.assign(timestamp=tgt_filt["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC"))
            .reset_index(drop=True),
            use_container_width=True, height=400,
        )
    with col_wx:
        st.markdown("**Weather: hourly by state**")
        st.caption(f"{len(wx_filt):,} rows · hourly · {len(selected_states)} states")
        st.dataframe(
            wx_filt.assign(timestamp=wx_filt["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC"))
            .reset_index(drop=True),
            use_container_width=True, height=400,
        )

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — Predictions vs Actual
# ──────────────────────────────────────────────────────────────────────────

if pipeline_ready:
    tab_pred = tab_objects[1]
    tab_horizon = tab_objects[2]
    tab_wf = tab_objects[3]

    modelling   = load_modelling_table()
    predictions = load_predictions()
    monthly     = load_monthly_metrics()
    summary     = load_eval_summary()

    mod_filt  = modelling[(modelling["timestamp"] >= d0) & (modelling["timestamp"] <= d1)].copy()
    pred_filt = predictions[(predictions["timestamp"] >= d0) & (predictions["timestamp"] <= d1)].copy()

    # Filter to h=1 and h=4 for plots
    has_horizon = "forecast_horizon" in pred_filt.columns
    pred_h1 = pred_filt[pred_filt["forecast_horizon"] == 1].copy() if has_horizon else pred_filt.copy()
    pred_h4 = pred_filt[pred_filt["forecast_horizon"] == 4].copy() if has_horizon else pred_filt.copy()

    mod_rs     = resample_df(mod_filt, freq) if freq != "15min" else mod_filt.copy()
    pred_h1_rs = resample_df(pred_h1, freq) if freq != "15min" else pred_h1.copy()
    pred_h4_rs = resample_df(pred_h4, freq) if freq != "15min" else pred_h4.copy()

    with tab_pred:
        # Compute per-horizon KPIs from predictions data
        def _horizon_kpis(df: pd.DataFrame, h: int) -> dict:
            """Compute MAE/nMAE for a specific horizon step."""
            sub = df[df["forecast_horizon"] == h] if "forecast_horizon" in df.columns else df
            if sub.empty:
                return {"bl_mae": 0, "bl_nmae": 0, "lgb_mae": 0, "lgb_nmae": 0}
            actual = sub["actual_solar_mw"].values
            daytime = actual > 0
            denom = float(actual[daytime].mean()) if daytime.any() else 1.0
            result = {}
            for prefix, col in [("bl", "pred_baseline_mw"), ("lgb", "predicted_solar_mw")]:
                if col in sub.columns:
                    mae = float(abs(actual - sub[col].values).mean())
                    result[f"{prefix}_mae"] = mae
                    result[f"{prefix}_nmae"] = mae / denom if denom > 0 else 0
                else:
                    result[f"{prefix}_mae"] = 0
                    result[f"{prefix}_nmae"] = 0
            return result

        h1 = _horizon_kpis(pred_filt, 1)
        h4 = _horizon_kpis(pred_filt, 4)

        # KPI row: h=1 (15 min ahead)
        st.markdown("##### h=1 (15 min ahead)")
        c0, c1, c2, c3, c4 = st.columns(5)
        c0.metric("Baseline MAE", f"{h1['bl_mae']:.0f} MW")
        c1.metric("Baseline nMAE", f"{h1['bl_nmae']:.3f}")
        c2.metric("LightGBM MAE", f"{h1['lgb_mae']:.0f} MW",
                  delta=f"{h1['lgb_mae'] - h1['bl_mae']:.0f} MW")
        c3.metric("LightGBM nMAE", f"{h1['lgb_nmae']:.3f}",
                  delta=f"{h1['lgb_nmae'] - h1['bl_nmae']:.3f}")
        skill_h1 = 1 - h1["lgb_mae"] / h1["bl_mae"] if h1["bl_mae"] > 0 else 0
        c4.metric("Skill score", f"{skill_h1:.1%}")

        # KPI row: h=4 (1 hour ahead)
        st.markdown("##### h=4 (1 hour ahead)")
        d0_, d1_, d2_, d3_, d4_ = st.columns(5)
        d0_.metric("Baseline MAE", f"{h4['bl_mae']:.0f} MW")
        d1_.metric("Baseline nMAE", f"{h4['bl_nmae']:.3f}")
        d2_.metric("LightGBM MAE", f"{h4['lgb_mae']:.0f} MW",
                   delta=f"{h4['lgb_mae'] - h4['bl_mae']:.0f} MW")
        d3_.metric("LightGBM nMAE", f"{h4['lgb_nmae']:.3f}",
                   delta=f"{h4['lgb_nmae'] - h4['bl_nmae']:.3f}")
        skill_h4 = 1 - h4["lgb_mae"] / h4["bl_mae"] if h4["bl_mae"] > 0 else 0
        d4_.metric("Skill score", f"{skill_h4:.1%}")

        # Time series and scatter for both h=1 and h=4
        C_LGBM_H4 = "#F97316"  # orange variant for h=4
        C_BASELINE_H4 = "#60A5FA"  # lighter blue for h=4

        for h_val, h_label, pred_hrs, pred_h_df, c_lgb, c_bl in [
            (1, "h=1 (15 min ahead)", pred_h1_rs, pred_h1, C_LGBM, C_BASELINE),
            (4, "h=4 (1 hour ahead)", pred_h4_rs, pred_h4, C_LGBM_H4, C_BASELINE_H4),
        ]:
            st.markdown(f"#### Solar generation: actual vs predictions — {h_label}")
            pred_fig = go.Figure()

            if "actual_solar_mw" in mod_rs.columns:
                pred_fig.add_trace(go.Scatter(
                    x=mod_rs["timestamp"], y=mod_rs["actual_solar_mw"],
                    name="Actual (SMARD)", line=dict(color=C_ACTUAL, width=1.5), mode="lines",
                ))
            if show_proxy and "proxy_solar_mw" in mod_rs.columns:
                pred_fig.add_trace(go.Scatter(
                    x=mod_rs["timestamp"], y=mod_rs["proxy_solar_mw"],
                    name="Synthetic proxy", line=dict(color=C_PROXY, width=1, dash="dot"), mode="lines",
                ))
            if len(pred_hrs) > 0:
                if show_baseline and "pred_baseline_mw" in pred_hrs.columns:
                    pred_fig.add_trace(go.Scatter(
                        x=pred_hrs["timestamp"], y=pred_hrs["pred_baseline_mw"],
                        name="Baseline (Ridge)", line=dict(color=c_bl, width=1.2, dash="dash"), mode="lines",
                    ))
                if show_lgbm and "predicted_solar_mw" in pred_hrs.columns:
                    pred_fig.add_trace(go.Scatter(
                        x=pred_hrs["timestamp"], y=pred_hrs["predicted_solar_mw"],
                        name="LightGBM", line=dict(color=c_lgb, width=1.5), mode="lines",
                    ))

            pred_fig.update_layout(
                xaxis_title="Date (UTC)", yaxis_title="Solar generation (MW)",
                hovermode="x unified", height=420,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(pred_fig, use_container_width=True)

            # Scatter
            if len(pred_h_df) > 0:
                st.markdown(f"#### Scatter: predicted vs actual — {h_label}")
                sc_cols = st.columns(2)
                for sc_col, model_col, model_name, color in [
                    (sc_cols[0], "pred_baseline_mw",  "Baseline (Ridge)", c_bl),
                    (sc_cols[1], "predicted_solar_mw", "LightGBM",        c_lgb),
                ]:
                    if model_col not in pred_h_df.columns:
                        continue
                    mask    = pred_h_df["actual_solar_mw"] > 0
                    actual_ = pred_h_df.loc[mask, "actual_solar_mw"]
                    pred_   = pred_h_df.loc[mask, model_col]
                    if pred_.empty:
                        continue
                    max_val = float(max(actual_.max(), pred_.max()))
                    sc_fig = go.Figure()
                    sc_fig.add_trace(go.Scattergl(
                        x=actual_, y=pred_,
                        mode="markers", marker=dict(color=color, size=2, opacity=0.4),
                        name=model_name,
                    ))
                    sc_fig.add_trace(go.Scatter(
                        x=[0, max_val], y=[0, max_val],
                        mode="lines", line=dict(color="gray", dash="dash", width=1),
                        showlegend=False,
                    ))
                    sc_fig.update_layout(
                        title=model_name,
                        xaxis_title="Actual (MW)", yaxis_title="Predicted (MW)",
                        height=350, margin=dict(l=0, r=0, t=40, b=0),
                    )
                    sc_col.plotly_chart(sc_fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 3 — Horizon Analysis
    # ──────────────────────────────────────────────────────────────────────

    with tab_horizon:
        st.markdown("#### Forecast Error by Horizon Step (15 min → 4 h)")
        st.caption(
            "Each step = 15 minutes. Step 1 = 15 min ahead, step 16 = 4 hours ahead. "
            "Error should increase with horizon as origin features become staler."
        )

        if horizon_ready:
            horizon = load_horizon_metrics()

            model_colors = {"baseline": C_BASELINE, "lightgbm": C_LGBM}
            model_labels = {"baseline": "Baseline (Ridge)", "lightgbm": "LightGBM"}

            # MAE by horizon
            st.markdown("##### MAE by horizon step")
            mae_fig = go.Figure()
            for model in horizon["model"].unique():
                sub = horizon[horizon["model"] == model].sort_values("horizon_minutes")
                mae_fig.add_trace(go.Scatter(
                    x=sub["horizon_minutes"], y=sub["mae_mw"],
                    mode="lines+markers",
                    name=model_labels.get(model, model),
                    line=dict(color=model_colors.get(model, "#888"), width=2),
                    marker=dict(size=6),
                ))
            mae_fig.update_layout(
                xaxis_title="Forecast horizon (minutes)",
                yaxis_title="MAE (MW)",
                height=380, hovermode="x unified",
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(mae_fig, use_container_width=True)

            # nMAE by horizon
            st.markdown("##### nMAE by horizon step")
            nmae_fig = go.Figure()
            for model in horizon["model"].unique():
                sub = horizon[horizon["model"] == model].sort_values("horizon_minutes")
                nmae_fig.add_trace(go.Scatter(
                    x=sub["horizon_minutes"], y=sub["nmae"],
                    mode="lines+markers",
                    name=model_labels.get(model, model),
                    line=dict(color=model_colors.get(model, "#888"), width=2),
                    marker=dict(size=6),
                ))
            nmae_fig.update_layout(
                xaxis_title="Forecast horizon (minutes)",
                yaxis_title="nMAE",
                height=380, hovermode="x unified",
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(nmae_fig, use_container_width=True)

            # Horizon metrics table
            st.markdown("##### Horizon metrics table")
            display_cols = ["forecast_horizon", "horizon_minutes", "model", "mae_mw", "rmse_mw", "nmae", "n_samples"]
            display_cols = [c for c in display_cols if c in horizon.columns]
            st.dataframe(
                horizon[display_cols].sort_values(["model", "forecast_horizon"]),
                use_container_width=True, height=400,
            )
        else:
            st.info("Run the pipeline to generate horizon metrics.")

        # Time-of-day metrics
        if tod_ready:
            st.divider()
            st.markdown("#### nMAE by Time of Day (UTC)")
            st.caption("Aggregated across all months and horizons. Daytime hours only.")
            tod = load_tod_metrics()

            tod_fig = go.Figure()
            model_colors_tod = {"baseline": C_BASELINE, "lightgbm": C_LGBM}
            model_labels_tod = {"baseline": "Baseline (Ridge)", "lightgbm": "LightGBM"}
            for model in tod["model"].unique():
                sub = tod[(tod["model"] == model) & (tod["nmae"].notna()) & (tod["nmae"] < 10)]
                sub = sub.sort_values("hour")
                tod_fig.add_trace(go.Scatter(
                    x=sub["hour"], y=sub["nmae"],
                    mode="lines+markers",
                    name=model_labels_tod.get(model, model),
                    line=dict(color=model_colors_tod.get(model, "#888"), width=2),
                    marker=dict(size=5),
                ))
            tod_fig.update_layout(
                xaxis_title="Hour of day (UTC)", yaxis_title="nMAE",
                height=350, hovermode="x unified",
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(dtick=2),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(tod_fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 4 — Walk-Forward Monthly Metrics
    # ──────────────────────────────────────────────────────────────────────

    with tab_wf:
        st.markdown("#### Walk-Forward Monthly Evaluation (12 Rounds)")
        st.caption(
            "Each round: retrain on all data up to test month, predict next month. "
            "Frozen = Round 1 model (trained on 2024 only) applied to all months."
        )

        model_colors_wf = {
            "baseline": C_BASELINE,
            "lightgbm": C_LGBM,
            "frozen_lightgbm": C_FROZEN,
        }
        model_labels_wf = {
            "baseline": "Baseline (Ridge)",
            "lightgbm": "LightGBM (retrained)",
            "frozen_lightgbm": "LightGBM (frozen, Round 1)",
        }

        if not monthly.empty:
            # nMAE trend by round
            st.markdown("##### Monthly nMAE — retrained vs frozen")
            trend_fig = go.Figure()
            for model in monthly["model"].unique():
                grp = monthly[monthly["model"] == model].sort_values("round")
                trend_fig.add_trace(go.Scatter(
                    x=grp["round"], y=grp["nmae"],
                    mode="lines+markers",
                    name=model_labels_wf.get(model, model),
                    line=dict(color=model_colors_wf.get(model, "#888"), width=2),
                    marker=dict(size=6),
                ))
            trend_fig.update_layout(
                xaxis_title="Walk-forward round (test month)",
                yaxis_title="nMAE",
                height=400, hovermode="x unified",
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(dtick=1, tickvals=list(range(1, 13)),
                           ticktext=[f"M{m}" for m in range(1, 13)]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(trend_fig, use_container_width=True)

            # MAE bar chart by round
            st.markdown("##### Monthly MAE (MW)")
            mae_bar = go.Figure()
            for model in monthly["model"].unique():
                grp = monthly[monthly["model"] == model].sort_values("round")
                mae_bar.add_trace(go.Bar(
                    x=[f"M{int(r)}" for r in grp["round"]],
                    y=grp["mae_mw"],
                    name=model_labels_wf.get(model, model),
                    marker_color=model_colors_wf.get(model, "#888"),
                ))
            mae_bar.update_layout(
                barmode="group",
                xaxis_title="Test month", yaxis_title="MAE (MW)",
                height=380, margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(mae_bar, use_container_width=True)

            # Retrain benefit per month
            lgb_m = monthly[monthly["model"] == "lightgbm"].set_index("round")
            frozen_m_df = monthly[monthly["model"] == "frozen_lightgbm"].set_index("round")
            if not frozen_m_df.empty:
                st.markdown("##### Retrain benefit per month (nMAE reduction vs frozen)")
                common_rounds = sorted(set(lgb_m.index) & set(frozen_m_df.index))
                if common_rounds:
                    benefit = []
                    for r in common_rounds:
                        b = frozen_m_df.loc[r, "nmae"] - lgb_m.loc[r, "nmae"]
                        benefit.append({"round": r, "nmae_reduction": b})
                    ben_df = pd.DataFrame(benefit)
                    ben_fig = go.Figure()
                    colors = ["#22C55E" if v >= 0 else "#EF4444" for v in ben_df["nmae_reduction"]]
                    ben_fig.add_trace(go.Bar(
                        x=[f"M{int(r)}" for r in ben_df["round"]],
                        y=ben_df["nmae_reduction"],
                        marker_color=colors,
                        name="Retrain benefit",
                    ))
                    ben_fig.update_layout(
                        xaxis_title="Test month",
                        yaxis_title="nMAE reduction (frozen − retrained)",
                        height=320, margin=dict(l=0, r=0, t=20, b=0),
                    )
                    ben_fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(ben_fig, use_container_width=True)

            st.divider()
            st.markdown("#### Monthly metrics table")
            st.dataframe(monthly, use_container_width=True)
