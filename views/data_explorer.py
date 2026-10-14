import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from mp import ui
from mp.regimes import CLASSES

ctx = ui.get_ctx()
ui.header("Data Explorer", "Inspect market data, engineered features, data quality and exports.", ctx)
R, df, q = ctx.R, ctx.df, ctx.quality
data, T = R["data"], ui.T

st.subheader("Dataset summary")
c = st.columns(6)
for col, (l, v) in zip(c, [("Instrument", ctx.ticker), ("Source", ctx.source), ("First", f"{df.index[0]:%Y-%m-%d}"), ("Last", f"{df.index[-1]:%Y-%m-%d}"), ("Observations", f"{len(df):,}"), ("Missing cells (raw)", sum(q["missing"].values()))]):
    col.metric(l, v)
st.caption(f"Columns: {df.shape[1]} raw, {R['Xb'].shape[1]} engineered, {len(data):,} usable rows after warm-up.")

st.subheader("Raw data")
st.dataframe(df.reset_index().assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")).iloc[::-1], hide_index=True, use_container_width=True, height=280)

st.subheader("Engineered features, labels and model outputs")
extra = pd.DataFrame({"regime_rule_based": R["labels"], "regime_hmm_filtered": R["hmm_state"], "split": R["split"]}).join(
    pd.DataFrame(R["proba"]["Calibrated hybrid"], index=data.index, columns=[f"P({ui.short(c)})" for c in CLASSES]))
for h, e in R["transition"].items():
    if e.get("model") is not None: extra[f"P_change_{h}d"] = e["p_cal"]
full = R["Xa"].join(extra)
grp = st.multiselect("Feature groups", list(R["groups"]) + ["hmm / labels / probabilities"], default=["returns", "volatility"])
show = [c for g in grp if g in R["groups"] for c in R["groups"][g]] + (list(R["Xa"].columns[len(R["Xb"].columns):]) + list(extra.columns) if "hmm / labels / probabilities" in grp else [])
st.dataframe(full[show].iloc[::-1].reset_index() if show else pd.DataFrame(), hide_index=True, use_container_width=True, height=280)

st.subheader("Data quality")
a, b = st.columns(2)
with a:
    st.dataframe(pd.DataFrame({"Check": ["Rows in", "Rows after cleaning", "Duplicate dates (removed)", "Invalid rows (removed)", "Gaps > 5 calendar days", "Volume available"],
                               "Result": [q["rows_in"], q["rows_out"], q["duplicate_dates"], q["invalid_rows"], q["gaps_over_5d"], "yes" if q["has_volume"] else "no (volume features disabled)"]}), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame({"Column": list(q["missing"]), "Missing (raw)": list(q["missing"].values())}), hide_index=True, use_container_width=True)
with b:
    first = R["Xb"].apply(lambda s: s.first_valid_index()); wu = pd.DataFrame({"Feature": R["Xb"].columns, "First valid date": [f"{first[c]:%Y-%m-%d}" if first[c] is not None else "never" for c in R["Xb"].columns]})
    st.dataframe(wu, hide_index=True, use_container_width=True, height=300)
if not q["has_volume"]: ui.warn("No volume data: volume features are unavailable.")
if q["gaps_over_5d"]: ui.warn(f"{q['gaps_over_5d']} gap(s) longer than 5 calendar days in the price history.")
st.caption("Warm-up: rows before the longest rolling window completes are excluded from modelling.")

st.subheader("Visual analysis")
a, b = st.columns(2)
with a:
    corr = R["Xb"].corr()
    ui.show(go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1, colorscale="RdBu_r", hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>")).update_yaxes(autorange="reversed").update_layout(title=dict(text="Feature correlation", font=dict(size=12))), 420, False)
with b:
    f1 = st.selectbox("Feature distribution", list(R["Xb"].columns), index=0)
    ui.show(go.Figure(go.Histogram(x=R["Xb"][f1], nbinsx=60, marker_color=T["accent"])).update_layout(title=dict(text=f"Distribution of {f1}", font=dict(size=12))), 180, False)
    ui.show(go.Figure(go.Histogram(x=data["ret"], nbinsx=80, marker_color=T["accent2"])).update_layout(title=dict(text="Daily log returns", font=dict(size=12))).update_xaxes(tickformat=".1%"), 180, False)
fig = go.Figure([go.Scatter(x=data.index, y=data[c], name=c, line=dict(width=1.2)) for c in R["groups"]["volatility"] if c != "vol_ratio"]); ui.show(fig.update_yaxes(tickformat=".0%").update_layout(title=dict(text="Rolling volatility", font=dict(size=12))), 280)
c1, c2 = st.columns(2); xs, ys = c1.selectbox("Scatter X", list(R["Xb"].columns), index=0), c2.selectbox("Scatter Y", list(R["Xb"].columns), index=min(3, R["Xb"].shape[1] - 1))
sc = R["Xb"].iloc[::3]; ui.show(go.Figure(go.Scattergl(x=sc[xs], y=sc[ys], mode="markers", marker=dict(size=4, color=T["accent"], opacity=.6))).update_xaxes(title=xs).update_yaxes(title=ys), 320, False)

st.subheader("Export")
d = st.columns(4)
d[0].download_button("Cleaned market data", df.to_csv(), f"{ctx.ticker}_clean.csv", "text/csv")
d[1].download_button("Engineered features", R["Xb"].to_csv(), f"{ctx.ticker}_features.csv", "text/csv")
d[2].download_button("Prediction results", extra.to_csv(), f"{ctx.ticker}_predictions.csv", "text/csv")
d[3].download_button("Regime assignments", pd.DataFrame({"rule_based": R["labels"], "hmm_filtered": R["hmm_state"]}).to_csv(), f"{ctx.ticker}_regimes.csv", "text/csv")
