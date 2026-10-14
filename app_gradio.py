"""MarketPulse: Financial Market Regime Intelligence - Gradio Application.

Compatible with Hugging Face Spaces (sdk: gradio) and local execution.
Run locally:   python app_gradio.py
"""
from __future__ import annotations
import os
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from mp.data import clean, load_bundled, load_csv, load_yf_aux, load_yfinance
from mp.inference import BUNDLE_VERSION, DEFAULT_BUNDLE, InferenceError, Predictor
from mp.regimes import CLASSES

# Load trained predictor bundle
try:
    predictor = Predictor.load(DEFAULT_BUNDLE)
    bundle_info = predictor.info()
    model_loaded = True
except Exception as e:
    predictor = None
    bundle_info = {}
    model_loaded = False
    print(f"Warning: Could not load default model bundle: {e}")

# Color mappings for the 6 market regimes
REGIME_COLORS = {
    "Bullish / Low Volatility": "#10B981",    # Emerald
    "Bullish / High Volatility": "#06B6D4",   # Cyan
    "Sideways / Low Volatility": "#38BDF8",   # Sky Blue
    "Sideways / High Volatility": "#A855F7",  # Purple
    "Bearish / Low Volatility": "#F59E0B",    # Amber
    "Bearish / High Volatility": "#EF4444",   # Crimson Red
}

PRESET_DATES = {
    "2018-12-21: Q4 2018 Fed Rate-Hike Selloff": "2018-12-21",
    "2018-02-05: 'Vol-mageddon' VIX Spike": "2018-02-05",
    "2017-10-18: Goldilocks Low-Vol Bull Run": "2017-10-18",
    "2016-01-20: Oil & China Growth Panic": "2016-01-20",
    "2011-08-08: US Credit Rating Downgrade": "2011-08-08",
    "2009-03-09: Financial Crisis Cycle Bottom": "2009-03-09",
    "2008-11-20: Subprime Peak Systemic Panic": "2008-11-20",
    "2008-09-15: Lehman Brothers Collapse": "2008-09-15",
}


def create_regime_badge_html(res: dict) -> str:
    """Generates a stylish dark-themed HTML card summarizing the current regime."""
    cur_regime = res.get("current_regime", "Unknown")
    next_regime = res.get("predicted_next_regime", "Unknown")
    conf = res.get("confidence", 0.0) * 100
    days = res.get("days_in_current_regime", 1)
    as_of = res.get("as_of", "")
    split = res.get("data_split", "test").upper()
    hmm_state = res.get("hmm_state", "State 0")
    color = REGIME_COLORS.get(cur_regime, "#38BDF8")

    low_conf_badge = '<span style="background:#EF4444; color:#FFF; padding:2px 8px; border-radius:12px; font-size:11px; margin-left:8px;">LOW CONFIDENCE</span>' if res.get("low_confidence") else ''

    html = f"""
    <div style="background: linear-gradient(135deg, #111827 0%, #1F2937 100%); border: 1px solid #374151; border-radius: 12px; padding: 20px; color: #F9FAFB; font-family: system-ui, -apple-system, sans-serif; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #374151; padding-bottom: 12px; margin-bottom: 16px;">
            <div>
                <span style="font-size: 13px; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em;">As-Of Date</span>
                <div style="font-size: 18px; font-weight: 700; color: #F3F4F6;">{as_of} <span style="font-size: 12px; background:#374151; padding:2px 8px; border-radius:6px; color:#D1D5DB;">{split} SPLIT</span></div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 13px; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em;">Causal HMM State</span>
                <div style="font-size: 15px; font-weight: 600; color: #60A5FA;">{hmm_state}</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 12px;">
            <div style="background: rgba(17, 24, 39, 0.7); border-left: 4px solid {color}; padding: 12px; border-radius: 6px;">
                <div style="font-size: 12px; color: #9CA3AF; text-transform: uppercase;">Active Market Regime</div>
                <div style="font-size: 18px; font-weight: 800; color: {color}; margin-top: 4px;">{cur_regime}</div>
                <div style="font-size: 12px; color: #D1D5DB; margin-top: 2px;">Persistence: <b>{days}</b> trading days</div>
            </div>
            <div style="background: rgba(17, 24, 39, 0.7); border-left: 4px solid #38BDF8; padding: 12px; border-radius: 6px;">
                <div style="font-size: 12px; color: #9CA3AF; text-transform: uppercase;">Next-Session Forecast (t+1)</div>
                <div style="font-size: 18px; font-weight: 800; color: #38BDF8; margin-top: 4px;">{next_regime}</div>
                <div style="font-size: 12px; color: #D1D5DB; margin-top: 2px;">Calibrated Probability: <b>{conf:.1f}%</b> {low_conf_badge}</div>
            </div>
        </div>
        <div style="font-size: 11px; color: #6B7280; margin-top: 8px;">
            ⚠️ {res.get('disclaimer', 'Educational research output. Not financial advice.')}
        </div>
    </div>
    """
    return html


def plot_next_day_probabilities(probs: dict[str, float]) -> go.Figure:
    """Plots a horizontal bar chart of the calibrated hybrid model's probabilities."""
    labels = list(probs.keys())
    values = [probs[k] * 100 for k in labels]
    colors = [REGIME_COLORS.get(k, "#38BDF8") for k in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1F2937", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Next-Day Regime Probabilities (Calibrated Hybrid Ensemble)", font=dict(size=14, color="#E5E7EB")),
        xaxis=dict(title="Probability (%)", range=[0, max(max(values) * 1.15, 10)], gridcolor="#374151", zerolinecolor="#4B5563", tickfont=dict(color="#9CA3AF"), titlefont=dict(color="#9CA3AF")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#E5E7EB", size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        margin=dict(l=20, r=40, t=40, b=30),
        height=260
    )
    return fig


def plot_transition_risk(trans_dict: dict[str, float]) -> go.Figure:
    """Plots a bar chart showing the risk of regime transition over 5, 10, 20 days."""
    horizons = [f"{h} Days" for h in trans_dict.keys()]
    values = [trans_dict[h] * 100 for h in trans_dict.keys()]
    colors = ["#F59E0B" if v < 40 else "#EF4444" for v in values]

    fig = go.Figure(go.Bar(
        x=horizons,
        y=values,
        marker=dict(color=colors, line=dict(color="#1F2937", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="Horizon: %{x}<br>Transition Risk: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Regime Transition Probability (Horizon Forecast)", font=dict(size=14, color="#E5E7EB")),
        yaxis=dict(title="Probability (%)", range=[0, 105], gridcolor="#374151", zerolinecolor="#4B5563", tickfont=dict(color="#9CA3AF"), titlefont=dict(color="#9CA3AF")),
        xaxis=dict(tickfont=dict(color="#E5E7EB", size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        margin=dict(l=20, r=20, t=40, b=30),
        height=260
    )
    return fig


def plot_price_history(df: pd.DataFrame, as_of: str, regime: str) -> go.Figure:
    """Plots trailing price trajectory leading up to the evaluation date."""
    sub = df.loc[: pd.Timestamp(as_of)].tail(160)
    color = REGIME_COLORS.get(regime, "#38BDF8")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub.index,
        y=sub["close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#60A5FA", width=2),
        fill="tozeroy",
        fillcolor="rgba(96, 165, 250, 0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>"
    ))
    # Mark the latest evaluation day
    fig.add_trace(go.Scatter(
        x=[sub.index[-1]],
        y=[sub["close"].iloc[-1]],
        mode="markers",
        marker=dict(size=10, color=color, line=dict(color="#FFFFFF", width=2)),
        name="As-of Point",
        hovertemplate="As-of: %{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=f"Trailing Price Trajectory (Evaluated As-of {as_of})", font=dict(size=14, color="#E5E7EB")),
        xaxis=dict(gridcolor="#374151", zerolinecolor="#4B5563", tickfont=dict(color="#9CA3AF")),
        yaxis=dict(title="Price ($)", gridcolor="#374151", zerolinecolor="#4B5563", tickfont=dict(color="#9CA3AF"), titlefont=dict(color="#9CA3AF")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=30),
        height=280
    )
    return fig


def run_historical(date_pick: str | None, preset: str | None):
    """Executes inference on bundled S&P 500 dataset as of the selected date."""
    if predictor is None:
        return "<p style='color:red;'>Error: Model bundle not loaded.</p>", None, None, None, {}
    
    target_date = PRESET_DATES.get(preset) if preset else (date_pick or "2018-12-21")
    try:
        raw = load_bundled()
        as_of_ts = pd.Timestamp(target_date)
        if as_of_ts > raw.index[-1]:
            target_date = raw.index[-1].strftime("%Y-%m-%d")
        sub = raw.loc[:target_date]
        if len(sub) < 320:
            return f"<p style='color:red;'>Error: Need at least 320 trading days up to {target_date}.</p>", None, None, None, {}
        
        res = predictor.predict(sub.tail(1500))
        badge = create_regime_badge_html(res)
        fig_prob = plot_next_day_probabilities(res["next_day_regime_probabilities"])
        fig_trans = plot_transition_risk(res["regime_change_probability"])
        fig_price = plot_price_history(raw, res["as_of"], res["current_regime"])
        return badge, fig_prob, fig_trans, fig_price, res
    except Exception as exc:
        return f"<div style='color:#EF4444; padding:12px; background:#1F2937; border-radius:8px;'>Prediction failed: {exc}</div>", None, None, None, {"error": str(exc)}


def run_live_ticker(ticker_symbol: str, years: int):
    """Fetches live daily market data via yfinance and runs the causal regime pipeline."""
    if predictor is None:
        return "<p style='color:red;'>Error: Model bundle not loaded.</p>", None, None, None, {}
    
    symbol = ticker_symbol.strip().upper() or "SPY"
    end_dt = date.today().strftime("%Y-%m-%d")
    start_dt = (date.today() - timedelta(days=int(years * 365.25 + 60))).strftime("%Y-%m-%d")

    try:
        raw = load_yfinance(symbol, start=start_dt, end=end_dt)
        vix = load_yf_aux(start_dt, end_dt)
        if vix is not None and not vix.empty:
            raw = raw.join(vix)
        
        cleaned, q = clean(raw)
        if len(cleaned) < 320:
            return f"<div style='color:#F59E0B; padding:12px; background:#1F2937; border-radius:8px;'>Fetched {len(cleaned)} bars for {symbol}. The model requires at least 320 daily rows for 200-SMA + 60-day trend quantiles. Increase the lookback window.</div>", None, None, None, q

        res = predictor.predict(cleaned)
        badge = create_regime_badge_html(res)
        fig_prob = plot_next_day_probabilities(res["next_day_regime_probabilities"])
        fig_trans = plot_transition_risk(res["regime_change_probability"])
        fig_price = plot_price_history(cleaned, res["as_of"], res["current_regime"])
        return badge, fig_prob, fig_trans, fig_price, res
    except Exception as exc:
        return f"<div style='color:#EF4444; padding:12px; background:#1F2937; border-radius:8px;'>Live download failed for '{symbol}': {exc}</div>", None, None, None, {"error": str(exc)}


def run_csv_upload(file):
    """Parses user-uploaded CSV and runs regime classification."""
    if predictor is None:
        return "<p style='color:red;'>Error: Model bundle not loaded.</p>", None, None, None, {}
    if file is None:
        return "<p style='color:#9CA3AF;'>Please upload a CSV file with Date, Open, High, Low, Close, Volume.</p>", None, None, None, {}

    try:
        content = Path(file.name).read_bytes() if hasattr(file, "name") else file
        raw = load_csv(content)
        cleaned, q = clean(raw)
        if len(cleaned) < 320:
            return f"<div style='color:#F59E0B; padding:12px; background:#1F2937; border-radius:8px;'>Cleaned CSV contains {len(cleaned)} rows. Minimum 320 daily trading rows required.</div>", None, None, None, q

        res = predictor.predict(cleaned)
        badge = create_regime_badge_html(res)
        fig_prob = plot_next_day_probabilities(res["next_day_regime_probabilities"])
        fig_trans = plot_transition_risk(res["regime_change_probability"])
        fig_price = plot_price_history(cleaned, res["as_of"], res["current_regime"])
        return badge, fig_prob, fig_trans, fig_price, res
    except Exception as exc:
        return f"<div style='color:#EF4444; padding:12px; background:#1F2937; border-radius:8px;'>CSV processing error: {exc}</div>", None, None, None, {"error": str(exc)}


def get_benchmarks_df() -> pd.DataFrame:
    """Loads held-out test benchmarks comparing our Calibrated Hybrid against baselines."""
    p = Path(__file__).resolve().parent / "artifacts" / "tables" / "bench_main.csv"
    if p.exists():
        df = pd.read_csv(p)
        cols = ["model", "accuracy", "macro_f1", "log_loss", "brier", "ece"]
        avail = [c for c in cols if c in df.columns]
        sub = df[avail].copy()
        for col in ["accuracy", "macro_f1", "log_loss", "brier", "ece"]:
            if col in sub.columns:
                sub[col] = sub[col].round(3)
        return sub.rename(columns={
            "model": "Model / Baseline",
            "accuracy": "Accuracy (↑)",
            "macro_f1": "Macro F1 (↑)",
            "log_loss": "Log Loss (↓)",
            "brier": "Brier Score (↓)",
            "ece": "ECE Calibration (↓)"
        })
    return pd.DataFrame({"Notice": ["Benchmark table not generated yet. Run train.py."]})


def get_literature_df() -> pd.DataFrame:
    """Loads academic references and methodology mappings."""
    p = Path(__file__).resolve().parent / "artifacts" / "tables" / "bench_literature.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame({"Notice": ["Literature table not found."]})


# -------------------------------------------------------------
# Gradio UI Construction
# -------------------------------------------------------------
custom_css = """
body, .gradio-container { background-color: #0B0F19 !important; color: #F3F4F6 !important; }
.gr-button-primary { background: #2563EB !important; border: none !important; }
.gr-button-primary:hover { background: #1D4ED8 !important; }
.tabs { border-color: #374151 !important; }
"""

with gr.Blocks(title="MarketPulse - Financial Regime Intelligence", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css) as demo:
    gr.Markdown("""
    # 📈 MarketPulse: Financial Market Regime Intelligence
    **Causal, leakage-controlled classification of 6 market regimes and multi-horizon transition prediction.**
    
    *Built with a Temperature-Calibrated Ensemble (Logistic Regression, Random Forest, XGBoost, LightGBM, and Forward-Filtered Gaussian HMM).*
    """)

    with gr.Tabs():
        # --- TAB 1: HISTORICAL S&P 500 SIMULATION ---
        with gr.TabItem("🏛️ Historical Simulation (S&P 500 Benchmark)"):
            gr.Markdown("Inspect market regime classification and forward transition probabilities during historical macro events.")
            with gr.Row():
                with gr.Column(scale=1):
                    preset_dropdown = gr.Dropdown(
                        label="Historical Event Presets",
                        choices=list(PRESET_DATES.keys()),
                        value=list(PRESET_DATES.keys())[0]
                    )
                    date_input = gr.Textbox(
                        label="Or enter custom As-Of Date (YYYY-MM-DD)",
                        value="2018-12-21",
                        placeholder="1995-01-01 to 2018-12-21"
                    )
                    sim_btn = gr.Button("Analyze Historical Regime", variant="primary")
                with gr.Column(scale=2):
                    hist_badge = gr.HTML()
            
            with gr.Row():
                hist_prob_plot = gr.Plot(label="Next-Day Probabilities")
                hist_trans_plot = gr.Plot(label="Transition Risk")
            
            hist_price_plot = gr.Plot(label="Trailing Price History")
            
            with gr.Accordion("Raw JSON Response", open=False):
                hist_json = gr.JSON()

            # Wire up historical tab
            def _preset_change(val):
                return PRESET_DATES.get(val, "2018-12-21")
            
            preset_dropdown.change(fn=_preset_change, inputs=[preset_dropdown], outputs=[date_input])
            sim_btn.click(
                fn=run_historical,
                inputs=[date_input, preset_dropdown],
                outputs=[hist_badge, hist_prob_plot, hist_trans_plot, hist_price_plot, hist_json]
            )

        # --- TAB 2: LIVE TICKER ANALYSIS ---
        with gr.TabItem("🌐 Live Market Ticker (yfinance)"):
            gr.Markdown("Download live daily OHLCV bars for any ticker from Yahoo Finance, compute causal technical & cross-asset covariates, and classify the current regime.")
            with gr.Row():
                with gr.Column(scale=1):
                    ticker_text = gr.Textbox(label="Ticker Symbol", value="SPY", placeholder="e.g. SPY, QQQ, AAPL, GLD, BTC-USD")
                    lookback_slider = gr.Slider(label="Lookback History (Years)", minimum=2, maximum=10, value=4, step=1)
                    live_btn = gr.Button("Run Live Market Intelligence", variant="primary")
                with gr.Column(scale=2):
                    live_badge = gr.HTML()

            with gr.Row():
                live_prob_plot = gr.Plot(label="Next-Day Forecast")
                live_trans_plot = gr.Plot(label="Transition Risk")

            live_price_plot = gr.Plot(label="Live Price Trajectory")

            with gr.Accordion("Raw Output Data", open=False):
                live_json = gr.JSON()

            live_btn.click(
                fn=run_live_ticker,
                inputs=[ticker_text, lookback_slider],
                outputs=[live_badge, live_prob_plot, live_trans_plot, live_price_plot, live_json]
            )

        # --- TAB 3: CUSTOM CSV UPLOAD ---
        with gr.TabItem("📁 Custom OHLCV CSV Upload"):
            gr.Markdown("Upload proprietary or alternative asset daily OHLCV data. Requires at least 320 trading rows.")
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload OHLCV CSV", file_types=[".csv"])
                    csv_btn = gr.Button("Analyze Uploaded Data", variant="primary")
                with gr.Column(scale=2):
                    csv_badge = gr.HTML()

            with gr.Row():
                csv_prob_plot = gr.Plot(label="Next-Day Forecast")
                csv_trans_plot = gr.Plot(label="Transition Risk")

            csv_price_plot = gr.Plot(label="Price Trajectory")

            with gr.Accordion("Inspection Output", open=False):
                csv_json = gr.JSON()

            csv_btn.click(
                fn=run_csv_upload,
                inputs=[file_input],
                outputs=[csv_badge, csv_prob_plot, csv_trans_plot, csv_price_plot, csv_json]
            )

        # --- TAB 4: EMPIRICAL BENCHMARKS & MODEL CARD ---
        with gr.TabItem("📊 Empirical Benchmarks & Architecture"):
            gr.Markdown("""
            ### Rigorous Empirical Evaluation (Held-Out Test Split)
            Evaluated on held-out test split (2013-05 to 2018-12, n=1,420 days, zero test-set tuning).
            * **Key Insight:** Due to strong market regime stickiness, the **Persistence Baseline** scores highest on hard-label accuracy. However, our **Calibrated Hybrid Ensemble** outperforms all literature baselines in probabilistic quality, achieving a **Log-Loss of 0.391** (vs. 0.783 for Persistence) and an Expected Calibration Error (**ECE**) of **0.015** (near-perfect probability reliability).
            """)
            gr.DataFrame(value=get_benchmarks_df(), label="Held-Out Test Set Performance Comparison")

            gr.Markdown("### Academic Literature Cross-Reference")
            gr.DataFrame(value=get_literature_df(), label="Literature Framework Implementation")

    # Initial load trigger for historical tab
    demo.load(
        fn=run_historical,
        inputs=[date_input, preset_dropdown],
        outputs=[hist_badge, hist_prob_plot, hist_trans_plot, hist_price_plot, hist_json]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
