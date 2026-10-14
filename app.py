"""MarketPulse - entry point. Run with:  streamlit run app.py"""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="MarketPulse", page_icon=":material/monitoring:", layout="wide", initial_sidebar_state="expanded")
from mp import ui  # noqa: E402

LOGO = str(Path(__file__).parent / "assets" / "logo.svg")
st.logo(LOGO, icon_image=LOGO)
st.markdown(ui.CSS, unsafe_allow_html=True)

pages = [
    st.Page("views/overview.py", title="Market Overview", icon=":material/dashboard:", default=True),
    st.Page("views/regime_explorer.py", title="Regime Explorer", icon=":material/explore:"),
    st.Page("views/transition.py", title="Transition Intelligence", icon=":material/swap_horiz:"),
    st.Page("views/model_lab.py", title="Model Laboratory", icon=":material/science:"),
    st.Page("views/benchmarks.py", title="Benchmarks & Errors", icon=":material/fact_check:"),
    st.Page("views/feature_lab.py", title="Feature Lab", icon=":material/scatter_plot:"),
    st.Page("views/explain.py", title="Explainable AI", icon=":material/psychology:"),
    st.Page("views/data_explorer.py", title="Data Explorer", icon=":material/table_chart:"),
    st.Page("views/settings.py", title="Settings", icon=":material/settings:"),
]
nav = st.navigation(pages)
ui.bootstrap()
ui.sidebar()
nav.run()
