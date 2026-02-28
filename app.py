import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="GlucoGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# THEME STATE
# ────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ────────────────────────────────────────────────
# GLOBAL BUTTON STYLING
# ────────────────────────────────────────────────
st.markdown("""
    <style>
        /* Base button style - rounded + shadow */
        div.stButton > button,
        div.element-container[data-testid="stDownloadButton"] button {
            border-radius: 12px !important;
            padding: 0.65rem 1.2rem !important;
            font-weight: 550 !important;
            border: none !important;
            box-shadow: 0 3px 8px rgba(0,0,0,0.14) !important;
            transition: all 0.22s ease !important;
        }
        div.stButton > button:hover,
        div.element-container[data-testid="stDownloadButton"] button:hover {
            box-shadow: 0 6px 14px rgba(0,0,0,0.20) !important;
            transform: translateY(-2px) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# SIDEBAR – THEME TOGGLE
# ────────────────────────────────────────────────
st.sidebar.markdown("### 🌗 Theme Mode")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("☀️ Light", use_container_width=True,
                 type="primary" if not st.session_state.dark_mode else "secondary"):
        st.session_state.dark_mode = False
        st.rerun()

with col2:
    if st.button("🌙 Dark", use_container_width=True,
                 type="primary" if st.session_state.dark_mode else "secondary"):
        st.session_state.dark_mode = True
        st.rerun()

if st.session_state.dark_mode:
    st.sidebar.success("🌙 Dark mode active")
else:
    st.sidebar.info("☀️ Light mode active")

st.sidebar.caption("You can also use ⋮ → Settings → Theme")

# ────────────────────────────────────────────────
# THEME CSS
# ────────────────────────────────────────────────
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        .stApp { background-color: #0f1117 !important; color: #e2e8f0 !important; }
        section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #334155 !important; }
        h1, h2, h3, h4, h5, h6, p, div, span, label { color: #e2e8f0 !important; }
        .stSlider label, .stSlider div { color: #cbd5e1 !important; }
        .stNumberInput input, .stTextInput input, .stSelectbox select {
            background-color: #1e2530 !important; color: #e2e8f0 !important;
            border: 1px solid #475569 !important; border-radius: 10px !important;
        }
        button[kind="primary"], button[kind="secondary"] {
            background-color: #334155 !important; color: white !important;
        }
        button:hover { background-color: #475569 !important; }
        .stAlert, .stSuccess, .stError { background-color: #1e293b !important; color: #e2e8f0 !important; }
        footer, .stCaption { color: #94a3b8 !important; }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
        .stApp { background-color: #f9fafb !important; }
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e5e7eb !important;
        }
        h1, h2, h3, h4, h5, h6, p, div, span, label { color: #111827 !important; }
        .stSlider label, .stSlider div { color: #374151 !important; }

        .stNumberInput input, .stTextInput input, .stSelectbox select {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            border-radius: 10px !important;
        }

        /* Theme toggle buttons */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
            color: white !important;
        }
        div.stButton > button[kind="secondary"] {
            background-color: #f3f4f6 !important;
            color: #374151 !important;
            border: 1px solid #d1d5db !important;
        }

        /* Analyze Risk – red */
        button:contains("Analyze Risk"),
        button[kind="primary"]:has(> div > span > p:contains("Analyze Risk")) {
            background: #dc2626 !important;
            color: white !important;
        }
        button:contains("Analyze Risk"):hover {
            background: #b91c1c !important;
        }

        /* ──────────────────────────────────────────────── */
        /*   Download button - explicit light mode control   */
        /* ──────────────────────────────────────────────── */
        div.element-container[data-testid="stDownloadButton"] button,
        [data-testid="stDownloadButton"] button,
        button[kind="secondary"]:has(> div > span > p:contains("Download Report")),
        button:has(> div > span > p:contains("Download Report (.txt)")) {
            background-color: #f3f4f6 !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
        }

        div.element-container[data-testid="stDownloadButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover,
        button[kind="secondary"]:hover:has(> div > span > p:contains("Download Report")),
        button:hover:has(> div > span > p:contains("Download Report (.txt)")) {
            background-color: #e5e7eb !important;
            color: #111827 !important !important;
            border-color: #9ca3af !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
        }

        div.element-container[data-testid="stDownloadButton"] button:focus,
        div.element-container[data-testid="stDownloadButton"] button:active,
        [data-testid="stDownloadButton"] button:focus,
        [data-testid="stDownloadButton"] button:active,
        button:focus:has(> div > span > p:contains("Download Report")),
        button:active:has(> div > span > p:contains("Download Report")) {
            background-color: #cbd5e1 !important;
            color: #111827 !important;
            border-color: #6b7280 !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
            outline: none !important;
        }

        /* Block Streamlit dark fallback aggressively */
        div.element-container[data-testid="stDownloadButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {
            background-color: #e5e7eb !important;
        }

        /* Plotly gauge light mode fix */
        .js-plotly-plot .plotly .bg,
        .js-plotly-plot .plotly .paper-bg {
            fill: #ffffff !important;
            background: #ffffff !important;
        }
        .js-plotly-plot .plotly text,
        .js-plotly-plot .plotly .gauge-title,
        .js-plotly-plot .plotly .number text {
            fill: #111827 !important;
        }

        .stAlert, .stSuccess, .stError { border-radius: 10px !important; }
        footer, .stCaption { color: #6b7280 !important; }
    </style>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────
# HEADER + LOGO
# ────────────────────────────────────────────────
try:
    logo = Image.open("logo.png")
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(logo, width=110)
    with col2:
        st.title("GlucoGuard AI 🛡️")
        st.markdown("### Intelligent Diabetes Risk Prediction")
        st.markdown("Deep Learning · Accurate · Reliable")
except:
    st.title("GlucoGuard AI 🛡️")
    st.markdown("### Intelligent Diabetes Risk Prediction")

st.markdown("---")

# ────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model("diabetes_ann_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# ────────────────────────────────────────────────
# INPUTS
# ────────────────────────────────────────────────
st.sidebar.header("🧾 Patient Information")

pregnancies    = st.sidebar.slider("Pregnancies",            0, 20,  1)
glucose        = st.sidebar.slider("Glucose (mg/dL)",        0, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)",    0, 100, 20)
insulin        = st.sidebar.slider("Insulin (mu U/ml)",      0, 900, 80)
bmi            = st.sidebar.slider("BMI",                    0.0, 60.0, 25.0, step=0.1)
dpf            = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
age            = st.sidebar.slider("Age (years)",            10, 100, 30)

input_data = np.array([[pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age]])

input_data_scaled = scaler.transform(input_data)

# ────────────────────────────────────────────────
# PREDICTION
# ────────────────────────────────────────────────
if st.button("🔍 Analyze Risk", type="primary", use_container_width=True):

    prob = model.predict(input_data_scaled, verbose=0)[0][0]
    risk_pct = float(prob) * 100
    high_risk = prob > 0.5

    st.subheader("📊 Risk Assessment")

    if high_risk:
        st.error(f"**High Risk** of Diabetes — {risk_pct:.1f}%")
        level = "High Risk"
    else:
        st.success(f"**Low Risk** of Diabetes — {risk_pct:.1f}%")
        level = "Low Risk"

    st.markdown("#### Main Contributing Factors")
    factors = []
    if glucose >= 126:      factors.append("Elevated glucose")
    if bmi >= 30:           factors.append("High BMI")
    if age > 45:            factors.append("Age > 45")
    if dpf > 0.9:           factors.append("Strong family history")

    if factors:
        st.write(" • " + "\n • ".join(factors))
    else:
        st.write("No major alerting factors detected.")

    st.markdown("#### Risk Level Gauge")
    color = "#22c55e" if risk_pct < 40 else "#f59e0b" if risk_pct < 70 else "#ef4444"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk (%)"},
        number = {'font': {'size': 42}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'threshold': {
                'line': {'color': "red", 'width': 5},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20,r=20,t=40,b=20),
        paper_bgcolor = "#ffffff" if not st.session_state.dark_mode else "#0f1117",
        font = dict(color = "#111827" if not st.session_state.dark_mode else "#e2e8f0")
    )

    st.plotly_chart(fig, use_container_width=True, theme=None)

    report = f"""GLUCOGUARD AI REPORT
=============================
Date:               February 2026
Risk probability:   {risk_pct:.1f}%
Risk classification: {level}

Patient data:
  • Pregnancies:            {pregnancies}
  • Glucose:                {glucose} mg/dL
  • Blood Pressure:         {blood_pressure} mm Hg
  • Skin Thickness:         {skin_thickness} mm
  • Insulin:                {insulin} mu U/ml
  • BMI:                    {bmi:.1f}
  • Diabetes Pedigree:      {dpf:.2f}
  • Age:                    {age} years

Generated by GlucoGuard AI (TensorFlow + Streamlit)
"""

    st.download_button(
        label="📄 Download Report (.txt)",
        data=report,
        file_name="GlucoGuard_Report.txt",
        mime="text/plain",
        use_container_width=True
    )

# ────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2026 GlucoGuard AI  •  Built with Streamlit & TensorFlow")