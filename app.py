import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import random

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# -------------------------
# Session State for Dark Mode
# -------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Toggle Button
if st.toggle("🌙 Dark Mode"):
    st.session_state.dark_mode = True
else:
    st.session_state.dark_mode = False

# -------------------------
# CSS Styling
# -------------------------
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #0f1117;
            color: white;
        }
        .stButton>button {
            background: linear-gradient(90deg,#4f46e5,#7c3aed);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6em 1.2em;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg,#4338ca,#6d28d9);
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #111827;
        }
        .stButton>button {
            background: linear-gradient(90deg,#6366f1,#8b5cf6);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6em 1.2em;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg,#4f46e5,#7c3aed);
        }

        /* Download Button Fix */
        button[kind="secondary"] {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
        }

        button[kind="secondary"]:hover {
            background-color: #f3f4f6 !important;
            color: #111827 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# -------------------------
# App Title
# -------------------------
st.title("🩺 Diabetes Risk Prediction App")

st.write("Enter the patient details below:")

# -------------------------
# Input Fields
# -------------------------
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# -------------------------
# Predict Button
# -------------------------
if st.button("🔍 Predict"):

    # Fake probability for demo
    probability = random.uniform(0, 1)

    result = "Diabetic" if probability > 0.5 else "Not Diabetic"

    st.subheader(f"Prediction: {result}")

    # -------------------------
    # Plotly Gauge
    # -------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Diabetes Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if probability > 0.5 else "green"},
        }
    ))

    # Dark / Light Mode Gauge Fix
    if st.session_state.dark_mode:
        fig.update_layout(
            height=300,
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="white")
        )
    else:
        fig.update_layout(
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black")
        )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Download Report
    # -------------------------
    report = pd.DataFrame({
        "Feature": [
            "Pregnancies", "Glucose", "Blood Pressure",
            "Skin Thickness", "Insulin", "BMI",
            "Diabetes Pedigree Function", "Age"
        ],
        "Value": [
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age
        ]
    })

    csv = report.to_csv(index=False)

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="diabetes_report.csv",
        mime="text/csv"
    )