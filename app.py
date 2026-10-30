import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="HeartCare | AI Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f0f7ff, #ffffff);
}

.main .block-container {
    padding: 2rem 3rem;
}

.header {
    background: linear-gradient(120deg, #123b70, #168aad);
    padding: 35px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.header h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}

.header p {
    color: #e3f2fd;
    font-size: 16px;
}

.section {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #168aad;
    margin-bottom: 15px;
}

.stButton > button {
    background: linear-gradient(90deg, #1261a0, #168aad);
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background: #0b456b;
    color: white;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## ❤️ HeartCare AI")
    st.write("Smart Healthcare Prediction System")

    st.divider()

    st.markdown("### About Project")
    st.write(
        "This project uses Machine Learning "
        "to analyze health-related input data "
        "and generate a heart disease prediction."
    )

    st.divider()

    st.markdown("### Features")
    st.write("✔ Patient Information")
    st.write("✔ Health Parameters")
    st.write("✔ ML-Based Prediction")
    st.write("✔ Instant Results")

    st.divider()

    st.caption("Developed using Python & Streamlit")

# ---------------- HEADER ----------------

st.markdown("""
<div class="header">
    <h1>❤️ Heart Disease Prediction</h1>
    <p>AI-Powered Healthcare Analysis System</p>
    <p>Enter the required health details to get a model prediction.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PATIENT DETAILS ----------------

st.markdown("## 👤 Patient Information")

st.markdown('<div class="section">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (Years)", min_value=1, max_value=120, value=25
    )

with col2:
    sex = st.selectbox("Sex", ["M", "F"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HEALTH PARAMETERS ----------------

st.markdown("## 🩺 Health Parameters")

col1, col2 = st.columns(2)

with col1:
    chestpain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    restingbp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=0,
        max_value=300,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0,
        max_value=1000,
        value=200
    )

    fastingbs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        format_func=lambda x: "Normal (0)" if x == 0 else "Above 120 mg/dL (1)"
    )

    restingecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

with col2:
    maxhr = st.number_input(
        "Maximum Heart Rate",
        min_value=0,
        max_value=300,
        value=150
    )

    exerciseangina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"],
        format_func=lambda x: "No" if x == "N" else "Yes"
    )

    oldpeak = st.number_input(
        "Old Peak",
        min_value=-10.0,
        max_value=20.0,
        value=0.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# ---------------- PREDICTION ----------------

st.markdown("---")
st.markdown("## 🔍 AI Prediction")

st.write("Click below to analyze the entered parameters.")

if st.button("❤️ Predict Heart Disease", use_container_width=True):

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chestpain],
        "RestingBP": [restingbp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fastingbs],
        "RestingECG": [restingecg],
        "MaxHR": [maxhr],
        "ExerciseAngina": [exerciseangina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

    prediction = model.predict(input_data)

    st.markdown("### Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ Model Prediction: Heart Disease Detected")

        st.warning(
            "The model has predicted a possible heart disease pattern. "
            "Please consult a qualified healthcare professional."
        )

    else:

        st.success("✅ Model Prediction: No Heart Disease Detected")

        st.info(
            "The model did not identify a heart disease pattern "
            "in the provided input. This does not guarantee "
            "that heart disease is absent."
        )

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    <hr>
    <p>❤️ HeartCare AI | Machine Learning Project</p>
    <p>Built with Python, Pandas, Joblib & Streamlit</p>
    <p>Disclaimer: This is an educational project, not a medical diagnostic tool.</p>
</div>
""", unsafe_allow_html=True)
