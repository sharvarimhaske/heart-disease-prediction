import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
import os

# Base path automatically setup karne ke liye
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))

st.title("Heart Disease Prediction")

age = st.number_input("Age", min_value=1, max_value=120)

sex = st.selectbox("Sex", ["M", "F"])

chestpain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])

restingbp = st.number_input("Resting BP")

cholesterol = st.number_input("Cholesterol")

fastingbs = st.selectbox("Fasting Blood Sugar", [0,1])

restingecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])

maxhr = st.number_input("Maximum Heart Rate")

exerciseangina = st.selectbox("Exercise Angina", ["Y","N"])

oldpeak = st.number_input("Old Peak")

st_slope = st.selectbox("ST Slope", ["Up","Flat","Down"])

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age":[age],
        "Sex":[sex],
        "ChestPainType":[chestpain],
        "RestingBP":[restingbp],
        "Cholesterol":[cholesterol],
        "FastingBS":[fastingbs],
        "RestingECG":[restingecg],
        "MaxHR":[maxhr],
        "ExerciseAngina":[exerciseangina],
        "Oldpeak":[oldpeak],
        "ST_Slope":[st_slope]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Heart Disease : Yes")
    else:
        st.success("Heart Disease : No")