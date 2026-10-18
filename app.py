import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction System")
st.caption("AIC354 – Assignment 2 | SVC trained on the UCI Heart Disease (Cleveland) dataset. "
           "Educational demo – not a medical diagnosis.")

@st.cache_resource
def load_bundle():
    return joblib.load("heart_model.pkl")

bundle = load_bundle()
model, sex_enc, cp_enc = bundle["model"], bundle["sex_encoder"], bundle["cp_encoder"]

age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", list(sex_enc.classes_))
cp = st.selectbox("Chest pain type", list(cp_enc.classes_))
hr = st.slider("Maximum heart rate achieved (bpm)", 60, 210, 150)

if st.button("Predict"):
    row = pd.DataFrame([[age, sex_enc.transform([sex])[0], cp_enc.transform([cp])[0], hr]],
                       columns=["Age", "Sex", "ChestPainType", "MaxHeartRate"])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    if pred == 1:
        st.error(f"⚠️ Heart disease likely (probability {prob:.0%})")
    else:
        st.success(f"✅ Heart disease unlikely (probability of disease {prob:.0%})")
