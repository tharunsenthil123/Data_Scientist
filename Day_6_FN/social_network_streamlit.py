import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("social_network_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("🔮 Machine Learning Prediction App")
st.write("Enter given Details to get prediction:")

# -------- INPUT FIELDS (EDIT AS PER YOUR MODEL) --------
feature1 = st.number_input("Age",
                             min_value=18,
                             max_value=80,
                             value=18,
                             step=1)
feature2 = st.number_input("Salary", value=0.0)

# Combine inputs
input_data = np.array([[feature1, feature2]])

# -------- PREDICTION --------
if st.button("Predict"):
    prediction = model.predict(input_data)

    st.success(f"✅ Prediction: {prediction[0]}")
