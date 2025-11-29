# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction", layout="centered")

model = joblib.load("house_price_model.joblib")

st.title("🏡 California House Price Predictor")

st.write("""
This model predicts median house value in **$100,000 units**  
Actual price ≈ prediction × 100,000
""")

inputs = {}

feature_list = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

defaults = [5, 20, 4, 1, 800, 3, 35, -120]

for f, d in zip(feature_list, defaults):
    inputs[f] = st.number_input(f, value=d)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    pred = model.predict(df)[0]
    st.success(f"Predicted Value → ${pred * 100000:,.2f}")
