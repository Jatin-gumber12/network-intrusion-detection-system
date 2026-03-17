import streamlit as st
import pandas as pd
import joblib

model = joblib.load("../models/intrusion_model.pkl")

st.title("Network Intrusion Detection System")

file = st.file_uploader("Upload network traffic CSV")

if file:
    data = pd.read_csv(file)
    prediction = model.predict(data)
    st.write(prediction)