import streamlit as st
import joblib
import numpy as np

model = joblib.load('diabetes_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=0)


input_data = np.array([[pregnancies, glucose, skin_thickness, bmi, age]])
input_data_scaled = scaler.transform(input_data)


if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.success("The model predicts: Diabetic")
    else:
        st.success("The model predicts: Not Diabetic")
