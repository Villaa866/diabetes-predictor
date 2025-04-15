import streamlit as st
import numpy as np
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title and description
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter patient details to predict their risk of having diabetes.")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 0, 200, 120)
blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 33)

# Make prediction
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"The patient is likely to have diabetes. (Risk: {probability:.2%})")
    else:
        st.success(f"The patient is unlikely to have diabetes. (Risk: {probability:.2%})")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Eben Villa | Trained on the Pima Indians Diabetes Dataset")
