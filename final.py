import streamlit as st
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the pre-trained model from the .sav file
model = pickle.load(open('finalized_model.sav', 'rb'))

# Title of the Streamlit app
st.title("Heart Disease Prediction Model")

# Collect user inputs
st.header("Enter the Patient's Details:")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
serum_cholestrol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
rest_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
num_major_vessels = st.slider("Number of Major Vessels (0-3) Colored by Fluoroscopy", 0, 3, 0)
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Converting user inputs into the format 
sex = 1 if sex == "Male" else 0
fasting_blood_sugar = 1 if fasting_blood_sugar == "Yes" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0

# Mapping categorical inputs to numerical values
chest_pain_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-Anginal Pain": 2,
    "Asymptomatic": 3
}
rest_ecg_mapping = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
thal_mapping = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# Converting categorical inputs to numerical values using mappings
chest_pain_type = chest_pain_mapping[chest_pain_type]
rest_ecg = rest_ecg_mapping[rest_ecg]
slope = slope_mapping[slope]
thal = thal_mapping[thal]

# input data for prediction
input_data = (age, sex, chest_pain_type, resting_bp, serum_cholestrol, fasting_blood_sugar,
              rest_ecg, max_heart_rate, exercise_angina, st_depression, slope,
              num_major_vessels, thal)

# Convert input data to numpy array and reshape for the model
input_data_array = np.asarray(input_data).reshape(1, -1)

# Button for making predictions
if st.button("Test Results"):
    prediction = model.predict(input_data_array)
    
    # Display the result
    if prediction[0] == 1:
        st.error("The patient is at risk of heart disease.")
    else:
        st.success("The patient is not at risk of heart disease.")
