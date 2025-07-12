import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json

# Set page configuration
st.set_page_config(page_title="CardioPredict MVP", page_icon="🩺", layout="wide")

# Main title
st.title("CardioPredict: Cardiovascular Disease Prediction")

# Introduction
st.markdown("""
Welcome to **CardioPredict**, a tool to assess your risk of cardiovascular disease (CVD) based on personal health metrics. Enter your information below to get a prediction of your CVD risk. Navigate to the 'Data Browser' page to explore the dataset, 'Model Info' to learn about our prediction model, or 'About' to understand the purpose and creator of this app.
""")

# Load config file
try:
    with open('config.json') as f:
        config = json.load(f)
except (FileNotFoundError, json.decoder.JSONDecodeError):
    st.error("Configuration file missing or malformed. Please check `config.json`.")
    st.stop()

# Load pipeline model
model_path = config.get('model_path')
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Input form
st.header("Enter Your Information")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age_years = st.number_input("Age (years)", min_value=18, max_value=100, value=40, step=1,
                                    help="Enter your age in years (will be converted to days internally)")
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=1.0,
                                 help="Your height in centimeters")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=1.0,
                                 help="Your weight in kilograms")
        ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60.0, max_value=250.0, value=120.0,
                                step=1.0, help="Upper (systolic) blood pressure")
        ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40.0, max_value=150.0, value=80.0,
                                step=1.0, help="Lower (diastolic) blood pressure")
    with col2:
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3],
                                   format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
                                   help="Choose cholesterol level")
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3],
                            format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x],
                            help="Choose glucose level")
        smoke = st.selectbox("Smoking Status", options=[0, 1],
                             format_func=lambda x: {0: "Non-Smoker", 1: "Smoker"}[x],
                             help="Do you smoke?")
        alco = st.selectbox("Alcohol Consumption", options=[0, 1],
                            format_func=lambda x: {0: "No", 1: "Yes"}[x],
                            help="Do you regularly consume alcohol?")
        active = st.selectbox("Physical Activity", options=[0, 1],
                              format_func=lambda x: {0: "Inactive", 1: "Active"}[x],
                              help="Do you exercise regularly?")
        gender = st.selectbox("Gender", options=[1, 2],
                              format_func=lambda x: {1: "Female", 2: "Male"}[x],
                              help="Biological sex")
    
    submit = st.form_submit_button("Predict CVD Risk")

# Prediction logic
if submit:
    # Validate inputs
    if ap_hi <= ap_lo:
        st.error("Systolic blood pressure (ap_hi) must be greater than diastolic blood pressure (ap_lo).")
    elif weight / (height / 100) ** 2 < 10 or weight / (height / 100) ** 2 > 60:
        st.error("BMI (calculated from weight and height) must be between 10 and 60.")
    else:
        bmi = weight / (height / 100) ** 2
        age = age_years * 365
        gender_2 = 1 if gender == 2 else 0
        
        # Create input DataFrame with explicit feature order
        input_data = pd.DataFrame({
            'age': [age],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'cholesterol': [cholesterol],
            'gluc': [gluc],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active],
            'bmi': [bmi],
            'gender_2': [1 if gender == 2 else 0]
        })
        
        # Predict probability
        try:
            prob = model.predict_proba(input_data)[0]
            prediction = model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
        
        # Display results
        st.header("Your CVD Risk Prediction")
        st.markdown(f"""
        - **Probability of No CVD (Class 0)**: {prob[0]:.2%}  
        - **Probability of CVD (Class 1)**: {prob[1]:.2%}  
        - **Prediction**: {'🩺 CVD Present' if prediction == 1 else '✅ No CVD'}
        """)
        if prob[1] > 0.7:
            st.warning("⚠️ High risk of CVD. Please consult a healthcare provider.")
        elif prob[1] > 0.3:
            st.info("🟠 Moderate risk. Consider regular check-ups and healthy lifestyle changes.")
        else:
            st.success("🟢 Low risk. Keep maintaining a healthy lifestyle!")