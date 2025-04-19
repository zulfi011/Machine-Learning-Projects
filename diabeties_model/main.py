import pandas as pd
import numpy as np
import streamlit as st
import joblib 

data = pd.read_csv('preprocessed_data.csv')
model = joblib.load('model.pkl')

st.markdown("<h3 style='text-align: center;padding-top: 0'>Diabeties Classifier</h3>", unsafe_allow_html=True)

with st.container(border=True):
    pregnancies = st.number_input(label='Pregnancies if any',value=0,min_value=0)
    glucose = st.number_input(label='Glucose levels (70 - ≥126)mg/dl',min_value=0)
    bloodpressure = st.number_input(label='BloodPressure mm Hg',min_value=0)
    skin = st.number_input(label=f'Skin Thickness (common: {round(data['SkinThickness'].mean(),2)}) mm',min_value=0)
    inulin = st.number_input(label='Insulin μU/mL ',min_value=0)
    bmi = st.number_input(label='BMI kg/m²',min_value=0)
    age = st.number_input(label='Age',min_value=1)
    button = st.button('Outcome')
    if button:
        inputs = [pregnancies,glucose,bloodpressure,skin,inulin,bmi,age]
        prediction = model.predict([inputs])
        if prediction == 0:
            st.markdown(f"<h3 style='text-align:center;'>Negative, Non-diabetic</h3>",unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align:center;'>Positive, Diabetic</h3>",unsafe_allow_html=True)

