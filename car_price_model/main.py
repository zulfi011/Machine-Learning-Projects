import pandas as pd
import numpy as np
import streamlit as st
import joblib

data = pd.read_csv('car_cleaned_dataset.csv')
car_model = joblib.load('car_model.pkl')

st.markdown("<h3 style='text-align: center;padding-top: 0'>Car Price Predictor</h3>", unsafe_allow_html=True)


with st.container(border=True):
    col1,col2 = st.columns(2)

    with col1:
        brand_ops = data['company'].unique()
        brand = st.selectbox(label='Brand',options=brand_ops)

        year_min = data['year'].min()
        year = st.number_input(label='Year',min_value=year_min)

        km_driven = st.number_input(label='Kms Driven',min_value=0)

    with col2:
        if brand:
            model_sub = data[data['company']==brand]
            model_ops = model_sub['name'].unique().tolist()
            model = st.selectbox(label='Model',options=model_ops)

        if brand and model:
            fuel_sub = data[(data['company']==brand) & (data['name']==model)]
            fuel_ops = fuel_sub['fuel_type'].unique().tolist()
            fuel_type = st.selectbox(label='Fuel Type', options=fuel_ops)

    
    pred_values = [model,brand,year,km_driven,fuel_type]
    prediction = car_model.predict([pred_values])
    
    if prediction <= 0:
        st.markdown(f"<h3 style='text-align:center;'>plz input valid car info</h3>",unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align:center;'>₹ {np.round(prediction[0],2)}</h3>",unsafe_allow_html=True)