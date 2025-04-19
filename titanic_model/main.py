import streamlit as st
import joblib
import numpy as np
import pandas as pd

data = pd.read_csv('titanic_clean.csv')
model = joblib.load('model.pkl')

st.markdown("<h3 style='text-align: center;padding-top: 0'>Titanic Survival Predictor</h3>", unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        gender_ops = data['Sex'].unique()
        gender = st.selectbox(label='Gender',options=gender_ops)

        pclass_min = data['Pclass'].min()
        pclass_max = data['Pclass'].max()
        p_class = st.number_input(label='PClass',min_value=pclass_min,max_value=pclass_max)

        sibsp_min = data['SibSp'].min()
        sibsp_max = data['SibSp'].max()
        sibsp = st.number_input(label='Siblings/Spouses Aboard',min_value=sibsp_min,max_value=sibsp_max)

        cabin_ops = data['Cabin'].unique()
        cabin = st.selectbox(label='Cabin',options=cabin_ops)

        ticket_no_val = data['ticket_num'].values
        ticket_no = st.selectbox(label='Ticket no.',options=ticket_no_val)
        
    with col2:
        age_min = data['Age'].min()+1
        age_max = data['Age'].max()
        age = st.number_input(label='Age',min_value=age_min,max_value=age_max)

        fare_min = data['Fare'].min()
        fare_max = data['Fare'].max()
        fare = st.number_input(label='Fare',min_value=fare_min,max_value=fare_max)

        parch_min = data['Parch'].min()
        parch_max = data['Parch'].max()
        parch = st.number_input(label='Parents/Children Aboard',min_value=parch_min,max_value=parch_max)

        embarked_ops = data['Embarked'].unique()
        embarked = st.selectbox(label='Embarked',options=embarked_ops)

        ticket_cat_ops = data['ticket_cat'].unique()
        ticket_cap = st.selectbox(label='Ticket Type',options=ticket_cat_ops)
        
    
    cola, colb = st.columns([1, 3])  
    with cola:
        pred_btn = st.button("Predict")
    with colb:
        if pred_btn:
            pred_ins = [p_class,gender,age,sibsp,parch,fare,cabin,embarked,ticket_no,ticket_cap]
            model_pred = model.predict([pred_ins])
            message = ''
            if model_pred[0] == 1:
                message = '😊 Survived'
            else:
                message = '😢 Sorry'
            st.markdown(f"<h3 style='text-align: center;'>{message}</h3>", unsafe_allow_html=True)
