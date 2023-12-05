import streamlit as st
# from streamlit_option_menu import option_menu
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

depression_model = joblib.load('E:\\dev2\\healthinsurance\\notebook\\depression_model.sav')

# page title
if(2==2):
        st.title("Depression Disease Prediction")
        
        col1, col2, col3, col4, col5 = st.columns(5)  
        
        with col1:
            menthlth = st.text_input('Mental Health')
            
        with col2:
            poorhlth = st.text_input('Poor Health')
            
        with col3:
            physhlth = st.text_input('Physical Health')
            
        with col4:
            X_bmi5 = st.text_input('BMI')
            
        with col5:
            drvisits = st.text_input('Doctor Visits')
            
        with col1:
            X_llcpwt2 = st.text_input('Weight')
            
        with col2:
            X_vegesum = st.text_input('Vegetable Consumption Frequency')
            
        with col3:
            fc60_ = st.text_input('Frequency of Eating Fruits in a Day')
            
        with col4:
            maxvo2_ = st.text_input('Maximum Oxygen Consumption')
            
            
        with col2:
            X_impnph = st.text_input('Number of Phones Using')
            
    
        
        
        # code for Prediction
        depression_diagnosis = ''
        
        # creating a button for Prediction    
        if st.button("Depression Test Result"):
            # Modify the feature names accordingly
            depression_prediction = depression_model.predict([[
                float(menthlth), float(poorhlth), float(physhlth), float(X_bmi5), float(drvisits),
                float(X_llcpwt2), float(X_vegesum), float(fc60_), float(maxvo2_),
                float(X_impnph)
            ]])
            
            if depression_prediction[0] == 1:
                depression_diagnosis = "The person is predicted to have Depression"
            else:
                depression_diagnosis = "The person is predicted to not have Depression"
            
        st.success(depression_diagnosis)