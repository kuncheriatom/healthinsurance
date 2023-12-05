import streamlit as st
import joblib
import pandas as pd

# Load the trained models and encoders
heart_attack_model = joblib.load('heart_attack_model.pkl')
angina_model = joblib.load('angina_or_coronary_heart_disease_model.pkl')
stroke_model = joblib.load('stroke_model.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

def predict(input_data):
    # Transform the input data using the loaded OrdinalEncoder
    input_data_encoded = ordinal_encoder.transform([input_data])

    # Make predictions using the respective models
    heart_attack_prob = heart_attack_model.predict_proba(input_data_encoded)[:, 1][0]
    angina_prob = angina_model.predict_proba(input_data_encoded)[:, 1][0]
    stroke_prob = stroke_model.predict_proba(input_data_encoded)[:, 1][0]

    predictions = {
        'Heart Attack': heart_attack_prob,
        'Angina or Coronary Heart Disease': angina_prob,
        'Stroke': stroke_prob
    }

    return predictions

def main():
    st.title('Health Prediction App')
    st.write('Enter the patient details to get predictions.')

    # Collect user input
    user_input = {}
    user_input['bphigh4'] = st.selectbox('High Blood Pressure:', ['Yes', 'No'])
    user_input['toldhi2'] = st.selectbox('Told about High Blood Pressure:', ['Yes', 'No'])
    user_input['chcocncr'] = st.selectbox('Ever told had cancer:', ['Yes', 'No'])
    user_input['havarth3'] = st.selectbox('Ever told had arthritis:', ['Yes', 'No'])
    user_input['addepev2'] = st.selectbox('Ever told had depression:', ['Yes', 'No'])
    user_input['employ1'] = st.selectbox('Employment Status:', ['Employed', 'Unemployed', 'Retired', 'Other'])
    user_input['weight2'] = st.number_input('Weight (in pounds):', min_value=0)
    user_input['height3'] = st.number_input('Height (in inches):', min_value=0)
    user_input['renthom1'] = st.selectbox('Housing Situation:', ['Own', 'Rent', 'Other'])
    user_input['qlactlm2'] = st.selectbox('Physical activity level:', ['Yes', 'No'])
    user_input['diabete3'] = st.selectbox('Ever told had diabetes:', ['Yes', 'No'])
    user_input['smoke100'] = st.selectbox('Smoking Status:', ['Yes', 'No'])
    user_input['asthma3'] = st.selectbox('Ever told had asthma:', ['Yes', 'No'])
    user_input['alcday5'] = st.number_input('Alcohol consumption per day:', min_value=0)
    user_input['sex'] = st.selectbox('Gender:', ['Male', 'Female'])

    # Make prediction on button click
    if st.button('Make Prediction'):
        predictions = predict(user_input)
        st.write('Predictions:')
        st.write(predictions)

if __name__ == '__main__':
    main()
