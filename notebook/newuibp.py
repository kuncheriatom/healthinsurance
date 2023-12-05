import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
# import requests
# from io import BytesIO

# Load the model using joblib
# bp_model = joblib.load('E:\\healthinsurance-dev\\healthinsurance\\notebook\\bp_model.sav')

# Replace the URL with the raw URL of your model file on GitHub
depression_model = joblib.load('depression_model.sav')

# Download the model file
# response = requests.get(model_url)
# response.raise_for_status()  # Check for HTTP errors

# # Load the model from the downloaded content
# depression_model = joblib.load(BytesIO(response.content))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Bp Prediction', 'Heart Disease Prediction', 'Depression Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)

# Diabetes Prediction Page
if selected == 'Bp Prediction':
    # page title
    st.title('BP Prediction')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        physhlth = st.text_input('Physical Health')

    with col2:
        menthlth = st.text_input('Mental Health')

    with col3:
        poorhlth = st.text_input('Poor Health')

    with col1:
        bloodcho = st.text_input('Blood Cholesterol')

    with col2:
        cholchk = st.text_input('Cholesterol Check')

    with col3:
        toldhi2 = st.text_input('Told Health Indicator')

    with col1:
        weight2 = st.text_input('Weight')

    with col2:
        height3 = st.text_input('Height')

    with col3:
        sleptim1 = st.text_input('Sleep Time')

    with col1:
        smoke100 = st.text_input('Smoking Status')

    with col2:
        usenow3 = st.text_input('Current Alcohol Use')

    with col3:
        alcday5 = st.text_input('Alcohol Consumption Frequency')

    with col1:
        X_rfchol = st.text_input('Cholesterol Checked')

    with col2:
        X_drdxar1 = st.text_input('Arthritis Diagnosis')

    with col3:
        X_prace1 = st.text_input('Race')

    with col1:
        X_bmi5 = st.text_input('BMI')

    with col2:
        X_bmi5cat = st.text_input('BMI Category')

    with col3:
        X_rfbmi5 = st.text_input('BMI Check')

    with col1:
        X_rfsmok3 = st.text_input('Smoking Status Check')

    with col2:
        drnkany5 = st.text_input('Alcohol Consumption Check')

    with col3:
        X_drnkdy4 = st.text_input('Number of Days Drinking')

    with col1:
        X_drnkmo4 = st.text_input('Number of Months Drinking')

    with col2:
        X_rfdrwm4 = st.text_input('Check for Drinking Weekly')

    with col3:
        X_rfdrhv4 = st.text_input('Check for Heavy Drinking')

    with col1:
        X_frtresp = st.text_input('Fruit Consumption Response')

    with col2:
        X_vegresp = st.text_input('Vegetable Consumption Response')

    with col3:
        X_frutsum = st.text_input('Fruit Consumption Frequency')

    with col1:
        X_vegesum = st.text_input('Vegetable Consumption Frequency')

    with col2:
        X_frtlt1 = st.text_input('Fruit Consumption Less Than 1 Time/Day')

    with col3:
        X_veglt1 = st.text_input('Vegetable Consumption Less Than 1 Time/Day')

    with col1:
        X_frt16 = st.text_input('Fruit Consumption 16 or More Times/Day')

    with col2:
        genhlth = st.text_input('General Health')

    with col3:
        activity_product = st.text_input('Activity Product')
    
    # Make predictions when the user clicks a button
    if st.button('BP Test Result'):
        # Validate input and make predictions
        try:
            input_data = np.array([[
                float(physhlth), float(menthlth), float(poorhlth), bloodcho, float(cholchk),
                toldhi2, float(weight2), float(height3), float(sleptim1), smoke100,
                usenow3, float(alcday5), X_rfchol, X_drdxar1,
                X_prace1, float(X_bmi5), X_bmi5cat, X_rfbmi5, X_rfsmok3,
                drnkany5, float(X_drnkdy4), float(X_drnkmo4), X_rfdrwm4, X_rfdrhv4,
                X_frtresp, X_vegresp, float(X_frutsum), float(X_vegesum), X_frtlt1,
                X_veglt1, X_frt16, genhlth, float(activity_product)
            ]])

            

            # Assuming 'input_features' is the list of feature names provided by the user interface
            input_features = [
                'physhlth', 'menthlth', 'poorhlth', 'bloodcho', 'cholchk',
                'toldhi2', 'weight2', 'height3', 'sleptim1', 'smoke100',
                'usenow3', 'alcday5', 'X_rfhype5', 'X_rfchol', 'X_drdxar1',
                'X_prace1', 'X_bmi5', 'X_bmi5cat', 'X_rfbmi5', 'X_rfsmok3',
                'drnkany5', 'X_drnkdy4', 'X_drnkmo4', 'X_rfdrwm4', 'X_rfdrhv4',
                'X_frtresp', 'X_vegresp', 'X_frutsum', 'X_vegesum', 'X_frtlt1',
                'X_veglt1', 'X_frt16', 'genhlth', 'activity_product'
            ]

            # Identify numeric and non-numeric features based on user input
           # Identify numeric and non-numeric features based on user input
            numeric_features = [feature for feature in input_features if isinstance(input_data[0][input_features.index(feature)], (int, float))]
            non_numeric_features = [feature for feature in input_features if feature not in numeric_features]
            column_trans = make_column_transformer(
                (RobustScaler(), numeric_features),
                (OneHotEncoder(handle_unknown='ignore'), non_numeric_features)
            )


            # Handle missing values and scaling
            # numeric_transformer = Pipeline(steps=[
            #     ('imputer', SimpleImputer(strategy='mean')),
            #     ('scaler', StandardScaler())
            # ])

            # preprocessor = ColumnTransformer(
            #     transformers=[
            #         ('num', numeric_transformer, numeric_features),
            #         ('non_num', 'drop', non_numeric_features)
            #     ]
            # )
            # print(input_features)
          # Transform the numeric features
            numeric_input_transformed = column_trans.named_transformers_['robustscaler'].transform(input_data[:, :len(numeric_features)])

            # Transform the non-numeric features
            non_numeric_input_transformed = column_trans.named_transformers_['onehotencoder'].transform(input_data[:, len(numeric_features):])

            # Combine the transformed features
            input_transformed = np.hstack([numeric_input_transformed, non_numeric_input_transformed])

            # Make predictions
            # prediction = bp_model.predict(input_transformed)

            
            # st.success(f'The predicted class is: {prediction[0]}')

        except ValueError:
            st.error('Please enter valid numeric values for all fields.')
    
    
#     # code for Prediction
#     bp_diagnosis = ''
    
#     # creating a button for Prediction
    
#     if st.button('BP Test Result'):
#         bp_prediction = bp_model.predict([[
#     'physhlth', 'menthlth', 'poorhlth', 'bloodcho', 'cholchk', 'toldhi2', 'weight2',
#     'height3', 'sleptim1', 'smoke100', 'usenow3', 'alcday5', 'X_rfhype5', 'X_rfchol', 'X_drdxar1',
#     'X_prace1', 'X_bmi5', 'X_bmi5cat', 'X_rfbmi5', 'X_rfsmok3', 'drnkany5', 'X_drnkdy4', 'X_drnkmo4',
#     'X_rfdrwm4', 'X_rfdrhv4', 'X_frtresp', 'X_vegresp', 'X_frutsum', 'X_vegesum', 'X_frtlt1', 'X_veglt1',
#     'X_frt16', 'genhlth', 'activity_product'
# ]])
        
#         if (bp_prediction[0] == 1):
#           bp_diagnosis = 'The person is having  bp'
#         else:
#           bp_diagnosis = 'The person is not having bp'
        
    # st.success(bp_diagnosis)




# # Heart Disease Prediction Page
# if (selected == 'Heart Disease Prediction'):
    
#     # page title
#     st.title('Heart Disease Prediction')
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         age = st.text_input('Age')
        
#     with col2:
#         sex = st.text_input('Sex')
        
#     with col3:
#         cp = st.text_input('Chest Pain types')
        
#     with col1:
#         trestbps = st.text_input('Resting Blood Pressure')
        
#     with col2:
#         chol = st.text_input('Serum Cholestoral in mg/dl')
        
#     with col3:
#         fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
#     with col1:
#         restecg = st.text_input('Resting Electrocardiographic results')
        
#     with col2:
#         thalach = st.text_input('Maximum Heart Rate achieved')
        
#     with col3:
#         exang = st.text_input('Exercise Induced Angina')
        
#     with col1:
#         oldpeak = st.text_input('ST depression induced by exercise')
        
#     with col2:
#         slope = st.text_input('Slope of the peak exercise ST segment')
        
#     with col3:
#         ca = st.text_input('Major vessels colored by flourosopy')
        
#     with col1:
#         thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
#     # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        # heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        # if (heart_prediction[0] == 1):
        #   heart_diagnosis = 'The person is having heart disease'
        # else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Depression Prediction Page
# Depression Prediction Page
if selected == "Depression Prediction":
    
    # page title
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
        
    with col5:
        X_cholchk = st.text_input('Cholesterol Check')
        
    with col1:
        X_frt16 = st.text_input('Fruit Consumption 16 or More Times/Day')
        
    with col2:
        X_impnph = st.text_input('Number of Phones Using')
        
    with col3:
        X_incomg = st.text_input('Income Group')
        
    with col4:
        X_pacat1 = st.text_input('Physical Activity Categories')
        
    with col5:
        X_prace1 = st.text_input('Race')
        
    with col1:
        X_rfhype5 = st.text_input('High Blood Pressure Check')
        
    with col2:
        X_rfseat2 = st.text_input('Seatbelt Use Check')
        
    with col3:
        X_rfsmok3 = st.text_input('Smoking Status Check')
        
    with col4:
        X_smoker3 = st.text_input('Smoker Status Check')
        
    with col5:
        X_state = st.text_input('State')
        
    with col1:
        X_veg23 = st.text_input('Frequency of Eating Vegetables in a Day')
    
    
    # code for Prediction
    depression_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Depression Test Result"):
        # Modify the feature names accordingly
        depression_prediction = depression_model.predict([[
            float(menthlth), float(poorhlth), float(physhlth), float(X_bmi5), float(drvisits),
            float(X_llcpwt2), float(X_vegesum), float(fc60_), float(maxvo2_), float(X_cholchk),
            float(X_frt16), float(X_impnph), float(X_incomg), float(X_pacat1), float(X_prace1),
            float(X_rfhype5), float(X_rfseat2), float(X_rfsmok3), float(X_smoker3), float(X_state),
            float(X_veg23)
        ]])
        
        if depression_prediction[0] == 1:
            depression_diagnosis = "The person is predicted to have Depression"
        else:
            depression_diagnosis = "The person is predicted to not have Depression"
        
    st.success(depression_diagnosis)
