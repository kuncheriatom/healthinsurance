import streamlit as st
from streamlit_option_menu import option_menu
import urllib.request
import joblib 

# Load the model using joblib
# bp_model = joblib.load('E:\\healthinsurance-dev\\healthinsurance\\notebook\\bp_model.sav')
# Specify the raw GitHub content URL of the model file
url_depression = 'https://raw.githubusercontent.com/kuncheriatom/healthinsurance/dev/model/depression_model.sav'
filename_dep = 'depression_model.sav'

# Download the file from the URL
urllib.request.urlretrieve(url_depression, filename_dep)

# Load the .sav file using joblib
try:
    depression_model = joblib.load(filename_dep)
    print("Depression Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

url_bp = 'https://raw.githubusercontent.com/kuncheriatom/healthinsurance/dev/model/bp_model.sav'
filename_bp = 'bp_model.sav'
urllib.request.urlretrieve(url_bp, filename_bp)
try:
    bp_model = joblib.load(filename_bp)
    print("Bp Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

url_arth = 'https://raw.githubusercontent.com/kuncheriatom/healthinsurance/dev/model/arthritis_model.sav'
filename_art = 'arthritis_model.sav'
urllib.request.urlretrieve(url_arth, filename_art)
try:
    arthritis_model = joblib.load(filename_art)
    print("Arthritis Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

# Replace the URL with the raw URL of your model file on GitHub
# bp_model = joblib.load('C:\\Users\\sachu\\Desktop\\Project\\devcode\\healthinsurance\\src\\ModelSavFiles\\bp_model.sav')
# depression_model = joblib.load('C:\\Users\\sachu\\Desktop\\Project\\devcode\\healthinsurance\\src\\ModelSavFiles\\depression_model.sav')
# overallhealth_model = joblib.load('C:\\Users\\sachu\\Desktop\\Project\\devcode\\healthinsurance\\src\\ModelSavFiles\\overallhealth_model.sav')
# arthritis_model = joblib.load('C:\\Users\\sachu\\Desktop\\Project\\devcode2\\healthinsurance\\src\\ModelSavFiles\\arthritis_model.sav')

# Download the model file
# response = requests.get(model_url)
# response.raise_for_status()  # Check for HTTP errors

# # Load the model from the downloaded content
# depression_model = joblib.load(BytesIO(response.content))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Bp Prediction', 'Overall Health Prediction', 'Depression Prediction','Arthritis Prediction'],
                          icons=['activity','heart','mind','cancer','health'],
                          default_index=0)

# Diabetes Prediction Page
if selected == "Bp Prediction":
    
    # page title
    st.title("Bp Prediction")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        X_age80 = st.text_input('Age')
        
    with col2:
        maxvo2_ = st.text_input('Maximum Oxygen Consumption')
        
    with col3:
        fc60_ = st.text_input('Frequency of Eating Fruits in a Day')
        
    with col4:
        X_bmi5 = st.text_input('BMI')
        
        
    with col1:
        weight2 = st.text_input('Weight')
        
    with col2:
        X_llcpwt = st.text_input('Alcohol consumption in ml')
        
    with col3:
        X_psu = st.text_input('Cig usage per day')
        
    with col4:
        X_ststr = st.text_input('Stratum Weight')
        
    with col5:
        X_vegesum = st.text_input('Vegetable Consumption Frequency')
        
    with col1:
        drvisits = st.text_input('Doctor Visits')
        
    
    # code for Prediction
    bp_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("BP Test Result"):
        # Modify the feature names accordingly
        bp_prediction = bp_model.predict([[
            float(X_age80), float(maxvo2_), float(fc60_), float(X_bmi5),
            float(weight2), float(X_llcpwt), float(X_psu), float(X_ststr), float(X_vegesum),
            float(drvisits)
        ]])
        
        if bp_prediction[0] == 'Yes':
            bp_diagnosis = "The person is predicted to have BP"
        elif bp_prediction[0] == 'prehypertensive':
            bp_diagnosis = "The person is told borderline or pre-hypertensive"
        elif bp_prediction[0] == 'No':
            bp_diagnosis = "The person is not predicted to have BP"
        else:
             bp_diagnosis = "Unknown"
    
    st.success(bp_diagnosis)
    



        
        
     
# Overall Health Prediction Page
if selected == "Overall Health Prediction":
    
    # page title
    st.title("Overall Health Prediction")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        physhlth = st.text_input('Physical Health')
        
    with col2:
        poorhlth = st.text_input('Poor Health')
        
    with col3:
        menthlth = st.text_input('Mental Health')
        
    with col4:
        X_bmi5 = st.text_input('BMI')
        
    with col5:
        drvisits = st.text_input('Doctor Visits')
        
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        maxvo2_ = st.text_input('Maximum Oxygen Consumption')
        
    with col7:
        fc60_ = st.text_input('Frequency of Eating Fruits in a Day')
        
    with col8:
        X_llcpwt = st.text_input('Weight')
        
    with col9:
        X_ststr = st.text_input('Stress Level')
        
    with col10:
        sleptim1 = st.text_input('Sleep Duration')
        
    col11, col12, col13, col14, col15 = st.columns(5)
    
    with col11:
        X_age80 = st.text_input('Age')
        
    with col12:
        X_drnkmo4 = st.text_input('Monthly Alcohol Consumption')
        
    with col13:
        X_strwt = st.text_input('Strength Training Frequency')
        
    with col14:
        X_vegesum = st.text_input('Vegetable Consumption Frequency')
        
   
    
    # code for Prediction
    overallhealth_diagnosis = ''
    
    # # creating a button for Prediction    
    # if st.button("Overall Health Test Result"):
    #     # Modify the feature names accordingly
    #     overallhealth_prediction = overallhealth_model.predict([[
    #         float(physhlth), float(poorhlth), float(menthlth), float(X_bmi5), float(drvisits),
    #         float(maxvo2_), float(fc60_), float(X_llcpwt), float(X_ststr),
    #         float(sleptim1), float(X_age80), float(X_drnkmo4),  float(X_strwt), float(X_vegesum)
    #     ]])
        
    #     if overallhealth_prediction[0] == 'Good or Better Health':
    #         overallhealth_diagnosis = "The person is predicted to have Overall Good Health"
    #     else:
    #         overallhealth_diagnosis = "The person is predicted to have Overall Health Issues"
        
    # st.success(overallhealth_diagnosis)




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
        
        if depression_prediction[0] == 'Yes':
            depression_diagnosis = "The person is predicted to have Depression"
        else:
            depression_diagnosis = "The person is predicted to not have Depression"
        
    st.success(depression_diagnosis)

    # Arthritis Prediction Page
if selected == "Arthritis Prediction":
    
    # page title
    st.title("Arthritis Prediction")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        X_age80 = st.text_input('Age')
        
    with col2:
        fc60_ = st.text_input('Frequency of Eating Fruits in a Day')
        
    with col3:
        maxvo2_ = st.text_input('Maximum Oxygen Consumption')
        
    with col4:
        physhlth = st.text_input('Physical Health')
        
    with col5:
        X_bmi5 = st.text_input('BMI')
        
    col6, col7, col8, col9, col10 = st.columns(5)
    
    with col6:
        drvisits = st.text_input('Doctor Visits')
        
    with col7:
        poorhlth = st.text_input('Poor Health')
        
    with col8:
        X_llcpwt = st.text_input('Weight')
        
    with col9:
        X_psu = st.text_input('Somek per day')
        
    with col10:
        X_llcpwt2 = st.text_input('Weight for Age')
        
    col11, col12, col13, col14, col15 = st.columns(5)
    
    with col11:
        X_strwt = st.text_input('Strength Training Frequency')
        
    with col12:
        X_wt2rake = st.text_input('Physical Activity')
        
    # code for Prediction
    arthritis_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Arthritis Test Result"):
        # Modify the feature names accordingly
        arthritis_prediction = arthritis_model.predict([[
            float(X_age80), float(fc60_), float(maxvo2_), float(physhlth), float(X_bmi5),
            float(drvisits), float(poorhlth), float(X_llcpwt), float(X_psu),
            float(X_llcpwt2), float(X_strwt), float(X_wt2rake)
        ]])
        
        if arthritis_prediction[0] == 'Yes':
            arthritis_diagnosis = "The person is predicted to have Arthritis"
        else:
            arthritis_diagnosis = "The person is predicted to not have Arthritis"
        
    st.success(arthritis_diagnosis)



