from flask import Flask, render_template, request, redirect, jsonify, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models and encoders
heart_attack_model = joblib.load('heart_attack_model.pkl')
angina_model = joblib.load('angina_or_coronary_heart_disease_model.pkl')
stroke_model = joblib.load('stroke_model.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    input_data = request.get_json()

    print("Input Data:", input_data)  # Add this line for debugging

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

    # Render the result page with predictions
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
