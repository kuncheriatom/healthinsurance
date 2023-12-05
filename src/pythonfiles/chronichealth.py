import pickle
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import warnings

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Assume df is your dataframe with relevant columns

# List of chronic conditions
chronic_conditions = ['Heart Attack', 'Angina Or Coronary Heart Disease', 'Stroke']

for condition in chronic_conditions:
    # Encode target variable
    label_encoder = LabelEncoder()
    df[condition] = label_encoder.fit_transform(df[condition])

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Split the data
    X = df.drop([condition], axis=1)  # Features
    y = df[condition]  # Target variable

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print(f"Results for {condition}:")
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Classification Report:")
    print(classification_rep)
    print("\n")

# Define the features for chronic conditions
selected_features_chronic_conditions = ['bphigh4', 'toldhi2', 'chcocncr', 'havarth3', 'addepev2', 'employ1', 'weight2', 'height3', 'renthom1', 'qlactlm2', 'diabete3', 'smoke100', 'asthma3', 'alcday5', 'sex']

# Define the target variables for each chronic condition
chronic_conditions = ['Heart Attack', 'Angina Or Coronary Heart Disease', 'Stroke']

for condition in chronic_conditions:
    print(f"\nResults for {condition}:")

    # Encode the target variable
    label_encoder = LabelEncoder()
    df[condition] = label_encoder.fit_transform(df[condition])

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Separate features and target variable
    X = df[selected_features_chronic_conditions]
    y = df[condition]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform oversampling using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_resampled, y_resampled)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_mat}")
    print(f"Classification Report:\n{classification_rep}")

    # Save the model using pickle
    with open(f'{condition.lower()}_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
