import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np

# Load dataset
df = pd.read_csv('/content/Clean_dataset.csv')

# Define features and target for anxiety analysis
anxiety_features = [
    'GenHealth', 'MentalHealth', 'ExerAny2', 'HaveArthritis3', 'AdDepressionEver2'
]

anxiety_target_variable = 'HaveAnxiety'

# Create a new dataframe with relevant columns for anxiety analysis
selected_data_anxiety = df[anxiety_features].copy()

# Create a new column 'HaveAnxiety' based on conditions
selected_data_anxiety['HaveAnxiety'] = 0  # Initialize to 0
selected_data_anxiety.loc[
    (selected_data_anxiety['GenHealth'] == 'Good') |
    (selected_data_anxiety['MentalHealth'] == 'Good') |
    (selected_data_anxiety['ExerAny2'] == 'Yes') |
    (selected_data_anxiety['HaveArthritis3'] == 'Yes') |
    (selected_data_anxiety['AdDepressionEver2'] == 1),
    'HaveAnxiety'
] = 1

# Assuming 'HaveAnxiety' is the target column for anxiety analysis
anxiety_target = selected_data_anxiety[anxiety_target_variable]

import numpy as np

# Identify numeric and non-numeric columns for anxiety analysis
anxiety_numeric_features = df[anxiety_features].select_dtypes(include=np.number).columns
anxiety_non_numeric_features = df[anxiety_features].select_dtypes(exclude=np.number).columns

# Handle missing values and scaling for anxiety analysis
anxiety_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

anxiety_preprocessor = ColumnTransformer(
    transformers=[
        ('num', anxiety_numeric_transformer, anxiety_numeric_features),
        ('non_num', 'drop', anxiety_non_numeric_features)
    ]
)

# Transform the anxiety features
anxiety_features_scaled = anxiety_preprocessor.fit_transform(df[anxiety_features])

# Split the data into training and testing sets for anxiety analysis
X_train_anxiety, X_test_anxiety, y_train_anxiety, y_test_anxiety = train_test_split(
    anxiety_features_scaled, anxiety_target, test_size=0.2, random_state=42
)

# Train a RandomForestClassifier for anxiety analysis
model_anxiety = RandomForestClassifier(random_state=42)
model_anxiety.fit(X_train_anxiety, y_train_anxiety)

# Make predictions on the test set for anxiety analysis
y_pred_anxiety = model_anxiety.predict(X_test_anxiety)

# Evaluate the model for anxiety analysis
accuracy_anxiety = accuracy_score(y_test_anxiety, y_pred_anxiety)
report_anxiety = classification_report(y_test_anxiety, y_pred_anxiety)
conf_matrix_anxiety = confusion_matrix(y_test_anxiety, y_pred_anxiety)

print(f"Accuracy for Anxiety Analysis: {accuracy_anxiety}")
print("\nClassification Report for Anxiety Analysis:\n", report_anxiety)
print("\nConfusion Matrix for Anxiety Analysis:\n", conf_matrix_anxiety)

# Assuming X_train, X_test, y_train, y_test are your original data split
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_anxiety, y_train_anxiety)

# Assuming you have a RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# we have a trained model (RandomForestClassifier)
cv_scores = cross_val_score(model_anxiety, X_train_resampled, y_train_resampled, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Visualize class separability using PCA for anxiety analysis
pca_anxiety = PCA(n_components=min(anxiety_features_scaled.shape[0], anxiety_features_scaled.shape[1]))
features_pca_anxiety = pca_anxiety.fit_transform(anxiety_features_scaled)

# Create a DataFrame with the reduced features and target for visualization
features_pca_df_anxiety = pd.DataFrame(features_pca_anxiety, columns=[f'PC{i}' for i in range(1, features_pca_anxiety.shape[1] + 1)])
features_pca_df_anxiety['Target'] = anxiety_target.values

# Visualize class separability using a scatter plot for anxiety analysis
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC1', hue='Target', data=features_pca_df_anxiety, palette='viridis')
plt.title('Class Separability for Anxiety Analysis - PCA')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 1 (PC1)')
plt.show()
