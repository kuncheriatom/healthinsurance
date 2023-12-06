import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Specify the correct file path
file_path = 'C:/Lambton/Term 2/BDM 3014 - Introduction to Artificial Intelligence/Project/archive (5)/brfss2013.csv'

# Read the CSV file
df = pd.read_csv(file_path, encoding='latin1')

print(df.head())
print(df.shape)

# Select columns and preprocessing
selected_columns = ['physhlth', 'menthlth', 'poorhlth', 'sleptim1', 'weight2', 'height3', 'sex', 'bpmeds',
                    'cvdinfr4', 'cvdstrk3', 'asthma3', 'chckidny', 'diabete3', 'smoke100', 'usenow3', 'alcday5',
                    'fruit1', 'exerany2', 'genhlth']
df = df[selected_columns].drop_duplicates()
df.info()

# Define a threshold for the maximum number of missing values
threshold = 250000

# Get the columns with missing values counts
missing_values = df.isnull().sum()

# Find columns with more than the specified threshold
columns_to_drop = missing_values[missing_values > threshold].index

# Drop the columns from the dataset
df = df.drop(columns=columns_to_drop)

# Now, data will contain only columns with fewer than 250,000 missing values.

# Function to check if a value is numeric
def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Convert object-type columns to int or float based on numeric content in at least one row
for column in df.select_dtypes(include='object').columns:
    # Check if any row in the column has purely numeric values
    if any(df[column].apply(is_numeric)):
        # Convert to float
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # Convert to int if there are no decimal values, otherwise to float
        df[column] = df[column].astype(int) if all(pd.notna(value) and value.is_integer() for value in df[column]) else df[column].astype(float)

# Visualization for EDA
def plot_distributions(df, columns):
    for col in columns:
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

# Get the column names from the DataFrame
all_columns = df.columns

# Example: plot distributions for all columns in the DataFrame
plot_distributions(df, all_columns)

# Heatmap for all columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[all_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Pairplot for all columns
sns.pairplot(df[all_columns])
plt.show()

# Calculate BMI
df['bmi'] = df['weight2'] / (df['height3'] / 100) ** 2

# Specify the input features and output feature
input_features = ['physhlth', 'menthlth', 'poorhlth', 'sleptim1', 'weight2', 'height3', 'sex',
                  'cvdinfr4', 'cvdstrk3', 'asthma3', 'chckidny', 'diabete3', 'smoke100', 'usenow3', 'alcday5',
                  'fruit1', 'exerany2']
output_feature = 'genhlth'

# Separating the target variable and features
X = df[input_features]
y = df[output_feature]

# Label Encoding for the target variable 'genhlth'
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing and model pipeline
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestClassifier(n_estimators=100, random_state=0)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Training the model
clf.fit(X_train, y_train)

# Now call the modified function with the appropriate number of classes
classes = np.unique(y_train)  # Assuming y_train contains all possible classes
evaluate_model_multiclass(clf, X_test, y_test, classes)

# Cross-Validation
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-Validation Scores:", scores)

# Feature Importances
feature_importances = clf.named_steps['model'].feature_importances_

# Generate feature names after OneHotEncoder
onehot_features = clf.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(
    categorical_cols)

# Combine with numerical feature names
all_features = np.concatenate([numerical_cols, onehot_features])

# Plotting feature importances
if len(feature_importances) == len(all_features):
    sns.barplot(x=feature_importances, y=all_features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.show()
else:
    print(f"Length mismatch: {len(feature_importances)} feature importances and {len(all_features)} features")
