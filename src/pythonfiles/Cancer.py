#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

file_path = 'C:\\Users\\ashna\\Downloads\\archive (7)\\brfss2013.csv'
df = pd.read_csv(file_path, encoding='latin1')


# In[2]:


df.head()


# In[3]:


df.shape


# In[4]:


# Check and display columns with missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing Values Count for Columns with Missing Values:")
print(missing_values)


# In[5]:


missing_values = df.isnull().sum()
print("Missing Values Count for Each Column:")
pd.set_option('display.max_rows', None)  # Display all rows
print(missing_values)
pd.reset_option('display.max_rows')  # Reset row display options


# In[6]:


# Define a threshold for the maximum number of missing values
threshold = 250000

# Get the columns with missing values counts
missing_values = df.isnull().sum()

# Find columns with more than the specified threshold
columns_to_drop = missing_values[missing_values > threshold].index

# Drop the columns from the dataset
df = df.drop(columns=columns_to_drop)


# In[7]:


df


# In[8]:


missing_values = df.isnull().sum()
print("Missing Values Count for Each Column:")
pd.set_option('display.max_rows', None)  # Display all rows
print(missing_values)
pd.reset_option('display.max_rows')  # Reset row display options


# In[9]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
missing_values_subset = df[df['X_age80'] < 30].isnull().sum()
missing_values_subset= missing_values_subset[missing_values_subset > 0]
print("Missing Values Count for Columns with Missing Values:")
print(missing_values_subset)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[10]:


# Calculate missing values for columns in the subset of rows
missing_values_subset1 = df[(df['X_age80'] > 30) & (df['X_age80'] < 50)].isnull().sum()

# Filter for columns with missing values
missing_values_subset1 = missing_values_subset1[missing_values_subset1 > 0]

# Print the count of missing values for selected columns
print("Missing Values Count for Columns with Missing Values:")
print(missing_values_subset1)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[11]:


# Calculate missing values for columns in the subset of rows
missing_values_subset2 = df[(df['X_age80'] > 50)].isnull().sum()

# Filter for columns with missing values
missing_values_subset2 = missing_values_subset2[missing_values_subset2 > 0]

# Print the count of missing values for selected columns
print("Missing Values Count for Columns with Missing Values:")
print(missing_values_subset2)


# In[12]:


"""dropping columns having high chance of biasing or no contribution for the model"""

columns_to_drop =  ['ctelenum'
		,'pvtresd1'
		,'stateres'
		,'cellfon3'
		,'numadult'
		,'nummen'
		,'numwomen'
		,'numhhol2'
		,'cpdemo1'
		,'exerany2'
		,'exract11'
		,'exract21'
		,'qstver'
		,'qstlang'
		,'mscode'
		,'X_hcvu651'
        ,'X_imprace'
        ]

df = df.drop(columns=columns_to_drop)


# In[13]:


"""converting text column to numerial for convinence"""

replacement_mapping = {
    'Within past year': 1,
    'Within past 2 year': 2,
    'Within past 3 year': 3,
    '5 or more years ago': 5,
    'never': 0
}

columns_to_update = ['checkup1', 'cholchk']

# Fill null values with 0 before applying replacement_mapping
df[columns_to_update] = df[columns_to_update].fillna(0).replace(replacement_mapping)


# In[14]:


"""Using Mean to fill columns"""

# # Define the list of columns to be converted to numeric
columns_to_fill = [				'genhlth'
		,'physhlth'
		,'menthlth'
		,'poorhlth'
		,'checkup1'
		,'sleptim1'
		,'cholchk'
		,'children'
		,'weight2'
		,'height3'
		,'cpdemo4'
		,'alcday5'
		,'fruitju1'
		,'fruit1'
		,'fvbeans'
		,'fvgreen'
		,'fvorang'
		,'vegetab1'
		,'exeroft1'
		,'exerhmm1'
		,'hlthcvrg'
		,'drvisits'
		,'X_strwt'
		,'X_rawrake'
		,'X_wt2rake'
		,'X_llcpwt2'
		,'X_llcpwt'
		,'htin4'
		,'htm4'
		,'wtkg3'
		,'X_bmi5'
		,'ftjuda1_'
		,'frutda1_'
		,'beanday_'
		,'grenday_'
		,'orngday_'
		,'vegeda1_'
		,'X_frutsum'
		,'X_vegesum'
		,'metvl11_'
		,'metvl21_'
		,'maxvo2_'
         ,'fc60_'
		,'padur1_'
		,'pafreq1_'
		,'X_minac11'
		,'X_minac21'
		,'strfreq_'
		,'pamiss1_'
		,'pamin11_'
		,'pamin21_'
		,'pa1min_'
		,'pavig11_'
		,'pavig21_'
		,'pa1vigm_']

# # Convert the specified columns to numeric
df[columns_to_fill] = df[columns_to_fill].apply(pd.to_numeric, errors='coerce')
# Calculate the mean of each column
column_means = df[columns_to_fill].mean()

# Fill NaN values in the specified columns with the mean of each respective column
df[columns_to_fill] = df[columns_to_fill].fillna(column_means)


# In[15]:


columns_to_update = ['hlthpln1', 'persdoc2', 'medcost', 'bloodcho', 'cvdinfr4', 'cvdcrhd4', 'cvdstrk3', 'asthma3', 'chcscncr', 'chcocncr', 'chccopd1', 'havarth3', 'addepev2', 'chckidny', 'diabete3', 'veteran3', 'qlactlm2', 'useequip', 'blind', 'decide', 'diffwalk', 'diffdres', 'diffalon','hlthpln1']

# Update missing values to 'No' in the specified columns for rows where 'X_age80' is less than 30
df.loc[df['X_age80'] < 30, columns_to_update].fillna('No')

# Define the columns to update and their corresponding values
update_values = {
    'marital': 'Never married',
    'employ1': 'Out of work',
    'renthom1': 'Rent'
}


# In[16]:


# Update the missing values in the specified columns where 'X_age80' is less than 30
for column, value in update_values.items():
    df.loc[df['X_age80'] < 30, column] = df.loc[df['X_age80'] < 30, column].fillna(value)

columns_to_update_unknown = [
    'seatbelt', 'flushot6', 'tetanus', 'pneuvac3', 'hivtst6', 'X_impnph',
    'X_prace1', 'X_mrace1', 'X_hispanc', 'X_race', 'X_raceg21', 'X_racegr3',
    'X_race_g1', 'X_ageg5yr', 'X_age65yr', 'X_age_g', 'X_chldcnt', 'X_educag',
    'X_misfrtn', 'X_misvegn', 'X_frtresp', 'X_vegresp', 'X_pacat1', 'X_paindx1',
    'X_pa150r2', 'X_pa300r2', 'X_pa30021', 'X_pastrng', 'X_parec1', 'X_pastae1',
    'X_lmtact1', 'X_lmtwrk1', 'X_lmtscl1', 'X_rfseat2', 'bphigh4', 'marital',
    'educa', 'income2', 'renthom1', 'genhlth', 'persdoc2', 'medcost', 'bloodcho',
    'cvdinfr4', 'cvdcrhd4', 'cvdstrk3'
]

for column in columns_to_update_unknown:
    df[column] = df[column].fillna('unknown')


# In[17]:


import numpy as np

df['asthnow'] = np.where(df['asthma3'] == 'No', 'No', 'Yes')

# Update 'bpmeds' column based on 'bphigh4' condition
df.loc[df['bphigh4'] == 'No', 'bpmeds'] = 'No'
df.loc[df['bphigh4'] != 'No', 'bpmeds'] = 'Yes'

# Update 'asthnow' column based on 'asthma3' condition
df.loc[df['asthma3'] == 'No', 'asthnow'] = 'No'
df.loc[df['asthma3'] != 'No', 'asthnow'] = 'Yes'
# Update "toldhi2" column based on "bloodcho"
df['toldhi2'] = np.where(df['bloodcho'] == 'No', 'Yes', 'No')
# Update 'employ1' column to 'Out of Work' where data has missing values
df.loc[df['employ1'].isnull(), 'employ1'] = 'Out of Work'
# Update "internet" column to 'Yes' where data points are missing
df['internet'].fillna('Yes', inplace=True)
# Update "renthom1" column to 'Rent' where data points are missing
df['renthom1'].fillna('Rent', inplace=True)
missing_values = df.isnull().sum()
print("Missing Values Count for Each Column:")
pd.set_option('display.max_rows', None)  # Display all rows
print(missing_values)
pd.reset_option('display.max_rows')  # Reset row display options


# In[18]:


df.info


# In[19]:


# Filter and get the list of numerical columns
numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()

# Replace missing values in numerical columns with their mean
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Replace missing values in categorical columns with their mode
categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
from scipy.stats import zscore

# 'df' is the DataFrame with numerical columns
numerical_columns = df.select_dtypes(include=[float, int]).columns.tolist()

# Calculating z-scores for each numerical column
z_scores = np.abs(zscore(df[numerical_columns]))

# Defining a threshold for identifying outliers (e.g., z_score_threshold = 3)
z_score_threshold = 3

# Find and remove rows containing outliers
outlier_rows = np.any(z_scores > z_score_threshold, axis=1)
df_no_outliers = df[~outlier_rows]

# Print the number of rows before and after handling outliers
print(f"Number of rows before handling outliers: {len(df)}")
print(f"Number of rows after handling outliers: {len(df_no_outliers)}")


# In[20]:


# Save the preprocessed DataFrame to a new CSV file
#df.to_csv('preprocessed_dataset.csv', index=False)


# In[21]:


df.shape


# In[22]:


missing_values = df.isnull().sum()
print("Missing Values Count for Each Column:")
pd.set_option('display.max_rows', None)  # Display all rows
print(missing_values)
pd.reset_option('display.max_rows')  # Reset row display options


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Define features and target variable
features = [
    "age",
    "smoke100",
    "smokday2",
    "alcday5",
    "genhlth",
    "physhlth",
    "bphigh4",
    "diabete3",
    "cvdinfr4", 
    "cvdcrhd4",
    "cvdstrk3", 
    "asthma3", 
    "chcscncr",  
    "chccopd1",
    "exeroft1", 
    "exerhmm1",
    "fruit1", 
    "vege1",
    "bmi5"
]
target_variable = 'chcocncr'
target = df[target_variable]

# Exclude the target variable from features
features = df.columns.difference([target_variable])

# Identify numeric and non-numeric columns
numeric_features = df[features].select_dtypes(include=np.number).columns
non_numeric_features = df[features].select_dtypes(exclude=np.number).columns

# Handle missing values and scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('non_num', 'drop', non_numeric_features)
    ]
)

features_scaled = preprocessor.fit_transform(df[features])

# Feature Ranking using SelectKBest
selector = SelectKBest(score_func=f_classif, k=15)  # Adjust the value of k as needed
features_selected = selector.fit_transform(features_scaled, target)
selected_feature_indices = selector.get_support(indices=True)
selected_features = df[features].columns[selected_feature_indices]
print("Top selected features:", selected_features)

# correlation_with_target = df[selected_features].apply(lambda x: x.corr(df[target]))
# print("Correlation with Target Variable:\n", correlation_with_target)

# Correlation Heatmap
correlation_matrix = correlation_matrix = pd.DataFrame(features_selected, columns=selected_features).corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Feature Combination
#df['activity_product'] = df['exeroft1'] * df['exerhmm1']
# Train a RandomForestClassifier to see feature importances
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Display feature importances
feature_importances = pd.Series(model.feature_importances_, index=numeric_features)
feature_importances = feature_importances.sort_values(ascending=False)
print("\nFeature Importances:\n", feature_importances)

# Select top k features based on importance
k = 10  # You can adjust this value
top_k_features = feature_importances.head(k).index
print(f"Top {k} features based on importance:", top_k_features)


# In[27]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


# Define features and target
features =[ "age",
    "smoke100",
    "smokday2",
    "alcday5",
    "genhlth",
    "physhlth",
    "bphigh4",
    "diabete3",
    "cvdinfr4", 
    "cvdcrhd4",
    "cvdstrk3", 
    "asthma3", 
    "chcscncr",  
    "chccopd1",
    "exeroft1", 
    "exerhmm1",
    "fruit1", 
    "vege1",
    "bmi5"
    ]
target_variable = 'chcocncr'
target = df[target_variable]

if target_variable not in df.columns:
    raise KeyError(f"Target variable '{target_variable}' not found in DataFrame.")

# Exclude the target variable from features
features = df.columns.difference([target_variable, 'smokday2', 'bmi5', 'vege1', 'age'])

# Identify numeric and non-numeric columns
numeric_features = df[features].select_dtypes(include=np.number).columns
non_numeric_features = df[features].select_dtypes(exclude=np.number).columns


# Handle missing values and scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('non_num', 'drop', non_numeric_features)
    ]
)

# Transform the features
features_scaled = preprocessor.fit_transform(df[features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# Visualize class separability using PCA
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Add the target variable to the reduced features for visualization
features_pca_df = pd.DataFrame(features_pca, columns=['PC1', 'PC2'])
features_pca_df['Target'] = target.values

# Visualize class separability using a scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Target', data=features_pca_df, palette='viridis')
plt.title('Class Separability - PCA')
plt.show()


# In[ ]:




