import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
df = pd.read_csv(r'C:\Users\kumar\Downloads/brfss2020.csv', encoding='latin1')

df = pd.read_csv(r'C:\Users\kumar\Downloads/brfss2020.csv')
df.info()
df.head()

df_selected = df[['SEXVAR', '_AGE65YR', 'GENHLTH', '_RFBMI5', '_TOTINDA', '_SMOKER3', 'DRNKANY5', 'DIABETE4', 'CVDCRHD4', 'CVDINFR4']]
df_selected.head()

df_selected.isnull().sum()
df_selected = df_selected.dropna()

df_selected['SEXVAR'].value_counts()
df_selected = df_selected.rename(columns={"SEXVAR":"Gender"})
sns.countplot(x='Gender', data=df_selected)

df_selected['_IMPRACE'].value_counts()
df_selected = df_selected.rename(columns={"_IMPRACE":"Race"})
sns.countplot(x='Race', hue='Gender', data=df_selected)

df_selected['_AGE65YR'].value_counts()
df_selected.drop(df_selected[df_selected['_AGE65YR'] == 3].index, inplace = True)
df_selected = df_selected.rename(columns={"_AGE65YR":"Age"})
sns.countplot(x='Age', hue='Gender', data=df_selected)

df_selected['GENHLTH'].value_counts()
df_selected = df_selected[(df_selected['GENHLTH'] != 7) & (df_selected['GENHLTH'] != 9)]
df_selected = df_selected.rename(columns={"GENHLTH":"General Health Status"})
sns.countplot(x='General Health Status', hue='Gender', data=df_selected)

df_selected['_RFBMI5'].value_counts()
df_selected.drop(df_selected[df_selected['_RFBMI5'] == 9].index, inplace = True)
df_selected = df_selected.rename(columns={"_RFBMI5":"Obesity and Overweight Status"})
sns.countplot(x='Obesity and Overweight Status', hue='Gender', data=df_selected)

df_selected['_TOTINDA'].value_counts()
df_selected.drop(df_selected[df_selected['_TOTINDA'] == 9].index, inplace = True)
df_selected = df_selected.rename(columns={"_TOTINDA":"Physical Activity Status"})
sns.countplot(x='Physical Activity Status', hue='Gender', data=df_selected)

df_selected['_SMOKER3'].value_counts()
df_selected.drop(df_selected[df_selected['_SMOKER3'] == 9].index, inplace = True)
chg = {2 : 1, 3 : 2, 4 : 3}
df_selected['_SMOKER3'].replace(to_replace=chg, inplace=True)
df_selected = df_selected.rename(columns={"_SMOKER3" : "Tobacco Usage"})
sns.countplot(x='Tobacco Usage', hue='Gender', data=df_selected)

df_selected['DRNKANY5'].value_counts()
df_selected = df_selected[(df_selected['DRNKANY5'] != 7) & (df_selected['DRNKANY5'] != 9)]
df_selected = df_selected.rename(columns={"DRNKANY5":"Alcohol Usage"})
sns.countplot(x='Alcohol Usage', hue='Gender', data=df_selected)

df_selected['DIABETE4'].value_counts()
df_selected = df_selected[(df_selected['DIABETE4'] != 7) & (df_selected['DIABETE4'] != 9)]
chg = {3 : 2, 4 : 1}
df_selected['DIABETE4'].replace(to_replace=chg, inplace = True)
df_selected = df_selected.rename(columns={"DIABETE4":"Diabetes Status"})
sns.countplot(x='Diabetes Status', hue='Gender', data=df_selected)

df_selected['CVDSTRK3'].value_counts()
df_selected = df_selected[(df_selected['CVDSTRK3'] != 7) & (df_selected['CVDSTRK3'] != 9)]
df_selected = df_selected.rename(columns={"CVDSTRK3" : "Stroke Status"})
sns.countplot(x='Stroke Status', hue='Gender', data=df_selected)

df_selected['CVDCRHD4'].value_counts()
df_selected = df_selected[(df_selected['CVDCRHD4'] != 7) & (df_selected['CVDCRHD4'] != 9)]
df_selected = df_selected.rename(columns={"CVDCRHD4" : "Coronary Heart Disease Status"})
sns.countplot(x='Coronary Heart Disease Status', hue='Gender', data=df_selected)

df_selected['CVDINFR4'].value_counts()
df_selected = df_selected[(df_selected['CVDINFR4'] != 7) & (df_selected['CVDINFR4'] != 9)]
chg = {1 : 1, 2 : 2}
df_selected['CVDINFR4'].replace(to_replace=chg, inplace = True)
df_selected = df_selected.rename(columns={"CVDINFR4" : "Heart Attack Status"})
sns.countplot(x='Heart Attack Status', hue='Gender', data=df_selected)

df_selected.info()
df_selected.head()
df_selected.hist(alpha=0.5, figsize=(12, 12))
plt.show()

df_selected.to_csv('preprocessed_dataset.csv', index=False)

X = df_selected.drop(columns=['Heart Attack Status'])
y = df_selected['Heart Attack Status']  

scaled_X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
X = pd.DataFrame(scaled_X, columns=X.columns)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

param_grid = {'n_neighbors': np.arange(5, 50),'leaf_size': np.arange(30, 50),'weights' : ['uniform', 'distance'], 'metric' : ['euclidean', 'manhattan']}
knn_rscv = RandomizedSearchCV(KNeighborsClassifier(), param_grid, n_iter = 50, verbose = True, cv = 3, n_jobs = -1, scoring = 'accuracy', return_train_score = True)
knn_rscv.fit(X, y)

#Checking top performing n_neighbors value
best_param = knn_rscv.best_params_
print("top performing n_neighbors value", best_param)
#Checking mean score for the top performing value of n_neighbors
knn_rscv.best_score_

knn = KNeighborsClassifier(n_neighbors = best_param['n_neighbors'], weights = best_param['weights'], metric = best_param['metric'], leaf_size = best_param['leaf_size']) 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(X_test)
accuracy = accuracy_score(y_test.values, y_pred)
print(f"Accuracy of model: {accuracy}\n")
print(classification_report(y_test, y_pred))

knn_cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = knn_cm)
disp.plot()


#Manual testing
'''
Paramters

Gender Information. 1 - Male 2 - Female.

Race Information. 1 - White, 2 - Black, 3 - Asian, 4 - American Indian/Alaskan Native, 5 - Hispanic, 6 - Other race

Age of the participants. 1 - Younger Ones 2 - Participants aged 65 and older

General health Information. 1 means excellent 2 - very good 3 - good 4 - fair 5 - poor

BMI Score 1 - BMI score of 25 and below 2 - BMI score of 25 and above.

Physical activity or exercise. 1 - Done in past 30 days 2 - Did not do

Tobacco usage. 1 - smoker 2 - former smoker 3 - non-smoker.

Alcohol Consumption. 1 - Atleast one drink of alcohol in the past 30 days 2- No Alcohol

Diabetes. 1 - Has Diabetes 2 - Not Has Diabetes

Stroke. 1 - Had atleast one time in their life 2- Did not had it.

Heart Disease. 1 - Had coronary heart disease. 2 - Did not Had

'''

result = knn.predict([[2.0,6.0,1.0,5.0,1.0,2.0,3.0,1.0,1.0,1.0,1.0]])
if result==1:
    print("Don't panic, try to consult a cardiolagist")
else:
    print("Congratulations, You are healthy")

